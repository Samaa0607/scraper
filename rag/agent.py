"""
A simple legal research agent on top of the RAG pipeline.

    question -> agent (Claude + tools) -> decides whether legal retrieval is needed
             -> search / retrieve / get details -> grounded answer + sources

Tools (all read-only, local data only):
  1. search_legislation       - find legislations by name/keyword in SQLite
  2. retrieve_legal_passages  - semantic search over the chunk index (optionally within one legislation)
  3. get_legislation_details  - metadata and summary of one legislation

The loop is a plain, readable tool-use loop limited to AGENT_MAX_STEPS model calls.
Passages are numbered globally across the whole run so the answer can cite them
as [1], [2], ... and the API can return exactly which passages were used.
Without an LLM configured, the agent falls back to the plain RAG pipeline.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from config import AGENT_MAX_STEPS
from database import get_legislation_by_serial, search_legislations
from rag import llm
from rag.rag import (
    GROUNDING_RULES,
    RAGPipeline,
    disclaimer,
    format_sources,
    insufficient_message,
    is_insufficient,
    retrieval_only_result,
)

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are a legal research assistant for Saudi legislation from the Ministry of Justice Laws & Regulations portal (laws.moj.gov.sa). You only know what your tools return; the local database contains a limited set of scraped legislations.

How to work:
- If the message is not a legal question (e.g. a greeting or a question about what you can do), answer briefly without tools.
- For any legal question, use the tools before answering. Typically: retrieve_legal_passages with a focused query; use search_legislation / get_legislation_details when the question names a specific legislation, then retrieve passages within it using its serial.
- Passages returned by retrieve_legal_passages are numbered [n]; cite them with these numbers.
- The legislation list and details tools only help you find the right legislation; legal statements must come from retrieved passages.

Rules for the final answer:
""" + GROUNDING_RULES

TOOLS = [
    {
        "name": "search_legislation",
        "description": (
            "Search the local legislation database by name or keyword (Arabic). Returns matching "
            "legislations with serial, title, type, status and source URL. Use it to find which "
            "legislation a question refers to. Does not return legal text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Legislation name or keyword, e.g. 'نظام الإفلاس'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieve_legal_passages",
        "description": (
            "Semantic search over the full text of the stored legislations. Returns the most relevant "
            "numbered passages (articles) with their legislation title and source URL. Optionally "
            "restrict to one legislation by serial. This is the only source of legal text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, ideally in Arabic"},
                "serial": {"type": "string", "description": "Optional legislation serial to search within"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_legislation_details",
        "description": (
            "Get metadata for one legislation by serial: title, type, status, issue/publication dates, "
            "summary, source URL and size. Does not return the full text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "serial": {"type": "string", "description": "Legislation serial from search results"},
            },
            "required": ["serial"],
        },
    },
]


class LegalAgent:
    def __init__(self, rag: Optional[RAGPipeline] = None, max_steps: int = AGENT_MAX_STEPS):
        self.rag = rag or RAGPipeline()
        self.max_steps = max_steps

    # ------------------------------------------------------------------ tools

    def tool_search_legislation(self, query: str) -> Dict[str, Any]:
        rows = search_legislations(query, limit=10)
        return {
            "results": [
                {
                    "serial": r["serial"],
                    "title": r["title"],
                    "legislation_type": r["legislation_type"],
                    "status": r["status"],
                    "source_url": r["source_url"],
                    "has_content": bool((r["content"] or "").strip()),
                }
                for r in rows
            ]
        }

    def tool_retrieve_legal_passages(self, query: str, serial: Optional[str], passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        chunks = self.rag.retrieve(query, serial=serial or None)
        known = {p["chunk_id"]: i for i, p in enumerate(passages, 1)}
        out = []
        for c in chunks:
            if c["chunk_id"] not in known:
                passages.append(c)
                known[c["chunk_id"]] = len(passages)
            ref = known[c["chunk_id"]]
            out.append({
                "ref": ref,
                "title": c.get("title"),
                "chapter": c.get("chapter"),
                "heading": c.get("heading"),
                "source_url": c.get("source_url"),
                "serial": c.get("serial"),
                "text": c.get("text"),
            })
        if not out:
            return {"passages": [], "note": "No sufficiently relevant passages found."}
        return {"passages": out}

    def tool_get_legislation_details(self, serial: str) -> Dict[str, Any]:
        row = get_legislation_by_serial(serial)
        if row is None:
            return {"error": f"No legislation with serial {serial!r} in the database."}
        content = row["content"] or ""
        return {
            "serial": row["serial"],
            "title": row["title"],
            "legislation_type": row["legislation_type"],
            "status": row["status"],
            "issue_date": row["issue_date"],
            "publication_date": row["publication_date"],
            "summary": row["description"],
            "source_url": row["source_url"],
            "content_chars": len(content),
            "article_count": len(re.findall(r"^#{3,}\s*المادة", content, flags=re.M)),
        }

    def _run_tool(self, name: str, args: Dict[str, Any], passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if name == "search_legislation":
            return self.tool_search_legislation(str(args.get("query", "")))
        if name == "retrieve_legal_passages":
            return self.tool_retrieve_legal_passages(str(args.get("query", "")), args.get("serial"), passages)
        if name == "get_legislation_details":
            return self.tool_get_legislation_details(str(args.get("serial", "")))
        raise ValueError(f"Unknown tool {name}")

    # ------------------------------------------------------------------- loop

    def ask(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty")

        if not llm.is_available():
            # No LLM: the "agent" degrades to a single retrieval step.
            result = self.rag.answer(question)
            result["steps"] = [{"tool": "retrieve_legal_passages", "input": {"query": question},
                                "result": f"{len(result['passages'])} passages"}]
            return result

        system = AGENT_SYSTEM_PROMPT.format(insufficient=insufficient_message(question))
        messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]
        passages: List[Dict[str, Any]] = []
        steps: List[Dict[str, Any]] = []
        answer = ""

        for _ in range(self.max_steps):
            try:
                response = llm.create_message(system, messages, tools=TOOLS)
            except llm.LLMError as e:
                return self._fallback(question, passages, steps, str(e))
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                answer = llm.response_text(response)
                if not answer:
                    return self._fallback(question, passages, steps, "The model returned an empty answer")
                break

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_uses:
                try:
                    output = self._run_tool(block.name, dict(block.input or {}), passages)
                    is_error = "error" in output
                except Exception as e:
                    logger.exception(f"Tool {block.name} failed")
                    output, is_error = {"error": str(e)}, True
                steps.append({"tool": block.name, "input": dict(block.input or {}), "result": _summarize(output)})
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, ensure_ascii=False),
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": results})
        else:
            answer = insufficient_message(question) + " (agent step limit reached)"

        # Return only the passages the answer actually cites (all retrieved ones if none are cited)
        cited_refs = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        used = [p for i, p in enumerate(passages, 1) if i in cited_refs] or passages
        payload = format_sources(used)
        # Keep the reference numbers the answer uses
        ref_of = {p["chunk_id"]: i for i, p in enumerate(passages, 1)}
        for p in payload["passages"]:
            p["ref"] = ref_of[p["chunk_id"]]
        for s in payload["sources"]:
            for c in s["chunks"]:
                c["ref"] = ref_of[c["chunk_id"]]

        if is_insufficient(answer, question):
            mode = "insufficient_context"
        else:
            mode = "agent" if passages else "no_retrieval"
        return {
            "question": question,
            "answer": answer,
            "mode": mode,
            **payload,
            "steps": steps,
            "disclaimer": disclaimer(question),
        }


    def _fallback(self, question: str, passages: List[Dict[str, Any]], steps: List[Dict[str, Any]], error: str) -> Dict[str, Any]:
        """LLM failed mid-run: return the passages retrieved so far (or a fresh retrieval)."""
        logger.error(f"Agent LLM call failed, returning retrieved passages only: {error}")
        chunks = passages or self.rag.retrieve(question)
        if not passages:
            steps = steps + [{"tool": "retrieve_legal_passages", "input": {"query": question},
                              "result": f"{len(chunks)} passages"}]
        if chunks:
            result = retrieval_only_result(question, format_sources(chunks), llm_error=error)
        else:
            result = {"answer": insufficient_message(question), "mode": "insufficient_context",
                      **format_sources([]), "llm_error": error}
        return {"question": question, **result, "steps": steps, "disclaimer": disclaimer(question)}


def _summarize(output: Dict[str, Any]) -> str:
    if "error" in output:
        return f"error: {output['error']}"
    if "results" in output:
        return f"{len(output['results'])} legislations"
    if "passages" in output:
        return f"{len(output['passages'])} passages"
    return output.get("title") or "ok"
