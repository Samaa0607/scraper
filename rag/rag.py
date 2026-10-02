"""
RAG pipeline:

    question -> question embedding -> vector similarity search
             -> top relevant legal chunks -> context -> LLM -> grounded answer + sources

If no retrieved chunk is relevant enough, the pipeline answers that the
available legal data is insufficient instead of calling the LLM.
If no LLM is configured, it returns the retrieved passages (with citations).
"""

import logging
import re
from typing import Any, Dict, List, Optional

from config import MIN_RELEVANCE_SCORE, TOP_K
from rag import llm
from rag.vector_store import Embedder, VectorStore

logger = logging.getLogger(__name__)

ARABIC_RE = re.compile(r"[؀-ۿ]")

DISCLAIMER_AR = "تنبيه: هذه الإجابة مولدة آليًا من نصوص الأنظمة المتاحة لأغراض تعليمية، وليست استشارة قانونية."
DISCLAIMER_EN = "Note: this answer is generated automatically from the available legislation texts for educational purposes and is not legal advice."

INSUFFICIENT_AR = "البيانات القانونية المتاحة غير كافية للإجابة عن هذا السؤال."
INSUFFICIENT_EN = "The available legal data is insufficient to answer this question."

# Grounding rules shared by the plain RAG prompt and the agent prompt.
GROUNDING_RULES = """- Use ONLY the numbered legal passages you were given. Do not use general knowledge, and never invent legislation names, article numbers, conditions, deadlines, dates, penalties or procedures.
- Cite every legal statement with the passage number(s) and the legislation + article it comes from, e.g. (نظام الإفلاس، المادة الثالثة عشرة) [3].
- Keep the legal text and your explanation clearly separate: put any wording copied from a passage inside «» quotation marks; everything outside «» is your own short explanation of the cited text.
- If the passages do not contain enough information to answer the question, reply starting with exactly: "{insufficient}" and then say in one sentence what is missing. Do not answer partially from general knowledge.
- Answer in the language of the question (Arabic questions get Arabic answers). Be clear and concise.
- Write plain text: no Markdown headings, bold or tables. Simple numbered lines are fine.
- This is legal information, not legal advice; do not present it as advice for the user's specific case."""

SYSTEM_PROMPT = """You are a legal information assistant for Saudi legislation published on the Ministry of Justice Laws & Regulations portal (laws.moj.gov.sa). You answer questions using retrieved passages from that legislation.

Rules:
""" + GROUNDING_RULES


def is_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


def insufficient_message(question: str) -> str:
    return INSUFFICIENT_AR if is_arabic(question) else INSUFFICIENT_EN


def is_insufficient(answer: str, question: str) -> bool:
    """True if the LLM answer is the "insufficient data" reply (tolerates quotes/whitespace)."""
    head = (answer or "").strip().lstrip("\"'«“ ")[:200]
    return head.startswith(insufficient_message(question).rstrip("."))


def disclaimer(question: str) -> str:
    return DISCLAIMER_AR if is_arabic(question) else DISCLAIMER_EN


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as numbered passages with their source."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.get('title') or ''}"
        if c.get("chapter"):
            header += f" — {c['chapter']}"
        header += f"\nSource: {c.get('source_url') or 'N/A'}"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def format_sources(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build the citation payload:
      sources  - one entry per legislation used (title, URL, matched chunks)
      passages - one entry per retrieved chunk, numbered as cited in the answer
    """
    sources: Dict[str, Dict[str, Any]] = {}
    passages = []
    for i, c in enumerate(chunks, 1):
        key = c.get("serial") or c.get("title")
        if key not in sources:
            sources[key] = {
                "title": c.get("title"),
                "serial": c.get("serial"),
                "source_url": c.get("source_url"),
                "legislation_type": c.get("legislation_type"),
                "status": c.get("status"),
                "issue_date": c.get("issue_date"),
                "chunks": [],
            }
        sources[key]["chunks"].append({
            "ref": i,
            "chunk_id": c.get("chunk_id"),
            "heading": c.get("heading"),
            "score": c.get("score"),
        })
        passages.append({
            "ref": i,
            "chunk_id": c.get("chunk_id"),
            "title": c.get("title"),
            "chapter": c.get("chapter"),
            "heading": c.get("heading"),
            "source_url": c.get("source_url"),
            "score": c.get("score"),
            "text": c.get("text"),
        })
    return {"sources": list(sources.values()), "passages": passages}


def retrieval_only_result(
    question: str,
    cited: Dict[str, List[Dict[str, Any]]],
    llm_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer made of the retrieved passages only (no LLM configured, or the LLM call failed)."""
    arabic = is_arabic(question)
    if llm_error:
        intro = ("تعذر توليد إجابة من نموذج اللغة حاليًا. هذه أكثر النصوص النظامية صلة بالسؤال:" if arabic
                 else "The LLM could not generate an answer right now. These are the most relevant legal passages:")
    else:
        intro = ("لم يتم إعداد نموذج اللغة (ANTHROPIC_API_KEY). هذه أكثر النصوص النظامية صلة بالسؤال:" if arabic
                 else "No LLM is configured (ANTHROPIC_API_KEY). These are the most relevant legal passages:")
    lines = [intro]
    for p in cited["passages"]:
        excerpt = p["text"][:400] + ("…" if len(p["text"]) > 400 else "")
        lines.append(f"[{p['ref']}] {p['title']} — {p['heading'] or ''}\n{excerpt}")
    result = {"answer": "\n\n".join(lines), "mode": "retrieval_only", **cited}
    if llm_error:
        result["llm_error"] = llm_error
    return result


class RAGPipeline:
    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        min_score: float = MIN_RELEVANCE_SCORE,
    ):
        self.store = store if store is not None else VectorStore.load()
        self.embedder = embedder or Embedder(self.store.meta.get("embedding_model") or Embedder().model_name)
        self.min_score = min_score

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        serial: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top_k chunks whose similarity is above the relevance threshold."""
        question = (question or "").strip()
        if not question or len(self.store) == 0:
            return []
        query_vector = self.embedder.encode_query(question)
        hits = self.store.search(query_vector, top_k=top_k, serial=serial)
        relevant = [h for h in hits if h["score"] >= self.min_score]
        logger.info(
            f"Retrieved {len(hits)} chunks, {len(relevant)} above threshold {self.min_score} "
            f"(best score: {hits[0]['score'] if hits else 'n/a'})"
        )
        return relevant

    def generate(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a grounded answer from already-retrieved chunks."""
        if not chunks:
            return {"answer": insufficient_message(question), "mode": "insufficient_context", **format_sources([])}

        cited = format_sources(chunks)

        if not llm.is_available():
            return retrieval_only_result(question, cited)

        system = SYSTEM_PROMPT.format(insufficient=insufficient_message(question))
        user = f"Legal passages:\n\n{build_context(chunks)}\n\n---\n\nQuestion: {question}"
        try:
            response = llm.create_message(system, [{"role": "user", "content": user}])
            answer = llm.response_text(response)
            if not answer:
                raise llm.LLMError("The model returned an empty answer")
        except llm.LLMError as e:
            # Never lose the retrieved passages because the LLM call failed
            logger.error(f"LLM generation failed, returning retrieved passages only: {e}")
            return retrieval_only_result(question, cited, llm_error=str(e))

        mode = "insufficient_context" if is_insufficient(answer, question) else "llm"
        return {"answer": answer, "mode": mode, **cited}

    def answer(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """Full RAG: retrieve relevant chunks, then generate a grounded answer with sources."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty")
        chunks = self.retrieve(question, top_k=top_k)
        result = self.generate(question, chunks)
        return {"question": question, **result, "disclaimer": disclaimer(question)}
