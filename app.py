"""
Simple Flask backend for the legal RAG / agent demo.

Endpoints:
  POST /api/ask                    {"question": "...", "mode": "agent"|"rag"} -> answer + sources
  GET  /api/search?q=...           semantic passage search (no LLM)
  GET  /api/legislations           list stored legislations (metadata only)
  GET  /api/legislations/<serial>  one legislation including full content
  GET  /api/health                 index / LLM status
  GET  /                           demo web page

Run:
    python app.py        # local: http://127.0.0.1:5000
    gunicorn "app:build_app()" --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:$PORT   # production (Render)
"""

import logging
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

from config import LOG_FORMAT, TOP_K
from database import get_all_legislations, get_legislation_by_serial
from rag import llm
from rag.rag import RAGPipeline, format_sources

logger = logging.getLogger("app")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_QUESTION_CHARS = 2000


def create_app(rag: RAGPipeline = None, agent=None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.json.ensure_ascii = False  # keep Arabic readable in JSON responses
    # Declare the charset explicitly: Windows PowerShell 5.1 (Invoke-RestMethod)
    # otherwise decodes JSON as Latin-1 and shows Arabic as mojibake.
    app.json.mimetype = "application/json; charset=utf-8"
    state = {"rag": rag, "agent": agent}

    def get_rag() -> RAGPipeline:
        if state["rag"] is None:
            state["rag"] = RAGPipeline()  # raises FileNotFoundError if the index is missing
        return state["rag"]

    def get_agent():
        if state["agent"] is None:
            from rag.agent import LegalAgent
            state["agent"] = LegalAgent(get_rag())
        return state["agent"]

    def error(message: str, status: int):
        return jsonify({"error": message}), status

    @app.errorhandler(FileNotFoundError)
    def index_missing(e):
        return error(f"{e}", 503)

    @app.errorhandler(llm.LLMError)
    def llm_failed(e):
        return error(f"LLM error: {e}", 502)

    @app.get("/")
    def index_page():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/health")
    def health():
        try:
            meta = get_rag().store.meta
            index = {"ready": True, **meta}
        except FileNotFoundError as e:
            index = {"ready": False, "error": str(e)}
        return jsonify({"index": index, "llm_available": llm.is_available(), "llm_model": llm.LLM_MODEL})

    @app.post("/api/ask")
    def ask():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return error("Request body must be JSON: {\"question\": \"...\"}", 400)
        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            return error("'question' must be a non-empty string", 400)
        if len(question) > MAX_QUESTION_CHARS:
            return error(f"'question' is too long (max {MAX_QUESTION_CHARS} characters)", 400)
        mode = body.get("mode", "agent")
        if mode not in ("agent", "rag"):
            return error("'mode' must be 'agent' or 'rag'", 400)

        result = get_agent().ask(question) if mode == "agent" else get_rag().answer(question)
        return jsonify(result)

    @app.get("/api/search")
    def search():
        q = (request.args.get("q") or "").strip()
        if not q:
            return error("Query parameter 'q' is required", 400)
        try:
            top_k = max(1, min(int(request.args.get("top_k", TOP_K)), 20))
        except ValueError:
            return error("'top_k' must be an integer", 400)
        chunks = get_rag().retrieve(q, top_k=top_k)
        return jsonify({"query": q, **format_sources(chunks)})

    @app.get("/api/legislations")
    def list_legislations():
        rows = get_all_legislations()
        return jsonify([
            {
                "serial": r["serial"],
                "title": r["title"],
                "legislation_type": r["legislation_type"],
                "status": r["status"],
                "issue_date": r["issue_date"],
                "publication_date": r["publication_date"],
                "source_url": r["source_url"],
                "content_chars": len(r["content"] or ""),
            }
            for r in rows
        ])

    @app.get("/api/legislations/<serial>")
    def legislation_detail(serial):
        row = get_legislation_by_serial(serial)
        if row is None:
            return error(f"Legislation {serial!r} not found", 404)
        return jsonify(dict(row))

    return app


def build_app() -> Flask:
    """
    Application factory used by both `python app.py` and gunicorn (production).
    Loads the index and embedding model once at startup so the first question is fast.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)

    rag = None
    try:
        rag = RAGPipeline()
        rag.embedder.encode_query("تهيئة")
        logger.info(f"Index ready: {len(rag.store)} chunks; LLM available: {llm.is_available()}")
    except FileNotFoundError as e:
        logger.warning(f"{e} (the API will return 503 until the index is built)")

    return create_app(rag=rag)


if __name__ == "__main__":
    build_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
