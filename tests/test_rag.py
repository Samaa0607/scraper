from types import SimpleNamespace

import numpy as np

from rag import llm
from rag.rag import INSUFFICIENT_AR, RAGPipeline, build_context
from rag.vector_store import VectorStore


def test_retrieval_returns_relevant_chunks(rag_pipeline):
    hits = rag_pipeline.retrieve("ما هي التسوية الوقائية للمدين؟")
    assert hits
    assert hits[0]["title"] == "نظام الإفلاس"
    assert "التسوية الوقائية" in hits[0]["text"]
    assert hits == sorted(hits, key=lambda h: -h["score"])

    labor = rag_pipeline.retrieve("كم مدة الإجازة السنوية للعامل؟")
    assert labor[0]["title"] == "نظام العمل"


def test_retrieval_can_be_restricted_to_one_legislation(rag_pipeline):
    hits = rag_pipeline.retrieve("العامل الإجازة", serial="SER-BANK")
    assert all(h["serial"] == "SER-BANK" for h in hits)


def test_empty_index_and_unrelated_question_are_handled(rag_pipeline, fake_embedder):
    assert rag_pipeline.retrieve("") == []
    result = rag_pipeline.answer("ما هي عقوبة مخالفة أنظمة المرور الجوي؟")
    assert result["mode"] == "insufficient_context"
    assert result["answer"] == INSUFFICIENT_AR
    assert result["sources"] == []

    empty = RAGPipeline(store=VectorStore(np.zeros((0, 0), np.float32), [], {}), embedder=fake_embedder)
    assert empty.retrieve("أي سؤال") == []
    assert empty.answer("أي سؤال")["mode"] == "insufficient_context"


def test_vector_store_save_load_round_trip(rag_pipeline, tmp_path):
    rag_pipeline.store.save(str(tmp_path))
    loaded = VectorStore.load(str(tmp_path))
    assert len(loaded) == len(rag_pipeline.store)
    assert loaded.chunks[0]["title"] == rag_pipeline.store.chunks[0]["title"]


def test_without_llm_returns_passages_with_sources(rag_pipeline):
    result = rag_pipeline.answer("ما هي التسوية الوقائية؟")
    assert result["mode"] == "retrieval_only"
    assert result["sources"][0]["title"] == "نظام الإفلاس"
    assert result["sources"][0]["source_url"].endswith("/SER-BANK")
    assert result["passages"][0]["chunk_id"].startswith("SER-BANK-")
    assert "ليست استشارة قانونية" in result["disclaimer"]


def test_with_llm_answer_is_grounded_in_context(rag_pipeline, monkeypatch):
    captured = {}

    def fake_create(system, messages, tools=None):
        captured["system"], captured["user"] = system, messages[0]["content"]
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text="التسوية الوقائية إجراء لتيسير اتفاق المدين مع دائنيه [1].")])

    monkeypatch.setattr(llm, "is_available", lambda: True)
    monkeypatch.setattr(llm, "create_message", fake_create)
    result = rag_pipeline.answer("ما هي التسوية الوقائية؟")

    assert result["mode"] == "llm"
    assert "[1]" in result["answer"]
    assert "[1] نظام الإفلاس" in captured["user"]
    assert "https://laws.moj.gov.sa/ar/legislation/SER-BANK" in captured["user"]
    assert INSUFFICIENT_AR in captured["system"]


def test_build_context_numbers_passages():
    ctx = build_context([{"title": "نظام", "source_url": "u1", "text": "نص1"}, {"title": "لائحة", "source_url": "u2", "text": "نص2"}])
    assert ctx.index("[1] نظام") < ctx.index("[2] لائحة")
