from types import SimpleNamespace

from rag import llm
from rag.agent import LegalAgent


def tool_call(id_, name, **inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def text(t):
    return SimpleNamespace(type="text", text=t)


def scripted_llm(monkeypatch, responses):
    """Replace the LLM with a fixed sequence of responses; record the messages sent."""
    sent = []

    def fake_create(system, messages, tools=None):
        sent.append(list(messages))
        content = responses[len(sent) - 1]
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(stop_reason=stop, content=content)

    monkeypatch.setattr(llm, "is_available", lambda: True)
    monkeypatch.setattr(llm, "create_message", fake_create)
    return sent


def test_agent_uses_tools_and_returns_cited_sources(rag_pipeline, monkeypatch):
    sent = scripted_llm(monkeypatch, [
        [tool_call("t1", "search_legislation", query="الإفلاس")],
        [tool_call("t2", "retrieve_legal_passages", query="افتتاح التسوية الوقائية", serial="SER-BANK"),
         tool_call("t3", "get_legislation_details", serial="SER-BANK")],
        [text("يجوز للمدين طلب افتتاح إجراء التسوية الوقائية [1].")],
    ])
    result = LegalAgent(rag_pipeline).ask("متى يجوز طلب التسوية الوقائية؟")

    assert result["mode"] == "agent"
    assert [s["tool"] for s in result["steps"]] == ["search_legislation", "retrieve_legal_passages", "get_legislation_details"]
    assert len(result["passages"]) == 1 and result["passages"][0]["ref"] == 1
    assert result["sources"][0]["title"] == "نظام الإفلاس"
    assert result["sources"][0]["source_url"].endswith("/SER-BANK")
    # both tool results went back in ONE user message
    last_user = sent[2][-1]
    assert last_user["role"] == "user" and len(last_user["content"]) == 2
    assert "التسوية الوقائية" in last_user["content"][0]["content"]


def test_agent_answers_directly_without_retrieval(rag_pipeline, monkeypatch):
    scripted_llm(monkeypatch, [[text("مرحبًا! يمكنني الإجابة عن أسئلة حول الأنظمة المخزنة.")]])
    result = LegalAgent(rag_pipeline).ask("مرحبا")
    assert result["mode"] == "no_retrieval" and result["sources"] == [] and result["steps"] == []


def test_agent_reports_tool_errors_to_the_model(rag_pipeline, monkeypatch):
    sent = scripted_llm(monkeypatch, [
        [tool_call("t1", "get_legislation_details", serial="NOPE")],
        [text("البيانات القانونية المتاحة غير كافية للإجابة عن هذا السؤال.")],
    ])
    result = LegalAgent(rag_pipeline).ask("ما تفاصيل النظام NOPE؟")
    assert sent[1][-1]["content"][0]["is_error"] is True
    assert result["mode"] == "insufficient_context"


def test_agent_stops_at_step_limit(rag_pipeline, monkeypatch):
    scripted_llm(monkeypatch, [[tool_call(f"t{i}", "search_legislation", query="نظام")] for i in range(10)])
    result = LegalAgent(rag_pipeline, max_steps=3).ask("سؤال")
    assert len(result["steps"]) == 3
    assert "step limit" in result["answer"]


def test_agent_without_llm_falls_back_to_rag(rag_pipeline):
    result = LegalAgent(rag_pipeline).ask("ما هي التسوية الوقائية؟")
    assert result["mode"] == "retrieval_only"
    assert result["sources"] and result["steps"][0]["tool"] == "retrieve_legal_passages"
