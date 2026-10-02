"""
LLM integration tests using the REAL anthropic SDK with a fake HTTP transport.

No network and no real API key: requests are captured and canned API responses
are returned. This verifies the exact request Claude would receive (system
prompt, retrieved context, tool results) and how responses/errors are handled.
"""

import json

import anthropic
import httpx2
import pytest

from app import create_app
from rag import llm
from rag.agent import LegalAgent
from rag.rag import INSUFFICIENT_AR

QUESTION = "ما هي شروط افتتاح إجراء التسوية الوقائية؟"


def message(content, stop_reason="end_turn"):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 50}}


class FakeAnthropic:
    """Queues canned HTTP responses and records every request body."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def handler(self, request):
        self.requests.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        status, body = self.responses.pop(0)
        return httpx2.Response(status, json=body)


@pytest.fixture
def fake_api(monkeypatch):
    def install(*responses):
        fake = FakeAnthropic(responses)
        client = anthropic.Anthropic(
            api_key="test-key-not-real",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(fake.handler)),
        )
        monkeypatch.setattr(llm, "_client", client)
        monkeypatch.setattr(llm, "is_available", lambda: True)
        return fake
    return install


def test_rag_sends_retrieved_context_and_returns_grounded_arabic_answer(rag_pipeline, fake_api):
    answer = "يجوز للمدين طلب افتتاح إجراء التسوية الوقائية «إذا كان يتوقع أن يعاني من اضطراب أوضاعه المالية» (نظام الإفلاس، المادة الثانية) [1]."
    api = fake_api((200, message([{"type": "thinking", "thinking": "", "signature": "s"},
                                  {"type": "text", "text": answer}])))
    result = rag_pipeline.answer(QUESTION)

    assert result["mode"] == "llm"
    assert result["answer"] == answer
    assert result["sources"][0]["title"] == "نظام الإفلاس"
    assert result["sources"][0]["source_url"].endswith("/SER-BANK")

    req = api.requests[0]
    body = req["body"]
    assert req["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert INSUFFICIENT_AR in body["system"]
    assert "Use ONLY the numbered legal passages" in body["system"]
    user_text = body["messages"][0]["content"]
    assert "[1] نظام الإفلاس" in user_text            # retrieved context is sent
    assert "طلب افتتاح إجراء التسوية الوقائية" in user_text
    assert user_text.rstrip().endswith(QUESTION)
    assert "tools" not in body


def test_rag_insufficient_answer_is_detected(rag_pipeline, fake_api):
    fake_api((200, message([{"type": "text", "text": f"«{INSUFFICIENT_AR}» لا تتناول النصوص هذا الموضوع."}])))
    assert rag_pipeline.answer(QUESTION)["mode"] == "insufficient_context"


def test_out_of_scope_question_never_reaches_the_llm(rag_pipeline, fake_api):
    api = fake_api()
    result = rag_pipeline.answer("ما هي أحكام ضريبة القيمة المضافة؟")
    assert result["mode"] == "insufficient_context" and result["answer"] == INSUFFICIENT_AR
    assert api.requests == []


@pytest.mark.parametrize("status", [401, 429, 500])
def test_llm_failure_falls_back_to_retrieved_passages(rag_pipeline, fake_api, status):
    fake_api((status, {"type": "error", "error": {"type": "api_error", "message": "boom"}}))
    result = rag_pipeline.answer(QUESTION)
    assert result["mode"] == "retrieval_only"
    assert result["llm_error"] and "test-key" not in result["llm_error"]
    assert result["sources"] and result["passages"]


def test_agent_tool_loop_with_real_sdk(rag_pipeline, fake_api):
    api = fake_api(
        (200, message([
            {"type": "thinking", "thinking": "", "signature": "sig1"},
            {"type": "tool_use", "id": "toolu_1", "name": "retrieve_legal_passages",
             "input": {"query": "شروط افتتاح التسوية الوقائية"}},
        ], stop_reason="tool_use")),
        (200, message([{"type": "text", "text": "يجوز للمدين التقدم بطلب افتتاح إجراء التسوية الوقائية (نظام الإفلاس، المادة الثانية) [1]."}])),
    )
    result = LegalAgent(rag_pipeline).ask(QUESTION)

    assert result["mode"] == "agent"
    assert result["steps"][0]["tool"] == "retrieve_legal_passages"
    assert result["sources"][0]["title"] == "نظام الإفلاس"
    assert [p["ref"] for p in result["passages"]] == [1]

    first, second = (r["body"] for r in api.requests)
    assert {t["name"] for t in first["tools"]} == {"search_legislation", "retrieve_legal_passages", "get_legislation_details"}
    # Second call: assistant turn (with thinking block echoed back) + tool result with the legal text
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][1]["content"][0]["type"] == "thinking"
    tool_result = second["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    passages = json.loads(tool_result["content"])["passages"]
    assert passages[0]["ref"] == 1 and "التسوية الوقائية" in passages[0]["text"]
    assert passages[0]["source_url"].endswith("/SER-BANK")


def test_agent_llm_failure_falls_back_to_retrieval(rag_pipeline, fake_api):
    fake_api((401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}))
    result = LegalAgent(rag_pipeline).ask(QUESTION)
    assert result["mode"] == "retrieval_only"
    assert "credentials" in result["llm_error"]
    assert result["sources"] and result["steps"]


def test_api_with_llm_returns_valid_json(rag_pipeline, fake_api):
    fake_api(
        (200, message([{"type": "tool_use", "id": "toolu_1", "name": "retrieve_legal_passages",
                        "input": {"query": QUESTION}}], stop_reason="tool_use")),
        (200, message([{"type": "text", "text": "الإجابة [1]."}])),
    )
    client = create_app(rag=rag_pipeline).test_client()
    res = client.post("/api/ask", json={"question": QUESTION})
    assert res.status_code == 200 and res.is_json
    data = res.get_json()
    assert data["answer"] == "الإجابة [1]." and data["mode"] == "agent" and data["sources"]


def test_whitespace_api_key_is_not_treated_as_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert llm.is_available() is False
