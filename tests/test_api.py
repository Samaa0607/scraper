import pytest

from app import create_app


@pytest.fixture
def client(rag_pipeline):
    app = create_app(rag=rag_pipeline)
    app.testing = True
    return app.test_client()


def test_ask_returns_answer_and_sources(client):
    res = client.post("/api/ask", json={"question": "ما هي التسوية الوقائية؟"})
    assert res.status_code == 200 and res.is_json
    assert res.headers["Content-Type"] == "application/json; charset=utf-8"  # Windows PowerShell decoding
    data = res.get_json()
    assert data["answer"] and data["sources"][0]["title"] == "نظام الإفلاس"
    assert "نظام الإفلاس" in res.get_data(as_text=True)  # Arabic not \u-escaped


def test_ask_rag_mode_and_insufficient(client):
    data = client.post("/api/ask", json={"question": "قواعد الملاحة الجوية", "mode": "rag"}).get_json()
    assert data["mode"] == "insufficient_context" and data["sources"] == []


@pytest.mark.parametrize("body", [None, {}, {"question": ""}, {"question": 5}, {"question": "x", "mode": "bad"}])
def test_ask_rejects_invalid_requests_with_json_error(client, body):
    res = client.post("/api/ask", json=body) if body is not None else client.post("/api/ask", data="not json")
    assert res.status_code == 400 and "error" in res.get_json()


def test_search_and_legislation_endpoints(client):
    data = client.get("/api/search?q=الإجازة السنوية").get_json()
    assert data["passages"][0]["title"] == "نظام العمل"
    assert client.get("/api/search").status_code == 400

    items = client.get("/api/legislations").get_json()
    assert {i["serial"] for i in items} == {"SER-BANK", "SER-LABOR", "SER-EMPTY"}
    assert "content" not in items[0]

    detail = client.get("/api/legislations/SER-BANK").get_json()
    assert detail["title"] == "نظام الإفلاس" and "المادة الأولى" in detail["content"]
    assert client.get("/api/legislations/NOPE").status_code == 404


def test_health_and_frontend(client):
    health = client.get("/api/health").get_json()
    assert health["index"]["ready"] is True and health["llm_available"] is False
    page = client.get("/")
    assert page.status_code == 200 and "المساعد القانوني" in page.get_data(as_text=True)


def test_missing_index_returns_503(tmp_path, monkeypatch):
    import rag.vector_store as vs
    monkeypatch.setattr(vs, "INDEX_DIR", str(tmp_path / "none"))
    app = create_app()
    res = app.test_client().post("/api/ask", json={"question": "سؤال"})
    assert res.status_code == 503 and "build_index.py" in res.get_json()["error"]


def test_build_app_production_factory(rag_pipeline, monkeypatch):
    """The gunicorn entry point (app:build_app()) loads the index at startup and serves requests."""
    import app as app_module
    monkeypatch.setattr(app_module, "RAGPipeline", lambda: rag_pipeline)
    client = app_module.build_app().test_client()
    assert client.get("/api/health").get_json()["index"]["ready"] is True
    res = client.post("/api/ask", json={"question": "ما هي التسوية الوقائية؟", "mode": "rag"})
    assert res.status_code == 200 and res.get_json()["sources"]


def test_build_app_without_index_still_starts(monkeypatch):
    import app as app_module

    def missing():
        raise FileNotFoundError("No vector index found. Run: python build_index.py")
    monkeypatch.setattr(app_module, "RAGPipeline", missing)
    client = app_module.build_app().test_client()
    assert client.get("/").status_code == 200
