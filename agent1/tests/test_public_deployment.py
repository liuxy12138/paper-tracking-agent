from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from competitive_research_agent.config import load_config
from competitive_research_agent import webapp


def test_container_service_urls_override_local_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MILVUS_URI", "http://milvus:19530")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("DATABASE_ENABLED", "false")
    config = load_config()
    assert config.rag.milvus_uri == "http://milvus:19530"
    assert config.memory.milvus_uri == "http://milvus:19530"
    assert config.memory.redis_url == "redis://redis:6379/0"
    assert config.database.enabled is False


def test_public_config_only_exposes_safe_fields(monkeypatch):
    config = SimpleNamespace(
        topic="research",
        api_key="top-secret",
        rag=SimpleNamespace(
            llm_model="glm-4-flash",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            collection_name="research_collection",
            milvus_token="another-secret",
        ),
    )
    monkeypatch.setattr(webapp, "get_agent", lambda: SimpleNamespace(config=config))
    response = TestClient(webapp.app).get("/api/config")
    assert response.status_code == 200
    assert response.json()["config"]["llm_model"] == "glm-4-flash"
    assert "top-secret" not in response.text
    assert "another-secret" not in response.text
