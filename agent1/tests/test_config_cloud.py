from __future__ import annotations

from agent_main import _redacted_config
from competitive_research_agent.config import load_config


def test_zilliz_environment_configures_document_and_memory_milvus(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZILLIZ_CLOUD_URI", "https://example.zillizcloud.com")
    monkeypatch.setenv("ZILLIZ_CLOUD_TOKEN", "secret-token")

    config = load_config()

    assert config.rag.milvus_uri == "https://example.zillizcloud.com"
    assert config.memory.milvus_uri == "https://example.zillizcloud.com"
    assert config.rag.milvus_token == "secret-token"
    assert config.memory.milvus_token == "secret-token"

    shown = _redacted_config(config)
    assert shown["rag"]["milvus_token"] == "***"
    assert shown["memory"]["milvus_token"] == "***"
