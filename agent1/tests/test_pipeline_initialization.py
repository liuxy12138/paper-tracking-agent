from __future__ import annotations

from competitive_research_agent.config import AgentConfig
from competitive_research_agent.pipeline import CompetitiveResearchAgent


def test_rebuild_only_initialization_skips_redis_and_workflow(monkeypatch):
    monkeypatch.setattr("competitive_research_agent.pipeline.ResearchDocumentRAG", lambda **kwargs: object())
    monkeypatch.setattr("competitive_research_agent.pipeline.create_document_store", lambda *args: object())
    monkeypatch.setattr("competitive_research_agent.pipeline.create_trace_store", lambda *args: None)

    agent = CompetitiveResearchAgent(AgentConfig(), initialize_workflow=False)

    assert agent.thread_history is None
    assert agent.long_term_memory is None
    assert agent.workflow is None


def test_warmup_models_preloads_parser_and_reranker(monkeypatch):
    parser_calls: list[str] = []
    reranker_calls: list[bool] = []

    class FakeRAG:
        def warmup(self, *, include_reranker: bool = False):
            reranker_calls.append(include_reranker)

    monkeypatch.setattr("competitive_research_agent.pipeline.ResearchDocumentRAG", lambda **kwargs: FakeRAG())
    monkeypatch.setattr("competitive_research_agent.pipeline.create_document_store", lambda *args: object())
    monkeypatch.setattr("competitive_research_agent.pipeline.create_trace_store", lambda *args: None)
    monkeypatch.setattr(
        "competitive_research_agent.pipeline.ResearchDocumentParser.warmup_backend",
        lambda backend: parser_calls.append(backend),
    )

    agent = CompetitiveResearchAgent(AgentConfig(), initialize_workflow=False)
    agent.warmup_models()

    assert parser_calls == ["docling"]
    assert reranker_calls == [True]
