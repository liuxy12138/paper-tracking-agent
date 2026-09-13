__all__ = [
    "AgentConfig",
    "CompetitiveResearchAgent",
    "LanguageDetector",
    "ResearchDocumentParser",
    "ResearchDocumentRAG",
    "load_config",
]


def __getattr__(name: str):
    if name in {"AgentConfig", "load_config"}:
        from .config import AgentConfig, load_config

        return {"AgentConfig": AgentConfig, "load_config": load_config}[name]
    if name in {"LanguageDetector", "ResearchDocumentParser"}:
        from .parser import LanguageDetector, ResearchDocumentParser

        return {
            "LanguageDetector": LanguageDetector,
            "ResearchDocumentParser": ResearchDocumentParser,
        }[name]
    if name == "CompetitiveResearchAgent":
        from .pipeline import CompetitiveResearchAgent

        return CompetitiveResearchAgent
    if name == "ResearchDocumentRAG":
        from .rag import ResearchDocumentRAG

        return ResearchDocumentRAG
    raise AttributeError(f"module 'competitive_research_agent' has no attribute {name!r}")
