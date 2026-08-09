__all__ = [
    "AgentConfig",
    "ArxivCrawler",
    "DailyResearchAgent",
    "LanguageDetector",
    "UniversalPaperParser",
    "UniversalPaperRAG",
    "load_config",
]


def __getattr__(name: str):
    if name in {"AgentConfig", "load_config"}:
        from .config import AgentConfig, load_config

        return {"AgentConfig": AgentConfig, "load_config": load_config}[name]
    if name == "ArxivCrawler":
        from .crawler import ArxivCrawler

        return ArxivCrawler
    if name in {"LanguageDetector", "UniversalPaperParser"}:
        from .parser import LanguageDetector, UniversalPaperParser

        return {
            "LanguageDetector": LanguageDetector,
            "UniversalPaperParser": UniversalPaperParser,
        }[name]
    if name == "DailyResearchAgent":
        from .pipeline import DailyResearchAgent

        return DailyResearchAgent
    if name == "UniversalPaperRAG":
        from .rag import UniversalPaperRAG

        return UniversalPaperRAG
    raise AttributeError(f"module 'paper_agent' has no attribute {name!r}")
