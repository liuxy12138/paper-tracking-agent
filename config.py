from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class PathConfig:
    base_dir: str = "./runtime_data"
    pdf_dir: str = "./runtime_data/pdfs"
    vector_dir: str = "./runtime_data/vectordb"
    report_dir: str = "./runtime_data/reports"
    metadata_path: str = "./runtime_data/paper_index.json"
    long_term_memory_path: str = "./runtime_data/memory/long_term_memory.json"
    thread_history_path: str = "./runtime_data/memory/thread_history.json"
    checkpoint_path: str = "./runtime_data/memory/langgraph_checkpoints.sqlite"

    def ensure_directories(self) -> None:
        directories = {
            self.base_dir,
            self.pdf_dir,
            self.vector_dir,
            self.report_dir,
            str(Path(self.long_term_memory_path).parent),
            str(Path(self.thread_history_path).parent),
            str(Path(self.checkpoint_path).parent),
        }
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


@dataclass
class KeywordConfig:
    core: list[str] = field(
        default_factory=lambda: [
            "LEO satellite",
            "low earth orbit",
            "satellite positioning",
            "satellite navigation",
            "GNSS",
            "GPS satellite",
            "Starlink",
            "mega-constellation",
            "satellite constellation",
            "orbit determination",
            "satellite tracking",
            "Doppler positioning",
        ]
    )
    extended: list[str] = field(
        default_factory=lambda: [
            "positioning",
            "navigation",
            "orbit",
            "satellite",
            "constellation",
            "tracking",
            "localization",
            "geolocation",
            "ephemeris",
            "ionosphere",
            "PPP",
            "RTK",
        ]
    )
    exclude: list[str] = field(
        default_factory=lambda: [
            "quantum",
            "molecular",
            "gene",
            "protein",
            "COVID",
            "psychology",
        ]
    )


@dataclass
class SearchConfig:
    max_results: int = 5
    relevance_threshold: float = 0.3
    query: Optional[str] = None
    semantic_top_k: int = 6
    query_rewrite_count: int = 3


@dataclass
class RagConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "glm-4-flash"
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_chunks_per_paper: int = 60
    retrieval_top_k: int = 6
    hybrid_alpha: float = 0.75


@dataclass
class GraphConfig:
    enable_query_rewrite: bool = True
    enable_reflection: bool = True
    max_reflection_rounds: int = 1
    default_thread_id: str = "default-thread"
    default_user_id: str = "default-user"
    max_history_messages: int = 6
    max_memory_items: int = 4
    use_sqlite_checkpointer: bool = True


@dataclass
class AgentConfig:
    topic: str = "low earth orbit satellite positioning and navigation"
    api_key: str = ""
    paths: PathConfig = field(default_factory=PathConfig)
    keywords: KeywordConfig = field(default_factory=KeywordConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(data: dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        topic=data.get("topic", AgentConfig.topic),
        api_key=data.get("api_key", ""),
        paths=PathConfig(**data.get("paths", {})),
        keywords=KeywordConfig(**data.get("keywords", {})),
        search=SearchConfig(**data.get("search", {})),
        rag=RagConfig(**data.get("rag", {})),
        graph=GraphConfig(**data.get("graph", {})),
    )


def load_config(config_path: Optional[str] = None) -> AgentConfig:
    default_config = AgentConfig()
    merged = default_config.to_dict()

    config_file = Path(config_path) if config_path else Path("agent_config.json")
    if config_file.exists():
        override = json.loads(config_file.read_text(encoding="utf-8"))
        merged = _deep_merge(merged, override)

    config = _build_config(merged)

    env_api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if env_api_key:
        config.api_key = env_api_key

    config.paths.ensure_directories()
    return config
