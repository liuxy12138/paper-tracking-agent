from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class PathConfig:
    base_dir: str = "./runtime_data"
    document_dir: str = "./runtime_data/documents"
    report_dir: str = "./runtime_data/reports"
    metadata_path: str = "./runtime_data/document_index.json"
    checkpoint_path: str = "./runtime_data/memory/langgraph_checkpoints.sqlite"
    performance_log_path: str = "./runtime_data/performance/runs.jsonl"
    performance_report_dir: str = "./runtime_data/performance/reports"
    agent_metrics_dir: str = "./runtime_data/metrics"

    def ensure_directories(self) -> None:
        directories = {
            self.base_dir,
            self.document_dir,
            self.report_dir,
            str(Path(self.checkpoint_path).parent),
            str(Path(self.performance_log_path).parent),
            self.performance_report_dir,
            self.agent_metrics_dir,
        }
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


@dataclass
class KeywordConfig:
    core: list[str] = field(
        default_factory=lambda: [
            "industry report",
            "competitive analysis",
            "market landscape",
            "white paper",
            "product documentation",
            "pricing",
            "business model",
            "technical architecture",
            "market size",
            "customer demand",
            "product differentiation",
            "roadmap",
        ]
    )
    extended: list[str] = field(
        default_factory=lambda: [
            "competitor",
            "product",
            "feature",
            "pricing model",
            "go-to-market",
            "market trend",
            "technology route",
            "limitation",
            "adoption",
            "benchmark",
            "announcement",
            "case study",
        ]
    )
    exclude: list[str] = field(
        default_factory=lambda: [
            "fiction",
            "entertainment",
            "sports",
            "recipe",
            "medical diagnosis",
        ]
    )


@dataclass
class SearchConfig:
    max_results: int = 5
    relevance_threshold: float = 0.3
    query: Optional[str] = None
    semantic_top_k: int = 6
    query_rewrite_count: int = 3
    web_enabled: bool = True
    web_provider: str = "tavily"
    web_api_key: str = ""
    web_endpoint: str = "https://api.tavily.com/search"
    web_search_depth: str = "advanced"
    web_timeout_seconds: float = 20.0
    web_max_download_bytes: int = 30 * 1024 * 1024
    web_user_agent: str = "CompetitiveResearchAgent/1.0"


@dataclass
class RagConfig:
    parser_backend: str = "docling"
    parser_fallback_backend: str = "pypdf"
    embedding_model: str = "BAAI/bge-m3"
    embedding_cache_dir: str = "./runtime_data/model_cache"
    llm_model: str = "glm-4-flash"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_timeout_seconds: float = 5.0
    collection_name: str = "industry_research_knowledge_bge_m3"
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_chunks_per_document: int = 60
    retrieval_top_k: int = 6
    hybrid_alpha: float = 0.75
    bm25_top_k: int = 12
    enable_dense_retrieval: bool = True
    enable_bm25_retrieval: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rerank_top_k: int = 12
    enable_model_rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_use_fp16: bool = False
    reranker_batch_size: int = 8
    reranker_max_length: int = 512
    reranker_weight: float = 0.85
    reranker_fallback: bool = True


@dataclass
class GraphConfig:
    enable_query_rewrite: bool = True
    enable_reflection: bool = True
    max_reflection_rounds: int = 1
    reflection_retry_score_threshold: float = 0.6
    reflection_rule_score_weight: float = 0.65
    reflection_min_evidence_items: int = 3
    reflection_min_source_count: int = 2
    reflection_min_citation_coverage: float = 0.5
    reflection_min_retrieval_score: float = 0.45
    reflection_enforce_quality_gates: bool = True
    reflection_evidence_count_weight: float = 0.25
    reflection_source_count_weight: float = 0.20
    reflection_citation_coverage_weight: float = 0.30
    reflection_retrieval_score_weight: float = 0.25
    default_thread_id: str = "default-thread"
    default_user_id: str = "default-user"
    max_history_messages: int = 6
    max_memory_items: int = 4
    use_sqlite_checkpointer: bool = True


@dataclass
class ToolConfig:
    timeout_seconds: float = 30.0
    max_retries: int = 1


@dataclass
class ObservabilityConfig:
    enabled: bool = True
    persist_jsonl: bool = True


@dataclass
class MemoryConfig:
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_key_prefix: str = "industry_research_agent:thread"
    redis_ttl_seconds: int = 604800
    redis_socket_timeout_seconds: float = 5.0
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_timeout_seconds: float = 5.0
    collection_name: str = "user_long_term_memory_bge_m3"
    min_similarity: float = 0.45
    duplicate_similarity: float = 0.92
    candidate_limit: int = 12
    enable_llm_compression: bool = True
    interaction_summary_max_chars: int = 600
    enable_thread_compression: bool = True
    thread_summary_max_chars: int = 1200


@dataclass
class DatabaseConfig:
    enabled: bool = True
    url: str = "mysql+pymysql://root:password@127.0.0.1:3306/industry_research_agent?charset=utf8mb4"
    init_schema: bool = True


@dataclass
class AgentConfig:
    topic: str = "AI coding agent industry and competitive landscape"
    api_key: str = ""
    paths: PathConfig = field(default_factory=PathConfig)
    keywords: KeywordConfig = field(default_factory=KeywordConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

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
        tools=ToolConfig(**data.get("tools", {})),
        observability=ObservabilityConfig(**data.get("observability", {})),
        memory=MemoryConfig(**data.get("memory", {})),
        database=DatabaseConfig(**data.get("database", {})),
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

    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_api_key:
        config.search.web_api_key = tavily_api_key

    database_url = (os.getenv("DATABASE_URL") or os.getenv("MYSQL_URL") or "").strip()
    if database_url:
        config.database.url = database_url
        config.database.enabled = True

    cloud_uri = os.getenv("ZILLIZ_CLOUD_URI", "").strip()
    cloud_token = os.getenv("ZILLIZ_CLOUD_TOKEN", "").strip()
    if cloud_uri:
        config.rag.milvus_uri = cloud_uri
        config.memory.milvus_uri = cloud_uri
    if cloud_token:
        config.rag.milvus_token = cloud_token
        config.memory.milvus_token = cloud_token

    milvus_uri = os.getenv("MILVUS_URI", "").strip()
    if milvus_uri:
        config.rag.milvus_uri = milvus_uri
        config.memory.milvus_uri = milvus_uri

    milvus_token = os.getenv("MILVUS_TOKEN", "").strip()
    if milvus_token:
        config.rag.milvus_token = milvus_token
        config.memory.milvus_token = milvus_token

    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        config.memory.redis_url = redis_url

    database_enabled = os.getenv("DATABASE_ENABLED", "").strip().lower()
    if database_enabled in {"true", "1", "yes"}:
        config.database.enabled = True
    elif database_enabled in {"false", "0", "no"}:
        config.database.enabled = False

    config.paths.ensure_directories()
    return config
