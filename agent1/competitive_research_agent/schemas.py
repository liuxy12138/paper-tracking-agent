from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolCallSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class PlanSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    objective: str = ""
    steps: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallSpec] = Field(default_factory=list)
    answer_format: str = "grounded answer"
    memory_summary: list[str] = Field(default_factory=list)


class ReflectionSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    passed: bool = True
    score: float = 0.0
    issues: list[str] = Field(default_factory=list)
    should_retry: bool = False
    retry_focus: str = ""
    llm_score: float = 0.0
    deterministic_score: float = 0.0
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    retry_recommended: bool = False
    retry_blocked_by_limit: bool = False
    llm_passed: bool = True
    llm_retry_recommended: bool = False


class EvidenceItemSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: str = ""
    source: str = "unknown"
    title: str = "unknown"
    score: float = 0.0
    section: str = ""
    origin: str = "vector_db"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error", "timeout"] = "success"
    result_preview: str = ""
    elapsed_ms: int = 0
    attempt: int = 1
    error_type: str = ""


class SearchKnowledgeBaseArgs(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=20)
    industry: str = ""
    company: str = ""
    document_type: str = ""


class IngestResearchDocumentArgs(BaseModel):
    file_path: str
    document_id: str = ""
    title: str = ""
    industry: str = ""
    company: str = ""
    product_line: str = ""
    document_type: str = ""


class ParsePdfArgs(BaseModel):
    file_path: str


class ExtractKeywordsArgs(BaseModel):
    text: str
    max_keywords: int = Field(default=8, ge=1, le=30)


class SearchWebArgs(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)
    include_domains: list[str] = Field(default_factory=list)


class CollectResearchPdfsArgs(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)
    include_domains: list[str] = Field(default_factory=list)
    industry: str = ""
    company: str = ""
    product_line: str = ""
    document_type: str = "white_paper"


TOOL_ARG_SCHEMAS = {
    "search_knowledge_base": SearchKnowledgeBaseArgs,
    "ingest_research_document": IngestResearchDocumentArgs,
    "parse_pdf": ParsePdfArgs,
    "extract_keywords": ExtractKeywordsArgs,
    "search_web": SearchWebArgs,
    "collect_research_pdfs": CollectResearchPdfsArgs,
}
