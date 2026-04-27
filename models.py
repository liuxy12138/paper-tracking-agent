from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    pdf_path: str
    relevance_score: float
    summary: str = ""
    published: str = ""
    source_url: str = ""
    indexed: bool = False
    added_at: str = ""
    notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    title: str
    score: float = 0.0
    section: str = ""
    origin: str = "vector_db"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolExecutionRecord:
    name: str
    args: dict[str, Any]
    status: str
    result_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowResult:
    question: str
    answer: str
    sources: list[str]
    plan: dict[str, Any]
    reflection: dict[str, Any]
    tool_history: list[dict[str, Any]]
    report_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    topic: str
    downloaded_count: int
    indexed_count: int
    skipped_count: int
    review_path: Optional[str]
    message: str
    workflow_result: Optional[WorkflowResult] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload
