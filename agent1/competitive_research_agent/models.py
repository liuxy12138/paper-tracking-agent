from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ResearchDocumentRecord:
    document_id: str
    title: str
    file_path: str
    relevance_score: float
    summary: str = ""
    published: str = ""
    source_url: str = ""
    indexed: bool = False
    added_at: str = ""
    notes: dict[str, str] = field(default_factory=dict)
    industry: str = ""
    company: str = ""
    product_line: str = ""
    document_type: str = ""

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
    elapsed_ms: int = 0
    attempt: int = 1
    error_type: str = ""

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
    evidence: list[dict[str, Any]] = field(default_factory=list)
    report_path: Optional[str] = None
    performance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BriefGenerationResult:
    topic: str
    report_path: Optional[str]
    message: str
    workflow_result: Optional[WorkflowResult] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class WorkflowExecutionError(RuntimeError):
    question: str
    performance: dict[str, Any]
    error_type: str
    message: str

    def __post_init__(self) -> None:
        super().__init__(self.message)
