from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


def normalize_text(text: Any) -> str:
    return str(text or "").casefold()


def contains_any(text: str, candidates: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(item) in normalized for item in candidates if item)


def coverage_ratio(text: str, expected_items: list[str]) -> float:
    if not expected_items:
        return 0.0
    normalized = normalize_text(text)
    hits = sum(1 for item in expected_items if normalize_text(item) in normalized)
    return round(hits / len(expected_items), 4)


def _document_match_terms(document: dict[str, Any]) -> list[str]:
    file_path = str(document.get("file_path", ""))
    filename = Path(file_path).name if file_path else ""
    return [
        str(document.get("document_id", "")),
        str(document.get("title", "")),
        filename,
        Path(filename).stem if filename else "",
    ]


def source_matches_document(source_text: str, document: dict[str, Any]) -> bool:
    return contains_any(source_text, _document_match_terms(document))


def recall_at_k(
    sources: list[str],
    expected_document_ids: list[str],
    documents_by_id: dict[str, dict[str, Any]],
    k: int,
) -> float:
    if not expected_document_ids:
        return 0.0

    top_sources = sources[:k]
    source_blob = "\n".join(top_sources)
    hits = 0
    for document_id in expected_document_ids:
        document = documents_by_id.get(document_id)
        if document and source_matches_document(source_blob, document):
            hits += 1
    return round(hits / len(expected_document_ids), 4)


def citation_accuracy(sources: list[str], documents_by_id: dict[str, dict[str, Any]]) -> float:
    if not sources:
        return 0.0
    known_documents = list(documents_by_id.values())
    matched = 0
    for source in sources:
        if any(source_matches_document(source, document) for document in known_documents):
            matched += 1
    return round(matched / len(sources), 4)


def tool_success_rate(tool_history: list[dict[str, Any]]) -> float:
    if not tool_history:
        return 0.0
    successes = sum(1 for item in tool_history if item.get("status") == "success")
    return round(successes / len(tool_history), 4)


@dataclass
class EvalScores:
    recall_at_3: float
    recall_at_5: float
    citation_accuracy: float
    keyword_coverage: float
    answer_point_coverage: float
    tool_success_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def score_eval_item(
    item: dict[str, Any],
    result: dict[str, Any],
    documents_by_id: dict[str, dict[str, Any]],
) -> EvalScores:
    answer = str(result.get("answer", ""))
    sources = [str(source) for source in result.get("sources", [])]
    tool_history = result.get("tool_history", [])

    return EvalScores(
        recall_at_3=recall_at_k(
            sources=sources,
            expected_document_ids=item.get("expected_documents", []),
            documents_by_id=documents_by_id,
            k=3,
        ),
        recall_at_5=recall_at_k(
            sources=sources,
            expected_document_ids=item.get("expected_documents", []),
            documents_by_id=documents_by_id,
            k=5,
        ),
        citation_accuracy=citation_accuracy(sources, documents_by_id),
        keyword_coverage=coverage_ratio(answer, item.get("expected_keywords", [])),
        answer_point_coverage=coverage_ratio(answer, item.get("answer_points", [])),
        tool_success_rate=tool_success_rate(tool_history),
    )


def average_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    score_rows = [row.get("scores", {}) for row in rows]
    if not score_rows:
        return {}

    keys = sorted({key for scores in score_rows for key in scores})
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(scores.get(key, 0.0)) for scores in score_rows]
        summary[key] = round(sum(values) / len(values), 4)
    return summary
