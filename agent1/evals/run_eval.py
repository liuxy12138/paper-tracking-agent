from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics import average_scores, score_eval_item


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = ROOT / "evals" / "documents" / "documents.json"
DEFAULT_QUESTIONS_PATH = ROOT / "evals" / "questions.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "evals" / "results"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_baseline_scores(questions: list[dict[str, Any]]) -> dict[str, float]:
    # A zero-retrieval baseline makes regressions obvious without requiring a model call.
    return {
        "recall_at_3": 0.0,
        "recall_at_5": 0.0,
        "citation_accuracy": 0.0,
        "keyword_coverage": 0.0,
        "answer_point_coverage": 0.0,
        "tool_success_rate": 0.0,
        "question_count": float(len(questions)),
    }


def find_previous_summary(output_dir: Path, current_run_id: str) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if not output_dir.exists():
        return None
    for path in output_dir.iterdir():
        summary_path = path / "summary.json"
        if not path.is_dir() or path.name == current_run_id or not summary_path.exists():
            continue
        candidates.append(summary_path)
    if not candidates:
        return None
    latest = sorted(candidates, key=lambda item: item.parent.name)[-1]
    return load_json(latest)


def compare_averages(current: dict[str, float], previous: dict[str, float]) -> dict[str, dict[str, float]]:
    keys = sorted(set(current) | set(previous))
    return {
        key: {
            "current": round(float(current.get(key, 0.0)), 4),
            "previous": round(float(previous.get(key, 0.0)), 4),
            "delta": round(float(current.get(key, 0.0)) - float(previous.get(key, 0.0)), 4),
        }
        for key in keys
    }


def collect_bad_cases(rows: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
    bad_cases: list[dict[str, Any]] = []
    for row in rows:
        scores = row.get("scores", {})
        if row.get("error"):
            bad_cases.append(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "reason": "runtime_error",
                    "error": row.get("error", ""),
                    "scores": scores,
                }
            )
            continue
        weak_metrics = {
            key: value
            for key, value in scores.items()
            if key != "tool_success_rate" and float(value) < threshold
        }
        if weak_metrics:
            bad_cases.append(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "reason": "low_score",
                    "weak_metrics": weak_metrics,
                    "expected_documents": row.get("expected_documents", []),
                    "sources": row.get("result", {}).get("sources", [])[:5],
                    "scores": scores,
                }
            )
    return bad_cases


def build_agent(config_path: str | None):
    sys.path.insert(0, str(ROOT))
    from competitive_research_agent.config import load_config
    from competitive_research_agent.pipeline import CompetitiveResearchAgent

    config = load_config(config_path)
    return CompetitiveResearchAgent(config)


def ingest_eval_documents(agent: Any, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ingested: list[dict[str, Any]] = []
    for document in documents:
        file_path = ROOT / document["file_path"]
        item = {
            "document_id": document["document_id"],
            "file_path": str(file_path),
            "status": "pending",
            "error": "",
        }
        try:
            agent.ingest_document(
                str(file_path),
                document_id=document["document_id"],
                title=document.get("title") or document["document_id"],
            )
            item["status"] = "success"
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
        ingested.append(item)
    return ingested


def run_eval(
    *,
    config_path: str | None,
    documents_path: Path,
    questions_path: Path,
    output_dir: Path,
    limit: int | None,
    ingest_documents: bool,
    dry_run: bool,
) -> dict[str, Any]:
    documents = load_json(documents_path)
    questions = load_jsonl(questions_path)
    if limit:
        questions = questions[:limit]

    documents_by_id = {document["document_id"]: document for document in documents}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp

    if dry_run:
        payload = {
            "mode": "dry_run",
            "document_count": len(documents),
            "question_count": len(questions),
            "document_ids": sorted(documents_by_id),
            "question_ids": [item["id"] for item in questions],
            "baseline": build_baseline_scores(questions),
        }
        write_json(run_dir / "dry_run.json", payload)
        return payload

    agent = build_agent(config_path)
    ingest_results = ingest_eval_documents(agent, documents) if ingest_documents else []

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {item['id']} {item['question']}")
        try:
            result = agent.ask(
                question=item["question"],
                thread_id=f"eval-{timestamp}-{item['id']}",
                user_id="eval-user",
            )
            scores = score_eval_item(item, result, documents_by_id).to_dict()
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "category": item.get("category", ""),
                    "expected_documents": item.get("expected_documents", []),
                    "scores": scores,
                    "result": result,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "category": item.get("category", ""),
                    "expected_documents": item.get("expected_documents", []),
                    "scores": {},
                    "result": {},
                    "error": str(exc),
                }
            )

    averages = average_scores(rows)
    previous_summary = find_previous_summary(output_dir, timestamp)
    comparison = {}
    if previous_summary:
        comparison = compare_averages(averages, previous_summary.get("averages", {}))

    summary = {
        "run_id": timestamp,
        "document_count": len(documents),
        "question_count": len(questions),
        "ingest_documents": ingest_documents,
        "ingest_results": ingest_results,
        "baseline": build_baseline_scores(questions),
        "averages": averages,
        "comparison_to_previous": comparison,
        "error_count": sum(1 for row in rows if row.get("error")),
        "bad_case_count": len(collect_bad_cases(rows)),
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "results.json", rows)
    write_json(run_dir / "bad_cases.json", collect_bad_cases(rows))
    write_markdown_report(run_dir / "report.md", summary, rows)
    write_bad_case_report(run_dir / "bad_cases.md", collect_bad_cases(rows))
    if getattr(agent, "trace_store", None):
        agent.trace_store.save_eval_run(summary, rows)
    return summary


def write_markdown_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Evaluation Report",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Documents: {summary['document_count']}",
        f"- Questions: {summary['question_count']}",
        f"- Errors: {summary['error_count']}",
        "",
        "## Average Scores",
        "",
    ]
    for key, value in summary.get("averages", {}).items():
        lines.append(f"- `{key}`: {value}")

    if summary.get("comparison_to_previous"):
        lines.extend(["", "## Compared With Previous Run", ""])
        for key, values in summary["comparison_to_previous"].items():
            lines.append(
                f"- `{key}`: current={values['current']} previous={values['previous']} delta={values['delta']}"
            )

    lines.extend(["", "## Baseline", ""])
    for key, value in summary.get("baseline", {}).items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Per-question Scores", ""])
    for row in rows:
        lines.append(f"### {row['id']} {row['question']}")
        if row.get("error"):
            lines.append(f"- Error: `{row['error']}`")
            lines.append("")
            continue
        for key, value in row.get("scores", {}).items():
            lines.append(f"- `{key}`: {value}")
        sources = row.get("result", {}).get("sources", [])
        if sources:
            lines.append("- Sources:")
            for source in sources[:5]:
                lines.append(f"  - {source}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bad_case_report(path: Path, bad_cases: list[dict[str, Any]]) -> None:
    lines = ["# Bad Case Report", ""]
    if not bad_cases:
        lines.append("No bad cases under the current threshold.")
    for item in bad_cases:
        lines.append(f"## {item.get('id', '')} {item.get('question', '')}")
        lines.append(f"- Reason: `{item.get('reason', '')}`")
        if item.get("error"):
            lines.append(f"- Error: `{item['error']}`")
        if item.get("weak_metrics"):
            for key, value in item["weak_metrics"].items():
                lines.append(f"- Weak `{key}`: {value}")
        if item.get("expected_documents"):
            lines.append(f"- Expected documents: {', '.join(item['expected_documents'])}")
        if item.get("sources"):
            lines.append("- Retrieved sources:")
            for source in item["sources"]:
                lines.append(f"  - {source}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evals for the Industry Competitive Research Agent.")
    parser.add_argument("--config", default=None, help="Optional path to agent_config.json.")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH), help="Path to documents.json.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH), help="Path to questions.jsonl.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for eval results.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions.")
    parser.add_argument("--ingest-documents", action="store_true", help="Ingest evaluation documents before asking questions.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate inputs and write a dry-run file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_eval(
        config_path=args.config,
        documents_path=Path(args.documents),
        questions_path=Path(args.questions),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        ingest_documents=args.ingest_documents,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
