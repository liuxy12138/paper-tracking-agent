from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from run_eval import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_DOCUMENTS_PATH,
    DEFAULT_QUESTIONS_PATH,
    collect_bad_cases,
    load_json,
    load_jsonl,
    score_eval_item,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).resolve().parents[1]


VARIANTS = {
    "dense_only": {
        "rag": {
            "enable_dense_retrieval": True,
            "enable_bm25_retrieval": False,
            "enable_model_rerank": False,
        }
    },
    "hybrid": {
        "rag": {
            "enable_dense_retrieval": True,
            "enable_bm25_retrieval": True,
            "enable_model_rerank": False,
        }
    },
    "hybrid_rerank": {
        "rag": {
            "enable_dense_retrieval": True,
            "enable_bm25_retrieval": True,
            "enable_model_rerank": True,
        }
    },
}


def _apply_overrides(target: Any, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dict__"):
            _apply_overrides(current, value)
            continue
        setattr(target, key, value)


def _average_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    score_rows = [row.get("scores", {}) for row in rows]
    if not score_rows:
        return {}
    keys = sorted({key for row in score_rows for key in row})
    return {
        key: round(sum(float(row.get(key, 0.0)) for row in score_rows) / len(score_rows), 4)
        for key in keys
    }


def build_agent(config_path: str | None, variant_name: str):
    import sys

    sys.path.insert(0, str(ROOT))
    from competitive_research_agent.config import load_config
    from competitive_research_agent.pipeline import CompetitiveResearchAgent

    config = load_config(config_path)
    _apply_overrides(config, copy.deepcopy(VARIANTS[variant_name]))
    return CompetitiveResearchAgent(config), config


def run_variant(
    *,
    config_path: str | None,
    variant_name: str,
    questions: list[dict[str, Any]],
    documents_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    agent, config = build_agent(config_path, variant_name)
    rows: list[dict[str, Any]] = []
    for item in questions:
        try:
            result = agent.ask(
                question=item["question"],
                thread_id=f"ablation-{variant_name}-{item['id']}",
                user_id="ablation-user",
            )
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "category": item.get("category", ""),
                    "scores": score_eval_item(item, result, documents_by_id).to_dict(),
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
                    "scores": {},
                    "result": {},
                    "error": str(exc),
                }
            )
    summary = {
        "run_id": variant_name,
        "variant": variant_name,
        "document_count": len(documents_by_id),
        "question_count": len(questions),
        "ingest_documents": False,
        "collection_name": config.rag.collection_name,
        "averages": _average_scores(rows),
        "error_count": sum(1 for row in rows if row.get("error")),
        "bad_case_count": len(collect_bad_cases(rows)),
        "settings": VARIANTS[variant_name],
    }
    return summary, rows


def _improvement(current: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return round(((current - baseline) / baseline) * 100, 2)


def write_ablation_report(path: Path, aggregate: dict[str, Any]) -> None:
    baseline_name = aggregate["baseline"]
    lines = [
        "# Retrieval Ablation Report",
        "",
        f"- Run ID: `{aggregate['run_id']}`",
        f"- Baseline: `{baseline_name}`",
        f"- Questions: {aggregate['question_count']}",
        "",
        "## Variants",
        "",
    ]
    for variant_name, payload in aggregate["variants"].items():
        averages = payload.get("averages", {})
        lines.append(f"### {variant_name}")
        for key, value in averages.items():
            improvement = payload.get("improvements", {}).get(key)
            if improvement is None:
                lines.append(f"- `{key}`: {value}")
            else:
                lines.append(f"- `{key}`: {value} ({improvement:+.2f}% vs {baseline_name})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval ablation for the Industry Competitive Research Agent.")
    parser.add_argument("--config", default=None, help="Optional path to agent_config.json.")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH), help="Path to documents.json.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH), help="Path to questions.jsonl.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for eval results.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions.")
    parser.add_argument("--baseline", default="dense_only", choices=sorted(VARIANTS), help="Baseline variant.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_json(Path(args.documents))
    questions = load_jsonl(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]
    documents_by_id = {document["document_id"]: document for document in documents}

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"ablation_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "run_id": run_id,
        "baseline": args.baseline,
        "question_count": len(questions),
        "variants": {},
    }
    baseline_recall = {}

    for variant_name in VARIANTS:
        summary, rows = run_variant(
            config_path=args.config,
            variant_name=variant_name,
            questions=questions,
            documents_by_id=documents_by_id,
        )
        variant_dir = output_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_json(variant_dir / "summary.json", summary)
        write_json(variant_dir / "results.json", rows)
        write_json(variant_dir / "bad_cases.json", collect_bad_cases(rows))
        write_markdown_report(variant_dir / "report.md", summary, rows)
        aggregate["variants"][variant_name] = summary
        if variant_name == args.baseline:
            baseline_recall = dict(summary.get("averages", {}))

    for variant_name, payload in aggregate["variants"].items():
        improvements = {}
        for key, value in payload.get("averages", {}).items():
            improvements[key] = _improvement(float(value), float(baseline_recall.get(key, 0.0)))
        payload["improvements"] = improvements

    write_json(output_dir / "aggregate.json", aggregate)
    write_ablation_report(output_dir / "aggregate.md", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
