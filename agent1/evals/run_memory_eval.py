from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = ROOT / "evals" / "memory_cases.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "evals" / "results"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_agent(config_path: str | None):
    sys.path.insert(0, str(ROOT))
    from competitive_research_agent.config import load_config
    from competitive_research_agent.pipeline import CompetitiveResearchAgent

    config = load_config(config_path)
    return CompetitiveResearchAgent(config)


def evaluate_case(store, case: dict[str, Any], top_k: int) -> dict[str, Any]:
    user_id = f"memory-eval-{case['id']}"
    store.delete_user_memories(user_id)
    store.remember_interaction(
        user_id=user_id,
        question=case["seed_question"],
        answer=case["seed_answer"],
        topic="memory evaluation",
    )
    hits = store.search(user_id=user_id, query=case["query"], limit=top_k)
    expected_substrings = [str(item) for item in case.get("expected_substrings", [])]
    expected_kind = str(case.get("expected_kind", ""))
    expected_memory_key = str(case.get("expected_memory_key", ""))

    def matches(hit: dict[str, Any]) -> bool:
        text = str(hit.get("text", ""))
        kind_ok = not expected_kind or str(hit.get("kind", "")) == expected_kind
        key_ok = not expected_memory_key or str(hit.get("memory_key", "")) == expected_memory_key
        text_ok = all(part in text for part in expected_substrings) if expected_substrings else True
        return kind_ok and key_ok and text_ok

    any_match = any(matches(hit) for hit in hits)
    top1_match = bool(hits) and matches(hits[0])
    return {
        "id": case["id"],
        "query": case["query"],
        "expected_kind": expected_kind,
        "expected_memory_key": expected_memory_key,
        "expected_substrings": expected_substrings,
        "top1_match": top1_match,
        "any_match": any_match,
        "returned_hits": hits,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    top1_hits = sum(1 for row in rows if row["top1_match"])
    any_hits = sum(1 for row in rows if row["any_match"])
    return {
        "case_count": total,
        "top1_accuracy": round(top1_hits / total, 4) if total else 0.0,
        "recall_at_k": round(any_hits / total, 4) if total else 0.0,
        "success_count": any_hits,
    }


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Memory Evaluation Report",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Top-1 Accuracy: {summary['top1_accuracy']}",
        f"- Recall@K: {summary['recall_at_k']}",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['id']}")
        lines.append(f"- Query: {row['query']}")
        lines.append(f"- Top-1 Match: {row['top1_match']}")
        lines.append(f"- Any Match: {row['any_match']}")
        if row["returned_hits"]:
            lines.append(f"- First Hit: {row['returned_hits'][0]['text']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-term memory evaluation for the Industry Competitive Research Agent.")
    parser.add_argument("--config", default=None, help="Optional path to agent_config.json.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="Path to memory cases jsonl.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for eval results.")
    parser.add_argument("--top-k", type=int, default=3, help="How many memory hits to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = build_agent(args.config)
    if agent.long_term_memory is None:
        raise SystemExit("Long-term memory is not initialized.")
    cases = load_jsonl(Path(args.cases))
    rows = [evaluate_case(agent.long_term_memory, case, args.top_k) for case in cases]
    summary = summarize(rows)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"memory_eval_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "results.json", rows)
    write_markdown(output_dir / "report.md", summary, rows)
    print(json.dumps({"run_id": run_id, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
