from __future__ import annotations

import json
import math
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


_CURRENT_COLLECTOR: ContextVar[Any] = ContextVar(
    "competitive_research_agent_performance_collector",
    default=None,
)


class PerformanceCollector:
    def __init__(self):
        self.started_at = time.perf_counter()
        self.node_timings: dict[str, list[int]] = {}
        self.operation_timings: dict[str, list[int]] = {}
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.llm_calls = 0
        self.counters: dict[str, int] = {}
        self.values: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, name: str, *, category: str = "operations") -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            target = self.node_timings if category == "nodes" else self.operation_timings
            with self._lock:
                target.setdefault(name, []).append(elapsed_ms)

    def record_llm_call(self, usage: dict[str, int] | None = None) -> None:
        with self._lock:
            self.llm_calls += 1
            if not usage:
                return
            for key in self.token_usage:
                self.token_usage[key] += int(usage.get(key, 0) or 0)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + int(amount)

    def record_value(self, name: str, value: float) -> None:
        with self._lock:
            self.values.setdefault(name, []).append(float(value))

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_ms": int((time.perf_counter() - self.started_at) * 1000),
            "nodes": {name: sum(values) for name, values in self.node_timings.items()},
            "node_calls": {name: len(values) for name, values in self.node_timings.items()},
            "operations": {name: sum(values) for name, values in self.operation_timings.items()},
            "operation_calls": {name: len(values) for name, values in self.operation_timings.items()},
            "token_usage": dict(self.token_usage),
            "llm_calls": self.llm_calls,
            "counters": dict(self.counters),
            "values": {name: list(items) for name, items in self.values.items()},
        }


def set_current_collector(collector: PerformanceCollector) -> Token:
    return _CURRENT_COLLECTOR.set(collector)


def reset_current_collector(token: Token) -> None:
    _CURRENT_COLLECTOR.reset(token)


def current_collector() -> PerformanceCollector | None:
    return _CURRENT_COLLECTOR.get()


@contextmanager
def measure_current(name: str, *, category: str = "operations") -> Iterator[None]:
    collector = current_collector()
    if collector is None:
        yield
        return
    with collector.measure(name, category=category):
        yield


class PerformanceStore:
    def __init__(self, log_path: str, report_dir: str):
        self.log_path = Path(log_path)
        self.report_dir = Path(report_dir)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows[-limit:] if limit else rows

    def generate_report(self, limit: int | None = None) -> dict[str, Any]:
        rows = self.load(limit)
        summary = summarize_performance(rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.report_dir / f"performance_{timestamp}.json"
        markdown_path = self.report_dir / f"performance_{timestamp}.md"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(render_markdown(summary), encoding="utf-8")
        return {
            **summary,
            "json_path": str(json_path.resolve()),
            "markdown_path": str(markdown_path.resolve()),
        }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return round(float(ordered[index]), 2)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "avg": round(sum(values) / len(values), 2),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": round(max(values), 2),
    }


def summarize_performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    node_names = sorted({name for row in rows for name in row.get("performance", {}).get("nodes", {})})
    operation_names = sorted({name for row in rows for name in row.get("performance", {}).get("operations", {})})
    counter_names = sorted({name for row in rows for name in row.get("performance", {}).get("counters", {})})
    value_names = sorted({name for row in rows for name in row.get("performance", {}).get("values", {})})
    return {
        "run_count": len(rows),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_ms": _stats([row.get("performance", {}).get("total_ms", 0) for row in rows]),
        "nodes": {
            name: _stats(
                [
                    row["performance"]["nodes"][name]
                    for row in rows
                    if name in row.get("performance", {}).get("nodes", {})
                ]
            )
            for name in node_names
        },
        "operations": {
            name: _stats(
                [
                    row["performance"]["operations"][name]
                    for row in rows
                    if name in row.get("performance", {}).get("operations", {})
                ]
            )
            for name in operation_names
        },
        "tokens": {
            key: _stats([row.get("performance", {}).get("token_usage", {}).get(key, 0) for row in rows])
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "counters": {
            name: {
                "sum": int(
                    sum(row.get("performance", {}).get("counters", {}).get(name, 0) for row in rows)
                ),
                "avg": round(
                    sum(row.get("performance", {}).get("counters", {}).get(name, 0) for row in rows) / len(rows),
                    4,
                )
                if rows
                else 0.0,
            }
            for name in counter_names
        },
        "values": {
            name: _stats(
                [
                    float(item)
                    for row in rows
                    for item in row.get("performance", {}).get("values", {}).get(name, [])
                ]
            )
            for name in value_names
        },
        "quality_score": _stats([row.get("quality_score", 0) for row in rows]),
        "tool_elapsed_ms": _stats(
            [
                sum(int(item.get("elapsed_ms") or 0) for item in row.get("tool_history", []))
                for row in rows
            ]
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Performance Report",
        "",
        f"- Runs: {summary['run_count']}",
        f"- Generated: {summary['generated_at']}",
        "",
        "## Total Latency",
        "",
        _format_stats(summary["total_ms"], unit="ms"),
        "",
        "## Node Latency",
        "",
    ]
    for name, stats in summary["nodes"].items():
        lines.append(f"- `{name}`: {_format_stats(stats, unit='ms')}")
    lines.extend(["", "## Retrieval And Other Operations", ""])
    for name, stats in summary["operations"].items():
        lines.append(f"- `{name}`: {_format_stats(stats, unit='ms')}")
    lines.extend(["", "## Tokens", ""])
    for name, stats in summary["tokens"].items():
        lines.append(f"- `{name}`: {_format_stats(stats)}")
    if summary["counters"]:
        lines.extend(["", "## Counters", ""])
        for name, stats in summary["counters"].items():
            lines.append(f"- `{name}`: sum={stats['sum']}, avg={stats['avg']}")
    if summary["values"]:
        lines.extend(["", "## Derived Rates And Scores", ""])
        for name, stats in summary["values"].items():
            lines.append(f"- `{name}`: {_format_stats(stats)}")
    lines.extend(
        [
            "",
            "## Quality And Tools",
            "",
            f"- `quality_score`: {_format_stats(summary['quality_score'])}",
            f"- `tool_elapsed_ms`: {_format_stats(summary['tool_elapsed_ms'], unit='ms')}",
        ]
    )
    return "\n".join(lines)


def _format_stats(stats: dict[str, Any], unit: str = "") -> str:
    suffix = unit
    return (
        f"avg={stats['avg']}{suffix}, p50={stats['p50']}{suffix}, "
        f"p95={stats['p95']}{suffix}, p99={stats['p99']}{suffix}, max={stats['max']}{suffix}"
    )
