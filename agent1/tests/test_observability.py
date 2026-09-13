from __future__ import annotations

import json

from competitive_research_agent.observability import PerformanceCollector, PerformanceStore, summarize_performance


def test_collector_accumulates_calls_and_tokens():
    collector = PerformanceCollector()
    with collector.measure("retrieval", category="nodes"):
        pass
    with collector.measure("retrieval", category="nodes"):
        pass
    collector.record_llm_call({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})
    collector.increment("retry_trigger_count")
    collector.record_value("structured_parse_success_rate", 1.0)

    snapshot = collector.snapshot()

    assert snapshot["node_calls"]["retrieval"] == 2
    assert snapshot["llm_calls"] == 1
    assert snapshot["token_usage"]["total_tokens"] == 14
    assert snapshot["counters"]["retry_trigger_count"] == 1
    assert snapshot["values"]["structured_parse_success_rate"] == [1.0]


def test_store_generates_json_and_markdown_report(tmp_path):
    store = PerformanceStore(
        str(tmp_path / "runs.jsonl"),
        str(tmp_path / "reports"),
    )
    for total_ms in (100, 200, 300):
        store.append(
            {
                "quality_score": 0.8,
                "performance": {
                    "total_ms": total_ms,
                    "nodes": {"retrieval": total_ms // 2},
                    "operations": {"milvus_search": 10},
                    "token_usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                },
                "tool_history": [{"elapsed_ms": 12}],
            }
        )

    report = store.generate_report()

    assert report["run_count"] == 3
    assert report["total_ms"]["p50"] == 200
    assert report["total_ms"]["p99"] == 300
    assert (tmp_path / "runs.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert json.loads(open(report["json_path"], encoding="utf-8").read())["run_count"] == 3
    assert "# Performance Report" in open(report["markdown_path"], encoding="utf-8").read()


def test_missing_node_does_not_count_as_zero():
    summary = summarize_performance(
        [
            {
                "performance": {
                    "total_ms": 100,
                    "nodes": {"reflection": 30},
                    "counters": {"retry_trigger_count": 1},
                    "values": {"structured_parse_success_rate": [1.0]},
                }
            },
            {"performance": {"total_ms": 80, "nodes": {}}},
        ]
    )

    assert summary["nodes"]["reflection"]["count"] == 1
    assert summary["nodes"]["reflection"]["avg"] == 30
    assert summary["counters"]["retry_trigger_count"]["sum"] == 1
    assert summary["values"]["structured_parse_success_rate"]["avg"] == 1.0
