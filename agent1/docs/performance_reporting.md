# Automatic Performance Reporting

Each completed `ask` or daily workflow automatically appends one JSON record to:

```text
runtime_data/performance/runs.jsonl
```

The record includes total latency, node latency and call counts, retrieval operation
latency, tool latency, LLM call count and token usage, and the reflection quality score.
Reflection retries are included in the node call counts and accumulated latency.

Generate an aggregate report:

```bash
python agent_main.py performance-report
python agent_main.py performance-report --limit 100
```

The command writes JSON and Markdown reports under:

```text
runtime_data/performance/reports/
```

No manual timing or spreadsheet is required. P50, P95, and P99 become meaningful only
after collecting enough representative requests. Token counts depend on the LLM provider
returning usage metadata.
