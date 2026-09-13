from __future__ import annotations

import argparse
import json
import sys
import re

from competitive_research_agent.config import load_config


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _redacted_config(config) -> dict:
    payload = config.to_dict()
    if payload["rag"].get("milvus_token"):
        payload["rag"]["milvus_token"] = "***"
    if payload["memory"].get("milvus_token"):
        payload["memory"]["milvus_token"] = "***"
    if payload.get("api_key"):
        payload["api_key"] = "***"
    if payload.get("search", {}).get("web_api_key"):
        payload["search"]["web_api_key"] = "***"
    if payload.get("database", {}).get("url"):
        payload["database"]["url"] = re.sub(r"(?<=://)([^:/@]+):([^@]+)@", r"\1:***@", payload["database"]["url"])
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industry Competitive Research Agent")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=[
            "run",
            "ask",
            "chat",
            "ingest",
            "rebuild-index",
            "migrate-memory",
            "show-config",
            "show-graph",
            "performance-report",
            "check-milvus",
            "web-search",
            "collect-pdfs",
        ],
        help="generate a research brief, ask questions, chat, ingest a document, or inspect the system",
    )
    parser.add_argument("--config", default=None, help="Path to agent_config.json")
    parser.add_argument("--question", default="", help="Question for ask command")
    parser.add_argument("--thread-id", default=None, help="Conversation thread id")
    parser.add_argument("--user-id", default=None, help="User id for long-term memory")
    parser.add_argument("--file", default=None, help="Local PDF path for ingest command")
    parser.add_argument("--document-id", default=None, help="Optional document id for ingest command")
    parser.add_argument("--title", default=None, help="Optional document title for ingest command")
    parser.add_argument("--limit", type=int, default=None, help="Only aggregate the latest N performance runs")
    parser.add_argument("--query", default="", help="Web/PDF research query")
    parser.add_argument("--max-results", type=int, default=None, help="Maximum web search results")
    parser.add_argument("--include-domain", action="append", default=[], help="Restrict web results to a domain; repeatable")
    return parser


def _build_agent(
    config_path: str | None,
    *,
    allow_rag_schema_mismatch: bool = False,
    initialize_workflow: bool = True,
):
    from competitive_research_agent.pipeline import CompetitiveResearchAgent

    config = load_config(config_path)
    return (
        CompetitiveResearchAgent(
            config,
            allow_rag_schema_mismatch=allow_rag_schema_mismatch,
            initialize_workflow=initialize_workflow,
        ),
        config,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "show-config":
        print(json.dumps(_redacted_config(config), ensure_ascii=False, indent=2))
        return

    if args.command == "performance-report":
        from competitive_research_agent.observability import PerformanceStore

        store = PerformanceStore(
            config.paths.performance_log_path,
            config.paths.performance_report_dir,
        )
        print(json.dumps(store.generate_report(limit=args.limit), ensure_ascii=False, indent=2))
        return

    if args.command == "check-milvus":
        from pymilvus import MilvusClient

        kwargs = {
            "uri": config.rag.milvus_uri,
            "timeout": config.rag.milvus_timeout_seconds,
        }
        if config.rag.milvus_token:
            kwargs["token"] = config.rag.milvus_token
        try:
            client = MilvusClient(**kwargs)
            collections = client.list_collections()
        except Exception as exc:
            raise SystemExit(f"Milvus connection failed: {exc}") from exc
        print(
            json.dumps(
                {
                    "connected": True,
                    "uri": config.rag.milvus_uri,
                    "collections": collections,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "ask" and not args.question.strip():
        raise SystemExit("Please provide --question for the ask command.")
    if args.command == "ingest" and not args.file:
        raise SystemExit("Please provide --file for the ingest command.")
    if args.command == "migrate-memory" and not args.file:
        raise SystemExit("Please provide the legacy long-term-memory JSON with --file.")
    if args.command in {"web-search", "collect-pdfs"} and not args.query.strip():
        raise SystemExit("Please provide --query for web research.")

    agent, config = _build_agent(
        args.config,
        allow_rag_schema_mismatch=args.command == "rebuild-index",
        initialize_workflow=args.command not in {"rebuild-index", "web-search", "collect-pdfs"},
    )

    if args.command == "show-graph":
        print(agent.show_graph())
        return

    if args.command == "ingest":
        result = agent.ingest_document(args.file, document_id=args.document_id, title=args.title)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "web-search":
        result = agent.search_web(args.query, max_results=args.max_results, include_domains=args.include_domain)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "collect-pdfs":
        result = agent.collect_research_pdfs(args.query, max_results=args.max_results, include_domains=args.include_domain)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "migrate-memory":
        result = agent.migrate_legacy_memory(args.file)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "rebuild-index":
        result = agent.rebuild_index()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        result = agent.ask(
            question=args.question,
            thread_id=args.thread_id,
            user_id=args.user_id,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "chat":
        thread_id = args.thread_id or config.graph.default_thread_id
        user_id = args.user_id or config.graph.default_user_id
        print(f"Interactive chat started. thread_id={thread_id} user_id={user_id}")
        print("Type 'exit' to leave.")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() == "exit":
                break
            if not question:
                continue
            result = agent.ask(question=question, thread_id=thread_id, user_id=user_id)
            print("\nAgent:")
            print(result["answer"])
        return

    result = agent.generate_brief()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
