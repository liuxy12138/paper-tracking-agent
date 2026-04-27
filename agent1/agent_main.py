from __future__ import annotations

import argparse
import json

from paper_agent.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper Research Agent")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "ask", "chat", "ingest", "show-config", "show-graph"],
        help="run daily pipeline, ask questions, chat, ingest pdf, or inspect config/graph",
    )
    parser.add_argument("--config", default=None, help="Path to agent_config.json")
    parser.add_argument("--question", default="", help="Question for ask command")
    parser.add_argument("--thread-id", default=None, help="Conversation thread id")
    parser.add_argument("--user-id", default=None, help="User id for long-term memory")
    parser.add_argument("--file", default=None, help="Local PDF path for ingest command")
    parser.add_argument("--paper-id", default=None, help="Optional paper id for ingest command")
    parser.add_argument("--title", default=None, help="Optional paper title for ingest command")
    return parser


def _build_agent(config_path: str | None):
    from paper_agent.pipeline import DailyResearchAgent

    config = load_config(config_path)
    return DailyResearchAgent(config), config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "show-config":
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return

    agent, config = _build_agent(args.config)

    if args.command == "show-graph":
        print(agent.show_graph())
        return

    if args.command == "ingest":
        if not args.file:
            raise SystemExit("Please provide --file for the ingest command.")
        result = agent.ingest_local_pdf(args.file, paper_id=args.paper_id, title=args.title)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        if not args.question.strip():
            raise SystemExit("Please provide --question for the ask command.")
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

    result = agent.run_daily()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
