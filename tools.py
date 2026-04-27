from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain.tools import tool

from .config import SearchConfig
from .models import ToolExecutionRecord


class ResearchToolbox:
    def __init__(self, crawler, rag, parser_cls, paper_store):
        self.crawler = crawler
        self.rag = rag
        self.parser_cls = parser_cls
        self.paper_store = paper_store
        self._tool_map = self._build_tool_map()

    def _build_tool_map(self) -> dict[str, Any]:
        @tool
        def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
            """Search arXiv metadata without downloading PDFs."""
            return self.crawler.search_metadata(query=query, max_results=max_results)

        @tool
        def download_and_index_arxiv(
            query: str,
            max_results: int = 3,
            relevance_threshold: float = 0.25,
        ) -> list[dict]:
            """Download relevant arXiv papers and index them into the local RAG knowledge base."""
            records = self.crawler.search_and_download(
                SearchConfig(
                    max_results=max_results,
                    relevance_threshold=relevance_threshold,
                    query=query,
                )
            )
            indexed: list[dict] = []
            for record in records:
                existing = self.paper_store.get(record.paper_id)
                if existing and existing.indexed:
                    indexed.append(existing.to_dict())
                    continue
                self.rag.add_paper(
                    record.pdf_path,
                    metadata={
                        "paper_id": record.paper_id,
                        "published": record.published,
                        "source_url": record.source_url,
                    },
                )
                record.indexed = True
                self.paper_store.upsert(record)
                indexed.append(record.to_dict())
            return indexed

        @tool
        def semantic_search(query: str, top_k: int = 6) -> list[dict]:
            """Retrieve relevant chunks from the FAISS knowledge base."""
            return [item.to_dict() for item in self.rag.search(query=query, top_k=top_k)]

        @tool
        def parse_pdf(file_path: str) -> dict:
            """Parse a local PDF and extract title, abstract, and major sections."""
            normalized = Path(file_path).expanduser().resolve()
            parsed = self.parser_cls(str(normalized)).parse()
            return {
                "file_path": str(normalized),
                "title": parsed.get("title", ""),
                "language": parsed.get("language", ""),
                "abstract": parsed.get("abstract", "")[:1000],
                "sections": {
                    key: len(parsed.get(key, ""))
                    for key in ("introduction", "related_work", "method", "experiment", "conclusion")
                },
            }

        @tool
        def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
            """Extract lightweight keywords from arbitrary text."""
            cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\s\-]", " ", text.lower())
            tokens = [token for token in cleaned.split() if len(token) > 2]
            scores: dict[str, int] = {}
            for token in tokens:
                scores[token] = scores.get(token, 0) + 1
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            return [token for token, _ in ranked[:max_keywords]]

        return {
            "search_arxiv": search_arxiv,
            "download_and_index_arxiv": download_and_index_arxiv,
            "semantic_search": semantic_search,
            "parse_pdf": parse_pdf,
            "extract_keywords": extract_keywords,
        }

    def describe_tools(self) -> list[dict[str, str]]:
        return [
            {"name": name, "description": tool_obj.description}
            for name, tool_obj in self._tool_map.items()
        ]

    def execute_calls(self, calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        collected_results: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []

        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            tool_obj = self._tool_map.get(name)
            if tool_obj is None:
                history.append(
                    ToolExecutionRecord(
                        name=name,
                        args=args,
                        status="error",
                        result_preview="Unknown tool",
                    ).to_dict()
                )
                continue

            try:
                result = tool_obj.invoke(args)
                preview = json.dumps(result, ensure_ascii=False)[:220]
                history.append(
                    ToolExecutionRecord(
                        name=name,
                        args=args,
                        status="success",
                        result_preview=preview,
                    ).to_dict()
                )
                collected_results.append({"tool": name, "args": args, "result": result})
            except Exception as exc:  # pragma: no cover - defensive path
                history.append(
                    ToolExecutionRecord(
                        name=name,
                        args=args,
                        status="error",
                        result_preview=str(exc)[:220],
                    ).to_dict()
                )

        return collected_results, history
