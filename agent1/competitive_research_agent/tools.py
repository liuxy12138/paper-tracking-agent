from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextvars import copy_context
from pathlib import Path
from typing import Any

from langchain.tools import tool
from pydantic import ValidationError

from .config import ToolConfig
from .models import ToolExecutionRecord
from .schemas import TOOL_ARG_SCHEMAS, ToolCallSpec


class ResearchToolbox:
    def __init__(self, rag, parser_cls, tool_config: ToolConfig | None = None, web_research=None, ingest_callback=None):
        self.rag = rag
        self.parser_cls = parser_cls
        self.tool_config = tool_config or ToolConfig()
        self.web_research = web_research
        self.ingest_callback = ingest_callback
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._tool_map = self._build_tool_map()

    def _build_tool_map(self) -> dict[str, Any]:
        @tool
        def search_knowledge_base(
            query: str,
            top_k: int = 6,
            industry: str = "",
            company: str = "",
            document_type: str = "",
        ) -> list[dict]:
            """Retrieve citable evidence from the industry and competitor research knowledge base."""
            # Metadata filters are kept in the tool contract for planner clarity; retrieval currently ranks globally.
            return [item.to_dict() for item in self.rag.search(query=query, top_k=top_k)]

        @tool
        def ingest_research_document(
            file_path: str,
            document_id: str = "",
            title: str = "",
            industry: str = "",
            company: str = "",
            product_line: str = "",
            document_type: str = "",
        ) -> dict:
            """Parse and index a local industry report, competitor brief, product document, or announcement."""
            normalized = Path(file_path).expanduser().resolve()
            metadata = {
                "document_id": document_id or str(normalized),
                "source_url": str(normalized),
                "industry": industry,
                "company": company,
                "product_line": product_line,
                "document_type": document_type or "research_material",
            }
            parsed = self.rag.add_document(str(normalized), metadata=metadata)
            return {
                "document_id": metadata["document_id"],
                "title": title or parsed.get("title", str(normalized)),
                "source": str(normalized),
                "industry": industry,
                "company": company,
                "product_line": product_line,
                "document_type": metadata["document_type"],
                "summary": parsed.get("summary", "")[:1000],
            }

        @tool
        def parse_pdf(file_path: str) -> dict:
            """Parse a local PDF and extract title, summary, and major sections."""
            normalized = Path(file_path).expanduser().resolve()
            parsed = self.parser_cls(str(normalized)).parse()
            parser_type = getattr(self.parser_cls, "func", self.parser_cls)
            return {
                "file_path": str(normalized),
                "title": parsed.get("title", ""),
                "language": parsed.get("language", ""),
                "summary": parsed.get("summary", "")[:1000],
                "sections": {
                    key: len(parsed.get(key, ""))
                    for key in parser_type.SECTION_ALIASES
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

        @tool
        def search_web(query: str, max_results: int = 5, include_domains: list[str] | None = None) -> list[dict]:
            """Search the live web with Tavily for competitor, market, product, and white-paper evidence."""
            if self.web_research is None:
                raise RuntimeError("Web research client is not configured.")
            return self.web_research.search(query, max_results=max_results, include_domains=include_domains)

        @tool
        def collect_research_pdfs(
            query: str,
            max_results: int = 5,
            include_domains: list[str] | None = None,
            industry: str = "",
            company: str = "",
            product_line: str = "",
            document_type: str = "white_paper",
        ) -> list[dict]:
            """Find public research PDFs, download them safely, and index them into the knowledge base."""
            if self.web_research is None or self.ingest_callback is None:
                raise RuntimeError("PDF collection is not configured.")
            collected = []
            for item in self.web_research.search_pdfs(query, max_results=max_results, include_domains=include_domains):
                download = self.web_research.download_pdf(item["url"], title=item["title"])
                record = self.ingest_callback(
                    download["file_path"], document_id=download["sha256"], title=item["title"],
                    industry=industry, company=company, product_line=product_line,
                    document_type=document_type, source_url=item["url"],
                )
                collected.append(record)
            return collected

        return {
            "search_knowledge_base": search_knowledge_base,
            "ingest_research_document": ingest_research_document,
            "parse_pdf": parse_pdf,
            "extract_keywords": extract_keywords,
            "search_web": search_web,
            "collect_research_pdfs": collect_research_pdfs,
        }

    def describe_tools(self) -> list[dict[str, str]]:
        return [
            {"name": name, "description": tool_obj.description}
            for name, tool_obj in self._tool_map.items()
            if name in {"search_knowledge_base", "ingest_research_document", "parse_pdf", "extract_keywords", "search_web", "collect_research_pdfs"}
        ]

    def _validate_call(self, call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        spec = ToolCallSpec.model_validate(call)
        schema_cls = TOOL_ARG_SCHEMAS.get(spec.name)
        if schema_cls is None:
            return spec.name, spec.args
        args_model = schema_cls.model_validate(spec.args)
        return spec.name, args_model.model_dump()

    def _invoke_with_timeout(self, tool_obj: Any, args: dict[str, Any]) -> Any:
        context = copy_context()
        future = self._executor.submit(context.run, tool_obj.invoke, args)
        return future.result(timeout=self.tool_config.timeout_seconds)

    def execute_calls(self, calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        collected_results: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []

        for call in calls:
            try:
                name, args = self._validate_call(call)
            except ValidationError as exc:
                history.append(
                    ToolExecutionRecord(
                        name=str(call.get("name", "")),
                        args=call.get("args", {}) or {},
                        status="error",
                        result_preview=exc.errors(include_url=False).__repr__()[:220],
                        error_type="validation_error",
                    ).to_dict()
                )
                continue

            tool_obj = self._tool_map.get(name)
            if tool_obj is None:
                history.append(
                    ToolExecutionRecord(
                        name=name,
                        args=args,
                        status="error",
                        result_preview="Unknown tool",
                        error_type="unknown_tool",
                    ).to_dict()
                )
                continue

            max_attempts = max(1, self.tool_config.max_retries + 1)
            for attempt in range(1, max_attempts + 1):
                started_at = time.perf_counter()
                try:
                    result = self._invoke_with_timeout(tool_obj, args)
                    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                    preview = json.dumps(result, ensure_ascii=False)[:220]
                    history.append(
                        ToolExecutionRecord(
                            name=name,
                            args=args,
                            status="success",
                            result_preview=preview,
                            elapsed_ms=elapsed_ms,
                            attempt=attempt,
                        ).to_dict()
                    )
                    collected_results.append({"tool": name, "args": args, "result": result})
                    break
                except TimeoutError:
                    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                    history.append(
                        ToolExecutionRecord(
                            name=name,
                            args=args,
                            status="timeout",
                            result_preview=f"Tool exceeded {self.tool_config.timeout_seconds}s timeout.",
                            elapsed_ms=elapsed_ms,
                            attempt=attempt,
                            error_type="timeout",
                        ).to_dict()
                    )
                except Exception as exc:  # pragma: no cover - defensive path
                    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                    history.append(
                        ToolExecutionRecord(
                            name=name,
                            args=args,
                            status="error",
                            result_preview=str(exc)[:220],
                            elapsed_ms=elapsed_ms,
                            attempt=attempt,
                            error_type=exc.__class__.__name__,
                        ).to_dict()
                    )
                if attempt < max_attempts:
                    time.sleep(min(0.25 * attempt, 1.0))

        return collected_results, history
