from __future__ import annotations

import datetime as dt
import queue
import threading
import uuid
from functools import partial
from typing import Any, Iterator

from .config import AgentConfig
from .logging_utils import get_logger
from .llm import GLMClient
from .memory import LongTermMemoryStore, ThreadHistoryStore
from .models import BriefGenerationResult, ResearchDocumentRecord, WorkflowExecutionError
from .observability import PerformanceStore
from .parser import ResearchDocumentParser
from .rag import ResearchDocumentRAG
from .storage import create_document_store, create_trace_store
from .tools import ResearchToolbox
from .workflow import LangGraphResearchWorkflow, workflow_event_sink
from .web_research import WebResearchClient


class CompetitiveResearchAgent:
    def __init__(
        self,
        config: AgentConfig,
        *,
        allow_rag_schema_mismatch: bool = False,
        initialize_workflow: bool = True,
    ):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        self.rag = ResearchDocumentRAG(
            api_key=config.api_key,
            rag_config=config.rag,
            allow_schema_mismatch=allow_rag_schema_mismatch,
        )
        self.document_store = create_document_store(config.database, config.paths.metadata_path)
        self.trace_store = create_trace_store(config.database)
        self.performance_store = PerformanceStore(
            config.paths.performance_log_path,
            config.paths.performance_report_dir,
        )
        self.web_research = WebResearchClient(config.search, config.paths.document_dir)
        self.thread_history = None
        self.long_term_memory = None
        self.toolbox = None
        self.workflow = None
        if not initialize_workflow:
            return

        memory_llm = GLMClient(api_key=config.api_key, model=config.rag.llm_model)
        self.thread_history = ThreadHistoryStore(
            config=config.memory,
            llm=memory_llm,
        )
        self.long_term_memory = LongTermMemoryStore(
            config=config.memory,
            embeddings=self.rag.embeddings,
            llm=memory_llm,
        )
        self.toolbox = ResearchToolbox(
            rag=self.rag,
            parser_cls=partial(
                ResearchDocumentParser,
                backend=config.rag.parser_backend,
                fallback_backend=config.rag.parser_fallback_backend,
            ),
            tool_config=config.tools,
            web_research=self.web_research,
            ingest_callback=self.ingest_document,
        )
        self.workflow = LangGraphResearchWorkflow(
            config=config,
            rag=self.rag,
            toolbox=self.toolbox,
            thread_history=self.thread_history,
            long_term_memory=self.long_term_memory,
        )

    def warmup_models(self) -> None:
        try:
            ResearchDocumentParser.warmup_backend(self.config.rag.parser_backend)
        except Exception as exc:
            self.logger.warning("Parser warmup skipped: %s", exc)
        try:
            self.rag.warmup(include_reranker=True)
        except Exception as exc:
            self.logger.warning("RAG warmup skipped: %s", exc)

    def generate_brief(self) -> BriefGenerationResult:
        if self.workflow is None:
            raise RuntimeError("Workflow components were not initialized.")
        run_id = str(uuid.uuid4())
        self.logger.info("Competitive research brief generation started: %s", run_id)

        workflow_result = self.workflow.invoke(
            question=f"Generate an industry and competitor research brief about {self.config.topic} based on the current indexed evidence.",
            thread_id=f"brief-{run_id}",
            user_id=self.config.graph.default_user_id,
            mode="research_brief",
        )
        self._save_workflow_trace(
            workflow_result,
            thread_id=f"brief-{run_id}",
            user_id=self.config.graph.default_user_id,
            mode="research_brief",
        )
        self._save_performance(
            workflow_result,
            thread_id=f"brief-{run_id}",
            user_id=self.config.graph.default_user_id,
            mode="research_brief",
        )

        return BriefGenerationResult(
            topic=self.config.topic,
            report_path=workflow_result.report_path,
            message="Completed Agentic RAG research brief generation from indexed industry evidence.",
            workflow_result=workflow_result,
        )

    def ask(self, question: str, thread_id: str | None = None, user_id: str | None = None) -> dict:
        if self.workflow is None:
            raise RuntimeError("Workflow components were not initialized.")
        resolved_thread_id = thread_id or self.config.graph.default_thread_id
        resolved_user_id = user_id or self.config.graph.default_user_id
        try:
            result = self.workflow.invoke(
                question=question,
                thread_id=resolved_thread_id,
                user_id=resolved_user_id,
                mode="qa",
            )
        except WorkflowExecutionError as exc:
            self._save_failed_performance(
                question=question,
                performance=exc.performance,
                error_type=exc.error_type,
                error_message=exc.message,
                thread_id=resolved_thread_id,
                user_id=resolved_user_id,
                mode="qa",
            )
            raise
        self._save_workflow_trace(
            result,
            thread_id=resolved_thread_id,
            user_id=resolved_user_id,
            mode="qa",
        )
        self._save_performance(
            result,
            thread_id=resolved_thread_id,
            user_id=resolved_user_id,
            mode="qa",
        )
        return result.to_dict()

    def ask_stream(
        self,
        question: str,
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def emit(event: dict[str, Any]) -> None:
            events.put(event)

        def run() -> None:
            try:
                with workflow_event_sink(emit):
                    result = self.ask(question, thread_id=thread_id, user_id=user_id)
                emit({"event": "result", "result": result})
                emit({"event": "done"})
            except Exception as exc:
                emit(
                    {
                        "event": "error",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    }
                )
            finally:
                events.put(None)

        threading.Thread(target=run, name="agent-sse-stream", daemon=True).start()
        while True:
            event = events.get()
            if event is None:
                break
            yield event

    def ingest_document(self, file_path: str, document_id: str | None = None, title: str | None = None, industry: str = "", company: str = "", product_line: str = "", document_type: str = "", source_url: str = "") -> dict:
        resolved_source_url = source_url or file_path
        metadata = {"document_id": document_id or file_path, "source_url": resolved_source_url, "industry": industry, "company": company, "product_line": product_line, "document_type": document_type or "research_material"}
        parsed = self.rag.add_document(file_path, metadata=metadata)
        record = ResearchDocumentRecord(
            document_id=document_id or file_path,
            title=title or parsed.get("title", file_path),
            file_path=file_path,
            relevance_score=1.0,
            summary=parsed.get("summary", "")[:1200],
            source_url=resolved_source_url,
            indexed=True,
            added_at=dt.datetime.now().isoformat(timespec="seconds"),
            notes={"ingest_mode": "local_research_document"},
            industry=industry,
            company=company,
            product_line=product_line,
            document_type=document_type or "research_material",
        )
        self.document_store.upsert(record)
        return record.to_dict()

    def show_graph(self) -> str:
        if self.workflow is None:
            raise RuntimeError("Workflow components were not initialized.")
        return self.workflow.draw_mermaid()

    def rebuild_index(self) -> dict[str, int]:
        return self.rag.rebuild(self.document_store.all_records())

    def search_web(self, query: str, *, max_results: int | None = None, include_domains: list[str] | None = None) -> list[dict]:
        return self.web_research.search(query, max_results=max_results, include_domains=include_domains)

    def collect_research_pdfs(self, query: str, *, max_results: int | None = None, include_domains: list[str] | None = None) -> list[dict]:
        collected = []
        for item in self.web_research.search_pdfs(query, max_results=max_results, include_domains=include_domains):
            download = self.web_research.download_pdf(item["url"], title=item["title"])
            collected.append(
                self.ingest_document(
                    download["file_path"], document_id=download["sha256"], title=item["title"],
                    document_type="white_paper", source_url=item["url"],
                )
            )
        return collected

    def migrate_legacy_memory(self, file_path: str) -> dict[str, int]:
        if self.long_term_memory is None:
            raise RuntimeError("Long-term memory was not initialized.")
        return self.long_term_memory.migrate_legacy_json(file_path)

    def generate_performance_report(self, limit: int | None = None) -> dict:
        return self.performance_store.generate_report(limit=limit)

    def _save_workflow_trace(self, result, *, thread_id: str, user_id: str, mode: str) -> None:
        if not self.trace_store:
            return
        self.trace_store.save_workflow_result(
            run_id=str(uuid.uuid4()),
            thread_id=thread_id,
            user_id=user_id,
            mode=mode,
            result=result.to_dict(),
        )

    def _save_performance(self, result, *, thread_id: str, user_id: str, mode: str) -> None:
        if not self.config.observability.enabled or not self.config.observability.persist_jsonl:
            return
        self.performance_store.append(
            {
                "run_id": str(uuid.uuid4()),
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "thread_id": thread_id,
                "user_id": user_id,
                "mode": mode,
                "question": result.question,
                "quality_score": float(result.reflection.get("score", 0) or 0),
                "performance": result.performance,
                "tool_history": result.tool_history,
            }
        )

    def _save_failed_performance(
        self,
        *,
        question: str,
        performance: dict,
        error_type: str,
        error_message: str,
        thread_id: str,
        user_id: str,
        mode: str,
    ) -> None:
        if not self.config.observability.enabled or not self.config.observability.persist_jsonl:
            return
        self.performance_store.append(
            {
                "run_id": str(uuid.uuid4()),
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "thread_id": thread_id,
                "user_id": user_id,
                "mode": mode,
                "question": question,
                "quality_score": 0.0,
                "performance": performance,
                "tool_history": [],
                "error_type": error_type,
                "error_message": error_message,
            }
        )
