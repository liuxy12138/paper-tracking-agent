from __future__ import annotations

import datetime as dt

from .config import AgentConfig
from .crawler import ArxivCrawler
from .logging_utils import get_logger
from .memory import LongTermMemoryStore, ThreadHistoryStore
from .models import PaperRecord, PipelineResult
from .parser import UniversalPaperParser
from .rag import UniversalPaperRAG
from .storage import PaperIndexStore
from .tools import ResearchToolbox
from .workflow import LangGraphResearchWorkflow


class DailyResearchAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        self.crawler = ArxivCrawler(
            download_dir=config.paths.pdf_dir,
            keyword_config=config.keywords,
        )
        self.rag = UniversalPaperRAG(
            api_key=config.api_key,
            persist_dir=config.paths.vector_dir,
            rag_config=config.rag,
        )
        self.paper_store = PaperIndexStore(config.paths.metadata_path)
        self.thread_history = ThreadHistoryStore(config.paths.thread_history_path)
        self.long_term_memory = LongTermMemoryStore(
            path=config.paths.long_term_memory_path,
            embeddings=self.rag.embeddings,
        )
        self.toolbox = ResearchToolbox(
            crawler=self.crawler,
            rag=self.rag,
            parser_cls=UniversalPaperParser,
            paper_store=self.paper_store,
        )
        self.workflow = LangGraphResearchWorkflow(
            config=config,
            rag=self.rag,
            toolbox=self.toolbox,
            thread_history=self.thread_history,
            long_term_memory=self.long_term_memory,
        )

    def _index_records(self, records: list[PaperRecord]) -> tuple[int, int]:
        indexed_count = 0
        skipped_count = 0

        for record in records:
            existing = self.paper_store.get(record.paper_id)
            if existing and existing.indexed:
                skipped_count += 1
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
            record.added_at = dt.datetime.now().isoformat(timespec="seconds")
            self.paper_store.upsert(record)
            indexed_count += 1

        return indexed_count, skipped_count

    def run_daily(self) -> PipelineResult:
        self.logger.info("Daily pipeline started at %s", dt.datetime.now().isoformat(timespec="seconds"))
        records = self.crawler.search_and_download(self.config.search)
        indexed_count, skipped_count = self._index_records(records)

        workflow_result = self.workflow.invoke(
            question=f"Generate a daily research brief about {self.config.topic} based on the current indexed evidence.",
            thread_id=f"daily-{dt.date.today().isoformat()}",
            user_id=self.config.graph.default_user_id,
            mode="daily_report",
        )

        return PipelineResult(
            topic=self.config.topic,
            downloaded_count=len(records),
            indexed_count=indexed_count,
            skipped_count=skipped_count,
            review_path=workflow_result.report_path,
            message="Completed daily ingestion and multi-agent report generation.",
            workflow_result=workflow_result,
        )

    def ask(self, question: str, thread_id: str | None = None, user_id: str | None = None) -> dict:
        result = self.workflow.invoke(
            question=question,
            thread_id=thread_id or self.config.graph.default_thread_id,
            user_id=user_id or self.config.graph.default_user_id,
            mode="qa",
        )
        return result.to_dict()

    def ingest_local_pdf(self, file_path: str, paper_id: str | None = None, title: str | None = None) -> dict:
        metadata = {"paper_id": paper_id or file_path, "source_url": file_path}
        parsed = self.rag.add_paper(file_path, metadata=metadata)
        record = PaperRecord(
            paper_id=paper_id or file_path,
            title=title or parsed.get("title", file_path),
            pdf_path=file_path,
            relevance_score=1.0,
            summary=parsed.get("abstract", "")[:1200],
            indexed=True,
            added_at=dt.datetime.now().isoformat(timespec="seconds"),
            notes={"ingest_mode": "local_pdf"},
        )
        self.paper_store.upsert(record)
        return record.to_dict()

    def show_graph(self) -> str:
        return self.workflow.draw_mermaid()
