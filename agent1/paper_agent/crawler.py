from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import arxiv

from .config import KeywordConfig, SearchConfig
from .logging_utils import get_logger
from .models import PaperRecord


class ArxivCrawler:
    def __init__(self, download_dir: str, keyword_config: KeywordConfig | None = None):
        self.download_dir = os.path.abspath(download_dir)
        self.keyword_config = keyword_config or KeywordConfig()
        self.logger = get_logger(self.__class__.__name__)
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)

    def _build_query(self, query: str | None) -> str:
        if query:
            return query
        core_terms = self.keyword_config.core[:5]
        return " OR ".join(f'"{term}"' for term in core_terms)

    def _calculate_relevance(self, title: str, abstract: str) -> tuple[float, list[str]]:
        text_lower = f"{title} {abstract}".lower()
        title_lower = title.lower()
        score = 0.0
        matched_keywords: list[str] = []

        for keyword in self.keyword_config.core:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                score += 0.8 if keyword_lower in title_lower else 0.4
                matched_keywords.append(f"core:{keyword}")

        for keyword in self.keyword_config.extended:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                score += 0.4 if keyword_lower in title_lower else 0.2
                matched_keywords.append(f"extended:{keyword}")

        for keyword in self.keyword_config.exclude:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                score -= 0.3
                matched_keywords.append(f"exclude:{keyword}")

        return max(0.0, min(1.0, score / 5)), matched_keywords

    def _safe_filename(self, title: str) -> str:
        return re.sub(r"[^\w\-_\. ]", "_", title).strip()[:80] or "paper"

    def _extract_paper_id(self, result: arxiv.Result) -> str:
        entry_id = getattr(result, "entry_id", "")
        return entry_id.rstrip("/").split("/")[-1] if entry_id else self._safe_filename(result.title)

    def search_metadata(self, query: str, max_results: int = 5) -> list[dict]:
        client = arxiv.Client(page_size=20, delay_seconds=2, num_retries=3)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        records: list[dict] = []
        for result in client.results(search):
            title = result.title.strip()
            abstract = result.summary.replace("\n", " ").strip()
            relevance, matched = self._calculate_relevance(title, abstract)
            records.append(
                {
                    "paper_id": self._extract_paper_id(result),
                    "title": title,
                    "summary": abstract[:1200],
                    "published": getattr(result, "published", dt.datetime.min).isoformat(),
                    "source_url": getattr(result, "entry_id", ""),
                    "pdf_url": getattr(result, "pdf_url", ""),
                    "relevance_score": relevance,
                    "matched_keywords": matched[:10],
                }
            )
        return records

    def search_and_download(self, search_config: SearchConfig) -> list[PaperRecord]:
        query = self._build_query(search_config.query)
        client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
        search = arxiv.Search(
            query=query,
            max_results=search_config.max_results * 5,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        self.logger.info(
            "Searching arXiv | query=%s | threshold=%.2f | target=%s",
            query,
            search_config.relevance_threshold,
            search_config.max_results,
        )

        collected: list[PaperRecord] = []
        for result in client.results(search):
            if len(collected) >= search_config.max_results:
                break

            title = result.title.strip()
            abstract = result.summary.replace("\n", " ").strip()
            relevance_score, matched_keywords = self._calculate_relevance(title, abstract)
            if relevance_score < search_config.relevance_threshold:
                continue

            safe_name = self._safe_filename(title)
            expected_path = os.path.join(self.download_dir, f"{safe_name}.pdf")

            if os.path.exists(expected_path):
                pdf_path = expected_path
            else:
                downloaded_path = result.download_pdf(dirpath=self.download_dir)
                if safe_name not in downloaded_path:
                    renamed_path = os.path.join(self.download_dir, f"{safe_name}.pdf")
                    os.replace(downloaded_path, renamed_path)
                    pdf_path = renamed_path
                else:
                    pdf_path = downloaded_path

            collected.append(
                PaperRecord(
                    paper_id=self._extract_paper_id(result),
                    title=title,
                    pdf_path=os.path.abspath(pdf_path),
                    relevance_score=relevance_score,
                    summary=abstract[:1200],
                    published=getattr(result, "published", dt.datetime.min).isoformat(),
                    source_url=getattr(result, "entry_id", ""),
                    notes={"matched_keywords": ", ".join(matched_keywords[:10])},
                )
            )

        self.logger.info("Search complete. downloaded_or_reused=%s", len(collected))
        return collected

    def search_daily(
        self,
        max_results: int = 5,
        relevance_threshold: float = 0.3,
        query: str | None = None,
    ) -> tuple[list[str], list[dict]]:
        records = self.search_and_download(
            SearchConfig(
                max_results=max_results,
                relevance_threshold=relevance_threshold,
                query=query,
            )
        )
        return [record.pdf_path for record in records], [record.to_dict() for record in records]
