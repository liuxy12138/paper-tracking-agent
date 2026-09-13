from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .config import SearchConfig


class WebResearchClient:
    """Tavily search plus bounded, PDF-only downloads for research ingestion."""

    def __init__(self, config: SearchConfig, document_dir: str, *, opener: Callable[..., Any] = urlopen):
        self.config = config
        self.document_dir = Path(document_dir)
        self.document_dir.mkdir(parents=True, exist_ok=True)
        self._opener = opener

    @property
    def available(self) -> bool:
        return bool(self.config.web_enabled and self.config.web_api_key)

    def _ensure_available(self) -> None:
        if not self.config.web_enabled:
            raise RuntimeError("Web search is disabled in search.web_enabled.")
        if self.config.web_provider.lower() != "tavily":
            raise ValueError(f"Unsupported web search provider: {self.config.web_provider}")
        if not self.config.web_api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")

    def search(self, query: str, *, max_results: int | None = None, include_domains: list[str] | None = None, search_depth: str | None = None) -> list[dict[str, Any]]:
        self._ensure_available()
        payload: dict[str, Any] = {
            "api_key": self.config.web_api_key,
            "query": query,
            "max_results": max(1, min(max_results or self.config.max_results, 20)),
            "search_depth": search_depth or self.config.web_search_depth,
            "include_answer": False,
            "include_raw_content": False,
        }
        if include_domains:
            payload["include_domains"] = include_domains
        request = Request(self.config.web_endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "User-Agent": self.config.web_user_agent}, method="POST")
        with self._opener(request, timeout=self.config.web_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        results = []
        for item in body.get("results", []):
            url = str(item.get("url", "")).strip()
            if url:
                results.append({"title": str(item.get("title", url)), "url": url, "content": str(item.get("content", "")), "score": float(item.get("score", 0.0) or 0.0), "is_pdf": urlparse(url).path.lower().endswith(".pdf")})
        return results

    def search_pdfs(self, query: str, *, max_results: int | None = None, include_domains: list[str] | None = None) -> list[dict[str, Any]]:
        pdf_query = query if "filetype:pdf" in query.lower() else f"{query} filetype:pdf"
        return [item for item in self.search(pdf_query, max_results=max_results, include_domains=include_domains) if item["is_pdf"]]

    def download_pdf(self, url: str, *, title: str = "") -> dict[str, str]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only HTTP(S) PDF URLs are supported.")
        hostname = parsed.hostname.casefold()
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise ValueError("Local network URLs are not allowed.")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("Private or local network URLs are not allowed.")
        request = Request(url, headers={"User-Agent": self.config.web_user_agent})
        with self._opener(request, timeout=self.config.web_timeout_seconds) as response:
            content_type = str(response.headers.get("Content-Type", "")).lower()
            length = response.headers.get("Content-Length")
            if length and int(length) > self.config.web_max_download_bytes:
                raise ValueError("PDF exceeds the configured download size limit.")
            data = response.read(self.config.web_max_download_bytes + 1)
        if len(data) > self.config.web_max_download_bytes:
            raise ValueError("PDF exceeds the configured download size limit.")
        if not data.startswith(b"%PDF-"):
            raise ValueError(f"Downloaded resource is not a PDF (Content-Type: {content_type or 'unknown'}).")
        digest = hashlib.sha256(data).hexdigest()
        source_name = Path(parsed.path).stem or title or "research-document"
        safe_name = re.sub(r"[^0-9A-Za-z._-]+", "-", source_name).strip("-.")[:80] or "research-document"
        target = self.document_dir / f"{safe_name}-{digest[:12]}.pdf"
        if not target.exists():
            target.write_bytes(data)
        return {"file_path": str(target.resolve()), "sha256": digest, "source_url": url}
