from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from competitive_research_agent.config import SearchConfig
from competitive_research_agent.web_research import WebResearchClient


class FakeResponse:
    def __init__(self, data: bytes, headers: dict[str, str] | None = None):
        self._stream = io.BytesIO(data)
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_tavily_search_normalizes_results(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        body = {"results": [{"title": "White paper", "url": "https://example.com/report.pdf", "content": "Market evidence", "score": 0.91}]}
        return FakeResponse(json.dumps(body).encode())

    client = WebResearchClient(SearchConfig(web_api_key="secret"), str(tmp_path), opener=opener)
    results = client.search_pdfs("agent market", max_results=3)

    assert results[0]["is_pdf"] is True
    assert results[0]["score"] == 0.91
    assert b"filetype:pdf" in calls[0][0].data


def test_pdf_download_is_content_addressed_and_deduplicated(tmp_path):
    pdf = b"%PDF-1.7\nminimal-test"
    client = WebResearchClient(SearchConfig(web_api_key="secret"), str(tmp_path), opener=lambda request, timeout: FakeResponse(pdf, {"Content-Type": "application/pdf"}))
    first = client.download_pdf("https://example.com/white-paper.pdf")
    second = client.download_pdf("https://example.com/white-paper.pdf")

    assert first == second
    assert Path(first["file_path"]).exists()
    assert len(list(tmp_path.glob("*.pdf"))) == 1


def test_pdf_download_rejects_non_pdf_and_local_urls(tmp_path):
    client = WebResearchClient(SearchConfig(web_api_key="secret"), str(tmp_path), opener=lambda request, timeout: FakeResponse(b"<html></html>", {"Content-Type": "text/html"}))
    with pytest.raises(ValueError, match="not a PDF"):
        client.download_pdf("https://example.com/report.pdf")
    with pytest.raises(ValueError, match="Local network"):
        client.download_pdf("http://localhost/report.pdf")
