from __future__ import annotations

import importlib
import sys
import types

from competitive_research_agent.parser import LanguageDetector, ResearchDocumentParser


def test_language_detector_recognizes_mixed_content():
    text = "本文系统对比主要产品的功能、定价、客户和技术路线，并分析 AI coding assistant capabilities."

    assert LanguageDetector.detect(text) == "mixed"


def test_docling_markdown_extracts_competitive_research_sections(monkeypatch):
    parser = ResearchDocumentParser("competitor-brief.pdf", backend="docling", fallback_backend="pypdf")
    markdown = """# AI Coding Assistants Competitive Brief

## 执行摘要
本文系统对比 Cursor、GitHub Copilot 等编码助手的产品定位、功能和客户群体。

## 核心功能
这些产品支持代码补全、仓库理解以及 Agent mode and tool execution.

## Pricing
The products use individual and enterprise subscription plans.

## 技术架构
The products combine code models, retrieval and tool execution.

## 风险
Enterprise adoption depends on security, governance and cost controls.
"""
    monkeypatch.setattr(
        parser,
        "_parse_with_docling",
        lambda: parser._build_result(markdown, [], parser_backend="docling", markdown=True),
    )

    parsed = parser.parse()

    assert parsed["parser_backend"] == "docling"
    assert parsed["language"] == "mixed"
    assert parsed["title"] == "AI Coding Assistants Competitive Brief"
    assert "Cursor" in parsed["summary"]
    assert "Agent mode" in parsed["features"]
    assert "subscription plans" in parsed["pricing"]
    assert "retrieval" in parsed["technology"]
    assert "governance" in parsed["limitations"]


def test_parser_falls_back_to_pypdf(monkeypatch):
    parser = ResearchDocumentParser("competitor-brief.pdf", backend="docling", fallback_backend="pypdf")
    monkeypatch.setattr(
        parser,
        "_parse_with_docling",
        lambda: (_ for _ in ()).throw(RuntimeError("docling unavailable")),
    )
    monkeypatch.setattr(
        parser,
        "_parse_with_pypdf",
        lambda: {"title": "Fallback", "parser_backend": "pypdf"},
    )

    parsed = parser.parse()

    assert parsed["parser_backend"] == "pypdf"
    assert parsed["parser_fallback_reason"] == "docling unavailable"


def test_markdown_document_is_parsed_without_pdf_backend(tmp_path):
    document = tmp_path / "product-brief.md"
    document.write_text(
        "# Product Brief\n\n## Pricing\nTeam subscriptions and enterprise contracts.",
        encoding="utf-8",
    )

    parsed = ResearchDocumentParser(str(document)).parse()

    assert parsed["parser_backend"] == "markdown"
    assert parsed["title"] == "Product Brief"
    assert "enterprise contracts" in parsed["pricing"]


def test_pypdf_style_competitor_sections_and_title_are_recognized():
    parser = ResearchDocumentParser("market-landscape.pdf", backend="pypdf", fallback_backend="")
    text = """AI Coding Assistant Market Landscape 2026
Competitive Intelligence Team
EXECUTIVE SUMMARY
This report compares leading AI coding assistants for enterprise buyers.
1. MARKET OVERVIEW
The market is growing across individual developer and enterprise segments.
2. PRICING
Vendors combine monthly subscriptions with enterprise contracts.
3. COMPETITIVE LANDSCAPE
Products differ in agent workflows, IDE coverage, governance and deployment options.
"""

    parsed = parser._build_result(text, [], parser_backend="pypdf", markdown=False)

    assert parsed["title"] == "AI Coding Assistant Market Landscape 2026"
    assert parsed["summary"].startswith("This report compares")
    assert parsed["market"].startswith("The market is growing")
    assert parsed["pricing"].startswith("Vendors combine")
    assert parsed["competition"].startswith("Products differ")


def test_docling_converter_is_cached(monkeypatch):
    import competitive_research_agent.parser as parser_module

    created = {"count": 0}

    class FakeConverter:
        def convert(self, _file_path):
            raise AssertionError("convert should not be called in this cache test")

    def build_converter():
        created["count"] += 1
        return FakeConverter()

    fake_docling = types.ModuleType("docling")
    fake_converter_module = types.ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = build_converter
    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)

    parser_module = importlib.reload(parser_module)
    parser_module._get_docling_converter.cache_clear()

    first = parser_module._get_docling_converter()
    second = parser_module._get_docling_converter()

    assert first is second
    assert created["count"] == 1
