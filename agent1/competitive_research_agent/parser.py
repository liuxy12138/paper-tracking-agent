from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader


LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_docling_converter():
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError(
            "Docling is not installed. Install project dependencies or set rag.parser_backend to pypdf."
        ) from exc
    return DocumentConverter()


class LanguageDetector:
    @staticmethod
    def detect(text: str) -> str:
        sample = text[:4000]
        if not sample:
            return "en"
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", sample))
        latin_count = len(re.findall(r"[A-Za-z]", sample))
        if chinese_count and latin_count and min(chinese_count, latin_count) / max(chinese_count, latin_count) >= 0.1:
            return "mixed"
        return "zh" if chinese_count > latin_count else "en"


class ResearchDocumentParser:
    """Parse competitive-research documents with Docling and a PyPDF fallback."""

    SECTION_ALIASES = {
        "overview": (
            "executive summary", "overview", "company overview", "product overview",
            "introduction", "about us", "概述", "执行摘要", "公司简介", "产品概览", "背景",
        ),
        "market": (
            "market", "market overview", "market size", "market trends", "industry trends",
            "market landscape", "industry landscape", "市场", "市场规模", "市场趋势", "行业趋势", "市场格局",
        ),
        "product": (
            "product", "products", "product portfolio", "solution", "solutions", "offering",
            "产品", "产品组合", "解决方案", "服务",
        ),
        "features": (
            "features", "capabilities", "key features", "functionality", "use cases",
            "功能", "核心功能", "能力", "应用场景",
        ),
        "pricing": (
            "pricing", "plans", "packages", "subscription", "commercial model", "business model",
            "定价", "价格", "套餐", "订阅", "商业模式", "收费模式",
        ),
        "customers": (
            "customers", "target customers", "target users", "customer segments", "personas",
            "客户", "目标客户", "目标用户", "用户画像", "客户群体",
        ),
        "competition": (
            "competition", "competitors", "competitive analysis", "competitive landscape",
            "comparison", "alternatives", "竞品", "竞争分析", "竞争格局", "产品对比", "替代方案",
        ),
        "technology": (
            "technology", "technical architecture", "architecture", "implementation", "security",
            "integration", "技术", "技术架构", "系统架构", "实现方案", "安全", "集成",
        ),
        "limitations": (
            "limitations", "constraints", "risks", "challenges", "known issues",
            "局限", "限制", "风险", "挑战", "已知问题",
        ),
        "roadmap": (
            "roadmap", "future plans", "future outlook", "strategy", "next steps",
            "路线图", "未来规划", "发展规划", "战略", "下一步",
        ),
        "case_studies": (
            "case studies", "customer stories", "success stories", "benchmarks", "results",
            "案例", "客户案例", "成功案例", "基准测试", "效果",
        ),
        "key_takeaways": (
            "key takeaways", "summary", "conclusion", "recommendations",
            "关键结论", "总结", "结论", "建议",
        ),
    }

    def __init__(
        self,
        file_path: str,
        backend: str = "docling",
        fallback_backend: str = "pypdf",
    ):
        self.file_path = file_path
        self.backend = backend.casefold()
        self.fallback_backend = fallback_backend.casefold()

    def parse(self) -> dict[str, Any]:
        suffix = Path(self.file_path).suffix.casefold()
        if suffix in {".md", ".markdown", ".txt"}:
            return self._parse_text_file(markdown=suffix in {".md", ".markdown"})
        try:
            return self._parse_backend(self.backend)
        except Exception as exc:
            if not self.fallback_backend or self.fallback_backend == self.backend:
                raise
            LOGGER.warning(
                "Document parser backend %s failed for %s; falling back to %s: %s",
                self.backend,
                self.file_path,
                self.fallback_backend,
                exc,
            )
            parsed = self._parse_backend(self.fallback_backend)
            parsed["parser_fallback_reason"] = str(exc)
            return parsed

    def _parse_backend(self, backend: str) -> dict[str, Any]:
        if backend == "docling":
            return self._parse_with_docling()
        if backend in {"pypdf", "pypdfloader"}:
            return self._parse_with_pypdf()
        raise ValueError(f"Unsupported parser backend: {backend}")

    @classmethod
    def warmup_backend(cls, backend: str) -> None:
        normalized = backend.casefold()
        if normalized == "docling":
            _get_docling_converter()

    def _parse_with_docling(self) -> dict[str, Any]:
        result = _get_docling_converter().convert(self.file_path)
        markdown = result.document.export_to_markdown()
        if not markdown.strip():
            raise ValueError("Docling returned no document content.")
        try:
            pages = self._load_pages()
        except Exception as exc:
            LOGGER.warning("Could not load page text for Docling output %s: %s", self.file_path, exc)
            pages = []
        return self._build_result(markdown, pages, parser_backend="docling", markdown=True)

    def _parse_with_pypdf(self) -> dict[str, Any]:
        pages = self._load_pages()
        full_text = "\n".join(page["content"] for page in pages)
        if not full_text.strip():
            raise ValueError("PyPDF returned no document content.")
        return self._build_result(full_text, pages, parser_backend="pypdf", markdown=False)

    def _parse_text_file(self, *, markdown: bool) -> dict[str, Any]:
        text = Path(self.file_path).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("Text document contains no content.")
        backend = "markdown" if markdown else "text"
        return self._build_result(text, [], parser_backend=backend, markdown=markdown)

    def _load_pages(self) -> list[dict[str, Any]]:
        pages = PyPDFLoader(self.file_path).load()
        return [
            {
                "page_number": int(page.metadata.get("page", index)) + 1,
                "content": page.page_content,
            }
            for index, page in enumerate(pages)
        ]

    def _build_result(
        self,
        full_text: str,
        pages: list[dict[str, Any]],
        *,
        parser_backend: str,
        markdown: bool,
    ) -> dict[str, Any]:
        sections = self._extract_markdown_sections(full_text) if markdown else {}
        parsed: dict[str, Any] = {
            "title": self._extract_title(full_text, markdown=markdown),
            "full_text": full_text,
            "language": LanguageDetector.detect(full_text),
            "pages": pages,
            "parser_backend": parser_backend,
        }
        for section_name, aliases in self.SECTION_ALIASES.items():
            parsed[section_name] = sections.get(section_name) or self._extract_section(full_text, aliases)
        parsed["summary"] = parsed.get("overview", "") or self._extract_inline_summary(full_text)
        return parsed

    def _extract_markdown_sections(self, text: str) -> dict[str, str]:
        matches = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", text))
        extracted: dict[str, list[str]] = {}
        for index, match in enumerate(matches):
            section_name = self._classify_heading(match.group(1))
            if not section_name:
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[match.end() : end].strip()
            if content:
                extracted.setdefault(section_name, []).append(content)
        return {name: "\n\n".join(parts) for name, parts in extracted.items()}

    def _classify_heading(self, heading: str) -> str | None:
        normalized = re.sub(r"^[\d.\s、一二三四五六七八九十]+", "", heading).strip().casefold()
        normalized = re.sub(r"[*_`:#]+", "", normalized).strip()
        for section_name, aliases in self.SECTION_ALIASES.items():
            if any(normalized == alias or normalized.startswith(f"{alias} ") for alias in aliases):
                return section_name
        return None

    def _extract_title(self, text: str, *, markdown: bool) -> str:
        if markdown:
            heading = re.search(r"(?m)^\s{0,3}#\s+(.+?)\s*$", text)
            if heading:
                return heading.group(1).strip()[:300]
        candidates: list[str] = []
        for line in text.splitlines()[:40]:
            candidate = line.strip().strip("#").strip()
            if re.match(r"(?i)^(?:executive summary|overview|执行摘要|概述)\b", candidate):
                break
            if not 10 < len(candidate) < 220:
                continue
            if re.search(
                r"(?i)@|copyright|personal use|permission|doi:|^©",
                candidate,
            ):
                continue
            if candidate.count(",") >= 2:
                continue
            candidates.append(candidate)
        if candidates:
            return candidates[0][:300]
        return "Unknown Title"

    def _extract_section(self, text: str, aliases: tuple[str, ...]) -> str:
        alias_pattern = "|".join(re.escape(alias) for alias in aliases)
        heading_pattern = re.compile(
            rf"(?im)^\s*(?:#{1,6}\s*)?(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.\s、]*)?"
            rf"(?:{alias_pattern})\s*:?\s*$"
        )
        match = heading_pattern.search(text)
        if not match:
            return ""
        all_aliases = [alias for values in self.SECTION_ALIASES.values() for alias in values]
        all_alias_pattern = "|".join(re.escape(alias) for alias in all_aliases)
        next_heading = re.search(
            rf"(?im)^\s*(?:#{{1,6}}\s+|(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.\s、]+)?"
            rf"(?:{all_alias_pattern})\s*:?)\s*$",
            text[match.end() :],
        )
        end = match.end() + next_heading.start() if next_heading else min(match.end() + 12000, len(text))
        return text[match.end() : end].strip()

    def _extract_inline_summary(self, text: str) -> str:
        match = re.search(
            r"(?is)\b(?:executive summary|overview)\s*[—–:-]\s*(.*?)"
            r"(?=\n\s*(?:#{1,6}\s+|(?:1|I)[.\s]+[A-Z]))",
            text,
        )
        return match.group(1).strip() if match else ""
