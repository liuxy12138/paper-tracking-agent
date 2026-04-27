from __future__ import annotations

import re
from typing import Dict

from langchain_community.document_loaders import PyPDFLoader


class LanguageDetector:
    @staticmethod
    def detect(text: str) -> str:
        if not text:
            return "en"
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text[:1000])
        chinese_ratio = len(chinese_chars) / max(len(text[:1000]), 1)
        return "zh" if chinese_ratio > 0.15 else "en"


class UniversalPaperParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.full_text = ""
        self.language = "en"

    def parse(self) -> Dict:
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        self.full_text = "\n".join(page.page_content for page in pages)
        self.language = LanguageDetector.detect(self.full_text)
        return self._parse_chinese() if self.language == "zh" else self._parse_english()

    def _parse_chinese(self) -> Dict:
        return {
            "title": self._extract_title_chinese(),
            "abstract": self._extract_abstract_chinese(),
            "introduction": self._extract_section(["引言", "绪论", "研究背景"]),
            "related_work": self._extract_section(["相关工作", "文献综述", "研究现状"]),
            "method": self._extract_section(["方法", "算法", "模型"]),
            "experiment": self._extract_section(["实验", "结果", "评估"]),
            "conclusion": self._extract_section(["结论", "总结", "展望"]),
            "full_text": self.full_text,
            "language": "zh",
        }

    def _parse_english(self) -> Dict:
        return {
            "title": self._extract_title_english(),
            "abstract": self._extract_abstract_english(),
            "introduction": self._extract_section(["Introduction"]),
            "related_work": self._extract_section(["Related Work", "Background"]),
            "method": self._extract_section(["Method", "Approach", "Model Architecture"]),
            "experiment": self._extract_section(["Experiments", "Results", "Evaluation"]),
            "conclusion": self._extract_section(["Conclusion", "Discussion"]),
            "full_text": self.full_text,
            "language": "en",
        }

    def _extract_title_chinese(self) -> str:
        lines = self.full_text.split("\n")[:20]
        for line in lines:
            line = line.strip()
            if re.search(r"[\u4e00-\u9fff]", line) and 10 < len(line) < 100:
                return line
        return lines[0][:100] if lines else "未识别到标题"

    def _extract_title_english(self) -> str:
        lines = self.full_text.split("\n")[:20]
        for line in lines:
            line = line.strip()
            if line and 15 < len(line) < 150 and not re.search(r"@|\.com", line):
                if re.search(r"[A-Z]", line):
                    return line
        return lines[0][:100] if lines else "Unknown Title"

    def _extract_abstract_chinese(self) -> str:
        patterns = [
            r"摘要[：:]*\s*\n*(.*?)(?=\n\n|\n[一二三四五六七八九十]|\n参考文献|$)",
            r"【摘要】\s*(.*?)【",
            r"\[摘要\]\s*(.*?)\[",
        ]
        for pattern in patterns:
            match = re.search(pattern, self.full_text, re.DOTALL)
            if match:
                text = re.sub(r"\n+", " ", match.group(1).strip())
                if len(text) > 50:
                    return text[:1500]
        return ""

    def _extract_abstract_english(self) -> str:
        patterns = [
            r"Abstract[:\s]*\n(.*?)(?=\n1\.|Introduction|$)",
            r"ABSTRACT\s*\n(.*?)(?=\n1\.|INTRODUCTION|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, self.full_text, re.IGNORECASE | re.DOTALL)
            if match:
                text = re.sub(r"\n+", " ", match.group(1).strip())
                if len(text) > 100:
                    return text[:2000]
        return ""

    def _extract_section(self, section_names: list[str]) -> str:
        text = self.full_text
        for name in section_names:
            patterns = [
                rf"\n\s*{re.escape(name)}\s*\n",
                rf"\n\s*\d+\.\s*{re.escape(name)}\s*\n",
                rf"\n\s*\d+\.\d+\s*{re.escape(name)}\s*\n",
                rf"\n\s*[一二三四五六七八九十]+\s*[、.．]\s*{re.escape(name)}\s*\n",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if not match:
                    continue
                start = match.end()
                next_pattern = r"\n\s*(?:\d+\.\s*[A-Z]|参考文献|References|Conclusion|Acknowledgments|结论|致谢)"
                next_match = re.search(next_pattern, text[start : start + 8000], re.IGNORECASE)
                end = start + next_match.start() if next_match else min(start + 5000, len(text))
                content = re.sub(r"\n+", " ", text[start:end].strip())
                if len(content) > 100:
                    return content[:2500]
        return ""
