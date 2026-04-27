from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RagConfig
from .logging_utils import get_logger
from .models import RetrievedChunk
from .parser import UniversalPaperParser


class UniversalPaperRAG:
    def __init__(self, api_key: str, persist_dir: str, rag_config: RagConfig | None = None):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.api_key = api_key.strip()
        self.persist_dir = os.path.abspath(persist_dir)
        self.rag_config = rag_config or RagConfig()
        self.logger = get_logger(self.__class__.__name__)

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.rag_config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: FAISS | None = None
        self.paper_info: dict = {}
        self._load_vectorstore()

    def _load_vectorstore(self) -> None:
        faiss_file = os.path.join(self.persist_dir, "index.faiss")
        pkl_file = os.path.join(self.persist_dir, "index.pkl")
        if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
            self.logger.info("No persisted vectorstore found. A new index will be created.")
            return
        self.vectorstore = FAISS.load_local(
            self.persist_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.logger.info("Loaded vectorstore. ntotal=%s", self.vectorstore.index.ntotal)

    def _save_vectorstore(self) -> None:
        if self.vectorstore is not None:
            self.vectorstore.save_local(self.persist_dir)

    def _split_section(self, text: str, section_name: str, base_metadata: dict) -> list[Document]:
        if not text or len(text.strip()) < 80:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rag_config.chunk_size,
            chunk_overlap=self.rag_config.chunk_overlap,
        )
        chunks = splitter.split_text(text)
        documents: list[Document] = []
        for index, chunk in enumerate(chunks[: self.rag_config.max_chunks_per_paper]):
            if len(chunk.strip()) <= 100:
                continue
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={**base_metadata, "section": section_name, "chunk_id": index},
                )
            )
        return documents

    def _build_documents(self, file_path: str, metadata: dict | None = None) -> tuple[dict, list[Document]]:
        parser = UniversalPaperParser(file_path)
        paper_info = parser.parse()
        base_metadata = {
            "source": os.path.basename(file_path),
            "title": paper_info.get("title", "Unknown"),
            "type": "paper_chunk",
        }
        if metadata:
            base_metadata.update(metadata)

        documents: list[Document] = []
        for section_name in ("abstract", "introduction", "related_work", "method", "experiment", "conclusion"):
            documents.extend(self._split_section(paper_info.get(section_name, ""), section_name, base_metadata))

        if not documents:
            documents.extend(self._split_section(paper_info["full_text"], "full_text", base_metadata))
        else:
            documents.extend(self._split_section(paper_info["full_text"][:5000], "full_text_head", base_metadata))

        return paper_info, documents

    def add_paper(self, file_path: str, metadata: dict | None = None) -> dict:
        paper_info, documents = self._build_documents(file_path, metadata)
        if not documents:
            raise ValueError(f"Failed to extract valid content from: {file_path}")

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)

        self._save_vectorstore()
        self.logger.info("Indexed paper: %s | total_vectors=%s", paper_info["title"], self.vectorstore.index.ntotal)
        return paper_info

    def load_paper(self, file_path: str) -> dict:
        self.paper_info, documents = self._build_documents(file_path)
        if not documents:
            raise ValueError("No valid paper content extracted.")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self._save_vectorstore()
        return self.paper_info

    def _keyword_overlap(self, query: str, content: str) -> float:
        query_tokens = {token for token in query.lower().split() if len(token) > 2}
        if not query_tokens:
            return 0.0
        content_lower = content.lower()
        hits = sum(1 for token in query_tokens if token in content_lower)
        return hits / len(query_tokens)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not self.vectorstore:
            return []

        limit = top_k or self.rag_config.retrieval_top_k
        raw_results = self.vectorstore.similarity_search_with_score(query, k=limit)
        ranked: list[RetrievedChunk] = []
        for document, distance in raw_results:
            semantic_score = 1 / (1 + float(distance))
            lexical_score = self._keyword_overlap(query, document.page_content)
            combined_score = (semantic_score * self.rag_config.hybrid_alpha) + (
                lexical_score * (1 - self.rag_config.hybrid_alpha)
            )
            ranked.append(
                RetrievedChunk(
                    content=document.page_content,
                    source=document.metadata.get("source", "unknown"),
                    title=document.metadata.get("title", "unknown"),
                    score=round(combined_score, 4),
                    section=document.metadata.get("section", ""),
                    origin="vector_db",
                    metadata=document.metadata,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def search_many(self, queries: list[str], top_k: int | None = None) -> list[RetrievedChunk]:
        merged: dict[tuple[str, str, str], RetrievedChunk] = {}
        for query in queries:
            for item in self.search(query=query, top_k=top_k):
                key = (item.source, item.section, item.content[:80])
                existing = merged.get(key)
                if existing is None or item.score > existing.score:
                    merged[key] = item
        ranked = sorted(merged.values(), key=lambda entry: entry.score, reverse=True)
        return ranked[: (top_k or self.rag_config.retrieval_top_k)]

    def build_context(self, chunks: list[RetrievedChunk], max_chars: int = 4500) -> str:
        parts: list[str] = []
        current = 0
        for index, chunk in enumerate(chunks, start=1):
            section = f" | section={chunk.section}" if chunk.section else ""
            block = f"[{index}] title={chunk.title} | source={chunk.source}{section} | score={chunk.score}\n{chunk.content}\n"
            if current + len(block) > max_chars:
                break
            parts.append(block)
            current += len(block)
        return "\n".join(parts)

    def generate_daily_review(self, topic: str, save_dir: str | None = None) -> str:
        docs = self.search_many([topic], top_k=8)
        if not docs:
            raise ValueError("Knowledge base is empty, cannot generate review.")

        lines = [
            f"# Daily Research Brief - {topic}",
            "",
            f"Generated at: {dt.datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Key Evidence",
        ]
        for index, doc in enumerate(docs, start=1):
            lines.append(f"{index}. {doc.title} ({doc.section or 'section'}) score={doc.score}")
            lines.append(f"   {doc.content[:240].replace(chr(10), ' ')}")

        content = "\n".join(lines)
        target_dir = save_dir or os.getcwd()
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        report_path = os.path.join(target_dir, f"Daily_Review_{dt.date.today().isoformat()}.md")
        Path(report_path).write_text(content, encoding="utf-8")
        return report_path
