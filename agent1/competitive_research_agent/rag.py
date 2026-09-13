from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RagConfig
from .logging_utils import get_logger
from .models import RetrievedChunk
from .observability import measure_current
from .parser import ResearchDocumentParser


def _escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class ResearchDocumentRAG:
    def __init__(
        self,
        api_key: str,
        rag_config: RagConfig | None = None,
        client: Any = None,
        allow_schema_mismatch: bool = False,
    ):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.api_key = api_key.strip()
        self.rag_config = rag_config or RagConfig()
        self.logger = get_logger(self.__class__.__name__)
        Path(self.rag_config.embedding_cache_dir).mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.rag_config.embedding_model,
            cache_folder=self.rag_config.embedding_cache_dir,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.dimension = len(self.embeddings.embed_query("industry research knowledge dimension probe"))
        self.client = client or self._build_client()
        self.allow_schema_mismatch = allow_schema_mismatch
        self._reranker = None
        self._reranker_load_failed = False
        self._ensure_collection()

    def warmup(self, *, include_reranker: bool = False) -> None:
        if include_reranker and self.rag_config.enable_model_rerank:
            self._get_reranker()

    def _build_client(self):
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError(
                "Milvus research-document RAG requires pymilvus. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        kwargs: dict[str, Any] = {
            "uri": self.rag_config.milvus_uri,
            "timeout": self.rag_config.milvus_timeout_seconds,
        }
        if self.rag_config.milvus_token:
            kwargs["token"] = self.rag_config.milvus_token
        try:
            return MilvusClient(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to connect to Milvus at {self.rag_config.milvus_uri}. "
                "Start Milvus or update rag.milvus_uri."
            ) from exc

    def _ensure_collection(self) -> None:
        if self.client.has_collection(collection_name=self.rag_config.collection_name):
            try:
                self._validate_collection()
            except RuntimeError:
                if not self.allow_schema_mismatch:
                    raise
                self.logger.warning(
                    "Research document collection schema mismatch accepted temporarily for rebuild-index."
                )
            return

        try:
            from pymilvus import DataType, Function, FunctionType
        except ImportError as exc:
            raise RuntimeError("pymilvus is required to create the research document collection.") from exc

        multi_analyzer_params = {
            "analyzers": {
                "english": {"type": "english"},
                "chinese": {"type": "chinese"},
                "default": {"tokenizer": "icu"},
            },
            "by_field": "language",
            "alias": {"en": "english", "cn": "chinese"},
        }
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("document_id", DataType.VARCHAR, max_length=1024)
        schema.add_field("industry", DataType.VARCHAR, max_length=256)
        schema.add_field("company", DataType.VARCHAR, max_length=256)
        schema.add_field("product_line", DataType.VARCHAR, max_length=256)
        schema.add_field("document_type", DataType.VARCHAR, max_length=128)
        schema.add_field("language", DataType.VARCHAR, max_length=32)
        schema.add_field(
            "content",
            DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
            multi_analyzer_params=multi_analyzer_params,
        )
        schema.add_field("source", DataType.VARCHAR, max_length=1024)
        schema.add_field("title", DataType.VARCHAR, max_length=2048)
        schema.add_field("section", DataType.VARCHAR, max_length=128)
        schema.add_field("source_url", DataType.VARCHAR, max_length=2048)
        schema.add_field("published", DataType.VARCHAR, max_length=128)
        schema.add_field("page_start", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field("citation_preview", DataType.VARCHAR, max_length=1024)
        schema.add_field("metadata_json", DataType.VARCHAR, max_length=8192)
        schema.add_field("status", DataType.VARCHAR, max_length=32)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field("bm25_sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="content_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names=["bm25_sparse"],
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="bm25_sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": self.rag_config.bm25_k1,
                "bm25_b": self.rag_config.bm25_b,
            },
        )
        self.client.create_collection(
            collection_name=self.rag_config.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _validate_collection(self) -> None:
        description = self.client.describe_collection(collection_name=self.rag_config.collection_name)
        fields = {str(field.get("name", "")): field for field in description.get("fields", [])}
        required = {"chunk_id", "document_id", "language", "content", "embedding", "bm25_sparse"}
        if not required <= set(fields) or "user_id" in fields:
            raise RuntimeError(
                f"Milvus collection {self.rag_config.collection_name} does not match the research-document BM25 schema. "
                "Run `python agent_main.py rebuild-index` to recreate it."
            )
        dimension = int(fields["embedding"].get("params", {}).get("dim", self.dimension))
        if dimension != self.dimension:
            raise RuntimeError(
                f"Research document collection dimension mismatch: stored={dimension}, embedding_model={self.dimension}."
            )

    def rebuild(self, records: list) -> dict[str, int]:
        if self.client.has_collection(collection_name=self.rag_config.collection_name):
            self.client.drop_collection(collection_name=self.rag_config.collection_name)
        self._ensure_collection()
        indexed = 0
        failed = 0
        for record in records:
            try:
                self.add_document(
                    record.file_path,
                    metadata={
                        "document_id": record.document_id,
                        "published": record.published,
                        "source_url": record.source_url,
                    },
                )
                indexed += 1
            except Exception:
                failed += 1
        return {"indexed": indexed, "failed": failed}

    def _locate_page_range(self, text: str, pages: list[dict]) -> tuple[int | None, int | None]:
        if not text.strip() or not pages:
            return None, None
        snippet = re.sub(r"\s+", " ", text.strip()[:180])
        if not snippet:
            return None, None
        for page in pages:
            page_text = re.sub(r"\s+", " ", str(page.get("content", "")))
            if snippet[:80] in page_text:
                page_number = int(page.get("page_number", 0) or 0)
                return page_number, page_number
        return None, None

    def _split_section(
        self,
        text: str,
        section_name: str,
        base_metadata: dict,
        pages: list[dict] | None = None,
    ) -> list[Document]:
        if not text or len(text.strip()) < 80:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rag_config.chunk_size,
            chunk_overlap=self.rag_config.chunk_overlap,
        )
        chunks = splitter.split_text(text)
        documents: list[Document] = []
        page_start, page_end = self._locate_page_range(text, pages or [])
        for index, chunk in enumerate(chunks[: self.rag_config.max_chunks_per_document]):
            if len(chunk.strip()) <= 100:
                continue
            citation_preview = re.sub(r"\s+", " ", chunk.strip())[:260]
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **base_metadata,
                        "section": section_name,
                        "local_chunk_id": f"{section_name}-{index}",
                        "page_start": page_start,
                        "page_end": page_end,
                        "citation_preview": citation_preview,
                    },
                )
            )
        return documents

    def _build_documents(self, file_path: str, metadata: dict | None = None) -> tuple[dict, list[Document]]:
        parser = ResearchDocumentParser(
            file_path,
            backend=self.rag_config.parser_backend,
            fallback_backend=self.rag_config.parser_fallback_backend,
        )
        document_info = parser.parse()
        base_metadata = {
            "document_id": str((metadata or {}).get("document_id") or file_path),
            "source": os.path.basename(file_path),
            "title": document_info.get("title", "Unknown"),
            "type": "research_document_chunk",
        }
        if metadata:
            base_metadata.update(metadata)

        documents: list[Document] = []
        pages = document_info.get("pages", [])
        for section_name in ResearchDocumentParser.SECTION_ALIASES:
            documents.extend(
                self._split_section(
                    document_info.get(section_name, ""),
                    section_name,
                    base_metadata,
                    pages=pages,
                )
            )

        if not documents:
            documents.extend(self._split_section(document_info["full_text"], "full_text", base_metadata, pages=pages))
        else:
            documents.extend(
                self._split_section(
                    document_info["full_text"][:5000],
                    "full_text_head",
                    base_metadata,
                    pages=pages,
                )
            )
        return document_info, documents

    def _document_row(self, document: Document, embedding: list[float]) -> dict[str, Any]:
        metadata = dict(document.metadata)
        return {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(metadata.get("document_id", ""))[:1024],
            "industry": str(metadata.get("industry", ""))[:256],
            "company": str(metadata.get("company", ""))[:256],
            "product_line": str(metadata.get("product_line", ""))[:256],
            "document_type": str(metadata.get("document_type", ""))[:128],
            "language": self._detect_language(document.page_content),
            "content": document.page_content[:8192],
            "source": str(metadata.get("source", "unknown"))[:1024],
            "title": str(metadata.get("title", "unknown"))[:2048],
            "section": str(metadata.get("section", ""))[:128],
            "source_url": str(metadata.get("source_url", ""))[:2048],
            "published": str(metadata.get("published", ""))[:128],
            "page_start": int(metadata.get("page_start") or -1),
            "page_end": int(metadata.get("page_end") or -1),
            "citation_preview": str(metadata.get("citation_preview", ""))[:1024],
            "metadata_json": json.dumps(metadata, ensure_ascii=False)[:8192],
            "status": "active",
            "embedding": embedding,
        }

    def add_document(self, file_path: str, metadata: dict | None = None) -> dict:
        with measure_current("document_parse_and_chunk"):
            document_info, documents = self._build_documents(file_path, metadata)
        if not documents:
            raise ValueError(f"Failed to extract valid content from: {file_path}")

        document_id = str((metadata or {}).get("document_id") or file_path)
        self.client.delete(
            collection_name=self.rag_config.collection_name,
            filter=f'document_id == "{_escape_filter_value(document_id)}"',
        )
        with measure_current("document_embedding"):
            embeddings = self.embeddings.embed_documents([document.page_content for document in documents])
        rows = [
            self._document_row(document, embedding)
            for document, embedding in zip(documents, embeddings)
        ]
        with measure_current("milvus_insert"):
            for start in range(0, len(rows), 100):
                self.client.insert(
                    collection_name=self.rag_config.collection_name,
                    data=rows[start : start + 100],
                )
        self.logger.info("Indexed research document in Milvus: %s | chunks=%s", document_info["title"], len(rows))
        return document_info

    def load_document(self, file_path: str) -> dict:
        return self.add_document(file_path, metadata={"document_id": file_path, "source_url": file_path})

    def _detect_language(self, text: str) -> str:
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin_count = len(re.findall(r"[A-Za-z]", text))
        if chinese_count and latin_count and min(chinese_count, latin_count) / max(chinese_count, latin_count) >= 0.1:
            return "default"
        return "cn" if chinese_count > latin_count else "en"

    def _row_to_document(self, row: dict[str, Any]) -> Document:
        metadata = json.loads(row.get("metadata_json") or "{}")
        metadata.update(
            {
                "document_id": row.get("document_id", ""),
                "industry": row.get("industry", ""),
                "company": row.get("company", ""),
                "product_line": row.get("product_line", ""),
                "document_type": row.get("document_type", ""),
                "language": row.get("language", ""),
                "source": row.get("source", "unknown"),
                "title": row.get("title", "unknown"),
                "section": row.get("section", ""),
                "source_url": row.get("source_url", ""),
                "published": row.get("published", ""),
                "page_start": row.get("page_start", -1),
                "page_end": row.get("page_end", -1),
                "citation_preview": row.get("citation_preview", ""),
            }
        )
        return Document(page_content=str(row.get("content", "")), metadata=metadata)

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        results = self.client.search(
            collection_name=self.rag_config.collection_name,
            data=[query],
            anns_field="bm25_sparse",
            filter='status == "active"',
            limit=top_k,
            output_fields=self._search_output_fields(),
            search_params={
                "metric_type": "BM25",
                "analyzer_name": self._detect_language(query),
                "drop_ratio_search": "0",
            },
        )
        hits = results[0] if results else []
        max_score = max((float(hit.get("distance", 0.0)) for hit in hits), default=1.0) or 1.0
        return [
            (self._row_to_document(hit.get("entity", {})), float(hit.get("distance", 0.0)) / max_score)
            for hit in hits
        ]

    def _search_output_fields(self) -> list[str]:
        return [
            "document_id",
            "industry",
            "company",
            "product_line",
            "document_type",
            "language",
            "content",
            "source",
            "title",
            "section",
            "source_url",
            "published",
            "page_start",
            "page_end",
            "citation_preview",
            "metadata_json",
        ]

    def _candidate_key(self, document: Document) -> tuple[str, str, str]:
        return (
            str(document.metadata.get("source", "unknown")),
            str(document.metadata.get("section", "")),
            document.page_content[:120],
        )

    def _document_to_chunk(
        self,
        document: Document,
        score: float,
        origin: str,
        score_breakdown: dict[str, float],
    ) -> RetrievedChunk:
        reason_parts = []
        if score_breakdown.get("semantic", 0) > 0:
            reason_parts.append(f"semantic={score_breakdown['semantic']:.3f}")
        if score_breakdown.get("bm25", 0) > 0:
            reason_parts.append(f"bm25={score_breakdown['bm25']:.3f}")
        if score_breakdown.get("model_rerank", 0) > 0:
            reason_parts.append(f"model_rerank={score_breakdown['model_rerank']:.3f}")
        retrieval_reason = "Matched by " + ", ".join(reason_parts) if reason_parts else f"Matched by {origin}"

        metadata = {
            **document.metadata,
            "retrieval_origin": origin,
            "retrieval_reason": retrieval_reason,
            "score_breakdown": {key: round(value, 4) for key, value in score_breakdown.items()},
        }
        return RetrievedChunk(
            content=document.page_content,
            source=document.metadata.get("source", "unknown"),
            title=document.metadata.get("title", "unknown"),
            score=round(score, 4),
            section=document.metadata.get("section", ""),
            origin=origin,
            metadata=metadata,
        )

    def _get_reranker(self):
        if getattr(self, "_reranker", None) is not None:
            return self._reranker
        if getattr(self, "_reranker_load_failed", False):
            return None
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                self.rag_config.reranker_model,
                max_length=self.rag_config.reranker_max_length,
                cache_folder=self.rag_config.embedding_cache_dir,
                trust_remote_code=True,
            )
            if self.rag_config.reranker_use_fp16:
                self._reranker.model.half()
            return self._reranker
        except Exception as exc:
            self._reranker_load_failed = True
            if not self.rag_config.reranker_fallback:
                raise RuntimeError(
                    f"Unable to load reranker model: {self.rag_config.reranker_model}"
                ) from exc
            self.logger.warning("Reranker unavailable; using hybrid-score fallback: %s", exc)
            return None

    def _model_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[float] | None:
        if not self.rag_config.enable_model_rerank or not candidates:
            return None
        reranker = self._get_reranker()
        if reranker is None:
            return None
        import torch

        pairs = [[query, candidate["document"].page_content] for candidate in candidates]
        with measure_current("model_rerank"):
            scores = reranker.predict(
                pairs,
                batch_size=self.rag_config.reranker_batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Sigmoid(),
            )
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(score) for score in scores]

    def _semantic_search(self, query: str, limit: int) -> list[tuple[Document, float]]:
        with measure_current("query_embedding"):
            query_embedding = self.embeddings.embed_query(query)
        with measure_current("milvus_search"):
            results = self.client.search(
                collection_name=self.rag_config.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                filter='status == "active"',
                limit=limit,
                output_fields=self._search_output_fields(),
                search_params={"metric_type": "COSINE", "params": {}},
            )
        return [
            (self._row_to_document(hit.get("entity", {})), float(hit.get("distance", 0.0)))
            for hit in (results[0] if results else [])
        ]

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        with measure_current("rag_search"):
            return self._search(query, top_k)

    def _search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        limit = top_k or self.rag_config.retrieval_top_k
        candidate_limit = max(limit, self.rag_config.rerank_top_k)
        candidates: dict[tuple[str, str, str], dict] = {}

        if self.rag_config.enable_dense_retrieval:
            for document, semantic_score in self._semantic_search(query, candidate_limit):
                key = self._candidate_key(document)
                candidates[key] = {
                    "document": document,
                    "semantic_score": max(semantic_score, candidates.get(key, {}).get("semantic_score", 0.0)),
                    "bm25_score": candidates.get(key, {}).get("bm25_score", 0.0),
                    "origin": "milvus",
                }

        if self.rag_config.enable_bm25_retrieval and self.rag_config.bm25_top_k > 0:
            with measure_current("milvus_bm25_search"):
                bm25_results = self._bm25_search(query, self.rag_config.bm25_top_k)
            for document, bm25_score in bm25_results:
                key = self._candidate_key(document)
                existing = candidates.get(key)
                if existing:
                    existing["bm25_score"] = max(float(bm25_score), existing.get("bm25_score", 0.0))
                    existing["origin"] = "milvus_dense+bm25"
                else:
                    candidates[key] = {
                        "document": document,
                        "semantic_score": 0.0,
                        "bm25_score": float(bm25_score),
                        "origin": "milvus_bm25",
                    }

        prepared: list[dict[str, Any]] = []
        for candidate in candidates.values():
            document = candidate["document"]
            semantic_score = float(candidate.get("semantic_score", 0.0))
            bm25_score = float(candidate.get("bm25_score", 0.0))
            hybrid_score = (semantic_score * self.rag_config.hybrid_alpha) + (
                bm25_score * (1 - self.rag_config.hybrid_alpha)
            )
            prepared.append(
                {
                    **candidate,
                    "semantic_score": semantic_score,
                    "bm25_score": bm25_score,
                    "hybrid_score": hybrid_score,
                }
            )

        prepared.sort(key=lambda item: item["hybrid_score"], reverse=True)
        prepared = prepared[: self.rag_config.rerank_top_k]
        model_scores = self._model_rerank(query, prepared)

        ranked: list[RetrievedChunk] = []
        for index, candidate in enumerate(prepared):
            document = candidate["document"]
            model_score = model_scores[index] if model_scores is not None else None
            if model_score is None:
                rerank_score = candidate["hybrid_score"]
            else:
                rerank_score = (model_score * self.rag_config.reranker_weight) + (
                    candidate["hybrid_score"] * (1 - self.rag_config.reranker_weight)
                )
            ranked.append(
                self._document_to_chunk(
                    document=document,
                    score=rerank_score,
                    origin=candidate["origin"],
                    score_breakdown={
                        "semantic": candidate["semantic_score"],
                        "bm25": candidate["bm25_score"],
                        "hybrid": candidate["hybrid_score"],
                        "model_rerank": model_score or 0.0,
                        "rerank": rerank_score,
                    },
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
