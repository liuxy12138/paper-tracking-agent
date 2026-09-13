from __future__ import annotations

import json
import re

import pytest
from langchain_core.documents import Document

from competitive_research_agent.config import RagConfig
from competitive_research_agent.rag import ResearchDocumentRAG


class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            1.0 if "竞品" in lowered or "coding assistant" in lowered else 0.0,
            1.0 if "定位" in lowered or "position" in lowered else 0.0,
            1.0 if "industry research knowledge dimension probe" in lowered else 0.5,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class FakeMilvusDocumentClient:
    def __init__(self):
        self.rows: list[dict] = []
        self.dropped = False
        self.query_called = False

    def has_collection(self, collection_name: str) -> bool:
        return True

    def describe_collection(self, collection_name: str):
        return {
            "fields": [
                {"name": "chunk_id"},
                {"name": "document_id"},
                {"name": "language"},
                {"name": "content"},
                {"name": "embedding", "params": {"dim": 3}},
                {"name": "bm25_sparse"},
            ]
        }

    def insert(self, collection_name: str, data: list[dict]):
        self.rows.extend(data)

    def delete(self, collection_name: str, ids=None, filter=None):
        if filter and 'document_id == "' in filter:
            document_id = filter.split('document_id == "', 1)[1].split('"', 1)[0]
            self.rows = [row for row in self.rows if row["document_id"] != document_id]

    def query(self, **kwargs):
        self.query_called = True
        raise AssertionError("Database BM25 must not scan all rows with query().")

    def search(self, **kwargs):
        query = kwargs["data"][0]
        hits = []
        for row in self.rows:
            if kwargs.get("anns_field") == "bm25_sparse":
                query_terms = set(re.findall(r"[a-z]+", query.casefold()))
                content_terms = set(re.findall(r"[a-z]+", row["content"].casefold()))
                score = len(query_terms & content_terms)
            else:
                score = sum(a * b for a, b in zip(query, row["embedding"]))
            if score <= 0:
                continue
            hits.append(
                {
                    "id": row["chunk_id"],
                    "distance": score,
                    "entity": {key: value for key, value in row.items() if key != "embedding"},
                }
            )
        hits.sort(key=lambda item: item["distance"], reverse=True)
        return [hits[: kwargs["limit"]]]

    def drop_collection(self, collection_name: str):
        self.rows = []
        self.dropped = True


def build_rag() -> ResearchDocumentRAG:
    rag = ResearchDocumentRAG.__new__(ResearchDocumentRAG)
    rag.rag_config = RagConfig(
        collection_name="industry_research_knowledge_bge_m3",
        retrieval_top_k=3,
        bm25_top_k=3,
        rerank_top_k=3,
        enable_model_rerank=False,
    )
    rag.embeddings = FakeEmbeddings()
    rag.client = FakeMilvusDocumentClient()
    rag.allow_schema_mismatch = False
    rag._reranker = None
    rag._reranker_load_failed = False
    return rag


def add_document(rag: ResearchDocumentRAG, content: str, document_id: str, title: str):
    document = Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "source": f"{document_id}.pdf",
            "title": title,
            "section": "abstract",
            "source_url": "",
            "published": "",
            "citation_preview": content[:100],
        },
    )
    embedding = rag.embeddings.embed_query(content)
    rag.client.insert(rag.rag_config.collection_name, [rag._document_row(document, embedding)])


def test_milvus_hybrid_search_returns_document_chunks_only():
    rag = build_rag()
    add_document(rag, "低轨竞品定位能够改善导航性能", "document-1", "Cursor capabilities")
    add_document(rag, "unrelated molecular biology evidence", "document-2", "Biology")

    results = rag.search("低轨竞品定位", top_k=2)

    assert results[0].title == "Cursor capabilities"
    assert results[0].origin in {"milvus", "milvus_dense+bm25"}
    assert results[0].metadata["document_id"] == "document-1"
    assert rag.client.query_called is False


def test_reindexing_same_document_replaces_old_chunks():
    rag = build_rag()
    add_document(rag, "old AI coding assistant content", "document-1", "Old")
    new_document = Document(
        page_content="new AI coding assistant content",
        metadata={"document_id": "document-1", "source": "document-1.pdf", "title": "New", "section": "abstract"},
    )
    rag.client.delete(
        collection_name=rag.rag_config.collection_name,
        filter='document_id == "document-1"',
    )
    rag.client.insert(
        rag.rag_config.collection_name,
        [rag._document_row(new_document, rag.embeddings.embed_query(new_document.page_content))],
    )

    assert len(rag.client.rows) == 1
    assert rag.client.rows[0]["title"] == "New"


def test_mixed_language_chunk_uses_default_icu_analyzer():
    rag = build_rag()

    assert rag._detect_language("本文介绍 AI coding assistant 的实现方法") == "default"
    assert rag._detect_language("AI coding assistant evidence") == "en"
    assert rag._detect_language("竞品定位方法研究") == "cn"


def test_document_collection_is_separate_from_memory_collection():
    config = RagConfig()

    assert config.collection_name == "industry_research_knowledge_bge_m3"
    assert config.collection_name != "user_long_term_memory_bge_m3"


def test_document_store_rejects_memory_collection_schema():
    rag = build_rag()
    rag.client.describe_collection = lambda collection_name: {
        "fields": [
            {"name": "memory_id"},
            {"name": "user_id"},
            {"name": "text"},
            {"name": "embedding", "params": {"dim": 3}},
        ]
    }

    with pytest.raises(RuntimeError, match="does not match the research-document BM25 schema"):
        rag._validate_collection()


def test_model_reranker_changes_final_order():
    class FakeReranker:
        def predict(self, pairs, batch_size, show_progress_bar, activation_fn):
            return [0.1 if "coding assistant" in passage else 0.95 for _, passage in pairs]

    rag = build_rag()
    rag.rag_config.enable_model_rerank = True
    rag.rag_config.reranker_weight = 1.0
    rag._reranker = FakeReranker()
    add_document(rag, "AI coding assistant evidence", "document-1", "Coding Assistant")
    add_document(rag, "positioning alternative evidence", "document-2", "Alternative")

    results = rag.search("AI coding assistant", top_k=2)

    assert results[0].title == "Alternative"
    assert results[0].metadata["score_breakdown"]["model_rerank"] == 0.95


def test_milvus_bm25_search_uses_sparse_index_and_language_analyzer():
    rag = build_rag()
    add_document(rag, "AI coding assistant evidence", "document-1", "Coding Assistant")

    results = rag._bm25_search("AI coding assistant", top_k=3)

    assert results[0][0].metadata["document_id"] == "document-1"
    assert results[0][1] == 1.0
    assert rag.client.query_called is False


def test_dense_only_mode_skips_bm25_search():
    rag = build_rag()
    rag.rag_config.enable_bm25_retrieval = False
    add_document(rag, "AI coding assistant evidence", "document-1", "Coding Assistant")

    results = rag.search("AI coding assistant", top_k=2)

    assert results
    assert rag.client.query_called is False


def test_bm25_only_mode_can_return_hits_without_dense():
    rag = build_rag()
    rag.rag_config.enable_dense_retrieval = False
    add_document(rag, "AI coding assistant evidence", "document-1", "Coding Assistant")

    results = rag.search("AI coding assistant", top_k=2)

    assert results[0].metadata["document_id"] == "document-1"


def test_collection_schema_creates_native_bm25_function_and_sparse_index():
    class FakeSchema:
        def __init__(self):
            self.fields = []
            self.functions = []

        def add_field(self, name, datatype, **kwargs):
            self.fields.append((name, datatype, kwargs))

        def add_function(self, function):
            self.functions.append(function)

    class FakeIndexes:
        def __init__(self):
            self.indexes = []

        def add_index(self, **kwargs):
            self.indexes.append(kwargs)

    class FakeCreateClient:
        def __init__(self):
            self.schema = FakeSchema()
            self.indexes = FakeIndexes()
            self.created = False

        def has_collection(self, collection_name):
            return False

        def create_schema(self, **kwargs):
            return self.schema

        def prepare_index_params(self):
            return self.indexes

        def create_collection(self, **kwargs):
            self.created = True

    rag = ResearchDocumentRAG.__new__(ResearchDocumentRAG)
    rag.rag_config = RagConfig()
    rag.dimension = 3
    rag.client = FakeCreateClient()
    rag.allow_schema_mismatch = False

    rag._ensure_collection()

    assert rag.client.created is True
    assert any(name == "bm25_sparse" for name, _, _ in rag.client.schema.fields)
    assert len(rag.client.schema.functions) == 1
    assert any(index["metric_type"] == "BM25" for index in rag.client.indexes.indexes)
