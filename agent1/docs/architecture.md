# Architecture

## Request Flow

```text
Question
  -> Planner
  -> Retrieval
  -> Analysis
  -> Summary
  -> Reflection
       -> Retrieval when quality is insufficient
       -> Finalize when quality passes or retry limit is reached
```

## Components

- `pipeline.py`: assembles RAG, memory, tools, persistence and workflow.
- `parser.py`: Docling-first document parsing with PyPDF fallback and competitive-research section normalization.
- `rag.py`: section chunking, BGE-M3 embeddings, Milvus Dense/BM25 retrieval and BGE reranking.
- `workflow.py`: LangGraph planning, retrieval, evidence analysis, answer generation and reflection.
- `memory.py`: Redis short-term history and Milvus long-term user memory.
- `storage.py`: JSON/MySQL document metadata, traces, evidence and evaluation history.
- `webapp.py`: FastAPI, SSE, document ingestion and research-brief endpoints.

## Storage Boundaries

- Milvus research collection: document chunks and vector indexes.
- Milvus memory collection: user preferences and reusable interaction summaries.
- Redis: recent messages and compressed conversation history.
- MySQL: `research_documents`, workflow runs, tool calls, evidence and evaluation records.

Research documents and user memories use separate collections to prevent preference text from contaminating business evidence retrieval.
