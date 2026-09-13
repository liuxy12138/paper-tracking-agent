# Database Design

## MySQL

### `research_documents`

Stores `document_id`, title, file path, relevance score, summary, publish time, source URL, index status and metadata.

### `qa_runs`

Stores question, answer, sources, planner JSON, reflection JSON and report path.

### `tool_calls`

Stores tool name, arguments, status, preview and latency.

### `retrieval_evidence`

Stores source, title, section, score, content preview and metadata for each run.

### `eval_runs` and `eval_items`

Store document counts, expected document IDs, scores, results and bad cases.

## Milvus

- `industry_research_knowledge_bge_m3`: document Dense vectors and native BM25 sparse vectors.
- `user_long_term_memory_bge_m3`: user-memory vectors.

MySQL is the structured persistence and observability layer; Milvus is the retrieval layer.
