# Milvus Document Schema

研究资料 collection 包含：

- `document_id`：资料级标识。
- `content`：启用多语言 analyzer 的 chunk 文本。
- `embedding`：BGE-M3 Dense 向量，Cosine 检索。
- `bm25_sparse`：Milvus BM25 Function 自动生成的稀疏向量。
- 行业、公司、产品线、资料类型、来源、章节和页码元数据。

从旧 schema 切换后执行：

```bash
python agent_main.py rebuild-index
```

该命令重建 `industry_research_knowledge_bge_m3` 并重新索引 `research_documents` 中记录的资料。
