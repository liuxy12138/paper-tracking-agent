# MySQL Persistence

MySQL 保存文档元数据和可观测数据：

- `research_documents`
- `qa_runs`
- `tool_calls`
- `retrieval_evidence`
- `eval_runs`
- `eval_items`

向量和 BM25 稀疏索引保存在 Milvus，不写入 MySQL。

配置：

```json
{
  "database": {
    "enabled": true,
    "url": "mysql+pymysql://root:password@127.0.0.1:3306/industry_research_agent?charset=utf8mb4",
    "init_schema": true
  }
}
```
