# 新手阅读指南

这个项目是一个行业与竞品分析 Agent。它把企业资料解析并写入知识库，在回答前检索证据，再通过反思节点检查证据是否充分。

推荐阅读顺序：

1. `competitive_research_agent/config.py`：默认主题、模型、数据库和阈值。
2. `models.py`、`schemas.py`：文档、工具调用、计划和反思数据结构。
3. `pipeline.py`：组件如何组装。
4. `workflow.py`：LangGraph 节点和条件边。
5. `parser.py`、`rag.py`：Docling 解析、章节切块、向量化和检索。
6. `memory.py`：Redis 与 Milvus 记忆。
7. `webapp.py`：API 与 SSE。

文档入库链路：

```text
竞品资料 -> Docling -> 业务章节归一 -> chunk -> BGE-M3 -> Milvus
```

问答链路：

```text
问题 -> Planner JSON -> 查询改写 -> Dense/BM25 -> Reranker
     -> 证据分析 -> 中文答案 -> Reflection -> 补检或结束
```

最常用命令：

```bash
python agent_main.py ingest --file D:\reports\product_brief.pdf --document-id product-brief
python agent_main.py ask --question "对比三家产品的定价与企业治理能力"
python agent_main.py run
```
