# Agent 知识指南

## 为什么需要 RAG

竞品信息分散在产品文档、白皮书、公告和行业报告中，而且更新频繁。系统先检索当前资料，再让模型基于证据回答，减少仅依赖模型参数记忆产生的无依据结论。

## Embedding 与混合检索

BGE-M3 将查询和文档 chunk 转成归一化向量，Milvus 使用 Cosine 做语义召回；Milvus BM25 补充产品名、套餐名和功能术语的精确匹配；BGE Reranker 对候选证据重新排序。

## Agentic Workflow

- Planner：输出目标、步骤、查询和工具调用。
- Retrieval：改写查询并检索知识库。
- Analysis：整理结论、证据、冲突和缺口。
- Summary：生成带来源标题引用的答案。
- Reflection：融合规则分和 LLM 分，决定是否补检。
- Finalize：保存会话、长期记忆和运行结果。

## 面试概括

> 我做的是一个面向行业与竞品分析的 Agentic RAG 系统。系统使用 Docling 解析多源企业资料，用 BGE-M3、Milvus Dense/BM25 和 BGE Reranker 构建可追溯检索链路，再通过 LangGraph 编排规划、检索、分析、生成和反思补检。Redis 与 Milvus 分别承担短期和长期记忆，FastAPI SSE 用于展示长任务过程，离线评测覆盖证据召回、引用准确率和答案完整性。
