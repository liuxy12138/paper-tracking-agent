# Industry Competitive Research Agent 面试深挖

## 项目定位

系统面向行业与竞品分析，将分散在行业报告、产品资料、白皮书、公告和技术文档中的信息转成可检索证据，支持竞品定位、功能对比、定价与商业模式、客户群体、技术路线、限制和路线图分析。

## 端到端链路

```text
资料上传
 -> Docling 解析，PyPDF 降级
 -> 竞品业务章节归一
 -> RecursiveCharacterTextSplitter 切块
 -> BGE-M3 归一化向量
 -> Milvus Dense + BM25
 -> BGE Reranker
 -> LangGraph 分析与生成
 -> Reflection 质量检查与补检
```

## 为什么 Docling 后还要章节归一

Docling 负责版面理解并输出结构化 markdown，但不同企业资料的标题写法不一致。项目将标题映射为 `overview`、`market`、`product`、`features`、`pricing`、`customers`、`competition`、`technology`、`limitations`、`roadmap`、`case_studies` 和 `key_takeaways`，方便按业务维度切块、检索和解释来源。

## 混合检索

Dense Retrieval 解决语义表达差异；BM25 提升产品名、套餐名、指标名和专有功能的精确召回；Reranker 使用 query-document 对重新评分。结果保留 semantic、BM25、hybrid 和 rerank 分数。

## Planner

Planner 先召回相关用户记忆，再输出 Pydantic 校验的 JSON：目标、步骤、查询、工具调用、答案格式和记忆摘要。异常结构会被规范化，LLM 不可用时走确定性默认计划。

## Reflection

确定性评分：

```text
evidence_count 25%
source_count 20%
citation_coverage 30%
retrieval_score 25%
```

最终分默认是确定性分 65% 加 LLM 分 35%。除了 0.6 总分阈值，还设置最低证据数、最低来源数、最低引用覆盖率和最低检索均分。程序统一生成 `passed` 与 `should_retry`，LLM 原始判断只用于观测。达到最大重试次数后进入 Finalize，同时记录质量未通过。

## 记忆

- Redis：最近消息、历史摘要、TTL。
- Milvus：偏好、画像、研究兴趣和交互摘要。
- 召回排序融合语义相关性、重要性、时效性和记忆类型。
- 相似记忆通过 Cosine 阈值去重，结构化偏好通过 `memory_key` 覆盖旧值。

## 工程化

- 工具参数由 Pydantic 校验。
- 工具执行具有超时、重试和状态记录。
- SSE 推送节点、Token、结果和错误事件。
- MySQL 保存文档、运行轨迹、工具调用、证据和评测。
- 评测覆盖 Recall@K、引用准确率、关键词覆盖率、答案要点覆盖率和工具成功率。

## 边界

- 当前主要接收 PDF/本地资料，尚未实现产品网页的定时刷新。
- 公司、产品线和资料类型已进入元数据，但检索过滤仍需进一步下推 Milvus。
- 引用覆盖率基于来源标题匹配，尚未做到逐条 claim entailment。
- 文档覆盖更新是先删除旧 chunk 再插入，尚未实现原子版本切换。

## 简历表述

- 基于 LangGraph 构建行业与竞品分析 Agentic RAG 工作流，覆盖规划、混合检索、证据分析、结构化生成和反思补检。
- 使用 Docling/PyPDF 解析企业资料，按竞品业务章节切块，采用 BGE-M3、Milvus Dense/BM25 和 BGE Reranker 构建可追溯检索链路。
- 设计 Redis 短期记忆与 Milvus 长期记忆，并通过确定性质量指标和 LLM 评分融合控制补检。
- 使用 FastAPI SSE、MySQL 轨迹存储和离线评测提升长任务可观测性与结果可验证性。
