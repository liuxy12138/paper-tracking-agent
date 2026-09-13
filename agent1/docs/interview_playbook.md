# Interview Playbook

## 30 秒介绍

这是一个行业与竞品分析 Agent。它将行业报告、产品资料、白皮书和公告解析入库，通过 Dense、BM25 和 Reranker 检索可引用证据，再由 LangGraph 编排 Planner、Retrieval、Analysis、Summary 和 Reflection。系统还包含 Redis/Milvus 记忆、SSE 过程推送和离线评测。

## 为什么使用 LangGraph

因为流程存在显式状态、多个阶段、条件分支和重试上限。Reflection 根据质量结果回到 Retrieval 或进入 Finalize，图结构比单条 Chain 更容易观测和控制。

## Planner 输出

```json
{
  "objective": "任务目标",
  "steps": ["规划", "检索", "分析", "生成"],
  "search_queries": ["产品功能", "定价", "企业治理"],
  "tool_calls": [{"name": "search_knowledge_base", "args": {"query": "产品功能", "top_k": 6}}],
  "answer_format": "structured research brief",
  "memory_summary": []
}
```

## Reflection 如何判断

确定性分由证据数量、来源数、引用覆盖率和检索分组成，默认占最终分 65%；LLM 语义评分占 35%。最终分低于 0.6，或任一质量门槛失败时建议重试。最大重试轮数阻止无限循环。

## Milvus 为什么有两个 Collection

研究资料和用户记忆属于不同数据域。资料 collection 用于事实证据，记忆 collection 用于偏好和历史关注方向，分开可以避免检索污染。
