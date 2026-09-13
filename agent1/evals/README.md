# 竞品分析评测集

```text
evals/
  documents/
    documents.json
    *_product_brief.md
    ai_coding_market_landscape.md
  questions.jsonl
  run_eval.py
  run_retrieval_ablation.py
```

评测资料是可审计的本地 fixture，用于验证产品定位、功能、定价、客户、技术、企业治理和风险等竞品分析维度。

指标：

- `recall_at_3` / `recall_at_5`
- `citation_accuracy`
- `keyword_coverage`
- `answer_point_coverage`
- `tool_success_rate`

运行：

```bash
python evals/run_eval.py --dry-run
python evals/run_eval.py --ingest-documents
python evals/run_retrieval_ablation.py
```

当前评测集包含 100 题，其中 75 题为跨文档问题、4 份本地示例资料。执行三模式对照实验需要可用的 Milvus、BGE 模型、GLM 凭据和可选 Redis/MySQL；--dry-run 仅校验评测输入，不产生引用准确率。0.44→0.49 属于待复现的目标对照结果，本仓库不将其标为当前实测。
