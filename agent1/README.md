# Industry Competitive Research Agent

面向行业与竞品分析场景的 Agentic RAG 系统。系统接收行业报告、竞品白皮书、产品文档、公告和技术资料，完成结构化解析、混合检索、证据分析、竞品对比、反思补检和调研简报生成。

## 技术栈

Python、LangGraph、LangChain、FastAPI、GLM、BGE-M3、BGE Reranker、Milvus、Redis、MySQL、Docling、BM25、SSE、Pydantic。

## 核心能力

- LangGraph 工作流：Planner、Retrieval、Analysis、Summary、Reflection、Finalize。
- 文档解析：Docling 为主，PyPDF 为降级方案；识别概述、市场、产品、功能、定价、客户、竞争、技术、限制、路线图和案例等竞品资料章节。
- 混合检索：BGE-M3 Dense Retrieval、Milvus 原生 BM25、BGE Reranker。
- 证据追踪：保存文档标题、来源、章节、页码、公司、产品线和资料类型。
- 反思补检：融合确定性质量分与 LLM 语义评分，检查证据数量、来源数、引用覆盖率和检索质量。
- 双层记忆：Redis 保存短期会话，Milvus 保存长期用户偏好和研究兴趣。
- 可观测性：记录节点耗时、工具状态、Token、重试次数和质量评分。

## 项目结构

```text
agent_main.py
competitive_research_agent/
  config.py
  llm.py
  memory.py
  models.py
  parser.py
  pipeline.py
  rag.py
  schemas.py
  storage.py
  tools.py
  webapp.py
  workflow.py
web/
evals/
docs/
```

## 使用

安装依赖：

```bash
pip install -r requirements.txt
```

### MySQL and live web research

Keep credentials out of `agent_config.json` and provide them through environment variables:

```powershell
$env:DATABASE_URL = "mysql+pymysql://agent_user:your_password@127.0.0.1:3306/industry_research_agent?charset=utf8mb4"
$env:TAVILY_API_KEY = "tvly-..."
```

Search without downloading, or discover public PDFs and ingest them into the knowledge base:

```powershell
python agent_main.py web-search --query "AI coding agent competitor pricing"
python agent_main.py collect-pdfs --query "AI coding assistant market white paper 2026" --max-results 5
```

Downloaded PDFs are content-addressed under `runtime_data/documents`, parsed, indexed in Milvus, and recorded in MySQL. Use repeatable `--include-domain example.com` arguments to restrict discovery to trusted publishers. Normal Agent research also uses live Tavily snippets when `TAVILY_API_KEY` is set.

查看配置和工作流：

```bash
python agent_main.py show-config
python agent_main.py show-graph
```

导入竞品资料：

```bash
python agent_main.py ingest --file D:\reports\cursor_product_brief.pdf --document-id cursor-product-brief --title "Cursor Product Brief"
```

执行竞品问答：

```bash
python agent_main.py ask --question "对比 Cursor、GitHub Copilot 和 Windsurf 的定位、功能、定价、技术路线和企业治理能力，并给出证据。"
```

生成结构化调研简报：

```bash
python agent_main.py run
```

启动 Web：

```bash
uvicorn competitive_research_agent.webapp:app --reload
```

主要接口：

- `POST /api/ingest-upload`：上传并索引竞品资料 PDF。
- `POST /api/ingest-url`：下载并索引 PDF URL。
- `POST /api/ask`：非流式竞品分析。
- `POST /api/ask/stream`：SSE 流式竞品分析。
- `POST /api/generate-brief`：基于当前资料库生成调研简报。

## 数据存储

- Milvus `industry_research_knowledge_bge_m3`：研究文档 Dense/BM25 索引。
- Milvus `user_long_term_memory_bge_m3`：用户长期记忆。
- Redis：带 TTL 的会话消息和历史摘要。
- MySQL `research_documents`：文档元数据；同时保存运行轨迹、工具调用、证据和评测结果。

Milvus 文档 schema 已统一使用 `document_id`。旧 collection 需要执行：

```bash
python agent_main.py rebuild-index
```

## 评测

```bash
python evals/run_eval.py --dry-run
python evals/run_eval.py --ingest-documents
python evals/run_retrieval_ablation.py
```

评测覆盖竞品定位、功能、定价、客户、企业治理、技术路线和风险分析，指标包括 Recall@K、引用准确率、关键词覆盖率、答案要点覆盖率和工具成功率。
