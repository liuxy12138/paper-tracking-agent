# Paper Research Agent

这是一个基于 LangGraph 构建的多 Agent 文献研究工作流项目，并提供了 FastAPI + 简单前端页面用于演示。

项目目标是把“论文抓取 + RAG 问答”的 demo，升级成一个更接近真实 Agent 应用开发场景的系统。

## 项目特点

- 基于 LangGraph 设计多 Agent 工作流
- 包含 `Planner / Retrieval / Analysis / Summary / Reflection` 五阶段
- 基于 `FAISS + Sentence-Transformers` 构建本地 RAG
- 支持 `Tool Calling`，可调用 arXiv 检索、论文下载入库、PDF 解析、语义检索等工具
- 支持短期记忆与长期记忆
- 支持自动反思与一次回退重试
- 提供 FastAPI 接口和可视化演示页面

## 系统架构

```text
用户问题
  -> Planner
  -> Retrieval
      -> query rewrite
      -> tool calling
      -> arXiv search / PDF parse / FAISS retrieval
  -> Analysis
  -> Summary
  -> Reflection
      -> 如结果不足则回退到 Retrieval 重试
  -> Final answer
```

## 项目结构

```text
agent_main.py
paper_agent/
  config.py
  crawler.py
  llm.py
  logging_utils.py
  memory.py
  models.py
  parser.py
  pipeline.py
  rag.py
  storage.py
  tools.py
  webapp.py
  workflow.py
web/
  templates/
  static/
```

## 技术栈

- Python
- LangGraph
- LangChain
- FAISS
- FastAPI
- Sentence-Transformers
- GLM-4
- ArXiv API
- RAG
- Tool Calling
- Memory
- Reflection

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置方式

将 `agent_config.example.json` 复制为 `agent_config.json`，然后填写你的配置。

至少需要配置：

- `api_key`：智谱 GLM 的 API Key

也可以直接使用环境变量：

```bash
set ZHIPU_API_KEY=your_key
```

## CLI 用法

查看配置：

```bash
python agent_main.py show-config
```

查看 LangGraph 工作流图：

```bash
python agent_main.py show-graph
```

运行每日论文抓取与日报流程：

```bash
python agent_main.py run
```

基于当前知识库提问：

```bash
python agent_main.py ask --question "对比当前低轨卫星定位相关论文的方法差异"
```

启动多轮对话：

```bash
python agent_main.py chat --thread-id interview-demo --user-id demo-user
```

导入本地 PDF：

```bash
python agent_main.py ingest --file D:\\papers\\sample.pdf --paper-id local-sample
```

## Web 演示页面

启动 FastAPI：

```bash
uvicorn paper_agent.webapp:app --reload
```

浏览器打开：

```text
http://127.0.0.1:8000
```

页面支持：

- 向多 Agent 工作流提问
- 运行每日工作流
- 上传并索引本地 PDF
- 查看工作流图
- 展示回答、规划、反思和工具调用记录

## 当前能力说明

这个项目目前已经具备以下能力：

1. 复杂问题分阶段执行
2. 检索增强问答
3. 本地知识库增量入库
4. 工具调用
5. 多轮对话
6. 长短期记忆
7. 自动反思和结果补救

## 后续可优化方向

- 增加更强的 reranker 提升检索质量
- 增加更稳定的持久化 checkpointer
- 增加数据库与用户系统
- 增加更完整的前端页面和任务历史管理
- 增加自动评测与可观测性能力
