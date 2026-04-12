# 论文追踪与RAG综述生成Agent

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange.svg)](https://github.com/facebookresearch/faiss)
[![GLM-4](https://img.shields.io/badge/LLM-GLM--4-purple.svg)](https://open.bigmodel.cn/)

## 📖 项目简介

一套**领域可配置**的学术论文自动化处理Pipeline。每日自动爬取ArXiv最新论文，通过三层关键词过滤算法精准筛选，基于FAISS构建增量式向量知识库，调用GLM-4大模型生成结构化研究简报。

> 🔧 支持通过修改配置文件快速切换研究领域（卫星导航 → 计算机视觉 → NLP）

## 🏗️ 系统架构

```
ArXiv爬虫 → 关键词过滤 → PDF解析（中英文自适应）→ 文本分块 
    ↓
FAISS向量库（增量持久化）← 多语言Embedding
    ↓
用户提问 / 每日定时 → RAG检索 → GLM-4生成综述 → Markdown简报
```

## ✨ 核心功能

- **智能论文筛选**：核心词/扩展词/排除词三级相关性打分
- **中英文自适应解析**：自动识别语言，提取标题、摘要、方法、结论
- **增量向量知识库**：FAISS本地持久化，每日追加不重建
- **多论文对比综述**：基于检索增强生成，输出核心趋势与方法对比

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install arxiv langchain langchain-community langchain-huggingface faiss-cpu pypdf sentence-transformers zhipuai
```

### 2. 配置API Key

在 `agent_main.py` 中替换你的智谱API Key：

```python
API_KEY = "your_zhipu_api_key_here"
```

### 3. 修改研究领域（可选）

在 `crawler.py` 中修改 `target_keywords` 字典，切换追踪领域：

```python
self.target_keywords = {
    "core": ["你的核心关键词"],
    "extended": ["扩展关键词"],
    "exclude": ["排除关键词"]
}
```

### 4. 运行

```bash
python agent_main.py
```

## 📁 目录结构

```
agent_paper/
├── agent_main.py          # 主程序入口
├── crawler.py             # ArXiv爬虫 + 关键词过滤
├── rag_system.py          # RAG系统 + 综述生成
├── pdfs/                  # 下载的论文PDF
├── vectordb/              # FAISS向量库持久化目录
├── Daily_Review_*.md      # 每日生成的简报
└── README.md
```

## 📊 输出示例

每日简报包含以下结构：

- **核心趋势**：今日论文共同关注的问题
- **亮点方法**：新颖的技术或模型
- **结论摘要**：每篇论文的核心贡献

## 🔮 后续优化方向

- [ ] 支持更多论文源（CVPR、NeurIPS等）
- [ ] 增加论文引用关系图谱
- [ ] Web可视化界面

## 📧 联系方式

如有问题或建议，欢迎提Issue。
