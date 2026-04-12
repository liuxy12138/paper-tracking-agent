"""
通用论文RAG系统 - 自动识别中英文
支持中文论文、英文论文、中英文混合论文
"""

import os
import sys
import io
import datetime
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像
import re
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from zhipuai import ZhipuAI
# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置Python默认编码
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class LanguageDetector:
    """语言检测器"""
    
    @staticmethod
    def detect(text: str) -> str:
        """检测文本语言，返回 'zh' 或 'en'"""
        if not text:
            return 'en'
        
        # 统计中文字符比例
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text[:1000])
        chinese_ratio = len(chinese_chars) / max(len(text[:1000]), 1)
        
        # 中文字符超过15%判断为中文
        if chinese_ratio > 0.15:
            return 'zh'
        return 'en'


class UniversalPaperParser:
    """读论文，解析论文 - 自动适配中英文"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.full_text = ""
        self.language = 'en'
        
    def parse(self) -> Dict:
        """解析论文，自动识别语言"""
        # 加载PDF
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        self.full_text = "\n".join([p.page_content for p in pages])
        
        # 检测语言
        self.language = LanguageDetector.detect(self.full_text)
        print(f"   🌐 检测到语言: {'中文' if self.language == 'zh' else 'English'}")
        
        # 根据语言选择解析规则
        if self.language == 'zh':
            return self._parse_chinese()
        else:
            return self._parse_english()
    
    def _parse_chinese(self) -> Dict:
        """解析中文论文"""
        return {
            "title": self._extract_title_chinese(),
            "abstract": self._extract_abstract_chinese(),
            "introduction": self._extract_section(["引言", "绪论", "1", "一、"]),
            "related_work": self._extract_section(["相关工作", "文献综述", "研究现状"]),
            "method": self._extract_section(["方法", "算法", "模型", "2", "二、"]),
            "experiment": self._extract_section(["实验", "结果", "评估", "3", "三、"]),
            "conclusion": self._extract_section(["结论", "总结", "展望", "4", "四、"]),
            "full_text": self.full_text,
            "language": "zh"
        }
    
    def _parse_english(self) -> Dict:
        """解析英文论文"""
        return {
            "title": self._extract_title_english(),
            "abstract": self._extract_abstract_english(),
            "introduction": self._extract_section(["Introduction", "1 Introduction"]),
            "related_work": self._extract_section(["Related Work", "Background", "2 Related Work"]),
            "method": self._extract_section(["Method", "Approach", "Model Architecture", "3 Method"]),
            "experiment": self._extract_section(["Experiments", "Results", "Evaluation", "4 Experiments"]),
            "conclusion": self._extract_section(["Conclusion", "Discussion", "5 Conclusion"]),
            "full_text": self.full_text,
            "language": "en"
        }
    
    def _extract_title_chinese(self) -> str:
        """提取中文标题"""
        if not self.full_text:
            return "未识别到标题"
        
        lines = self.full_text.split('\n')[:20]
        for line in lines:
            line = line.strip()
            # 中文标题特征：包含中文，不太长
            if re.search(r'[\u4e00-\u9fff]', line) and 10 < len(line) < 100:
                return line
        return lines[0][:100] if lines else "未识别到标题"
    
    def _extract_title_english(self) -> str:
        """提取英文标题"""
        lines = self.full_text.split('\n')[:20]
        for line in lines:
            line = line.strip()
            # 英文标题特征：大写字母多，不太长
            if line and 15 < len(line) < 150 and not re.search(r'@|\.com', line):
                if re.search(r'[A-Z]', line):
                    return line
        return lines[0][:100] if lines else "Unknown Title"
    
    def _extract_abstract_chinese(self) -> str:
        """提取中文摘要"""
        patterns = [
            r'摘要[：:]*\s*\n*(.*?)(?=\n\n|\n[一二三四五]|\n参考文献|$)',  # 匹配到空行或下一章节
        r'摘要\s*\n(.*?)(?=\n\n|\n[一二三四五]|\n参考文献|$)',  # 摘要换行
        r'【摘要】\s*(.*?)【',  # 【摘要】内容【
        r'\[摘要\]\s*(.*?)\[',  # [摘要]内容[
        r'^(.*?)$',  # 最后手段：取第一段
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.full_text, re.DOTALL)
            if match:
                text = match.group(1).strip()
                text = re.sub(r'\n+', ' ', text)
                if len(text) > 50:
                    return text[:1500]
        return ""
    
    def _extract_abstract_english(self) -> str:
        """提取英文摘要"""
        patterns = [
            r'Abstract[:\s]*\n(.*?)(?=\n1\.|Introduction|$)',
            r'Abstract\n(.*?)(?=\n1\.|Introduction|$)',
            r'ABSTRACT\s*\n(.*?)(?=\n1\.|INTRODUCTION|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.full_text, re.IGNORECASE | re.DOTALL)
            if match:
                text = match.group(1).strip()
                text = re.sub(r'\n+', ' ', text)
                if len(text) > 100:
                    return text[:2000]
        
        # 尝试找Abstract附近的内容
        if 'abstract' in self.full_text[:2000].lower():
            idx = self.full_text[:2000].lower().find('abstract')
            return self.full_text[idx:idx+1500]
        return ""
    
    def _extract_section(self, section_names: List[str]) -> str:
        """通用章节提取（中英文都适用）"""
        text = self.full_text
        
        for name in section_names:
            # 多种格式匹配
            patterns = [
                rf'\n\s*{re.escape(name)}\s*\n',
                rf'\n\s*\d+\.\s*{re.escape(name)}\s*\n',
                rf'\n\s*\d+\.\d+\s*{re.escape(name)}\s*\n',
                rf'\n\s*[一二三四五]\.\s*{re.escape(name)}\s*\n',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = match.end()
                    # 找下一个章节边界
                    next_pattern = r'\n\s*(?:\d+\.\s*[A-Z]|参考文献|References|Conclusion|Acknowledgments|结论|致谢)'
                    next_match = re.search(next_pattern, text[start:start+8000], re.IGNORECASE)
                    
                    if next_match:
                        end = start + next_match.start()
                    else:
                        end = min(start + 5000, len(text))
                    
                    content = text[start:end].strip()
                    content = re.sub(r'\n+', ' ', content)
                    if len(content) > 100:
                        return content[:2500]
        
        return ""


class UniversalPaperRAG:
    """建知识库和问答- 自动适配中英文"""
    
    def __init__(self, api_key: str,persist_dir="D:\agent_paper"):
        self.client = ZhipuAI(api_key=api_key)
        # 使用多语言embedding模型（同时支持中英文）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.paper_info = {}  #保留，用于单篇问答的上下文
        self.persist_dir = os.path.abspath(persist_dir)
    
        # 确保目录存在
        os.makedirs(self.persist_dir, exist_ok=True)
    
        # 检查向量库文件是否存在
        faiss_file = os.path.join(self.persist_dir, "index.faiss")
        pkl_file = os.path.join(self.persist_dir, "index.pkl")
    
        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            print(f"📀 加载本地持久化向量库: {self.persist_dir}")
            try:
                self.vectorstore = FAISS.load_local(
                    self.persist_dir, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
            )
                print(f"   ✅ 成功加载，当前索引数: {self.vectorstore.index.ntotal}")
            except Exception as e:
                print(f"   ⚠️ 加载失败: {e}")
                print("   🆕 将初始化新的向量库")
                self.vectorstore = None
        else:
            print(f"🆕 初始化新的向量库: {self.persist_dir}")
            self.vectorstore = None

    # ========== 新增：Agent 批量添加方法 ==========
    def add_paper(self, file_path: str):
        """Agent调用：将论文加入知识库（不覆盖原有）"""
        parser = UniversalPaperParser(file_path)
        paper_info = parser.parse()
        
        # 构建文档
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_text(paper_info['full_text'])
        
        for i, chunk in enumerate(chunks[:40]):
            if len(chunk) > 100:
                # 添加来源标识
                doc = Document(
                    page_content=chunk, 
                    metadata={
                        "type": "chunk", 
                        "id": i,
                        "source": os.path.basename(file_path),  # 新增：记录来源文件
                        "title": paper_info.get('title', 'Unknown')
                    }
                )
                documents.append(doc)
        
        # 追加到向量库
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)
        
        # 持久化
        self.vectorstore.save_local(self.persist_dir)
        print(f"   ✅ 已添加至知识库 (总索引: {self.vectorstore.index.ntotal})")

    def load_paper(self, file_path: str):
        """加载论文"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        print(f"\n📄 正在加载: {os.path.basename(file_path)}")
        
        # 解析论文
        parser = UniversalPaperParser(file_path)
        self.paper_info = parser.parse()
        
        # 打印信息
        print(f"   ✅ 标题: {self.paper_info['title'][:80]}")
        print(f"   ✅ 摘要: {self.paper_info['abstract'][:100]}..." if self.paper_info['abstract'] else "   ⚠️ 未提取到摘要")
        
        # 统计各章节
        for section in ['introduction', 'method', 'experiment', 'conclusion']:
            if self.paper_info.get(section) and len(self.paper_info[section]) > 50:
                print(f"   ✅ {section}: {len(self.paper_info[section])} 字符")
        
        # 构建文档列表
        documents = []
        
        # 添加全文切块
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_text(self.paper_info['full_text'])
        for i, chunk in enumerate(chunks[:40]):  # 限制数量
            if len(chunk) > 100:
                doc = Document(page_content=chunk, metadata={"type": "chunk", "id": i})
                documents.append(doc)
        
        # 添加各章节
        for section_name in ['abstract', 'introduction', 'method', 'experiment', 'conclusion']:
            content = self.paper_info.get(section_name)
            if content and len(content) > 100:
                doc = Document(page_content=content, metadata={"type": "section", "section": section_name})
                documents.append(doc)
        
        if not documents:
            raise ValueError("未能从论文中提取到有效内容")
        
        # 向量化
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print(f"   ✅ 知识库构建完成: {len(documents)} 个索引")
        
        return self.paper_info
    
        # ========== 原有的问答方法（保留并增强） ==========
    def ask(self, question: str) -> Dict:
        """交互式问答 - 支持从多篇论文中检索"""
        if not self.vectorstore:
            raise ValueError("请先加载论文")
        
        # 检索（现在可以从多篇论文中检索）
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(question)
        
        # 构建上下文（显示来源）
        context_parts = []
        sources = set()
        
        for doc in retrieved_docs[:4]:
            content = doc.page_content[:600]
            source = doc.metadata.get("source", "Unknown")
            sources.add(source)
            
            if doc.metadata.get("title"):
                context_parts.append(f"\n【来源：{doc.metadata['title'][:50]}】\n{content}")
            else:
                context_parts.append(f"\n{content}")
        
        context = "\n".join(context_parts)
        
        # 提示词（现在会告诉用户答案来自哪些论文）
        prompt = f"""基于以下论文内容回答问题。如果涉及多篇论文，请分别说明。

{context}

---
问题：{question}

【要求】用中文回答，并注明信息来自哪篇论文。"""
        
        response = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "你是专业的学术论文分析助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        
        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "sources": list(sources)  # 新增：返回引用来源
        }
    
    # ========== 原有的交互式聊天（保留） ==========
    def interactive_chat(self):
        """交互式问答界面（增强版：显示多篇论文来源）"""
        if not self.vectorstore:
            print("❌ 请先加载论文")
            return
        
        print("\n" + "="*60)
        print("🤖 通用论文问答系统（支持多篇论文检索）")
        print("="*60)
        print(f"📚 知识库索引数: {self.vectorstore.index.ntotal}")
        print("\n示例问题：")
        print("  • 对比这几篇论文的方法差异")
        print("  • RAG系统有哪些优化方向？")
        print("  • 各论文的实验结果如何？")
        print("\n输入 'exit' 退出")
        print("-"*60)
        
        while True:
            question = input("\n❓ 问题: ").strip()
            if question.lower() == 'exit':
                print("再见！")
                break
            if not question:
                continue
            
            print("🤔 分析中...")
            try:
                result = self.ask(question)
                print(f"\n📖 回答：\n{result['answer']}")
                if result.get('sources'):
                    print(f"\n📚 参考来源：{', '.join(result['sources'])}")
            except Exception as e:
                print(f"❌ 错误: {e}")
    def generate_daily_review(self, topic: str,save_dir:str=None) -> str:
        """生成每日简报/综述"""
    
        # 1. 检索与该主题最相关的内容 (从多篇论文中混合检索)
        if not self.vectorstore:
            return "知识库为空"
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})
        docs = retriever.invoke(topic)
    
        # 2. 按论文来源分组 (通过 metadata 里的 source 或者简单切片)
        context_parts = []
        for doc in docs:
            context_parts.append(doc.page_content)
    
        context = "\n---\n".join(context_parts)
    
        # 3. 专门的综述提示词
        prompt = f"""你是一个顶尖的科研助理。请根据从多篇最新论文中检索到的内容片段，写一份 **《今日{topic}领域简报》**。

要求：
1. **核心趋势**：今天这些论文共同关注什么问题？
2. **亮点方法**：有哪些新颖的技术或模型？（列出论文标题/片段中的方法名）
3. **结论摘要**：一句话总结每篇的核心贡献。
4. 输出格式要清晰，便于快速阅读。

检索到的论文片段如下：
{context}
"""
    
        response = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "你是专业的文献综述生成器。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        # 如果指定了保存目录，使用绝对路径
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f"Daily_Review_{datetime.date.today()}.md")
        else:
            filename = f"Daily_Review_{datetime.date.today()}.md"
    
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
    
        print(f"📝 简报已生成: {os.path.abspath(filename)}")
        return content

