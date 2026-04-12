from rag_system import UniversalPaperRAG
from crawler import ArxivCrawler
import datetime
import os

def run_daily_agent():
    print(f"🚀 低轨卫星定位研究Agent 启动 - {datetime.datetime.now()}")
    
    # 目录配置
    BASE_DIR = "D:/agent_paper"
    PDF_DIR = os.path.join(BASE_DIR, "pdfs")        # 存PDF
    VECTOR_DIR = os.path.join(BASE_DIR, "vectordb") # 存向量库
    
    # 创建目录
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    
    # 1. 爬取今日新论文
    crawler = ArxivCrawler(download_dir=PDF_DIR)  # PDF存在 pdfs/ 下
    
    new_papers, papers_info = crawler.search_daily(
        max_results=5,
        relevance_threshold=0.3
    )
    
    if not new_papers:
        print("📭 今日无相关新论文。")
        return
    
    # 2. 初始化 RAG 系统
    API_KEY = "yours API(here)"
    rag = UniversalPaperRAG(api_key=API_KEY, persist_dir=VECTOR_DIR)  # 向量库存 vectordb/
    
    # 3. 学习新论文
    print("\n📚 开始学习今日论文...")
    for i, pdf_path in enumerate(new_papers):
        try:
            print(f"\n[{i+1}/{len(new_papers)}] 处理: {os.path.basename(pdf_path)}")
            rag.add_paper(pdf_path)
        except Exception as e:
            print(f"   ⚠️ 解析失败: {e}")
    
    # 4. 生成综述文档
    print("\n📝 生成每日研究简报...")
    review = rag.generate_daily_review("低轨卫星 定位 导航", save_dir=BASE_DIR)
    print(f"✅ 简报已保存: {BASE_DIR}")
    # 把综述也保存到 BASE_DIR
    review_path = os.path.join(BASE_DIR, f"Daily_Review_{datetime.date.today()}.md")
    print(f"✅ 简报已保存: {review_path}")
    
    print("\n🏁 Agent 任务完成。")

if __name__ == "__main__":
    run_daily_agent()