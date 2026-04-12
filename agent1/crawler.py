import arxiv
import os
import ssl
import urllib.request
from typing import List, Dict, Tuple
import datetime
import re
import json

class ArxivCrawler:
    def __init__(self, download_dir="D:\agent_paper"):
        # 使用绝对路径，避免路径混乱
        self.download_dir = os.path.abspath(download_dir)
        os.makedirs(self.download_dir, exist_ok=True)
        print(f"📁 PDF存储位置: {self.download_dir}")
        
        # 配置SSL绕过
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        opener = urllib.request.build_opener(https_handler)
        urllib.request.install_opener(opener)
        
        # ============================================
        # 🎯 在这里修改你的研究领域关键词！
        # ============================================
        self.target_keywords = {
            # 核心关键词（必须匹配至少一个）
            "core": [
                "LEO satellite",
                "low earth orbit",
                "satellite positioning",
                "satellite navigation",
                "GNSS",
                "GPS satellite",
                "Starlink",
                "mega-constellation",
                "satellite constellation",
                "orbit determination",
                "satellite tracking",
                "Doppler positioning",
            ],
            
            # 扩展关键词（提升相关性权重）
            "extended": [
                "positioning",
                "navigation",
                "orbit",
                "satellite",
                "constellation",
                "tracking",
                "localization",
                "geolocation",
                "ephemeris",
                "ionosphere",
                "PPP",
                "RTK",
            ],
            
            # 排除关键词
            "exclude": [
                "quantum",
                "molecular",
                "gene",
                "protein",
                "COVID",
                "psychology",
            ]
        }
        
    def _calculate_relevance(self, title: str, abstract: str) -> Tuple[float, list]:
        """计算论文与目标领域的相关性分数"""
        text_lower = (title + " " + abstract).lower()
        title_lower = title.lower()
        
        score = 0.0
        matched_keywords = []
        
        # 检查核心关键词
        for keyword in self.target_keywords["core"]:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                if keyword_lower in title_lower:
                    score += 0.8
                    matched_keywords.append(f"🎯{keyword}(标题)")
                else:
                    score += 0.4
                    matched_keywords.append(f"✓{keyword}")
        
        # 检查扩展关键词
        for keyword in self.target_keywords["extended"]:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                if keyword_lower in title_lower:
                    score += 0.4
                    matched_keywords.append(f"📌{keyword}(标题)")
                else:
                    score += 0.2
                    matched_keywords.append(f"·{keyword}")
        
        # 检查排除关键词
        for keyword in self.target_keywords["exclude"]:
            if keyword.lower() in text_lower:
                score -= 0.3
                matched_keywords.append(f"❌{keyword}")
        
        # 归一化
        score = max(0, min(1, score / 5))
        
        return score, matched_keywords
    
    def search_daily(self, max_results: int = 5, relevance_threshold: float = 0.3, query: str = None):
        """
        搜索论文并下载PDF
        
        参数:
            max_results: 最多下载几篇
            relevance_threshold: 最低相关性阈值（0-1）
            query: 搜索关键词（可选）
        
        返回:
            (下载的PDF路径列表, 论文元信息列表)
        """
        
        # 如果没有指定query，自动构造
        if query is None:
            core_terms = self.target_keywords["core"][:5]
            query = " OR ".join([f'"{term}"' for term in core_terms])
        
        # 配置ArXiv客户端
        client = arxiv.Client(
            page_size=50,
            delay_seconds=5,
            num_retries=5
        )
        
        # 搜索（多搜一些用于过滤）
        search = arxiv.Search(
            query=query,
            max_results=max_results * 5,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        downloaded_paths = []
        papers_info = []
        
        print("=" * 70)
        print(f"🔍 正在搜索 ArXiv: {query[:100]}...")
        print(f"🎯 目标领域: 低轨卫星定位与导航")
        print(f"📅 日期: {datetime.date.today()}")
        print(f"📊 最低相关性阈值: {relevance_threshold}")
        print("=" * 70)
        
        downloaded_count = 0
        skipped_count = 0
        processed_count = 0
        
        try:
            for result in client.results(search):
                processed_count += 1
                
                if downloaded_count >= max_results:
                    break
                
                title = result.title
                abstract = result.summary.replace('\n', ' ')
                
                # 相关性过滤
                relevance_score, matched_keywords = self._calculate_relevance(title, abstract)
                
                if relevance_score < relevance_threshold:
                    skipped_count += 1
                    if skipped_count <= 3:
                        print(f"\n   ⏭️  跳过 (相关性 {relevance_score:.2f})")
                        print(f"      {title[:60]}...")
                    continue
                
                print(f"\n   📄 找到相关论文 {downloaded_count+1}/{max_results}")
                print(f"   标题: {title}")
                print(f"   相关性: {relevance_score:.2f}")
                print(f"   关键词: {', '.join(matched_keywords[:5])}")
                
                # 检查是否已下载
                safe_filename = re.sub(r'[^\w\-_\. ]', '_', title)[:60]
                expected_path = os.path.join(self.download_dir, f"{safe_filename}.pdf")
                
                if os.path.exists(expected_path):
                    print(f"   ✅ 已存在")
                    downloaded_paths.append(expected_path)
                    papers_info.append({
                        'title': title,
                        'pdf_path': expected_path,
                        'relevance_score': relevance_score
                    })
                    downloaded_count += 1
                    continue
                
                # 下载
                print(f"   ⏳ 下载中...")
                try:
                    pdf_path = result.download_pdf(dirpath=self.download_dir)
                    
                    # 重命名
                    if safe_filename not in pdf_path:
                        new_path = os.path.join(self.download_dir, f"{safe_filename}.pdf")
                        os.rename(pdf_path, new_path)
                        pdf_path = new_path
                    
                    print(f"   ✅ 下载成功")
                    downloaded_paths.append(pdf_path)
                    papers_info.append({
                        'title': title,
                        'pdf_path': pdf_path,
                        'relevance_score': relevance_score
                    })
                    downloaded_count += 1
                    
                except Exception as e:
                    print(f"   ❌ 下载失败: {e}")
                    
        except Exception as e:
            print(f"\n❌ 搜索出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 统计
        print("\n" + "=" * 70)
        print(f"📊 统计: 处理 {processed_count} 篇, 下载 {downloaded_count} 篇, 跳过 {skipped_count} 篇")
        print("=" * 70)
        
        return downloaded_paths, papers_info