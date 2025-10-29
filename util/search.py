from typing import Dict, List

from googlesearch import search

from config.logger_config import logger

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

# DuckDuckGo Search import with fallback
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
    logger.warning("DDGS not available. Please install with: pip install ddgs")


def google_search(question: str, max_results: int = 3) -> List[Dict[str, str]]:
    try:
        results = []
        for url in search(question, num_results=max_results, advanced=True):
            results.append({"title": url.title,  "body": url.description})
        return results
    except Exception as e:
        logger.warning(f"Google 搜索失败: {e}")
        return []  # 修复：返回空列表而不是空字符串

def ddgs_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(
                    query,
                    region="cn-zh",
                    safesearch="off",
                    max_results=max_results,
                    timelimit="y"            # 限定近一年
                ):
                    title = r.get("title", "")
                    body = r.get("body", "")
                    url = r.get("href", "")
                    results.append({"title": title, "body": body, "url": url})
            logger.info(f"ddgs search results: {results}")     
            return results        
        except Exception as e:
            logger.error(f"DDGS search failed: {e}")
            return []

def baidu_search(query: str, max_results: int = 3):
    """
    基于百度搜索的简易封装函数
    :param query: 搜索关键词（中文或英文）
    :param max_results: 返回结果条数（默认 3 条）
    :return: list[dict] -> [{'title': 标题, 'url': 链接, 'snippet': 摘要}]
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        )
    }

    search_url = f"https://www.baidu.com/s?wd={quote(query)}"
    resp = requests.get(search_url, headers=headers, timeout=10)
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for item in soup.select("div.result")[:max_results]:
        title_tag = item.select_one("h3 a")
        snippet_tag = item.select_one(".c-abstract, .content-right_8Zs40")

        title = title_tag.get_text(strip=True) if title_tag else ""
        url = title_tag["href"] if title_tag and "href" in title_tag.attrs else ""
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

        if title and url:
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })
    logger.info(f"baidu search results: {results}")    
    return results        