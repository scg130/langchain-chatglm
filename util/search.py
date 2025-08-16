from typing import Dict, List

from googlesearch import search

from config.logger_config import logger


def google_search(question: str, max_results: int = 3) -> List[Dict[str, str]]:
    try:
        results = []
        for url in search(question, num_results=max_results, advanced=True):
            results.append({"title": url.title,  "body": url.description})
        return results
    except Exception as e:
        logger.warning(f"Google 搜索失败: {e}")
        return ""
