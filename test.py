from googlesearch import search

from config.logger_config import logger


def _search(question: str):
    try:
        results = []
        for url in search(question, num_results=3, advanced=True):
            results.append(f"{url.title}\n{url.description}")
        return "\n".join(results)
    except Exception as e:
        logger.warning(f"Google 搜索失败: {e}")
        return ""


if __name__ == "__main__":
    print(_search("巴黎奥运会的乒乓球男单冠军是谁？"))
