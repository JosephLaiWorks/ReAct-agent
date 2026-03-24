import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()


def clean_text(text: str, max_len: int = 220) -> str:
    if not text:
        return ""

    # 移除 emoji / 奇怪 unicode，避免 Observation 太雜
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text


def search_web(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "Search error: TAVILY_API_KEY not found in .env"

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": 3,
        "include_answer": False,
        "include_raw_content": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return "No clear result found."

        lines = []
        for i, item in enumerate(results[:3], 1):
            title = clean_text(item.get("title", "No title"), 90)
            content = clean_text(item.get("content", "No content"), 220)
            source = item.get("url", "No URL")
            lines.append(f"[{i}] {title}\n{content}\nSource: {source}")

        return "\n\n".join(lines)

    except requests.RequestException as e:
        return f"Search error: {str(e)}"
    except Exception as e:
        return f"Unexpected search error: {str(e)}"