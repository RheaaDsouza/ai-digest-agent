from typing import List, Dict, Any
import arxiv
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from agent.config import (
    TAVILY_API_KEY,
    ARXIV_CATEGORIES,
    ARXIV_MAX_RESULTS,
    NEWS_MAX_RESULTS,
)

@tool
def fetch_news(query: str, max_results: int = NEWS_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Fetch news articles using Tavily."""
    if not TAVILY_API_KEY:
        print("TAVILY_API_KEY not set.")
        return []
    try:
        search = TavilySearch(
            api_key=TAVILY_API_KEY,
            max_results=max_results,
            topic="general",
            days=7,
            search_depth="advanced",
        )
        results = search.invoke({"query": query})
    except Exception as e:
        print(f"Tavily error: {e}")
        return []

    articles = []
    for r in results.get("results", []):
        articles.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],
            "source": "news",
        })
    return articles

@tool
def fetch_arxiv_papers(categories: List[str] = None) -> List[Dict[str, Any]]:
    """Fetch latest research papers from arXiv by category."""
    if categories is None:
        categories = ARXIV_CATEGORIES
    cat_query = " OR ".join(f"cat:{c.strip()}" for c in categories if c.strip())
    query = f"({cat_query})"
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "url": result.entry_id,
                "authors": [a.name for a in result.authors[:3]],
                "summary": result.summary[:500],
                "published": result.published.strftime("%Y-%m-%d"),
                "source": "arxiv",
            })
    except Exception as e:
        print(f"arXiv fetch error: {e}")
    return papers