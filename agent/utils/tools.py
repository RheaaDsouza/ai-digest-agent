from typing import List
import arxiv
from langchain_tavily import TavilySearch
from agent.config import (
    TAVILY_API_KEY,
    ARXIV_CATEGORIES,
    ARXIV_MAX_RESULTS,
    NEWS_MAX_RESULTS,
)

# Fetch news articles using Tavily search.
def fetch_news(query: str, max_results: int = NEWS_MAX_RESULTS):
    search = TavilySearch(
        api_key=TAVILY_API_KEY,
        max_results=max_results,
        topic="news",
        search_depth="advanced",
    )
    results = search.invoke({"query": query})
    
    articles = []
    for r in results.get("results", []):
        articles.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],  # Truncate for context
            "source": "news",
        })
    return articles

# Fetch latest arXiv papers by category.
def fetch_arxiv_papers(
    categories: List[str] = None,
    max_results: int = ARXIV_MAX_RESULTS,
):
   
    if categories is None:
        categories = ARXIV_CATEGORIES
    
    # Build query: (cat:cs.AI OR cat:cs.LG) AND submittedDate:[recent]
    cat_query = " OR ".join(f"cat:{c.strip()}" for c in categories if c.strip())
    query = f"({cat_query})"
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
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