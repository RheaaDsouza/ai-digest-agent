from typing import TypedDict, List, Optional, Set

# State that flows through the graph.
class AgentState(TypedDict):
    query: str
    news_articles: List[dict]
    research_papers: List[dict]
    curated_digest: str
    email_status: str
    error: Optional[str]
    seen_urls: Set[str]
