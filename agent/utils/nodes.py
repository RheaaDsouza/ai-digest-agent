from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.config import GOOGLE_API_KEY, GEMINI_MODEL, EMAIL_RECIPIENT
from agent.utils.tools import fetch_news, fetch_arxiv_papers

from agent.utils.email import send_email
from agent.utils.state import AgentState


SYSTEM_PROMPT = """You are a research assistant curating a daily digest.

Below are news articles and academic papers I've gathered. Create a digest using
this markdown structure:

# Daily Digest

## News Highlights

### [Article Title](URL)
One or two sentences summarizing the article.

### [Article Title](URL)
One or two sentences summarizing the article.

## Research Papers

### [Paper Title](URL)
**Authors:** Name, Name, Name
One or two sentences on what the paper is about.

---
If a section has no items, write "Nothing notable today." under it. 
Use the exact URLs provided. Do not invent links.

Here are the articles and papers:
{content}

"""

# Fetch both news articles and research papers
def fetch_content_node(state: AgentState) -> dict:
    print(f"Fetching news for: {state['query']}")
    news = fetch_news(state["query"])
    print(f"   Found {len(news)} news articles")
    
    print("Fetching arXiv papers...")
    papers = fetch_arxiv_papers()
    print(f"   Found {len(papers)} research papers")
    
    return {
        "news_articles": news,
        "research_papers": papers,
    }

# Use LLM to curate and summarize the fetched content
def curate_content_node(state: AgentState) -> dict:
    print(" Curating digest with LLM...")
    
    # Combine all content into a single string for the prompt
    content_parts = []
    
    for i, article in enumerate(state.get("news_articles", []), 1):
        content_parts.append(
            f"NEWS {i}:\n"
            f"Title: {article['title']}\n"
            f"URL: {article['url']}\n"
            f"Content: {article['content']}\n"
        )
    
    for i, paper in enumerate(state.get("research_papers", []), 1):
        authors = ", ".join(paper.get("authors", []))
        content_parts.append(
            f"PAPER {i}:\n"
            f"Title: {paper['title']}\n"
            f"Authors: {authors}\n"
            f"URL: {paper['url']}\n"
            f"Published: {paper.get('published', 'unknown')}\n"
            f"Abstract: {paper['summary']}\n"
        )
    
    combined = "\n---\n".join(content_parts)
    
    if not combined.strip():
        return {
            "curated_digest": "No content found for today's digest.",
            "error": "No articles or papers fetched",
        }
    
    llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model=GEMINI_MODEL,
        max_retries=3,
    )
    
    prompt = SYSTEM_PROMPT.format(content=combined)

    # response = llm.invoke([HumanMessage(content=prompt)])
    # return {"curated_digest": response.content}
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return {"curated_digest": content}

    except Exception as e:
        print(f"LLM failed after retries: {e}")
        return {
            "curated_digest": fallback(state),
            "error": f"LLM curation failed: {e}",
        }
    
def fallback(state: AgentState) -> str:
    lines = ["# Daily Digest (LLM unavailable)\n"]

    news = state.get("news_articles", [])
    if news:
        lines.append("## News")
        for a in news:
            lines.append(f"- [{a['title']}]({a['url']})")
    else:
        lines.append("## News\n_No articles fetched today._")

    papers = state.get("research_papers", [])
    if papers:
        lines.append("\n## Research Papers")
        for p in papers:
            authors = ", ".join(p.get("authors", []))
            lines.append(f"- [{p['title']}]({p['url']}) — {authors}")

    return "\n".join(lines)


def dedupe_node(state: AgentState) -> dict:
    seen = state.get("seen_urls", set())
    news = [a for a in state["news_articles"] if a["url"] not in seen]
    papers = [p for p in state["research_papers"] if p["url"] not in seen]
    print(f"🔁 Dedupe: {len(state['news_articles']) - len(news)} news, "
          f"{len(state['research_papers']) - len(papers)} papers filtered")

    return {
        "news_articles": news, 
        "research_papers": papers
    }


# Send the curated digest via email.
def send_email_node(state: AgentState) -> dict:
    print("Sending email...")
    
    digest = state.get("curated_digest", "")
    if not digest or "No content" in digest:
        return {"email_status": "skipped - no content"}
    
    try:
        send_email(
            to=EMAIL_RECIPIENT,
            subject="Your Daily AI & Research Digest",
            body=digest,
        )
        return {"email_status": "sent"}
    except Exception as e:
        print(f"Email error: {e}")
        return {"email_status": f"failed: {e}"}


# Conditional routing - only send if we have content.
def should_send_email(state: AgentState) -> str:
    digest = state.get("curated_digest", "")
    if digest and "No content found" not in digest:
        return "send_email"
    return "skip"