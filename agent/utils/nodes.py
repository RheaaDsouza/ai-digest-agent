from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from agent.config import GOOGLE_API_KEY, GEMINI_MODEL, EMAIL_RECIPIENT
from agent.utils.tools import fetch_news, fetch_arxiv_papers
from agent.utils.email import send_email
from agent.utils.state import AgentState

# Tools setup
tools = [fetch_news, fetch_arxiv_papers]
tool_node = ToolNode(tools)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    max_retries=5,
    timeout=60,
)

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a research assistant gathering material for a daily digest.

You have access to two tools:
- `search_news(query)`: Search recent news articles about a topic.
- `search_arxiv(categories)`: Fetch latest arXiv papers by category (comma-separated, e.g. "cs.AI,cs.LG").

Your task: gather enough material to create a comprehensive digest.
- Call tools as many times as you need.
- If the user's query is about AI research, focus on `search_arxiv`.
- If it's about general AI news, use `search_news`.
- When you have sufficient material, stop calling tools. Your final message (with no tool calls) will signal you are done.
"""

CURATION_PROMPT = """You are a research assistant curating a daily digest.

Below are news articles and academic papers I've gathered. Create a digest using EXACTLY this markdown structure:

Daily AI & Research Digest

**News Highlights**

[Article Title](URL)
One or two sentences summarizing the article.

**Research Papers**

[Paper Title](URL)
**Authors:** Name, Name, Name
One or two sentences on what the paper is about.

---

Only include items that are genuinely interesting or relevant. If a section has no items, write "_Nothing notable today._" under it. Use the exact URLs provided. Do not invent links.

Here are the articles and papers:
{content}

"""

def agent_node(state: AgentState) -> dict:
    """
    The LLM decides whether to call tools or stop.
    If it returns tool_calls, the graph routes to the tools node.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def write_digest_node(state: AgentState) -> dict:
    """
    After the agent loop finishes, take all the tool results
    and write a digest.
    """
    # Extract tool results from the message history
    tool_results = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_results.append(msg.content)
    
    if not tool_results:
        return {
            "curated_digest": "No new content found today.",
            "error": "Agent did not collect any data",
        }

    # Combine all tool results into a single string for the curation prompt
    combined = "\n---\n".join(str(tr) for tr in tool_results)

    prompt = CURATION_PROMPT.format(content=combined)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content
        if isinstance(raw, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            )
        else:
            content = raw
        return {"curated_digest": content}
    except Exception as e:
        print(f"LLM curation failed: {e}")
        return {
            "curated_digest": fallback_digest(state),
            "error": f"LLM curation failed: {e}",
        }

# Fallback digest in case LLM fails
def fallback_digest(state: AgentState) -> str:
    lines = ["Daily Digest (LLM unavailable)\n"]

    # Extract tool results from the message history
    tool_results = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_results.append(msg.content)

    if not tool_results:
        lines.append("No content collected.")
        return "\n".join(lines)

    # Tool results are lists of dicts (articles/papers).
    # Flatten them into a single list so we can render them uniformly.
    items = []
    for result in tool_results:
        if isinstance(result, list):
            items.extend(result)
        elif isinstance(result, dict):
            items.append(result)

    # Separate news from papers for section headers
    news = [i for i in items if isinstance(i, dict) and i.get("source") == "news"]
    papers = [i for i in items if isinstance(i, dict) and i.get("source") == "arxiv"]

    if news:
        lines.append("News")
        for a in news:
            title = a.get("title", "Untitled")
            url = a.get("url", "")
            if url:
                lines.append(f"- [{title}]({url})")
            else:
                lines.append(f"- {title}")
        lines.append("")

    if papers:
        lines.append("Research Papers")
        for p in papers:
            title = p.get("title", "Untitled")
            url = p.get("url", "")
            authors = ", ".join(p.get("authors", []))
            entry = f"- [{title}]({url})" if url else f"- {title}"
            if authors:
                entry += f" — {authors}"
            lines.append(entry)
        lines.append("")

    if not news and not papers:
        lines.append("No structured content found in tool results.")

    return "\n".join(lines)


# Send the curated digest via email.
def send_email_node(state: AgentState) -> dict:
    print("Sending email...")
    digest = state.get("curated_digest", "")
    if not digest or "No new content" in digest:
        return {"email_status": "skipped - no new content"}
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
    if digest and "No new content" not in digest:
        return "send_email"
    return "skip"