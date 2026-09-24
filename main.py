from langgraph.graph import StateGraph, START, END

from agent.utils.state import AgentState
from agent.utils.nodes import (
    dedupe_node,
    fetch_content_node,
    curate_content_node,
    send_email_node,
    should_send_email,
)
from agent.utils.persistence import load_seen_urls, save_seen_urls
from agent.config import NEWS_QUERY

# Build and compile the LangGraph
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("fetch", fetch_content_node)
    workflow.add_node("dedupe", dedupe_node)
    workflow.add_node("curate", curate_content_node)
    workflow.add_node("send_email", send_email_node)
    
    # Linear edges
    workflow.add_edge(START, "fetch")
    workflow.add_edge("fetch", "dedupe")
    workflow.add_edge("dedupe", "curate")
    
    # Conditional routing: only email if we have content
    workflow.add_conditional_edges(
        "curate",
        should_send_email,
        {
            "send_email": "send_email",
            "skip": END,
        },
    )
    
    workflow.add_edge("send_email", END)
    
    return workflow.compile()


def main(query: str = None):
    graph = build_graph()
    initial_state = {
        "query": query or NEWS_QUERY,
        "news_articles": [],
        "research_papers": [],
        "curated_digest": "",
        "email_status": "pending",
        "error": None,
        "seen_urls": load_seen_urls(), 
    }
    
    print("Starting agent...\n")
    result = graph.invoke(initial_state)

    # Only persist if the email actually sent
    if result["email_status"] == "sent":
        new_urls = {a["url"] for a in result["news_articles"]}
        new_urls |= {p["url"] for p in result["research_papers"]}
        save_seen_urls(result["seen_urls"] | new_urls)

    print(f"\n Done! Email status: {result['email_status']}")
    return result


if __name__ == "__main__":
    main()
