from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from agent.utils.state import AgentState
from agent.utils.nodes import (
    agent_node,
    tool_node,
    write_digest_node,
    send_email_node,
    should_send_email,
)
from agent.config import NEWS_QUERY

# Build and compile the LangGraph
def build_graph():
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("write_digest", write_digest_node)
    workflow.add_node("send_email", send_email_node)

    # Entry point
    workflow.add_edge(START, "agent")

    # The agentic loop
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: "write_digest", 
        },
    )
    workflow.add_edge("tools", "agent")  # Loop back to agent after tools run

    # decide whether to email
    workflow.add_conditional_edges(
        "write_digest",
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
        "messages": [HumanMessage(content=query or NEWS_QUERY)],
        "curated_digest": "",
        "email_status": "pending",
        "error": None,
    }

    print("\nStarting agent...\n")
    result = graph.invoke(initial_state)

    print(f"\nDone! Email status: {result['email_status']}")

    return result

if __name__ == "__main__":
    main()