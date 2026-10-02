from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages

# State that flows through the graph.
class AgentState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    curated_digest: str
    email_status: str
    error: Optional[str]