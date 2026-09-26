# AI Digest Agent

A daily email digest of AI news and arXiv papers.
The agent fetches recent articles and papers, filters out anything you've already seen, summarizes the rest, and sends it to your inbox.

This project was built as a hands-on way to learn [LangGraph](https://langchain-ai.github.io/langgraph/) and agent orchestration in Python.

## How it works

Each run does the following:

1. Fetches AI/ML news from Tavily search.
2. Fetches recent arXiv papers in AI, ML, and NLP categories.
3. Loads the previous seen URL set from a local JSON file.
4. Filters out items you already received.
5. Sends the remaining content to Gemini for a concise digest.
6. Falls back to a plain link list if the LLM fails.
7. Sends the digest by email.

## Workflow 

The main execution path is:

- main.py builds the graph and launches it.
- fetch_content_node calls both data sources:
  - Tavily for news articles
  - arXiv for recent papers
- dedupe_node removes URLs already stored in seen_urls.json.
- curate_content_node builds a markdown prompt and asks Gemini to summarize the new information.
- should_send_email checks whether there is a valid digest to send.
- send_email_node sends the content via Gmail SMTP.
- The result is persisted only after a successful send.

## Configuration

Create a .env file in the project root with the values below:

```bash
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_key
EMAIL_SENDER=you@gmail.com
EMAIL_PASSWORD=your_app_password #Use a Gmail App Password for EMAIL_PASSWORD, not your actual account password.
EMAIL_RECIPIENT=you@gmail.com
```

## Run

```bash
git clone https://github.com/your-username/ai-digest-agent.git
cd ai-digest-agent
uv sync
uv run python main.py
```

If you do not have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
