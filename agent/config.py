import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-3.6-flash"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Email
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Gmail app password
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")

# Agent 
NEWS_QUERY = "Latest developments in AI and machine learning"
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]
ARXIV_MAX_RESULTS = 3
NEWS_MAX_RESULTS = 3