import json
import os
from typing import Set

SEEN_FILE = os.getenv("SEEN_FILE", "seen_urls.json")

# Load previously seen URLs
def load_seen_urls() -> Set[str]:
    if not os.path.exists(SEEN_FILE):
        return set()
    try:
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not load {SEEN_FILE}: {e}")
        return set()

# Persist seen URLs to disk as a sorted JSON list
def save_seen_urls(urls: Set[str]) -> None:
    with open(SEEN_FILE, "w") as f:
        json.dump(sorted(urls), f, indent=2)
