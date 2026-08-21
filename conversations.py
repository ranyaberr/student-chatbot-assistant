import os
import json
import uuid 
from config import SYSTEM_PROMPT

DATA_FOLDER = "data"
DATA_FILE = os.path.join(DATA_FOLDER, "conversations.json")

# ---------------------------------------------------------
# Generic safe JSON loader (fixes empty/invalid file crashes)
# ---------------------------------------------------------
def safe_load_json(path, default):
    """Read JSON from disk. Returns `default` if file is missing, empty, or invalid."""
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return default
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return default

# ---------------------------------------------------------
# STEP 2: Helper functions to load/save conversations.json
# ---------------------------------------------------------
def load_conversations():
    """Read conversations.json from disk. If it doesn't exist/is invalid, return an empty dict."""
    return safe_load_json(DATA_FILE, {})

def save_conversations(conversations):
    """Write the conversations dictionary back to disk as JSON."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

def make_title(first_user_message, max_length=30):
    """Turn the first user message into a short title for the sidebar."""
    title = first_user_message.strip().replace("\n", " ")
    if len(title) > max_length:
        title = title[:max_length].rstrip() + "..."
    return title

def create_new_conversation(conversations):
    """Create a brand-new empty conversation and return its ID."""
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = {
        "title": "New conversation",
        "summary": "",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    }
    return conversation_id

def ensure_conversation_fields(conversations):
    """
    Make sure every conversation has a 'summary' field, even if it was
    saved by an older version of the app that didn't have this feature.
    """
    for conv in conversations.values():
        if "summary" not in conv:
            conv["summary"] = ""
    return conversations
