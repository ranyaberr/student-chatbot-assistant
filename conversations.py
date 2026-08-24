import os
import json
import uuid

from config import SYSTEM_PROMPT


DATA_FOLDER = "data"
DATA_FILE = os.path.join(
    DATA_FOLDER,
    "conversations.json"
)


def safe_load_json(path, default):
    """
    Safely load a JSON file.

    Returns default if the file:
    - does not exist
    - is empty
    - contains invalid JSON
    """

    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return default

    try:
        return json.loads(content)

    except json.JSONDecodeError as e:
        print(
            f"[ERROR] Malformed JSON in {path}: {e}"
        )
        return default


def load_conversations():
    return safe_load_json(
        DATA_FILE,
        {}
    )


def save_conversations(conversations):
    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    with open(
        DATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            conversations,
            f,
            indent=2,
            ensure_ascii=False
        )


def make_title(
    first_user_message,
    max_length=30
):
    title = (
        first_user_message
        .strip()
        .replace("\n", " ")
    )

    if len(title) > max_length:
        title = (
            title[:max_length]
            .rstrip()
            + "..."
        )

    return title


def create_new_conversation(conversations):
    conversation_id = str(uuid.uuid4())

    conversations[conversation_id] = {
        "title": "New conversation",
        "summary": "",
        "memory_processed_count": 0,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
    }

    return conversation_id


def ensure_conversation_fields(conversations):
    """
    Add missing fields to conversations created
    by older versions of the application.
    """

    for conv in conversations.values():

        if "summary" not in conv:
            conv["summary"] = ""

        if "memory_processed_count" not in conv:
            conv["memory_processed_count"] = 0

    return conversations