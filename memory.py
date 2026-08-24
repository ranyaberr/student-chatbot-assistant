# --------------------------------------------------------- 
# MEMORY: Helper functions for student memory (memories.json) 
# --------------------------------------------------------- 
import os 
import json 
import uuid 
 
 
from conversations import safe_load_json 
 
DATA_FOLDER = "data" 
MEMORY_FILE = os.path.join(DATA_FOLDER, "memories.json") 
 
def load_memories(): 
    return safe_load_json(MEMORY_FILE, []) 
 
def save_memories(memories): 
    os.makedirs(DATA_FOLDER, exist_ok=True) 
    with open(MEMORY_FILE, "w", encoding="utf-8") as f: 
        json.dump(memories, f, indent=2, ensure_ascii=False) 
 
# Small English/French stopwords list used to strip noise words before
# comparing text. Deliberately short and simple — this is keyword
# overlap, not NLP.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "am", "i", "me", "my",
    "what", "which", "who", "do", "does", "did", "you", "your", "to",
    "of", "in", "on", "for", "and", "or", "about", "that", "this",
    "le", "la", "les", "un", "une", "de", "des", "je", "tu", "il",
    "que", "qui", "quoi", "sur", "pour", "et", "ou",
}


def _keywords(text):
    words = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _similarity(text_a, text_b):
    """
    Jaccard-style overlap between the meaningful keywords of two
    strings. Returns a float in [0, 1]. Pure Python, no LLM.
    """
    keywords_a = _keywords(text_a)
    keywords_b = _keywords(text_b)
    if not keywords_a or not keywords_b:
        return 0.0
    intersection = keywords_a & keywords_b
    union = keywords_a | keywords_b
    return len(intersection) / len(union)


# Thresholds for deterministic fact deduplication/update. These are
# heuristics, not semantic understanding — tuned to catch obvious
# rewordings ("learning Java" vs "currently learning Java") without
# an LLM call.
DUPLICATE_SIMILARITY_THRESHOLD = 0.6
UPDATE_SIMILARITY_THRESHOLD = 0.3


def _classify_fact(new_fact, existing_memories):
    """
    Deterministically decide whether a new fact should be added,
    should update an existing memory, or should be skipped as a
    duplicate — using keyword-overlap similarity only. No LLM call.

    Returns a tuple: ("add", None) | ("update", index) | ("skip", None)
    """
    if not existing_memories:
        return ("add", None)

    best_index = None
    best_score = 0.0

    for i, memory in enumerate(existing_memories):
        score = _similarity(new_fact, memory["text"])
        if score > best_score:
            best_score = score
            best_index = i

    if best_score >= DUPLICATE_SIMILARITY_THRESHOLD:
        return ("skip", None)
    elif best_score >= UPDATE_SIMILARITY_THRESHOLD:
        return ("update", best_index)
    else:
        return ("add", None)


def store_new_memories(new_facts, conversation_id): 
    """
    Add new facts to memories.json, handling duplicates/updates using
    deterministic Python keyword-overlap matching. No LLM call.
    """ 
    if not new_facts: 
        return 
 
    memories = load_memories() 
 
    for fact in new_facts:
        action, index = _classify_fact(fact, memories)

        if action == "skip": 
            continue 
        elif action == "update": 
            memories[index]["text"] = fact 
            memories[index]["conversation_id"] = conversation_id 
        else:  # add 
            memories.append({ 
                "id": str(uuid.uuid4()), 
                "text": fact, 
                "conversation_id": conversation_id 
            }) 
 
    save_memories(memories) 
 
def retrieve_relevant_memories(current_question, all_memories, max_memories=5, fallback_count=3):
    """
    Select memories relevant to the current question using lightweight
    keyword overlap. No LLM call — pure Python string matching.

    A memory is considered relevant if it shares at least one
    meaningful keyword with the question. Results are ranked by number
    of overlapping keywords (most overlap first).

    Fallback: generic questions like "what do you know about me?" or
    "qu'est-ce que tu sais sur moi ?" strip down to almost no
    meaningful keywords after stopword removal, so they never overlap
    with specific stored facts. When keyword matching finds nothing,
    fall back to the most recently stored facts instead of returning
    an empty list — still 0 LLM calls, just a deterministic fallback.
    """
    if not all_memories:
        return []

    question_keywords = _keywords(current_question)

    scored = []
    if question_keywords:
        for memory in all_memories:
            memory_keywords = _keywords(memory["text"])
            overlap = question_keywords & memory_keywords
            if overlap:
                scored.append((len(overlap), memory["text"]))
        scored.sort(key=lambda pair: pair[0], reverse=True)

    if scored:
        return [text for _, text in scored[:max_memories]]

    # No keyword overlap (or no meaningful keywords at all) — fall
    # back to the most recently stored facts.
    return [m["text"] for m in all_memories[-fallback_count:]]