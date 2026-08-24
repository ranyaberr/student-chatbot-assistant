# ---------------------------------------------------------
# MEMORY: Helper functions for student memory (memories.json)
# ---------------------------------------------------------

import os
import json
import uuid

from conversations import safe_load_json


DATA_FOLDER = "data"
MEMORY_FILE = os.path.join(DATA_FOLDER, "memories.json")


# ---------------------------------------------------------
# LOAD / SAVE
# ---------------------------------------------------------

def load_memories():
    return safe_load_json(MEMORY_FILE, [])


def save_memories(memories):
    os.makedirs(DATA_FOLDER, exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            memories,
            f,
            indent=2,
            ensure_ascii=False
        )


# ---------------------------------------------------------
# KEYWORDS
# ---------------------------------------------------------

_STOPWORDS = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "am",
    "i", "me", "my", "mine",
    "what", "which", "who",
    "do", "does", "did",
    "you", "your",
    "to", "of", "in", "on", "for",
    "and", "or",
    "about", "that", "this",
    "how",

    # French
    "le", "la", "les",
    "un", "une",
    "des",
    "de", "du", "au", "aux",
    "je", "j", "tu", "il", "elle",
    "me", "moi", "mon", "ma", "mes",
    "ton", "ta", "tes",
    "que", "qui", "quoi",
    "sur", "pour",
    "et", "ou",
    "est", "suis",
    "dans", "avec",
    "quel", "quelle",
    "quels", "quelles",
    "comment",
}


def _keywords(text):
    """
    Extract simple meaningful words.

    This is intentionally lightweight.
    No LLM and no external NLP library.
    """

    words = "".join(
        c if c.isalnum() else " "
        for c in text.lower()
    ).split()

    return {
        word
        for word in words
        if word not in _STOPWORDS and len(word) > 2
    }


def _similarity(text_a, text_b):
    """
    Jaccard-style keyword similarity.
    """

    keywords_a = _keywords(text_a)
    keywords_b = _keywords(text_b)

    if not keywords_a or not keywords_b:
        return 0.0

    intersection = keywords_a & keywords_b
    union = keywords_a | keywords_b

    return len(intersection) / len(union)


# ---------------------------------------------------------
# MEMORY DEDUPLICATION
# ---------------------------------------------------------

DUPLICATE_SIMILARITY_THRESHOLD = 0.6
UPDATE_SIMILARITY_THRESHOLD = 0.3


def _classify_fact(new_fact, existing_memories):
    """
    Decide whether a new fact should be:

    - added
    - used to update an existing fact
    - ignored as a duplicate

    Uses deterministic keyword similarity only.
    """

    if not existing_memories:
        return ("add", None)

    best_index = None
    best_score = 0.0

    for i, memory in enumerate(existing_memories):

        score = _similarity(
            new_fact,
            memory["text"]
        )

        if score > best_score:
            best_score = score
            best_index = i

    if best_score >= DUPLICATE_SIMILARITY_THRESHOLD:
        return ("skip", None)

    if best_score >= UPDATE_SIMILARITY_THRESHOLD:
        return ("update", best_index)

    return ("add", None)


# ---------------------------------------------------------
# STORE NEW MEMORIES
# ---------------------------------------------------------

def store_new_memories(new_facts, conversation_id):
    """
    Store extracted facts.

    Existing storage behavior is preserved.
    """

    if not new_facts:
        return

    memories = load_memories()

    for fact in new_facts:

        action, index = _classify_fact(
            fact,
            memories
        )

        if action == "skip":
            continue

        elif action == "update":

            memories[index]["text"] = fact
            memories[index]["conversation_id"] = conversation_id

        else:

            memories.append({
                "id": str(uuid.uuid4()),
                "text": fact,
                "conversation_id": conversation_id
            })

    save_memories(memories)


# =========================================================
# PERSONAL INFORMATION RETRIEVAL
# =========================================================

# The important part of this fix.
#
# The chatbot may store:
#
#     "Name: Ranya"
#
# but the user may ask:
#
#     "c est quoi mon nom ?"
#
# These two strings have almost no lexical overlap.
#
# Therefore we identify the CATEGORY of the question first,
# then search the stored memories for facts belonging to that
# category.


_PERSONAL_CATEGORIES = {

    "name": {
        "question_keywords": {
            "nom",
            "appelle",
            "name",
            "whoami",
        },

        "fact_keywords": {
            "name",
            "nom",
            "appelle",
        },
    },

    "age": {
        "question_keywords": {
            "age",
            "âge",
            "ans",
            "vieux",
            "vieille",
            "old",
        },

        "fact_keywords": {
            "age",
            "âge",
            "ans",
        },
    },

    "studies": {
        "question_keywords": {
            "étudie",
            "etudes",
            "études",
            "filière",
            "filiere",
            "formation",
            "studying",
            "study",
            "major",
            "degree",
            "domaine",
        },

        "fact_keywords": {
            "étud",
            "etud",
            "study",
            "studying",
            "filière",
            "filiere",
            "formation",
            "course",
            "module",
            "student",
        },
    },

    "university": {
        "question_keywords": {
            "université",
            "universite",
            "école",
            "ecole",
            "fac",
            "school",
            "university",
            "college",
        },

        "fact_keywords": {
            "université",
            "universite",
            "école",
            "ecole",
            "school",
            "university",
            "college",
        },
    },

    "country": {
        "question_keywords": {
            "pays",
            "origine",
            "viens",
            "origine",
            "country",
            "from",
            "nationality",
            "nationalité",
        },

        "fact_keywords": {
            "country",
            "pays",
            "origine",
            "nationality",
            "nationalité",
            "maroc",
            "morocco",
        },
    },

    "project": {
        "question_keywords": {
            "projet",
            "project",
            "travaille",
            "travail",
            "working",
            "build",
            "construis",
        },

        "fact_keywords": {
            "project",
            "projet",
            "travail",
            "working",
            "building",
            "build",
        },
    },

    "goals": {
        "question_keywords": {
            "objectif",
            "objectifs",
            "goal",
            "goals",
            "but",
            "ambition",
        },

        "fact_keywords": {
            "objectif",
            "goal",
            "but",
            "ambition",
        },
    },

    "language": {
        "question_keywords": {
            "langue",
            "language",
            "parle",
            "préférence",
            "preference",
        },

        "fact_keywords": {
            "language",
            "langue",
            "préférence",
            "preference",
        },
    },
}


def _detect_personal_category(question):
    """
    Detect whether the question is asking about a known
    personal-information category.

    Returns:
        category name
        or None
    """

    question_words = _keywords(question)

    if not question_words:
        return None

    best_category = None
    best_score = 0

    for category, rules in _PERSONAL_CATEGORIES.items():

        category_words = rules["question_keywords"]

        overlap = question_words & category_words

        score = len(overlap)

        if score > best_score:
            best_score = score
            best_category = category

    return best_category


def _retrieve_category_memories(
    category,
    all_memories,
    max_memories
):
    """
    Retrieve memories belonging to a detected personal category.
    """

    if category not in _PERSONAL_CATEGORIES:
        return []

    fact_keywords = _PERSONAL_CATEGORIES[
        category
    ]["fact_keywords"]

    candidates = []

    for memory in all_memories:

        text = memory["text"].lower()

        score = 0

        for keyword in fact_keywords:

            if keyword in text:
                score += 1

        if score > 0:

            candidates.append(
                (
                    score,
                    memory["text"]
                )
            )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        text
        for _, text in candidates[:max_memories]
    ]


# ---------------------------------------------------------
# GENERAL MEMORY RETRIEVAL
# ---------------------------------------------------------

def retrieve_relevant_memories(
    current_question,
    all_memories,
    max_memories=5,
    fallback_count=3
):
    """
    Retrieve memories relevant to the user's question.

    Retrieval order:

    1. Personal-information category
    2. Normal keyword overlap
    3. Recent-memory fallback

    No LLM is used here.
    """

    if not all_memories:
        return []

    # -----------------------------------------------------
    # STEP 1 — PERSONAL CATEGORY
    # -----------------------------------------------------

    category = _detect_personal_category(
        current_question
    )

    if category is not None:

        category_memories = _retrieve_category_memories(
            category,
            all_memories,
            max_memories
        )

        if category_memories:

            print(
                f"[MEMORY DEBUG] "
                f"Detected category: {category}"
            )

            print(
                f"[MEMORY DEBUG] "
                f"Category memories: {category_memories}"
            )

            return category_memories

    # -----------------------------------------------------
    # STEP 2 — NORMAL KEYWORD MATCHING
    # -----------------------------------------------------

    question_keywords = _keywords(
        current_question
    )

    scored = []

    if question_keywords:

        for memory in all_memories:

            memory_keywords = _keywords(
                memory["text"]
            )

            overlap = (
                question_keywords
                & memory_keywords
            )

            if overlap:

                scored.append(
                    (
                        len(overlap),
                        memory["text"]
                    )
                )

        scored.sort(
            key=lambda pair: pair[0],
            reverse=True
        )

    if scored:

        results = [
            text
            for _, text
            in scored[:max_memories]
        ]

        print(
            f"[MEMORY DEBUG] "
            f"Keyword memories: {results}"
        )

        return results

    # -----------------------------------------------------
    # STEP 3 — FALLBACK
    # -----------------------------------------------------

    fallback = [
        memory["text"]
        for memory
        in all_memories[-fallback_count:]
    ]

    print(
        f"[MEMORY DEBUG] "
        f"Fallback memories: {fallback}"
    )

    return fallback