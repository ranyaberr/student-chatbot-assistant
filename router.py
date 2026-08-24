# ---------------------------------------------------------
# ROUTER: deterministic, LLM-free classification of user messages
# ---------------------------------------------------------
#
# This module decides WHAT context is needed to answer a message.
# It never calls an LLM. It uses keyword/phrase matching only.
#
# The chatbot's primary audience speaks French, so French phrases are
# the primary keyword set for each category. English equivalents are
# kept as a secondary set where reasonable.
#
# A single message can trigger multiple context needs at once
# (e.g. "Quels devoirs ai-je, et tu te souviens de mon projet ?"
# needs both CLASSROOM and MEMORY context).

import re
from dataclasses import dataclass

# ---- CLASSROOM ----
# Assignments / homework / deadlines (Classroom data).
CLASSROOM_KEYWORDS = [
    # French (primary)
    "devoir", "devoirs",
    "tâche", "tâches",
    "date limite",
    "rendu",
    "à rendre",
    "en retard",
    "delai", "délai",
    "echeance", "échéance",
    # English (secondary)
    "assignment", "assignments",
    "homework",
    "task", "tasks",
    "pending",
    "deadline", "deadlines",
    "due",
    "overdue",
    "classroom",
]

# ---- PREVIOUS_CONVERSATION ----
# Explicit references to an earlier, distinct conversation.
# NOTE: broad words like "avant" / "before" are deliberately excluded
# on their own — they cause false positives in ordinary sentences
# (e.g. "avant de commencer", "before I start"). Only specific,
# unambiguous phrases are used.
PREVIOUS_CONVERSATION_KEYWORDS = [
    # French (primary)
    "la dernière fois", "la derniere fois",
    "hier",
    "conversation précédente", "conversation precedente",
    "conversation d'avant",
    "dont on a parlé", "dont on a parle",
    "on a discuté", "on a discute",
    "on avait dit",
    "précédemment", "precedemment",
    "de quoi on a parle ",
    "De quoi on avait parle dans notre dernière conversation",
    "De quoi on a parlé dans la dernière conversation ",
    "on avait parlé de",
    # English (secondary)
    "last time",
    "previous conversation",
    "yesterday",
    "we discussed",
    "we talked about",
    "what did i say",
    "what did we say",
    "do you remember",
    "earlier conversation",
    "last conversation",
    "previously",
    "what did we talk about"
]

# ---- MEMORY ----
# Explicit requests to store or recall durable facts about the student.
MEMORY_KEYWORDS = [
    # French (primary) — explicit remember/recall requests
    "souviens-toi",
    "rappelle-toi",
    "comment je m appelle",
    "comment je m'appelle",
    "quel est mon nom",
    "c'est quoi mon nom",
    "c est quoi mon nom", 
    "tu connais mon nom",   
    "tu te souviens",
    "tu te rappelles",
    "n'oublie pas",
    "garde en tête", "garde en tete",
    "retiens que",
    "qu'est-ce que tu sais sur moi",
    "qu est ce que tu sais sur moi",
    "que sais-tu sur moi",
    "qu'est ce que tu sais de moi",
    "que sais-tu de moi",
    "qu'est-ce que tu te souviens de moi",
    "qu'est-ce que tu sais sur mes études", "qu'est-ce que tu sais sur mes etudes",
    "que sais-tu sur mes études", "que sais-tu sur mes etudes",
    # French (primary) — natural questions about the student's own
    # studies/projects/goals (specific phrases, not single broad words)
    "qu'est-ce que j'étudie", "qu'est-ce que j'etudie",
    "qu'est ce que j'étudie", "qu'est ce que j'etudie",
    "quelle est ma filière", "quelle est ma filiere",
    "sur quel projet",
    "quel langage de programmation",
    "quels sont mes objectifs",
    "mes objectifs",
    "sur mes études", "sur mes etudes",
    # English (secondary)
    "remember that",
    "remember this",
    "don't forget",
    "keep in mind",
    "note that",
    "what am i learning",
    "what programming language am i",
    "what do you know about me",
    "what do you remember about me",
    "what am i studying",
    "what project am i working on",
    "what are my goals",
]


@dataclass
class RouteResult:
    needs_classroom: bool
    needs_previous_conversation: bool
    needs_memory: bool

    @property
    def is_normal(self):
        """True when no extra context is required at all."""
        return not (
            self.needs_classroom
            or self.needs_previous_conversation
            or self.needs_memory
        )

    def label(self):
        """Human-readable label(s), useful for debug logging."""
        flags = []
        if self.needs_classroom:
            flags.append("CLASSROOM")
        if self.needs_previous_conversation:
            flags.append("PREVIOUS_CONVERSATION")
        if self.needs_memory:
            flags.append("MEMORY")
        return " + ".join(flags) if flags else "NORMAL"


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------
# Users type apostrophes and hyphens inconsistently depending on their
# keyboard/browser/OS ("qu'est-ce que" vs "qu est-ce que" vs
# "qu est ce que"). To avoid having to hardcode every spelling variant
# in the keyword lists, we normalize BOTH the keyword strings and the
# incoming user message the same way before matching:
#   1. Replace every apostrophe variant (straight + curly) with a space.
#   2. Replace every hyphen/dash variant (ASCII + typographic) with a space.
#   3. Collapse repeated whitespace into a single space and strip ends.
#
# This is purely textual normalization — no LLM, no new dependencies.

_APOSTROPHE_CHARS = "'’‘‛`´"
_HYPHEN_CHARS = "-‐‑‒–—―"

_APOSTROPHE_PATTERN = re.compile("[" + re.escape(_APOSTROPHE_CHARS) + "]")
_HYPHEN_PATTERN = re.compile("[" + re.escape(_HYPHEN_CHARS) + "]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize(text):
    """
    Normalize apostrophes and hyphens (of any common variant) to
    spaces, then collapse whitespace. Used on both keyword strings and
    the incoming user message so that matching is insensitive to
    punctuation style.
    """
    text = _APOSTROPHE_PATTERN.sub(" ", text)
    text = _HYPHEN_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _compile_patterns(keywords):
    """
    Compile each keyword/phrase into a word-boundary regex so that,
    e.g., "travail" does not match inside "travaille". The keyword is
    normalized first (apostrophes/hyphens -> spaces) so the compiled
    pattern matches against normalized message text. Still pure
    Python/regex — no LLM involved.
    """
    return [
        re.compile(r"\b" + re.escape(_normalize(keyword)) + r"\b", re.UNICODE)
        for keyword in keywords
    ]


_CLASSROOM_PATTERNS = _compile_patterns(CLASSROOM_KEYWORDS)
_PREVIOUS_CONVERSATION_PATTERNS = _compile_patterns(PREVIOUS_CONVERSATION_KEYWORDS)
_MEMORY_PATTERNS = _compile_patterns(MEMORY_KEYWORDS)


def _contains_any(text_lower, patterns):
    return any(pattern.search(text_lower) for pattern in patterns)


def route_message(user_message):
    """
    Classify a user message using plain Python logic (no LLM call).

    Returns a RouteResult with independent boolean flags, so a single
    message can require classroom data AND memory AND previous-chat
    context at the same time.
    """
    text = _normalize(user_message.lower())

    needs_classroom = _contains_any(text, _CLASSROOM_PATTERNS)
    needs_previous_conversation = _contains_any(text, _PREVIOUS_CONVERSATION_PATTERNS)
    needs_memory = _contains_any(text, _MEMORY_PATTERNS)

    return RouteResult(
        needs_classroom=needs_classroom,
        needs_previous_conversation=needs_previous_conversation,
        needs_memory=needs_memory,
    )