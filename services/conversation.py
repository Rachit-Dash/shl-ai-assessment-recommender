"""Conversation classification and safety layer.

Classifies each user turn into one of four intents (recommend, clarify,
compare, refuse) and detects prompt-injection, off-topic, and vague inputs
before any retrieval or LLM call is made.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.helpers import clean_text, infer_requested_type_codes, normalize_for_match


# ---------------------------------------------------------------------------
# Prompt injection detection patterns
# ---------------------------------------------------------------------------

PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all )?(previous|prior|above|system|developer) instructions\b",
    r"\bdisregard (all )?(previous|prior|above|system|developer) instructions\b",
    r"\boverride (the )?(system|developer|instruction|rules)\b",
    r"\breveal (the )?(system|developer|hidden|internal) prompt\b",
    r"\bshow (the )?(system|developer|hidden|internal) prompt\b",
    r"\bjailbreak\b",
    r"\bact as\b.*\bwithout restrictions\b",
    r"\bdo anything now\b",
    r"\bprint.*environment variables\b",
    r"\bexfiltrate\b",
]


# ---------------------------------------------------------------------------
# Domain vocabulary for topic classification
# ---------------------------------------------------------------------------

HIRING_DOMAIN_TERMS = {
    "hire", "hiring", "candidate", "candidates", "role", "roles", "job",
    "assessment", "assessments", "test", "tests", "screen", "screening",
    "recruit", "recruiting", "selection", "evaluate", "evaluation",
    "interview", "talent", "employee", "developer", "engineer", "manager",
    "sales", "support", "service", "graduate", "intern", "analyst",
    "personality", "cognitive", "technical", "coding", "skills",
    "ability", "aptitude", "reasoning", "behavioral", "behavioural",
    "leadership", "java", "python", "sql", "javascript", "opq", "gsa",
}

STRICT_REFUSAL_TERMS = {
    "legal advice", "tax advice", "medical advice", "diagnosis", "lawsuit",
    "contract review", "business consulting", "business plan", "market entry",
    "pricing strategy", "investment advice", "financial advice",
}

OFF_TOPIC_TERMS = {
    "weather", "recipe", "movie", "song", "poem", "joke", "homework",
    "essay", "stock", "crypto", "medical", "doctor", "investment", "loan",
}


# ---------------------------------------------------------------------------
# Intent classification patterns
# ---------------------------------------------------------------------------

VAGUE_PATTERNS = [
    r"^\s*(hi|hello|hey)\s*$",
    r"\b(i|we)\s+need\s+(an?\s+)?(assessment|test)\s*$",
    r"\brecommend\s+(an?\s+)?(assessment|test|assessments|tests)\s*$",
    r"\bwhich\s+(assessment|test|assessments|tests)\s*$",
    r"\bhelp\s+me\s+(choose|find)\s+(an?\s+)?(assessment|test)\s*$",
]

COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bdifference between\b",
    r"\bdifferences between\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bwhich is better\b",
]

REFINEMENT_PATTERNS = [
    r"^\s*(also|add|include|exclude|remove|only|instead|prefer|focus on|refine|narrow|update|change)\b",
    r"\bwith(out)?\s+(personality|cognitive|technical|coding|simulation|behavioral|behavioural)\b",
    r"\bmore\s+(technical|personality|cognitive|behavioral|behavioural|senior|junior)\b",
]


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConversationState:
    """Immutable snapshot of the classified conversation turn."""
    latest_user_message: str
    user_context: str
    intent: str
    requested_type_codes: List[str]
    is_refinement: bool = False
    is_comparison: bool = False
    refusal_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_latest_user_message(messages: List[Dict[str, str]]) -> str:
    """Extract the most recent user message from the conversation history."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return clean_text(message.get("content", ""))
    return ""


def get_user_context(messages: List[Dict[str, str]]) -> str:
    """Concatenate all user messages into a single context string for retrieval."""
    user_messages = [clean_text(message.get("content", "")) for message in messages if message.get("role") == "user"]
    return clean_text(" ".join(user_messages))


def is_prompt_injection(text: str) -> bool:
    """Return True if the text matches known prompt-injection attack patterns."""
    normalized = normalize_for_match(text)
    return any(re.search(pattern, normalized) for pattern in PROMPT_INJECTION_PATTERNS)


def is_comparison_request(text: str) -> bool:
    """Return True if the user is asking to compare two or more assessments."""
    normalized = normalize_for_match(text)
    return any(re.search(pattern, normalized) for pattern in COMPARISON_PATTERNS)


def is_refinement_request(text: str) -> bool:
    """Return True if the user is refining a previous recommendation set."""
    normalized = normalize_for_match(text)
    return any(re.search(pattern, normalized) for pattern in REFINEMENT_PATTERNS)


def is_off_topic(text: str) -> bool:
    """Return True if the message falls outside the hiring-assessment domain."""
    normalized = normalize_for_match(text)
    if not normalized:
        return False

    # Hard refusal for explicitly non-hiring advisory topics.
    if any(term in normalized for term in STRICT_REFUSAL_TERMS):
        return True

    # Soft check: off-topic words without any hiring-domain anchor.
    if any(term in normalized for term in OFF_TOPIC_TERMS):
        domain_hit = any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in HIRING_DOMAIN_TERMS)
        if not domain_hit:
            return True

    # Messages longer than 3 words with zero domain hits are likely off-topic.
    domain_hits = sum(1 for term in HIRING_DOMAIN_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized))
    if domain_hits == 0 and len(normalized.split()) > 3:
        return True
    return False


def is_vague_request(text: str, full_context: str) -> bool:
    """Return True if the request lacks enough specificity to retrieve assessments."""
    normalized = normalize_for_match(text)
    if not normalized:
        return True
    if any(re.search(pattern, normalized) for pattern in VAGUE_PATTERNS):
        return True

    context = normalize_for_match(full_context)
    role_terms = [
        "developer", "engineer", "manager", "sales", "support", "service",
        "analyst", "graduate", "intern", "leader", "java", "python", "sql",
        "javascript", "frontend", "backend", "fullstack", "finance",
        "accounting", "operations", "administrator", "customer",
    ]
    assessment_terms = [
        "technical", "coding", "personality", "cognitive", "behavioral",
        "behavioural", "ability", "aptitude", "reasoning", "simulation",
        "skills", "knowledge",
    ]
    has_role = any(re.search(rf"\b{re.escape(term)}\b", context) for term in role_terms)
    has_assessment_preference = any(re.search(rf"\b{re.escape(term)}\b", context) for term in assessment_terms)

    if "assessment" in normalized or "test" in normalized or "recommend" in normalized:
        return not (has_role or has_assessment_preference)

    return False


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_conversation(messages: List[Dict[str, str]]) -> ConversationState:
    """Classify the conversation into an intent and return a frozen state snapshot.

    Intent priority: refuse > compare > clarify > recommend.
    """
    latest = get_latest_user_message(messages)
    context = get_user_context(messages)
    requested_codes = infer_requested_type_codes(context)

    if not latest:
        return ConversationState(latest, context, "clarify", requested_codes)
    if is_prompt_injection(latest):
        return ConversationState(latest, context, "refuse", requested_codes, refusal_reason="prompt_injection")
    if is_off_topic(latest):
        return ConversationState(latest, context, "refuse", requested_codes, refusal_reason="off_topic")

    comparison = is_comparison_request(latest)
    refinement = is_refinement_request(latest)
    if comparison:
        return ConversationState(latest, context, "compare", requested_codes, is_refinement=refinement, is_comparison=True)
    if is_vague_request(latest, context):
        return ConversationState(latest, context, "clarify", requested_codes, is_refinement=refinement)
    return ConversationState(latest, context, "recommend", requested_codes, is_refinement=refinement)
