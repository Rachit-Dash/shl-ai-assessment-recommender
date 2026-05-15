"""Shared text-processing utilities for catalog matching and response formatting."""

import re
from typing import Any, Dict, List, Sequence


# SHL catalog test-type codes and their human-readable labels.
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


# Maps natural-language assessment terms to their catalog type codes.
FILTER_TO_CODES = {
    "ability": {"A"},
    "aptitude": {"A"},
    "cognitive": {"A"},
    "reasoning": {"A"},
    "numerical": {"A"},
    "verbal": {"A"},
    "deductive": {"A"},
    "inductive": {"A"},
    "skills": {"K", "S"},
    "skill": {"K", "S"},
    "technical": {"K", "S"},
    "coding": {"S", "K"},
    "programming": {"S", "K"},
    "knowledge": {"K"},
    "simulation": {"S"},
    "simulations": {"S"},
    "personality": {"P"},
    "behavior": {"P", "B", "C"},
    "behaviour": {"P", "B", "C"},
    "behavioral": {"P", "B", "C"},
    "behavioural": {"P", "B", "C"},
    "competency": {"C"},
    "competencies": {"C"},
    "judgement": {"B"},
    "judgment": {"B"},
    "situational": {"B"},
}


# Common English words and hiring-domain filler removed during tokenization
# to improve retrieval signal-to-noise ratio.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "i",
    "in", "is", "it", "of", "on", "or", "our", "the", "to", "we", "with",
    "need", "needs", "want", "wants", "looking", "hire", "hiring",
    "assessment", "assessments", "test", "tests", "candidate", "candidates",
    "role", "roles", "job", "jobs", "recommend", "recommendation",
}


def clean_text(value: Any) -> str:
    """Normalize whitespace while preserving ordinary punctuation."""
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_match(value: Any) -> str:
    """Lowercase, strip symbols, and normalize for fuzzy catalog matching."""
    text = clean_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> List[str]:
    """Split normalized text into meaningful tokens, removing stopwords."""
    normalized = normalize_for_match(value)
    if not normalized:
        return []
    return [token for token in normalized.split() if token not in STOPWORDS and len(token) > 1]


def parse_test_type_codes(test_type: Any) -> List[str]:
    """Extract single-letter SHL type codes (A, B, C, K, P, S …) from a test_type string."""
    text = clean_text(test_type).upper()
    codes = []
    for part in re.findall(r"\b[A-Z]\b", text):
        if part in TEST_TYPE_LABELS and part not in codes:
            codes.append(part)
    if not codes:
        for code, label in TEST_TYPE_LABELS.items():
            if label.lower() in text.lower():
                codes.append(code)
    return codes


def describe_test_type(test_type: Any) -> str:
    """Return a human-readable label for a test_type value, e.g. 'K - Knowledge & Skills'."""
    codes = parse_test_type_codes(test_type)
    if not codes:
        return clean_text(test_type) or "Unknown"
    labels = [f"{code} - {TEST_TYPE_LABELS[code]}" for code in codes]
    return ", ".join(labels)


def infer_requested_type_codes(text: str) -> List[str]:
    """Detect assessment-type intent words in user text and return matching codes."""
    normalized = normalize_for_match(text)
    requested = set()
    for word, codes in FILTER_TO_CODES.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            requested.update(codes)
    return sorted(requested)


def catalog_item_text(item: Dict[str, Any]) -> str:
    """Concatenate all catalog fields into a single searchable string."""
    fields = [
        item.get("name", ""),
        item.get("description", ""),
        item.get("category", ""),
        item.get("duration", ""),
        item.get("test_type", ""),
        item.get("remote_testing_support", ""),
        item.get("adaptive_support", ""),
    ]
    return clean_text(" ".join(clean_text(field) for field in fields))


def format_recommendation(item: Dict[str, Any]) -> Dict[str, str]:
    """Shape a catalog item into the strict API response recommendation schema."""
    return {
        "name": clean_text(item.get("name")),
        "url": clean_text(item.get("url")),
        "test_type": describe_test_type(item.get("test_type")),
    }


def item_has_any_code(item: Dict[str, Any], codes: Sequence[str]) -> bool:
    """Return True if the item's test_type contains at least one of the requested codes."""
    if not codes:
        return True
    item_codes = set(parse_test_type_codes(item.get("test_type")))
    return bool(item_codes.intersection(set(codes)))
