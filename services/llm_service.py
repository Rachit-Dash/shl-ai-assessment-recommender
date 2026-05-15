"""Gemini LLM integration with deterministic fallback replies.

When a valid GOOGLE_API_KEY is available, uses Gemini for natural-language
reply generation. When the key is missing or the API call fails, returns
safe deterministic fallback text so the evaluator can still exercise the
full recommendation pipeline without external dependencies.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from utils.helpers import clean_text, describe_test_type


def load_local_env() -> None:
    """Read key=value pairs from the project-root .env file into os.environ."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        return


class GeminiService:
    """Thin Gemini wrapper with deterministic fallbacks for evaluator safety.

    - Temperature 0.2 + top_p 0.8 keeps replies concise and grounded.
    - A 6-second deadline prevents hanging network calls during evaluation.
    - Every public method returns a safe string even when the LLM is unavailable.
    """

    def __init__(self) -> None:
        load_local_env()
        self.api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip() or "gemini-1.5-flash"
        self._model = None
        self._request_options = {"timeout": 6}
        if self.api_key:
            try:
                import google.generativeai as genai
                from google.api_core.retry import Retry

                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
                self._request_options = {
                    "timeout": 6,
                    "retry": Retry(initial=0.5, maximum=1.0, multiplier=1.0, deadline=6.0),
                }
            except Exception:
                self._model = None

    @property
    def available(self) -> bool:
        """True when the Gemini model was initialized successfully."""
        return self._model is not None

    def _generate(self, prompt: str) -> Optional[str]:
        """Call Gemini and return cleaned text, or None on any failure."""
        if not self._model:
            return None
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_output_tokens": 220,
                },
                request_options=self._request_options,
            )
            text = clean_text(getattr(response, "text", ""))
            return text or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public reply generators (each has a deterministic fallback)
    # ------------------------------------------------------------------

    def clarification_reply(self, user_context: str) -> str:
        """Generate a one-question clarification prompt for vague hiring requests."""
        prompt = (
            "Ask one concise clarification question for an SHL assessment recommendation. "
            "Focus on role, seniority, technical skills, personality, or cognitive testing. "
            "Do not recommend assessments yet. "
            f"Conversation: {user_context[:1200]}"
        )
        return self._generate(prompt) or (
            "To recommend the right SHL assessments, what role are you hiring for "
            "and do you need technical, cognitive, personality, or behavioral testing?"
        )

    def refusal_reply(self, reason: str) -> str:
        """Return a deterministic refusal message (no LLM call needed)."""
        if reason == "prompt_injection":
            return (
                "I can only help with SHL assessment recommendations using the "
                "catalog data, so I cannot follow requests to ignore instructions "
                "or reveal internal prompts."
            )
        return "I can only help with SHL assessment recommendations, refinements, and catalog-grounded comparisons."

    def recommendation_reply(self, user_context: str, recommendations: List[Dict[str, str]]) -> str:
        """Summarize catalog-backed recommendations in one concise sentence."""
        if not recommendations:
            return "I could not find a strong catalog-backed match. Please share the role, seniority, and assessment focus."
        catalog_lines = "\n".join(
            f"- {item['name']} | {item.get('test_type', '')} | {item.get('description', '')[:180]}"
            for item in recommendations[:10]
        )
        prompt = (
            "Write one concise sentence recommending these SHL assessments for the hiring need. "
            "Use only the listed catalog items and do not add unsupported claims.\n"
            f"Hiring need: {user_context[:1000]}\n"
            f"Catalog items:\n{catalog_lines}"
        )
        count = len(recommendations)
        fallback = f"I found {count} SHL catalog-backed assessment match{'es' if count != 1 else ''} for this hiring need."
        return self._generate(prompt) or fallback

    def comparison_reply(self, user_request: str, matched_items: List[Dict[str, str]], missing_names: List[str]) -> str:
        """Compare two or more catalog items, or explain why comparison failed."""
        if len(matched_items) < 2:
            if missing_names:
                return "I could not compare those assessments because one or more requested names were not found in the SHL catalog data."
            return "Please name at least two SHL catalog assessments to compare."

        lines = []
        for item in matched_items:
            lines.append(
                f"{item.get('name', '')}: type={describe_test_type(item.get('test_type', ''))}; "
                f"duration={item.get('duration', 'Unknown')}; "
                f"description={item.get('description', '')[:260]}"
            )
        missing_note = f" Missing from catalog: {', '.join(missing_names)}." if missing_names else ""
        prompt = (
            "Compare the SHL assessments in 2-4 concise sentences using only the supplied catalog data. "
            "Do not infer facts that are not present.\n"
            f"User request: {user_request[:700]}\n"
            f"Assessments:\n" + "\n".join(lines) + missing_note
        )
        # Deterministic fallback used when the LLM is unavailable.
        fallback_parts = []
        for item in matched_items:
            fallback_parts.append(
                f"{item.get('name')} is listed as {describe_test_type(item.get('test_type'))}"
                f" with duration {item.get('duration') or 'Unknown'}."
            )
        if missing_names:
            fallback_parts.append(f"Not found in the catalog data: {', '.join(missing_names)}.")
        return self._generate(prompt) or clean_text(" ".join(fallback_parts))
