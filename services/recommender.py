"""Orchestrator that wires conversation classification, retrieval, and LLM reply generation.

The SHLRecommender is the single entry-point called by the /chat endpoint.
It delegates to:
  - conversation.classify_conversation  → intent detection + safety
  - RetrievalEngine.search              → catalog-backed retrieval
  - GeminiService.*_reply               → natural-language response (with fallback)
"""

from typing import Any, Dict, List

from services.conversation import classify_conversation
from services.llm_service import GeminiService
from services.retrieval import RetrievalEngine
from utils.helpers import clean_text


class SHLRecommender:
    """Stateless recommender that turns a message list into a safe JSON response."""

    def __init__(self, retrieval_engine: RetrievalEngine, llm_service: GeminiService) -> None:
        self.retrieval = retrieval_engine
        self.llm = llm_service

    def respond(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Classify the conversation and produce a schema-compliant response dict."""
        state = classify_conversation(messages)

        if state.intent == "refuse":
            # Deterministic refusal — no retrieval, no LLM, no recommendations.
            return {
                "reply": self.llm.refusal_reply(state.refusal_reason or "off_topic"),
                "recommendations": [],
                "end_of_conversation": True,
            }

        if state.intent == "clarify":
            # Vague requests clarify first; recommendations stay empty until
            # the user provides enough hiring context to query the catalog.
            return {
                "reply": self.llm.clarification_reply(state.user_context),
                "recommendations": [],
                "end_of_conversation": False,
            }

        if state.intent == "compare":
            return self._compare(state.latest_user_message)

        return self._recommend(state.user_context, state.requested_type_codes)

    def _recommend(self, context: str, requested_type_codes: List[str]) -> Dict[str, Any]:
        """Retrieve catalog items and generate a recommendation reply."""
        results = self.retrieval.search(context, top_k=10, requested_codes=requested_type_codes)
        if not results:
            return {
                "reply": "I could not find a strong SHL catalog match. Please share the role, seniority, and whether you need technical, cognitive, personality, or behavioral testing.",
                "recommendations": [],
                "end_of_conversation": False,
            }
        response_items = self.retrieval.to_response_items(results)
        reply = self.llm.recommendation_reply(context, results)
        return {
            "reply": clean_text(reply),
            "recommendations": response_items[:10],
            "end_of_conversation": True,
        }

    def _compare(self, user_text: str) -> Dict[str, Any]:
        """Extract named assessments from the query and generate a comparison reply."""
        matched, missing = self.retrieval.extract_comparison_items(user_text)
        if len(matched) < 2:
            return {
                "reply": self.llm.comparison_reply(user_text, matched, missing),
                "recommendations": [],
                "end_of_conversation": False,
            }
        reply = self.llm.comparison_reply(user_text, matched, missing)
        return {
            "reply": clean_text(reply),
            "recommendations": self.retrieval.to_response_items(matched[:10]),
            "end_of_conversation": True,
        }
