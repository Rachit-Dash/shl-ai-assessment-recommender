"""FastAPI application — SHL AI Assessment Recommender.

Exposes GET /health and POST /chat endpoints with strict JSON schema
validation. All recommendations are catalog-backed with safe fallbacks.
"""

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field, HttpUrl

from services.llm_service import GeminiService
from services.recommender import SHLRecommender
from services.retrieval import RetrievalEngine
from utils.helpers import clean_text


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class Recommendation(BaseModel):
    name: str
    url: HttpUrl
    test_type: str

    class Config:
        extra = "forbid"


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

    class Config:
        extra = "forbid"


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL AI Assessment Recommender",
    version="1.0.0",
    description="Catalog-grounded SHL assessment recommendation API with semantic + keyword retrieval.",
)

retrieval_engine = RetrievalEngine()
llm_service = GeminiService()
recommender = SHLRecommender(retrieval_engine, llm_service)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Lightweight liveness probe."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Process a conversation and return catalog-backed assessment recommendations."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        response = recommender.respond(messages)
    except Exception:
        # Deterministic safe fallback — never surface internal errors to the caller.
        response = {
            "reply": "I could not process that request safely. Please share the hiring role and the assessment focus.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    safe_response = {
        "reply": clean_text(response.get("reply", "")) or "Please share the hiring role and assessment focus.",
        "recommendations": response.get("recommendations", []) or [],
        "end_of_conversation": bool(response.get("end_of_conversation", False)),
    }

    # Enforce the max-10 recommendation cap required by the assignment spec.
    if safe_response["recommendations"]:
        safe_response["recommendations"] = safe_response["recommendations"][:10]

    return ChatResponse(**safe_response)
