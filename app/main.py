"""
AstroAI — FastAPI application entry point.

Exposes POST /chat endpoint for the Astro Conversational Insight Agent.
"""

import logging
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import ChatRequest, ChatResponse
from app.services.agent import run_agent
from app.services.memory import (
    append_message,
    get_history,
    get_user_profile,
    save_user_profile,
)

# Load environment variables from .env file (e.g. OPENAI_API_KEY)
load_dotenv()

logger = logging.getLogger("astroai")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AstroAI starting up …")
    yield
    logger.info("AstroAI shutting down …")


app = FastAPI(
    title="AstroAI — Conversational Insight Agent",
    description="Multi-turn Vedic astrology assistant with RAG and conversation ownership",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────── Health Check ────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ───────────────────── Chat Endpoint ───────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Accepts a user message with session ID and profile,
    runs the LangGraph agent, and returns the response.
    """
    try:
        session_id = request.session_id
        user_message = request.message.strip()
        profile = request.user_profile.model_dump()

        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Persist / update user profile for the session
        save_user_profile(session_id, profile)

        # Retrieve conversation history (windowed)
        history = get_history(session_id)

        # Save the user message BEFORE running the agent
        append_message(session_id, "user", user_message)

        # Run the agent
        logger.info(f"[{session_id}] User: {user_message[:80]}…")
        result = run_agent(
            user_message=user_message,
            history=history,
            user_profile=profile,
        )

        # Save the assistant response
        append_message(session_id, "assistant", result["response"])

        logger.info(
            f"[{session_id}] Agent replied (retrieval={result['retrieval_used']})"
        )

        return ChatResponse(
            response=result["response"],
            zodiac=result["zodiac"],
            context_used=result["context_used"],
            retrieval_used=result["retrieval_used"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        # Safe fallback
        return ChatResponse(
            response="I'm sorry, I encountered an issue processing your request. Please try again.",
            zodiac="Unknown",
            context_used=[],
            retrieval_used=False,
        )
