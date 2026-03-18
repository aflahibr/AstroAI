"""
Pydantic schemas for the AstroAI API contract.
"""

from typing import Optional
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """User birth details for astrological analysis."""
    name: str = Field(..., description="User's name")
    birth_date: str = Field(..., description="Date of birth (YYYY-MM-DD)")
    birth_time: Optional[str] = Field(None, description="Time of birth (HH:MM)")
    birth_place: Optional[str] = Field(None, description="Place of birth")
    preferred_language: str = Field("en", description="Preferred language: 'en' or 'hi'")


class ChatRequest(BaseModel):
    """Input contract for the /chat endpoint."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User's message")
    user_profile: UserProfile = Field(..., description="User's birth details")


class ChatResponse(BaseModel):
    """Output contract for the /chat endpoint."""
    response: str = Field(..., description="Agent response text")
    zodiac: str = Field(..., description="User's zodiac sign")
    context_used: list[str] = Field(default_factory=list, description="Knowledge contexts used")
    retrieval_used: bool = Field(False, description="Whether RAG retrieval was used")
