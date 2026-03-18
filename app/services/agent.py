"""
AstroAI Agent — LangGraph-based agentic workflow.

Graph flow:
  1. classify_intent   →  Should we retrieve from RAG?
  2. retrieve          →  (conditional) Fetch from ChromaDB
  3. generate          →  Build final LLM response

The RAG tool is bound to the LLM so it can decide when to invoke it.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any, Sequence, TypedDict, Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.services.rag import retrieve_context
import logging

logger = logging.getLogger("astroai")
logging.basicConfig(level=logging.INFO)


# ─────────────────────── Zodiac Utility ────────────────────────

ZODIAC_DATES = [
    ("Capricorn", (1, 1), (1, 19)),
    ("Aquarius", (1, 20), (2, 18)),
    ("Pisces", (2, 19), (3, 20)),
    ("Aries", (3, 21), (4, 19)),
    ("Taurus", (4, 20), (5, 20)),
    ("Gemini", (5, 21), (6, 20)),
    ("Cancer", (6, 21), (7, 22)),
    ("Leo", (7, 23), (8, 22)),
    ("Virgo", (8, 23), (9, 22)),
    ("Libra", (9, 23), (10, 22)),
    ("Scorpio", (10, 23), (11, 21)),
    ("Sagittarius", (11, 22), (12, 21)),
    ("Capricorn", (12, 22), (12, 31)),
]


def get_zodiac_sign(birth_date_str: str) -> str:
    """Derive zodiac sign from a YYYY-MM-DD date string."""
    try:
        dt = datetime.strptime(birth_date_str, "%Y-%m-%d")
    except ValueError:
        return "Unknown"
    month, day = dt.month, dt.day
    for sign, (sm, sd), (em, ed) in ZODIAC_DATES:
        if (month == sm and day >= sd) or (month == em and day <= ed):
            return sign
    return "Unknown"


# ─────────────────────── RAG Tool (bound to LLM) ──────────────


@tool
def astro_knowledge_search(query: str) -> str:
    """Search the Vedic astrology knowledge base for factual information about
    ZODIAC traits, PLANETARY impacts, CAREER guidance, LOVE guidance,
    SPIRITUAL guidance, or NAKSHATRA mappings.
    Use this tool ONLY when the user asks a factual astrology question
    that requires specific knowledge. Do NOT use it for greetings,
    follow-ups, summaries, or questions about previous conversation."""
    results = retrieve_context(query, top_k=6)
    if not results:
        logger.info("No relevant astrological knowledge found.")
        return "No relevant astrological knowledge found."
    return "\n---\n".join(results)


# ─────────────────────── Agent State ───────────────────────────


def _merge_messages(
    left: list[BaseMessage], right: list[BaseMessage]
) -> list[BaseMessage]:
    return left + right


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], _merge_messages]
    user_profile: dict
    zodiac: str
    retrieval_used: bool
    context_used: list[str]


# ─────────────────────── LLM Setup ─────────────────────────────


def _build_llm():
    """Instantiate the ChatOpenAI model with the RAG tool bound."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    llm_with_tools = llm.bind_tools([astro_knowledge_search])
    return llm_with_tools


# ─────────────────────── Graph Nodes ───────────────────────────


def build_system_prompt(state: AgentState) -> str:
    """Construct the system prompt with user profile context."""
    profile = state["user_profile"]
    zodiac = state["zodiac"]
    lang = profile.get("preferred_language", "en")

    lang_instruction = ""
    if lang == "hi":
        lang_instruction = (
            "IMPORTANT: You MUST respond entirely in Hindi (Devanagari script). "
            "Do not use English at all in your response."
        )

    return f"""You are AstroAI, an expert Vedic astrology assistant.

User Profile:
- Name: {profile.get('name', 'User')}
- Date of Birth: {profile.get('birth_date', 'unknown')}
- Time of Birth: {profile.get('birth_time', 'unknown')}
- Place of Birth: {profile.get('birth_place', 'unknown')}
- Zodiac Sign (Sun): {zodiac}

{lang_instruction}

Guidelines:
1. Provide personalised astrological insights based on the user's zodiac sign and birth details.
2. You have access to an astro_knowledge_search tool. Use it whenever:
   - The user asks a factual question about planets, houses, signs, or transits.
   - The user asks about CAREER, LOVE, or HEALTH from an astrological perspective.
   - The user asks about Career guidance or predictions in career.
3. Do NOT use the tool for:
   - Greetings, pleasantries, or general conversation.
   - Summarising previous responses.
   - Questions like "why are you saying this" that refer to conversation context.
4. Be warm, insightful, and supportive.
5. Reference specific astrological concepts when relevant.
"""


def agent_node(state: AgentState) -> dict:
    """The main agent node — calls the LLM (which may invoke tools)."""
    llm = _build_llm()
    system = SystemMessage(content=build_system_prompt(state))
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    logger.info(f"Agent Node Used")
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute tool calls made by the LLM."""
    tool_executor = ToolNode([astro_knowledge_search])
    result = tool_executor.invoke(state)
    logger.info(f"Tool Node Used")

    # Mark retrieval as used and track contexts
    context_used = state.get("context_used", [])
    if result.get("messages"):
        for msg in result["messages"]:
            content = msg.content if hasattr(msg, "content") else str(msg)
            if content and content != "No relevant astrological knowledge found.":
                # Extract short descriptor from content
                snippet = content.strip()
                # snippet = content[:80].replace("\n", " ").strip()
                context_used.append(snippet)

    return {
        "messages": result.get("messages", []),
        "retrieval_used": True,
        "context_used": context_used,
    }


def should_use_tools(state: AgentState) -> str:
    """Router: check if the last AI message has tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ─────────────────────── Build Graph ───────────────────────────


def build_graph():
    """Construct and compile the LangGraph agent workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Conditional edge: agent → tools or end
    workflow.add_conditional_edges(
        "agent",
        should_use_tools,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tool execution, go back to agent for final response
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ─────────────────────── Run Agent ─────────────────────────────


def run_agent(
    user_message: str,
    history: list[dict],
    user_profile: dict,
) -> dict:
    """
    Execute the AstroAI agent for a single turn.

    Args:
        user_message: The current user message.
        history: Previous conversation history (list of {role, content}).
        user_profile: User profile dictionary.

    Returns:
        Dict with keys: response, zodiac, context_used, retrieval_used
    """
    # Derive zodiac
    zodiac = get_zodiac_sign(user_profile.get("birth_date", ""))

    # Build message list from history
    messages: list[BaseMessage] = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Add current message
    messages.append(HumanMessage(content=user_message))

    # Initial state
    initial_state: AgentState = {
        "messages": messages,
        "user_profile": user_profile,
        "zodiac": zodiac,
        "retrieval_used": False,
        "context_used": [],
    }

    # Run graph
    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # Extract final AI response
    ai_messages = [
        m for m in final_state["messages"] if isinstance(m, AIMessage) and m.content
    ]
    response_text = (
        ai_messages[-1].content if ai_messages else "I'm unable to respond right now."
    )

    return {
        "response": response_text,
        "zodiac": zodiac,
        "context_used": final_state.get("context_used", []),
        "retrieval_used": final_state.get("retrieval_used", False),
    }
