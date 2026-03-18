"""
Redis-based conversation memory with windowing and decay.
Manages session history for multi-turn conversations.
"""

import json
import os
import time
import redis

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MAX_HISTORY_TURNS = 20        # Maximum number of turns retained
WINDOW_SIZE = 10              # Number of recent turns sent to LLM
SESSION_TTL_SECONDS = 86400   # 24-hour session expiry (decay)


def get_redis_client() -> redis.Redis:
    """Create a Redis client connection."""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _session_key(session_id: str) -> str:
    return f"astro:session:{session_id}"


def _profile_key(session_id: str) -> str:
    return f"astro:profile:{session_id}"


def save_user_profile(session_id: str, profile: dict) -> None:
    """Store user profile in Redis for the session."""
    r = get_redis_client()
    key = _profile_key(session_id)
    r.set(key, json.dumps(profile))
    r.expire(key, SESSION_TTL_SECONDS)


def get_user_profile(session_id: str) -> dict | None:
    """Retrieve user profile from Redis."""
    r = get_redis_client()
    data = r.get(_profile_key(session_id))
    if data:
        return json.loads(data)
    return None


def append_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session history list in Redis."""
    r = get_redis_client()
    key = _session_key(session_id)
    message = json.dumps({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })
    r.rpush(key, message)
    r.expire(key, SESSION_TTL_SECONDS)

    # Trim to MAX_HISTORY_TURNS (each turn = 1 entry; user + assistant = 2 entries per turn)
    max_entries = MAX_HISTORY_TURNS * 2
    current_len = r.llen(key)
    if current_len > max_entries:
        r.ltrim(key, current_len - max_entries, -1)


def get_history(session_id: str, window: int | None = None) -> list[dict]:
    """
    Retrieve session conversation history.
    
    Args:
        session_id: The session identifier.
        window: Number of recent message pairs to return. Defaults to WINDOW_SIZE.
    
    Returns:
        List of message dicts with 'role' and 'content'.
    """
    r = get_redis_client()
    key = _session_key(session_id)
    
    if window is None:
        window = WINDOW_SIZE
    
    # Get last N*2 entries (user + assistant pairs)
    max_entries = window * 2
    all_messages = r.lrange(key, -max_entries, -1)
    
    history = []
    for msg_str in all_messages:
        msg = json.loads(msg_str)
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return history


def clear_session(session_id: str) -> None:
    """Clear a session's data from Redis."""
    r = get_redis_client()
    r.delete(_session_key(session_id))
    r.delete(_profile_key(session_id))


def get_session_summary(session_id: str) -> str:
    """Get a brief summary of conversation length for diagnostics."""
    r = get_redis_client()
    key = _session_key(session_id)
    length = r.llen(key)
    return f"Session {session_id}: {length} messages stored"
