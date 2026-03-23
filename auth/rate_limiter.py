# auth/rate_limiter.py
# ------------------------------------------------------------
# Per-user rate limiting for LLM and Tavily API calls.
#
# Tracks usage in Supabase rate_limits table.
# Checked before every LLM and Tavily call.
# Resets daily at midnight UTC automatically.
#
# Default limits per user per day:
#   - 10 LLM calls
#   - 10000 tokens
#   - 10 Tavily searches
#
# Admin users get 10x limits automatically.
#
# In production: limits are configurable per user in the
# profiles table. Enterprise users get higher limits.
# Rate limit checks use Redis for speed — Postgres as source
# of truth. Here we use Postgres directly for simplicity.
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Optional
from memory.supabase_client import get_admin_client


# ------------------------------------------------------------
# Limit constants
# Adjust these to control costs
# ------------------------------------------------------------

DEFAULT_LIMITS = {
    "max_llm_calls": 10,
    "max_tokens": 10000,
    "max_tavily_calls": 10
}

ADMIN_LIMITS = {
    "max_llm_calls": 500,
    "max_tokens": 1_000_000,
    "max_tavily_calls": 200
}


class RateLimitExceeded(Exception):
    """
    Raised when a user exceeds their daily rate limit.
    Caught by agents and UI to show user-friendly message.
    """
    def __init__(self, limit_type: str, current: int, max_val: int):
        self.limit_type = limit_type
        self.current = current
        self.max_val = max_val
        super().__init__(
            f"Rate limit exceeded: {limit_type} "
            f"({current}/{max_val} today)"
        )


# ------------------------------------------------------------
# Core rate limit functions
# ------------------------------------------------------------

def check_and_increment(
    user_id: str,
    llm_calls: int = 0,
    tokens: int = 0,
    tavily_calls: int = 0,
    is_admin: bool = False
) -> dict:
    """
    Checks rate limits and increments counters atomically.

    Call this BEFORE every LLM or Tavily API call.
    Raises RateLimitExceeded if any limit is exceeded.
    Returns updated usage dict on success.

    In production: this check happens in middleware before
    the request reaches the agent — not inside the agent.
    Uses Redis for sub-millisecond check latency.
    """
    client = get_admin_client()
    limits = ADMIN_LIMITS if is_admin else DEFAULT_LIMITS

    try:
        # Get or create today's rate limit row
        response = client.rpc(
            "get_or_create_rate_limit",
            {"p_user_id": user_id}
        ).execute()

        if not response.data:
            raise Exception("Could not get rate limit row")

        current = response.data

        # Check limits before incrementing
        if llm_calls > 0:
            new_total = current["llm_calls"] + llm_calls
            if new_total > limits["max_llm_calls"]:
                raise RateLimitExceeded(
                    "LLM calls",
                    current["llm_calls"],
                    limits["max_llm_calls"]
                )

        if tokens > 0:
            new_total = current["tokens_used"] + tokens
            if new_total > limits["max_tokens"]:
                raise RateLimitExceeded(
                    "tokens",
                    current["tokens_used"],
                    limits["max_tokens"]
                )

        if tavily_calls > 0:
            new_total = current["tavily_calls"] + tavily_calls
            if new_total > limits["max_tavily_calls"]:
                raise RateLimitExceeded(
                    "Tavily searches",
                    current["tavily_calls"],
                    limits["max_tavily_calls"]
                )

        # All checks passed — increment atomically
        updated = client.rpc(
            "increment_usage",
            {
                "p_user_id": user_id,
                "p_llm_calls": llm_calls,
                "p_tokens": tokens,
                "p_tavily_calls": tavily_calls
            }
        ).execute()

        return updated.data or current

    except RateLimitExceeded:
        raise
    except Exception as e:
        print(f"[rate_limiter] Check failed for {user_id}: {e}")
        # Fail open — don't block users if rate limiter breaks
        # Log the failure for investigation
        return {}


def get_usage_today(user_id: str) -> dict:
    """
    Returns today's usage stats for a user.
    Called by: UI sidebar to show remaining quota.
    """
    client = get_admin_client()

    try:
        response = client.rpc(
            "get_or_create_rate_limit",
            {"p_user_id": user_id}
        ).execute()

        data = response.data or {}

        return {
            "llm_calls": data.get("llm_calls", 0),
            "max_llm_calls": data.get("max_llm_calls", DEFAULT_LIMITS["max_llm_calls"]),
            "tokens_used": data.get("tokens_used", 0),
            "max_tokens": data.get("max_tokens", DEFAULT_LIMITS["max_tokens"]),
            "tavily_calls": data.get("tavily_calls", 0),
            "max_tavily_calls": data.get("max_tavily_calls", DEFAULT_LIMITS["max_tavily_calls"]),
            "date": data.get("date", datetime.now().date().isoformat())
        }

    except Exception as e:
        print(f"[rate_limiter] get_usage_today failed: {e}")
        return {}


def get_remaining(user_id: str, is_admin: bool = False) -> dict:
    """
    Returns remaining quota for today.
    Called by: UI to show users how much they have left.
    """
    usage = get_usage_today(user_id)
    limits = ADMIN_LIMITS if is_admin else DEFAULT_LIMITS

    return {
        "llm_calls_remaining": max(
            0, limits["max_llm_calls"] - usage.get("llm_calls", 0)
        ),
        "tokens_remaining": max(
            0, limits["max_tokens"] - usage.get("tokens_used", 0)
        ),
        "tavily_remaining": max(
            0, limits["max_tavily_calls"] - usage.get("tavily_calls", 0)
        )
    }