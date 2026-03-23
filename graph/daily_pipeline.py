# graph/daily_pipeline.py
# ------------------------------------------------------------
# The daily learning pipeline — LangGraph DAG.
# Updated to accept and pass user_id through all nodes.
#
# Orchestrates:
#   memory_loader → researcher → curator →
#   briefing → notifier → memory_writer
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END, START

from agents.researcher import researcher_agent, ResearcherOutput
from agents.curator import curator_agent, ChosenTopic
from agents.briefing import briefing_agent, DailyBrief
from auth.rate_limiter import check_and_increment, RateLimitExceeded

from memory.knowledge_store import (
    load_store,
    increment_sessions,
    get_known_topics,
    add_topic
)
from memory.supabase_client import get_admin_client
from notifications.gmail import send_daily_brief
from dotenv import load_dotenv
from errors.handler import handle_pipeline_error

load_dotenv()


# ------------------------------------------------------------
# State definition
# user_id and is_admin added — flow through every node
# ------------------------------------------------------------

class DailyPipelineState(TypedDict):
    # Auth context — set at pipeline start, never changes
    user_id: str
    is_admin: bool

    # Loaded at start — full knowledge store snapshot
    knowledge_snapshot: dict

    # Written by: researcher
    candidate_topics: Optional[ResearcherOutput]

    # Written by: curator
    chosen_topic: Optional[ChosenTopic]

    # Written by: briefing
    daily_brief: Optional[DailyBrief]

    # Written by: notifier
    email_sent: bool


# ------------------------------------------------------------
# Bookend nodes
# ------------------------------------------------------------
@handle_pipeline_error(fallback={})
def memory_loader(state: DailyPipelineState) -> dict:
    """
    First node — loads knowledge store for this user.
    """
    print("\n[Pipeline] ── Starting daily run ──")
    print(f"[memory_loader] User: {state.get('user_id', 'unknown')}")

    user_id = state.get("user_id")
    store = load_store(user_id=user_id)
    known = get_known_topics(store)

    print(f"[memory_loader] {len(known)} topics in knowledge store")

    return {
        "knowledge_snapshot": store,
        "candidate_topics": None,
        "chosen_topic": None,
        "daily_brief": None,
        "email_sent": False
    }

@handle_pipeline_error(fallback={"email_sent": False})
def notifier_node(state: DailyPipelineState) -> dict:
    print("\n[notifier] Sending daily brief...")

    brief = state.get("daily_brief")
    user_id = state.get("user_id")

    if not brief:
        print("[notifier] No brief in state — skipping")
        return {"email_sent": False}

    # Get user's actual email from Supabase
    recipient_email = None
    if user_id:
        try:
            client = get_admin_client()
            profile = client.table("profiles") \
                .select("email") \
                .eq("id", user_id) \
                .single() \
                .execute()

            if profile.data:
                recipient_email = profile.data["email"]
                print(f"[notifier] Recipient from Supabase: {recipient_email}")
            else:
                print(f"[notifier] No profile for user_id: {user_id}")

        except Exception as e:
            print(f"[notifier] Profile fetch failed: {e}")

    # Fallback to env var if Supabase fetch failed
    if not recipient_email:
        print("[notifier] Falling back to GMAIL_RECIPIENT env var")
        recipient_email = os.getenv("GMAIL_RECIPIENT")

    if not recipient_email:
        print("[notifier] No recipient — cannot send email")
        return {"email_sent": False}

    success = send_daily_brief(brief, recipient_email=recipient_email)
    return {"email_sent": success}



@handle_pipeline_error(fallback={})
def memory_writer(state: DailyPipelineState) -> dict:
    """
    Last node — commits everything to Supabase.
    Saves: chosen topic, daily brief, session count.
    """
    print("\n[memory_writer] Committing session to memory...")

    user_id = state.get("user_id")
    chosen_topic = state.get("chosen_topic")
    daily_brief = state.get("daily_brief")
    email_sent = state.get("email_sent", False)

    # Save chosen topic to knowledge store
    if chosen_topic and user_id:
        add_topic(
            topic=chosen_topic.title,
            user_id=user_id,
            confidence="low",
            connected_to=chosen_topic.knowledge_gaps_it_fills,
            source="daily_pipeline"
        )
        print(f"[memory_writer] Topic saved: '{chosen_topic.title}'")

    # Save daily brief to Supabase
    if daily_brief and user_id:
        from memory.knowledge_store import save_daily_brief
        save_daily_brief(
            user_id=user_id,
            brief=daily_brief,
            chosen_topic=chosen_topic
        )
        print(f"[memory_writer] Brief saved to Supabase")

    if user_id:
        increment_sessions(user_id=user_id)

    print(f"[memory_writer] Email delivered: {email_sent}")
    print("[Pipeline] ── Daily run complete ──\n")

    return {}


# ------------------------------------------------------------
# Conditional edge
# ------------------------------------------------------------

def should_continue(state: DailyPipelineState) -> str:
    if state.get("chosen_topic") is None:
        print("[Pipeline] No new topic — skipping briefing")
        return "skip"
    return "continue"


# ------------------------------------------------------------
# Rate-limited agent wrappers
#
# Wraps each LLM-calling agent with a rate limit check.
# Raises RateLimitExceeded before the API call if limit hit.
# In production: this becomes a decorator pattern so you
# don't have to wrap manually.
# ------------------------------------------------------------

def curator_node(state: DailyPipelineState) -> dict:
    """Curator agent with rate limit check."""
    user_id = state.get("user_id")
    is_admin = state.get("is_admin", False)

    try:
        check_and_increment(
            user_id=user_id,
            llm_calls=1,
            is_admin=is_admin
        )
    except RateLimitExceeded as e:
        print(f"[curator] Rate limit hit: {e}")
        return {"chosen_topic": None}

    return curator_agent(state)


def briefing_node(state: DailyPipelineState) -> dict:
    """Briefing agent with rate limit check."""
    user_id = state.get("user_id")
    is_admin = state.get("is_admin", False)

    try:
        check_and_increment(
            user_id=user_id,
            llm_calls=1,
            tavily_calls=1,
            is_admin=is_admin
        )
    except RateLimitExceeded as e:
        print(f"[briefing] Rate limit hit: {e}")
        return {"daily_brief": None}

    return briefing_agent(state)


# ------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------

def build_daily_pipeline() -> StateGraph:
    graph = StateGraph(DailyPipelineState)

    graph.add_node("memory_loader", memory_loader)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("curator", curator_node)
    graph.add_node("briefing", briefing_node)
    graph.add_node("notifier", notifier_node)
    graph.add_node("memory_writer", memory_writer)

    graph.add_edge(START, "memory_loader")
    graph.add_edge("memory_loader", "researcher")
    graph.add_edge("researcher", "curator")

    graph.add_conditional_edges(
        "curator",
        should_continue,
        {
            "continue": "briefing",
            "skip": "memory_writer"
        }
    )

    graph.add_edge("briefing", "notifier")
    graph.add_edge("notifier", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile()


# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------

def run_daily_pipeline(
    user_id: str,
    is_admin: bool = False
) -> dict:
    """
    Runs the full daily pipeline for a specific user.
    Called by APScheduler and the Streamlit UI.
    """
    if not user_id:
        raise ValueError("user_id is required to run the daily pipeline")

    pipeline = build_daily_pipeline()
    final_state = pipeline.invoke({
        "user_id": user_id,
        "is_admin": is_admin
    })
    return final_state


if __name__ == "__main__":
    import os
    test_user = os.getenv("TEST_USER_ID")
    if not test_user:
        print("Set TEST_USER_ID in .env to test")
    else:
        run_daily_pipeline(user_id=test_user, is_admin=True)