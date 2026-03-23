# graph/chat_pipeline.py
# ------------------------------------------------------------
# The chat session pipeline — interactive LangGraph DAG.
# Updated to accept and pass user_id through all nodes.
#
# Orchestrates:
#   session_loader → teacher (loop) → memory_agent
#                 → session_memory_writer → END
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

from agents.teacher import teacher_agent, build_system_prompt
from auth.rate_limiter import check_and_increment, RateLimitExceeded
from memory.knowledge_store import (
    load_store,
    get_topic_depth,
    update_sub_concepts,
    increment_sessions,
    save_session_transcript
)
from errors.handler import handle_pipeline_error
load_dotenv()


# ------------------------------------------------------------
# State definition
# user_id and is_admin added
# ------------------------------------------------------------

class ChatSessionState(TypedDict):
    # Auth context
    user_id: str
    is_admin: bool

    # Session context
    topic_title: str
    daily_brief: Optional[dict]
    knowledge_snapshot: dict
    topic_depth: Optional[dict]
    system_prompt: str

    # Conversation history
    messages: Annotated[list, add_messages]

    # Timeout detection
    last_activity: str

    # Session end outputs
    concepts_learned: list[dict]
    gaps_identified: list[str]

    # Tracks how many times compression has run this session
    compression_count: int

# ------------------------------------------------------------
# Token counting
#
# Estimates token count of current message history.
# 4 chars ≈ 1 token is a reliable approximation for English.
#
# In production: use tiktoken for exact counts.
# Set threshold at ~60% of model context window to leave
# room for system prompt and next response.
# ------------------------------------------------------------

TOKEN_THRESHOLD = 6000  # trigger compression at ~6k tokens


def _count_tokens_approximate(messages: list) -> int:
    """
    Estimates total tokens in message history.
    Called before every teacher turn.
    """
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4


# ------------------------------------------------------------
# Compression node
#
# Fires when message history exceeds TOKEN_THRESHOLD.
# Three steps:
#   1. Save full transcript to Supabase (nothing lost)
#   2. Summarize old messages into one context block
#   3. Replace old messages with summary + keep last 4
#
# Why keep last 4 messages verbatim?
# The teacher needs immediate conversational context —
# what was just said — to continue naturally. The summary
# handles everything before that.
#
# In production: threshold is configurable per user tier.
# The summary prompt is tuned for technical content —
# generic summarizers lose too much detail for a learning app.
# ------------------------------------------------------------
@handle_pipeline_error(fallback={})
def compression_node(state: ChatSessionState) -> dict:
    """
    Compresses message history when it gets too long.
    Runs as a conditional node BEFORE teacher_node.

    Reads:  state["messages"]
    Writes: state["messages"]        — compressed
            state["compression_count"] — incremented
    """
    messages = state.get("messages", [])
    token_estimate = _count_tokens_approximate(messages)

    # Only compress if above threshold
    if token_estimate < TOKEN_THRESHOLD:
        return {}

    print(f"\n[compression] Token estimate {token_estimate} "
          f"exceeds {TOKEN_THRESHOLD} — compressing...")

    user_id = state.get("user_id")
    topic = state.get("topic_title", "unknown")

    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1
    )

    # Step 1 — save full transcript before any compression
    full_transcript = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Sage'}: {m.content}"
        for m in messages
    ])

    if user_id:
        save_session_transcript(
            user_id=user_id,
            topic=topic,
            transcript=full_transcript,
            message_count=len(messages),
            tokens_used=token_estimate,
            compressed=True,
            session_type="chat_compression_checkpoint"
        )
        print(f"[compression] Full transcript saved to Supabase")

    # Step 2 — keep last 4 messages verbatim, summarize the rest
    messages_to_summarize = messages[:-4] if len(messages) > 4 else []
    messages_to_keep = messages[-4:] if len(messages) > 4 else messages

    if not messages_to_summarize:
        print("[compression] Not enough messages to compress — skipping")
        return {}

    conversation_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Sage'}: {m.content}"
        for m in messages_to_summarize
    ])

    summary_prompt = f"""Summarize this learning conversation about '{topic}'.

Your summary replaces the conversation history to save context space.
A teacher AI will read it to continue the session seamlessly.

CRITICAL — preserve exactly:
1. Every technical concept that was explained
2. What the student demonstrated understanding of
3. What questions were asked and how they were answered
4. Any code examples or analogies that were used
5. What gaps or confusions were identified
6. The current depth of the conversation

Conversation to summarize:
{conversation_text}

Write a dense technical summary that preserves all learning content.
Start with: 'In this session so far:'"""

    try:
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_text = response.content

        # Step 3 — replace old messages with summary block
        summary_message = AIMessage(
            content=f"[Session Context] {summary_text}"
        )
        compressed_messages = [summary_message] + messages_to_keep
        compression_count = state.get("compression_count", 0) + 1

        print(
            f"[compression] {len(messages_to_summarize)} messages → "
            f"1 summary block. "
            f"History: {len(messages)} → {len(compressed_messages)} messages"
        )

        return {
            "messages": compressed_messages,
            "compression_count": compression_count
        }

    except Exception as e:
        print(f"[compression] Failed: {e} — continuing without compression")
        return {}
    
# ------------------------------------------------------------
# Session loader
# ------------------------------------------------------------
@handle_pipeline_error(fallback={})
def session_loader(state: ChatSessionState) -> dict:
    """
    First node — loads knowledge context for this user,
    builds system prompt once.
    """
    print("\n[Chat] ── Session starting ──")

    user_id = state.get("user_id")
    topic_title = state.get("topic_title", "")
    daily_brief = state.get("daily_brief", {})

    store = load_store(user_id=user_id)
    topic_depth = get_topic_depth(
        topic_title,
        user_id=user_id,
        store=store
    )

    if topic_depth:
        print(f"[session_loader] Topic depth loaded:")
        print(f"  Known: {topic_depth['known_count']}")
        print(f"  Gaps:  {topic_depth['gap_count']}")
    else:
        print(f"[session_loader] Fresh topic — no prior depth")

    system_prompt = build_system_prompt(
        topic_title=topic_title,
        knowledge_snapshot=store,
        topic_depth=topic_depth,
        daily_brief=daily_brief
    )

    print(f"[session_loader] System prompt: {len(system_prompt)} chars")
    print("[session_loader] Ready\n")

    return {
            "knowledge_snapshot": store,
            "topic_depth": topic_depth,
            "system_prompt": system_prompt,
            "last_activity": datetime.now().isoformat(),
            "concepts_learned": [],
            "gaps_identified": [],
            "messages": [],
            "compression_count": 0
        }


# ------------------------------------------------------------
# Rate-limited teacher node
# ------------------------------------------------------------

def teacher_node(state: ChatSessionState) -> dict:
    """
    Teacher agent with rate limit check.
    Wraps teacher_agent with per-user quota enforcement.
    """
    user_id = state.get("user_id")
    is_admin = state.get("is_admin", False)

    try:
        check_and_increment(
            user_id=user_id,
            llm_calls=1,
            is_admin=is_admin
        )
    except RateLimitExceeded as e:
        print(f"[teacher] Rate limit hit: {e}")
        return {
            "messages": [AIMessage(
                content="You've reached your daily limit. "
                        "Your session has been saved. "
                        "Come back tomorrow to continue learning!"
            )],
            "last_activity": datetime.now().isoformat()
        }

    return teacher_agent(state)


# ------------------------------------------------------------
# Memory agent
# ------------------------------------------------------------
@handle_pipeline_error(fallback={"concepts_learned": [], "gaps_identified": []})
def memory_agent(state: ChatSessionState) -> dict:
    """
    Runs once at session end.
    Extracts structured knowledge from conversation.
    """
    print("\n[memory_agent] Extracting knowledge...")

    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1
    )

    conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Sage'}: {m.content}"
        for m in state["messages"]
    ])

    topic_title = state.get("topic_title", "unknown")

    prompt = f"""Analyze this learning conversation and extract knowledge updates.

TOPIC: {topic_title}

CONVERSATION:
{conversation}

Extract:
1. Sub-concepts the user DEMONSTRATED understanding of
   (they explained correctly, answered a question, made right connection)
2. Sub-concepts that SURFACED but user was unclear on
   (they asked what it means, got it wrong, said they weren't sure)

Be conservative — only mark as known with clear evidence.

Respond with ONLY valid JSON:
{{
    "known": [
        {{
            "name": "sub-concept name",
            "confidence": "low | medium | high",
            "notes": "one sentence on what they demonstrated"
        }}
    ],
    "gaps": [
        "gap concept name"
    ]
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        known = data.get("known", [])
        gaps = data.get("gaps", [])

        print(f"[memory_agent] {len(known)} known, {len(gaps)} gaps")
        return {
            "concepts_learned": known,
            "gaps_identified": gaps
        }

    except Exception as e:
        print(f"[memory_agent] Extraction failed: {e}")
        return {"concepts_learned": [], "gaps_identified": []}


# ------------------------------------------------------------
# Session memory writer
# ------------------------------------------------------------
@handle_pipeline_error(fallback={})
def session_memory_writer(
    state: ChatSessionState,
    user_id: Optional[str] = None
) -> dict:
    """
    Commits session knowledge to Supabase.
    Last node in the chat pipeline.
    """
    print("\n[session_memory_writer] Committing...")

    # Use passed user_id or fall back to state
    uid = user_id or state.get("user_id")
    topic_title = state.get("topic_title", "")
    known = state.get("concepts_learned", [])
    gaps = state.get("gaps_identified", [])
    messages = state.get("messages", [])

    if topic_title and (known or gaps) and uid:
        update_sub_concepts(
            topic=topic_title,
            known=known,
            gaps=gaps,
            user_id=uid
        )
        print(f"[session_memory_writer] Updated '{topic_title}':")
        print(f"  +{len(known)} known, +{len(gaps)} gaps")

    # Save full transcript
    if uid and messages:
        transcript = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Sage'}: {m.content}"
            for m in messages
        ])
        save_session_transcript(
            user_id=uid,
            topic=topic_title,
            transcript=transcript,
            message_count=len(messages),
            session_type="chat"
        )

    if uid:
        increment_sessions(user_id=uid)

    print("[Chat] ── Session complete ──\n")
    return {}


# ------------------------------------------------------------
# Routing
# ------------------------------------------------------------

def should_compress(state: ChatSessionState) -> str:
    """
    Conditional edge before teacher node.
    Checks if message history needs compression.

    Routes to compression_node if token count is high.
    Routes directly to teacher_node if not.

    This runs every turn but compression_node itself
    is a no-op if below threshold — so it's cheap.
    """
    messages = state.get("messages", [])
    if not messages:
        return "teach"

    token_estimate = _count_tokens_approximate(messages)
    if token_estimate >= TOKEN_THRESHOLD:
        print(f"[routing] Token estimate {token_estimate} — compressing")
        return "compress"

    return "teach"

def should_end_session(state: ChatSessionState) -> str:
    if not state.get("messages"):
        return "continue"

    last_human = next(
        (m for m in reversed(state["messages"])
         if isinstance(m, HumanMessage)),
        None
    )

    if last_human:
        exit_phrases = [
            "end session", "done", "quit",
            "exit", "goodbye", "bye"
        ]
        if any(p in last_human.content.lower() for p in exit_phrases):
            print("[Chat] Exit detected")
            return "end"

    return "continue"


# ------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------

def build_chat_pipeline() -> StateGraph:
    graph = StateGraph(ChatSessionState)

    graph.add_node("session_loader", session_loader)
    graph.add_node("compression", compression_node)
    graph.add_node("teacher", teacher_node)
    graph.add_node("memory_agent", memory_agent)
    graph.add_node("session_memory_writer", session_memory_writer)

    graph.add_edge(START, "session_loader")
    graph.add_edge("session_loader", "teacher")

    graph.add_conditional_edges(
        "session_loader",
        should_compress,
        {
            "compress": "compression",
            "teach": "teacher"
        }
    )

    graph.add_edge("compression", "teacher")

    graph.add_conditional_edges(
        "teacher",
        should_end_session,
        {
            "continue": "teacher",
            "end": "memory_agent"
        }
    )

    graph.add_edge("memory_agent", "session_memory_writer")
    graph.add_edge("session_memory_writer", END)

    return graph.compile()


# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------

def run_chat_turn(
    graph,
    current_state: dict,
    user_message: str,
    user_id: Optional[str] = None,
    is_admin: bool = False
) -> tuple[str, dict]:
    """
    Runs one turn of the chat pipeline.
    Returns (ai_response, updated_state).
    """
    # Ensure user_id is in state
    if user_id:
        current_state["user_id"] = user_id
        current_state["is_admin"] = is_admin

    current_state["messages"] = current_state.get("messages", []) + [
        HumanMessage(content=user_message)
    ]

    # Check compression before calling teacher
    token_estimate = _count_tokens_approximate(
        current_state.get("messages", [])
    )
    if token_estimate >= TOKEN_THRESHOLD:
        compression_result = compression_node(current_state)
        if compression_result:
            current_state["messages"] = compression_result.get(
                "messages", current_state["messages"]
            )
            current_state["compression_count"] = compression_result.get(
                "compression_count", 0
            )

    result = teacher_node(current_state)

    current_state["messages"] = current_state["messages"] + result["messages"]
    current_state["last_activity"] = result.get(
        "last_activity",
        datetime.now().isoformat()
    )

    exit_phrases = [
        "end session", "done", "quit",
        "exit", "goodbye", "bye"
    ]
    last_human = next(
        (m for m in reversed(current_state["messages"])
         if isinstance(m, HumanMessage)),
        None
    )
    if last_human and any(
        p in last_human.content.lower() for p in exit_phrases
    ):
        print("[Chat] Exit — running memory extraction...")
        memory_result = memory_agent(current_state)
        current_state.update(memory_result)
        session_memory_writer(
            current_state,
            user_id=user_id or current_state.get("user_id")
        )

    ai_messages = [
        m for m in current_state["messages"]
        if isinstance(m, AIMessage)
    ]
    last_response = ai_messages[-1].content if ai_messages else "..."

    return last_response, current_state


# ------------------------------------------------------------
# CLI test
# ------------------------------------------------------------

if __name__ == "__main__":
    import os
    from memory.knowledge_store import load_store

    test_user = os.getenv("TEST_USER_ID")
    if not test_user:
        print("Set TEST_USER_ID in .env")
        exit()

    store = load_store(user_id=test_user)
    topics = store.get("topics", [])
    test_topic = topics[-1]["topic"] if topics else "sparse attention"

    state = {
        "user_id": test_user,
        "is_admin": True,
        "topic_title": test_topic,
        "daily_brief": {},
        "knowledge_snapshot": store,
        "topic_depth": None,
        "system_prompt": "",
        "last_activity": datetime.now().isoformat(),
        "messages": [],
        "concepts_learned": [],
        "gaps_identified": []
    }

    state = session_loader(state)
    state["topic_title"] = test_topic
    state["user_id"] = test_user
    state["is_admin"] = True

    print(f"Topic: {test_topic}")
    print("Type 'end session' to finish\n")

    graph = build_chat_pipeline()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        response, state = run_chat_turn(
            graph, state, user_input,
            user_id=test_user,
            is_admin=True
        )
        print(f"\nSage: {response}\n")

        if any(p in user_input.lower()
               for p in ["end session", "quit", "exit", "bye"]):
            break
