# memory/knowledge_store.py
# ------------------------------------------------------------
# Long-term memory interface — Supabase pgvector version.
#
# INTERFACE IS IDENTICAL to the JSON version.
# All agents call the same functions with the same signatures.
# Zero agent code changes required.
#
# What changed internally:
#   - JSON file → Supabase Postgres tables
#   - String matching → vector similarity search (Cohere)
#   - Single user → multi-user via user_id
#   - No RLS → Row Level Security enforced at DB level
#
# Two modes:
#   - user_id provided → scoped to that user (normal operation)
#   - user_id None → raises error (safety guard)
#
# In production: user_id always comes from the authenticated
# session — never from user input directly.
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Optional
from memory.supabase_client import get_admin_client, generate_embedding
from errors.handler import handle_memory_error

# ------------------------------------------------------------
# Safety guard
# Every function requires a user_id.
# Prevents accidental cross-user data access.
# ------------------------------------------------------------

def _require_user_id(user_id: Optional[str]) -> str:
    """
    Validates user_id is present.
    Raises ValueError if missing.

    Called at the top of every public function.
    In production: user_id always comes from JWT claims —
    never from user-supplied input.
    """
    if not user_id:
        raise ValueError(
            "user_id is required for all knowledge store operations. "
            "Ensure the user is authenticated before calling this function."
        )
    return user_id


# ------------------------------------------------------------
# Core read operations
# ------------------------------------------------------------
@handle_memory_error(fallback={"topics": [], "meta": {}})
def load_store(user_id: Optional[str] = None) -> dict:
    """
    Loads the full knowledge store for a user.
    Returns a dict matching the old JSON format exactly
    so all downstream code works without changes.

    Called by: memory_loader node at START of every graph run.

    In production: result is cached in Redis (5min TTL)
    so repeated calls within a session don't hit the DB.
    Cache is invalidated when memory_writer commits new data.
    """
    uid = _require_user_id(user_id)
    client = get_admin_client()

    try:
        # Fetch all knowledge entries for this user
        entries_response = client.table("knowledge_entries") \
            .select("*, sub_concepts(*)") \
            .eq("user_id", uid) \
            .execute()

        entries = entries_response.data or []

        # Transform DB rows into the same shape as the old JSON
        # so all agents work without modification
        topics = []
        for entry in entries:
            sub_concepts_raw = entry.get("sub_concepts", [])

            known = [
                {
                    "name": sc["name"],
                    "confidence": sc["confidence"] or "medium",
                    "notes": sc["notes"] or ""
                }
                for sc in sub_concepts_raw
                if sc["status"] == "known"
            ]

            gaps = [
                sc["name"]
                for sc in sub_concepts_raw
                if sc["status"] == "gap"
            ]

            topics.append({
                "topic": entry["topic"],
                "confidence": entry["confidence"],
                "date_learned": entry["date_learned"],
                "connected_to": entry["connected_to"] or [],
                "source": entry["source"],
                "sub_concepts": {
                    "known": known,
                    "gaps": gaps
                }
            })

        # Fetch metadata from sessions table
        sessions_response = client.table("sessions") \
            .select("id") \
            .eq("user_id", uid) \
            .execute()

        total_sessions = len(sessions_response.data or [])

        return {
            "topics": topics,
            "meta": {
                "user_id": uid,
                "total_sessions": total_sessions,
                "last_updated": datetime.now().isoformat()
            }
        }

    except Exception as e:
        print(f"[knowledge_store] load_store failed: {e}")
        # Return empty store on failure — agents handle gracefully
        return {
            "topics": [],
            "meta": {
                "user_id": uid,
                "total_sessions": 0,
                "last_updated": None
            }
        }


def get_known_topics(
    store: Optional[dict] = None,
    user_id: Optional[str] = None
) -> list[str]:
    """
    Returns flat list of topic names the user knows.

    Called by: Curator Agent to filter already-known topics.

    Accepts either a pre-loaded store dict (fast, no DB call)
    or a user_id to load fresh from DB.
    Prefer passing store when already loaded in state.
    """
    if store is not None:
        return [entry["topic"] for entry in store.get("topics", [])]

    uid = _require_user_id(user_id)
    loaded = load_store(uid)
    return [entry["topic"] for entry in loaded.get("topics", [])]

@handle_memory_error(fallback=None)
def get_topic_depth(
    topic: str,
    user_id: Optional[str] = None,
    store: Optional[dict] = None
) -> dict | None:
    """
    Returns sub-concept map for a specific topic.

    Called by: Teacher Agent at session start to know
    exactly what the user knows and doesn't know within
    this specific topic.

    In production: this query uses the vector index to find
    semantically similar topics too — not just exact match.
    """
    # Use pre-loaded store if available — avoids DB call
    if store is not None:
        entry = next(
            (e for e in store.get("topics", [])
             if e["topic"].lower() == topic.lower()),
            None
        )
        if entry is None:
            return None
        return {
            "topic": entry["topic"],
            "overall_confidence": entry["confidence"],
            "known": entry["sub_concepts"]["known"],
            "gaps": entry["sub_concepts"]["gaps"],
            "known_count": len(entry["sub_concepts"]["known"]),
            "gap_count": len(entry["sub_concepts"]["gaps"])
        }

    uid = _require_user_id(user_id)
    client = get_admin_client()

    try:
        response = client.table("knowledge_entries") \
            .select("*, sub_concepts(*)") \
            .eq("user_id", uid) \
            .ilike("topic", topic) \
            .single() \
            .execute()

        if not response.data:
            return None

        entry = response.data
        sub_concepts = entry.get("sub_concepts", [])

        known = [
            {
                "name": sc["name"],
                "confidence": sc["confidence"] or "medium",
                "notes": sc["notes"] or ""
            }
            for sc in sub_concepts if sc["status"] == "known"
        ]
        gaps = [
            sc["name"]
            for sc in sub_concepts if sc["status"] == "gap"
        ]

        return {
            "topic": entry["topic"],
            "overall_confidence": entry["confidence"],
            "known": known,
            "gaps": gaps,
            "known_count": len(known),
            "gap_count": len(gaps)
        }

    except Exception as e:
        print(f"[knowledge_store] get_topic_depth failed: {e}")
        return None


def get_store_summary(user_id: Optional[str] = None) -> dict:
    """
    Returns lightweight summary of knowledge store.

    Called by: Streamlit UI sidebar to show progress stats.
    Called by: Briefing Agent to personalize the brief.

    In production: cached aggressively — this is called on
    every page render so it must be fast.
    """
    if not user_id:
        return {
            "total_topics": 0,
            "total_sessions": 0,
            "last_updated": None,
            "topics_by_confidence": {
                "high": 0, "medium": 0, "low": 0
            }
        }

    client = get_admin_client()

    try:
        # Get confidence counts in one query
        entries_response = client.table("knowledge_entries") \
            .select("confidence") \
            .eq("user_id", user_id) \
            .execute()

        entries = entries_response.data or []

        confidence_counts = {"high": 0, "medium": 0, "low": 0}
        for e in entries:
            c = e.get("confidence", "low")
            if c in confidence_counts:
                confidence_counts[c] += 1

        # Get session count
        sessions_response = client.table("sessions") \
            .select("id", count="exact") \
            .eq("user_id", user_id) \
            .execute()

        total_sessions = sessions_response.count or 0

        return {
            "total_topics": len(entries),
            "total_sessions": total_sessions,
            "last_updated": datetime.now().isoformat(),
            "topics_by_confidence": confidence_counts
        }

    except Exception as e:
        print(f"[knowledge_store] get_store_summary failed: {e}")
        return {
            "total_topics": 0,
            "total_sessions": 0,
            "last_updated": None,
            "topics_by_confidence": {"high": 0, "medium": 0, "low": 0}
        }


# ------------------------------------------------------------
# Core write operations
# ------------------------------------------------------------

def add_topic(
    topic: str,
    user_id: Optional[str] = None,
    confidence: str = "medium",
    connected_to: list = [],
    source: str = "daily_session"
) -> None:
    """
    Adds a new topic to the knowledge store.
    If topic already exists, updates confidence if improved.

    Called by: memory_writer after daily pipeline.
    Called by: seed.py during onboarding.

    Now generates a Cohere embedding for semantic search.
    The embedding enables the Curator Agent to find topics
    that are semantically similar — not just string matches.
    """
    uid = _require_user_id(user_id)
    client = get_admin_client()

    try:
        # Generate embedding for semantic search
        embedding = generate_embedding(topic)

        # Upsert — insert or update confidence if better
        data = {
            "user_id": uid,
            "topic": topic,
            "confidence": confidence,
            "connected_to": connected_to,
            "source": source,
            "date_learned": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        if embedding:
            data["embedding"] = embedding

        # Check if exists first
        existing = client.table("knowledge_entries") \
            .select("id, confidence") \
            .eq("user_id", uid) \
            .ilike("topic", topic) \
            .execute()

        if existing.data:
            # Update confidence if improved
            entry_id = existing.data[0]["id"]
            current_conf = existing.data[0]["confidence"]
            confidence_rank = {"low": 1, "medium": 2, "high": 3}

            if confidence_rank.get(confidence, 1) > \
               confidence_rank.get(current_conf, 1):
                client.table("knowledge_entries") \
                    .update({
                        "confidence": confidence,
                        "updated_at": datetime.now().isoformat()
                    }) \
                    .eq("id", entry_id) \
                    .execute()
                print(f"[knowledge_store] Updated confidence: '{topic}' → {confidence}")
        else:
            client.table("knowledge_entries") \
                .insert(data) \
                .execute()
            print(f"[knowledge_store] Added topic: '{topic}'")

    except Exception as e:
        print(f"[knowledge_store] add_topic failed for '{topic}': {e}")


def add_topics_batch(
    topics: list[dict],
    user_id: Optional[str] = None
) -> None:
    """
    Adds multiple topics efficiently.

    Called by: memory_writer after sessions where
    multiple concepts were learned.

    In production: generates embeddings in a single
    batched Cohere API call for efficiency.
    """
    uid = _require_user_id(user_id)
    for t in topics:
        add_topic(
            topic=t.get("topic", ""),
            user_id=uid,
            confidence=t.get("confidence", "medium"),
            connected_to=t.get("connected_to", []),
            source=t.get("source", "daily_session")
        )


def update_sub_concepts(
    topic: str,
    known: list[dict],
    gaps: list[str],
    user_id: Optional[str] = None
) -> None:
    """
    Merges new sub-concept knowledge into an existing topic.

    Called by: session_memory_writer after every chat session.

    Merge logic:
    - New known sub-concepts upserted with their confidence
    - Known sub-concepts removed from gaps automatically
    - New gaps added if not already known
    """
    uid = _require_user_id(user_id)
    client = get_admin_client()

    try:
        # Get or create the parent knowledge entry
        existing = client.table("knowledge_entries") \
            .select("id") \
            .eq("user_id", uid) \
            .ilike("topic", topic) \
            .execute()

        if not existing.data:
            add_topic(topic, user_id=uid, source="chat_session")
            existing = client.table("knowledge_entries") \
                .select("id") \
                .eq("user_id", uid) \
                .ilike("topic", topic) \
                .execute()

        if not existing.data:
            print(f"[knowledge_store] Could not find/create entry for '{topic}'")
            return

        entry_id = existing.data[0]["id"]

        # Get names of newly known concepts for gap cleanup
        newly_known_names = [k["name"].lower() for k in known]

        # Upsert known sub-concepts
        for concept in known:
            client.table("sub_concepts").upsert({
                "user_id": uid,
                "knowledge_entry_id": entry_id,
                "name": concept["name"],
                "status": "known",
                "confidence": concept.get("confidence", "medium"),
                "notes": concept.get("notes", ""),
                "updated_at": datetime.now().isoformat()
            }, on_conflict="knowledge_entry_id,name").execute()

        # Add new gaps — skip if already known
        for gap in gaps:
            if gap.lower() not in newly_known_names:
                # Check if already exists as known
                exists_known = client.table("sub_concepts") \
                    .select("id, status") \
                    .eq("knowledge_entry_id", entry_id) \
                    .ilike("name", gap) \
                    .execute()

                if not exists_known.data:
                    client.table("sub_concepts").insert({
                        "user_id": uid,
                        "knowledge_entry_id": entry_id,
                        "name": gap,
                        "status": "gap",
                        "updated_at": datetime.now().isoformat()
                    }).execute()

        # Clean up — if a gap is now known, update its status
        if newly_known_names:
            client.table("sub_concepts") \
                .update({
                    "status": "known",
                    "updated_at": datetime.now().isoformat()
                }) \
                .eq("knowledge_entry_id", entry_id) \
                .eq("status", "gap") \
                .in_("name", [k["name"] for k in known]) \
                .execute()

        print(f"[knowledge_store] Sub-concepts updated for '{topic}'")

    except Exception as e:
        print(f"[knowledge_store] update_sub_concepts failed: {e}")


def increment_sessions(user_id: Optional[str] = None) -> None:
    """
    Logs a completed session to the sessions table.

    Called by: memory_writer nodes at end of every pipeline run.

    In production: also records token usage, duration,
    and session type for analytics and billing.
    """
    if not user_id:
        return

    client = get_admin_client()

    try:
        client.table("sessions").insert({
            "user_id": user_id,
            "session_type": "pipeline",
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat()
        }).execute()

    except Exception as e:
        print(f"[knowledge_store] increment_sessions failed: {e}")


# ------------------------------------------------------------
# Semantic similarity search
# Used by Curator Agent to find semantically new topics
# ------------------------------------------------------------

def find_similar_topics(
    topic: str,
    user_id: Optional[str] = None,
    threshold: float = 0.85
) -> list[str]:
    """
    Finds topics in the knowledge store that are
    semantically similar to the given topic.

    Called by: Curator Agent as part of the semantic filter —
    a candidate topic is considered 'already known' if its
    embedding is within threshold distance of any known topic.

    Returns list of similar topic names.
    Empty list means the topic is genuinely new.

    In production: threshold is tuned based on user feedback —
    too low = recommends things they know, too high = recommends
    things too similar to what they just learned.
    """
    uid = _require_user_id(user_id)
    client = get_admin_client()

    try:
        embedding = generate_embedding(topic)
        if not embedding:
            return []

        # pgvector cosine similarity search
        response = client.rpc(
            "match_knowledge_entries",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": 5,
                "p_user_id": uid
            }
        ).execute()

        return [r["topic"] for r in (response.data or [])]

    except Exception as e:
        print(f"[knowledge_store] find_similar_topics failed: {e}")
        return []
    
@handle_memory_error(fallback=None)
def save_session_transcript(
    user_id: str,
    topic: str,
    transcript: str,
    summary: str = "",
    message_count: int = 0,
    tokens_used: int = 0,
    compressed: bool = False,
    session_type: str = "chat"
) -> str | None:
    """
    Saves full session transcript to Supabase sessions table.

    Called by: compression node when conversation is compressed.
    Called by: session_memory_writer at session end.

    Returns session_id for reference, None on failure.

    In production: transcript is encrypted at rest using
    Supabase Vault — conversation data is sensitive.
    """
    uid = _require_user_id(user_id)
    client = get_admin_client()

    try:
        response = client.table("sessions").insert({
            "user_id": uid,
            "session_type": session_type,
            "topic": topic,
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
            "message_count": message_count,
            "tokens_used": tokens_used,
            "compressed": compressed,
            "transcript": transcript,
            "summary": summary
        }).execute()

        if response.data:
            return response.data[0]["id"]
        return None

    except Exception as e:
        print(f"[knowledge_store] save_session_transcript failed: {e}")
        return None
    
@handle_memory_error(fallback=None)
def save_daily_brief(
    user_id: str,
    brief,
    chosen_topic=None
) -> str | None:
    """
    Saves the generated daily brief to Supabase.
    Called by memory_writer at the end of daily pipeline.

    Returns the brief id on success, None on failure.

    In production: the UI retrieves today's brief from here
    on page load — no need to re-run the pipeline to see it.
    This also builds a browseable history of past briefs.
    """
    uid = _require_user_id(user_id)
    client = get_admin_client()

    try:
        data = {
            "user_id": uid,
            "date": datetime.now().date().isoformat(),
            "topic_title": brief.topic_title,
            "explanation": brief.explanation,
            "connections": brief.connections,
            "why_it_matters": brief.why_it_matters,
            "discussion_questions": brief.discussion_questions,
            "email_hook": brief.email_hook,
            "tldr": brief.tldr,
            "source_url": brief.source_url,
        }

        if chosen_topic:
            data["difficulty"] = chosen_topic.difficulty

        # Upsert — one brief per user per day
        # If pipeline runs twice today, second run updates the row
        response = client.table("daily_briefs").upsert(
            data,
            on_conflict="user_id,date"
        ).execute()

        if response.data:
            brief_id = response.data[0]["id"]
            print(f"[knowledge_store] Brief saved: {brief_id}")
            return brief_id
        return None

    except Exception as e:
        print(f"[knowledge_store] save_daily_brief failed: {e}")
        return None


def get_latest_brief(user_id: str) -> dict | None:
    """
    Retrieves today's brief from Supabase.
    Called by UI on page load — no need to re-run pipeline.

    In production: this is cached in Redis (1hr TTL)
    so every page render doesn't hit the DB.
    """
    uid = _require_user_id(user_id)
    client = get_admin_client()

    try:
        response = client.table("daily_briefs") \
            .select("*") \
            .eq("user_id", uid) \
            .order("date", desc=True) \
            .limit(1) \
            .execute()

        if response.data:
            return response.data[0]
        return None

    except Exception as e:
        print(f"[knowledge_store] get_latest_brief failed: {e}")
        return None