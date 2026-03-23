# agents/curator.py
# ------------------------------------------------------------
# Curator Agent — selects the best topic to learn today.
# First real LLM call in the daily pipeline.
#
# Input  (from state): candidate_topics, knowledge_snapshot
# Output (to state):   chosen_topic
#
# Two-step process:
#   1. Rule-based filter  — cheap, fast, no LLM
#   2. LLM semantic pick  — nuanced, personalized final selection
#
# In production: the LLM call here is the core personalization
# engine. You'd fine-tune or prompt-engineer this heavily based
# on user engagement data — did they actually enjoy the topic
# the curator picked? That feedback loop improves selection.
# ------------------------------------------------------------

import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from errors.handler import handle_agent_error
load_dotenv()


# ------------------------------------------------------------
# Output schema
# ------------------------------------------------------------

class ChosenTopic(BaseModel):
    """
    The single topic the Curator selected for today.
    Contains enough context for the Briefing Agent to work with.

    In production: this object gets logged to a decisions table
    so you can audit why the system picked what it picked —
    critical for debugging personalization quality over time.
    """
    title: str
    summary: str
    why_chosen: str = Field(
        description="Why this topic is right for this user today"
    )
    knowledge_gaps_it_fills: list[str] = Field(
        description="Which gaps from the knowledge store this addresses"
    )
    what_you_already_know: str = Field(
        description="What the user already knows that connects to this topic"
    )
    source_url: str
    difficulty: str = Field(
        description="beginner | intermediate | advanced relative to user's level"
    )


# ------------------------------------------------------------
# Step 1 — Rule-based pre-filter
#
# Eliminates candidates that are obviously already known.
# Fast string matching before we spend tokens on LLM reasoning.
#
# In production: this step also checks a "seen topics" cache
# (Redis) so you never recommend something surfaced in the
# last 30 days even if it's technically not in the knowledge store.
# ------------------------------------------------------------

def _rule_based_filter(
    candidates: list,
    known_topics: list[str]
) -> list:
    """
    Filters out candidates whose titles closely match known topics.
    Uses lowercase comparison and partial string matching.
    Returns the filtered candidate list.
    """
    known_lower = [t.lower() for t in known_topics]
    filtered = []

    for candidate in candidates:
        title_lower = candidate.title.lower()

        # Check if any known topic is contained in the candidate title
        # or vice versa — catches "LoRA" vs "LoRA fine-tuning"
        is_known = any(
            known in title_lower or title_lower in known
            for known in known_lower
        )

        if not is_known:
            filtered.append(candidate)

    print(f"[Curator] Rule filter: {len(candidates)} → {len(filtered)} candidates")
    return filtered


# ------------------------------------------------------------
# Step 2 — LLM semantic selection
#
# Passes filtered candidates + full knowledge snapshot to Claude.
# Claude reasons about semantic overlap and picks the best topic.
#
# This is where personalization actually happens — Claude reads
# your knowledge graph and thinks about what would be most
# valuable to learn next given what you already know.
#
# In production: the system prompt here is your most valuable
# prompt asset. It gets versioned, A/B tested, and refined
# based on user satisfaction signals.
# ------------------------------------------------------------

def _llm_semantic_pick(
    candidates: list,
    knowledge_snapshot: dict
) -> ChosenTopic | None:
    """
    Uses Claude to semantically select the best topic for today.
    Returns a ChosenTopic or None if no good candidate exists.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3   # low temp — this is a selection task, not creative
    )

    # Serialize candidates for the prompt
    candidates_text = "\n\n".join([
        f"CANDIDATE {i+1}:\n"
        f"Title: {c.title}\n"
        f"Summary: {c.summary}\n"
        f"Relevance score: {c.relevance_score}\n"
        f"Related concepts: {', '.join(c.related_concepts)}\n"
        f"URL: {c.source_url}"
        for i, c in enumerate(candidates)
    ])

    # Serialize knowledge snapshot for the prompt
    known_topics = [
        f"- {e['topic']} (confidence: {e['confidence']}, "
        f"gaps: {e.get('sub_concepts', {}).get('gaps', [])})"
        for e in knowledge_snapshot.get("topics", [])
    ]
    knowledge_text = "\n".join(known_topics) if known_topics else "No topics seeded yet."

    system_prompt = """You are a personalized learning curator for an AI/ML engineer.
Your job is to select the single best topic for them to learn today.

SELECTION CRITERIA:
- The topic must be genuinely new — not semantically covered by what they already know
- It should connect naturally to their existing knowledge (not too foreign)
- It should fill a real gap or extend a concept they know partially
- Prefer topics with practical relevance to agentic AI systems
- Avoid topics that are too basic given their confidence levels

You must respond with a valid JSON object and nothing else.
No preamble, no explanation outside the JSON."""

    user_prompt = f"""Here is what this person currently knows:

{knowledge_text}

Here are today's candidate topics from the web:

{candidates_text}

Select the single best topic for them to learn today.
Respond with this exact JSON structure:

{{
    "title": "exact title from candidates",
    "summary": "2-3 sentence plain English summary of the topic",
    "why_chosen": "why this is the right topic for this person today",
    "knowledge_gaps_it_fills": ["gap1", "gap2"],
    "what_you_already_know": "what they already know that connects to this",
    "source_url": "exact URL from the candidate",
    "difficulty": "beginner | intermediate | advanced"
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Parse the JSON response
        raw = response.content.strip()

        # Strip markdown code fences if Claude adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        chosen = ChosenTopic(**data)

        print(f"[Curator] LLM selected: '{chosen.title}'")
        print(f"[Curator] Why: {chosen.why_chosen[:80]}...")
        return chosen

    except Exception as e:
        print(f"[Curator] LLM selection failed: {e}")
        return None


# ------------------------------------------------------------
# Fallback — if LLM fails, pick highest scoring candidate
#
# In production: fallback logic is non-negotiable. LLM calls
# fail, time out, or return malformed JSON. Always have a
# deterministic fallback that keeps the pipeline running.
# ------------------------------------------------------------

def _fallback_pick(candidates: list) -> ChosenTopic | None:
    """
    Deterministic fallback — picks highest Tavily relevance score.
    Used when the LLM call fails for any reason.
    """
    if not candidates:
        return None

    best = max(candidates, key=lambda c: c.relevance_score)
    return ChosenTopic(
        title=best.title,
        summary=best.summary,
        why_chosen="Selected by relevance score (fallback mode)",
        knowledge_gaps_it_fills=[],
        what_you_already_know="",
        source_url=best.source_url,
        difficulty="intermediate"
    )


# ------------------------------------------------------------
# Main agent function — LangGraph node
# ------------------------------------------------------------
@handle_agent_error(
    fallback={"chosen_topic": None},
    layer="agent"
)
def curator_agent(state: dict) -> dict:
    """
    Curator Agent — LangGraph node.

    Reads:  state["candidate_topics"], state["knowledge_snapshot"]
    Writes: state["chosen_topic"]
    """
    print("\n[Curator] Starting topic selection...")

    researcher_output = state.get("candidate_topics")
    knowledge_snapshot = state.get("knowledge_snapshot", {})

    if not researcher_output or not researcher_output.candidates:
        print("[Curator] No candidates received — skipping")
        return {"chosen_topic": None}

    candidates = researcher_output.candidates
    known_topics = [
        e["topic"] for e in knowledge_snapshot.get("topics", [])
    ]

    # Step 1 — cheap rule-based filter
    filtered = _rule_based_filter(candidates, known_topics)

    if not filtered:
        print("[Curator] All candidates already known — nothing new today")
        return {"chosen_topic": None}

    # Step 2 — LLM semantic selection
    chosen = _llm_semantic_pick(filtered, knowledge_snapshot)

    # Fallback if LLM fails
    if chosen is None:
        print("[Curator] Falling back to score-based selection")
        chosen = _fallback_pick(filtered)

    return {"chosen_topic": chosen}