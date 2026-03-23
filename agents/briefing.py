# agents/briefing.py
# ------------------------------------------------------------
# Briefing Agent — writes the personalized daily learning brief.
# Third node in the daily pipeline.
#
# Input  (from state): chosen_topic, knowledge_snapshot
# Output (to state):   daily_brief
#
# This agent's output serves two consumers:
#   1. Gmail notification (sent by notifications/gmail.py)
#   2. Streamlit chat UI (shown at session start)
#
# In production: this agent's system prompt is your core
# product differentiator. It gets A/B tested and refined
# based on engagement signals — did the user actually open
# the chat after reading the brief?
# ------------------------------------------------------------

import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from tavily import TavilyClient
from errors.handler import handle_agent_error
load_dotenv()


# ------------------------------------------------------------
# Output schema
# ------------------------------------------------------------

class DailyBrief(BaseModel):
    """
    The personalized daily learning brief.

    Structured into distinct sections so each consumer
    (Gmail, Streamlit UI) can render what it needs.

    In production: this object is stored in the DB so the
    UI can retrieve it without re-generating it, and so you
    can build a history of past briefs.
    """
    topic_title: str

    # The core explanation — personalized to knowledge level
    explanation: str = Field(
        description="3-4 paragraph explanation anchored to what the user already knows"
    )

    # Explicit knowledge graph connections
    connections: list[str] = Field(
        description="How this topic connects to things already in the knowledge store"
    )

    # What makes this topic matter right now
    why_it_matters: str = Field(
        description="1 paragraph on practical relevance to agentic AI systems"
    )

    # Socratic questions for the chat session
    discussion_questions: list[str] = Field(
        description="3 questions calibrated to the user's knowledge level"
    )

    # One-liner for the Gmail subject line
    email_hook: str = Field(
        description="One punchy sentence to use as email subject — makes them want to log in"
    )

    # TL;DR for the email body preview
    tldr: str = Field(
        description="2 sentence summary for the email body"
    )

    source_url: str


# ------------------------------------------------------------
# Content fetcher — grounds the brief in real source content
#
# Uses Tavily's get_search_context() which fetches, parses,
# and cleans the full page content from a URL.
# This is what separates a grounded brief from a hallucinated one.
#
# In production: fetched content gets cached (Redis, 24hr TTL)
# so if two users get the same topic, you don't fetch twice.
# Also adds resilience — if fetch fails, fall back to snippet.
# ------------------------------------------------------------

def _fetch_topic_content(url: str, topic_title: str) -> str:
    """
    Fetches full content from the source URL using Tavily.
    Falls back to a targeted search if URL fetch fails.

    In production: result is cached by URL hash so repeated
    runs don't re-fetch the same content.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    try:
        # get_search_context fetches + cleans full page content
        # max_tokens limits how much we pull — keeps prompt size manageable
        print(f"[Briefing] Fetching full content from source...")
        content = client.get_search_context(
            query=topic_title,
            search_depth="advanced",
            max_tokens=3000
        )
        print(f"[Briefing] Content fetched — {len(content)} chars")
        return content

    except Exception as e:
        print(f"[Briefing] URL fetch failed ({e}) — falling back to search")

        # Fallback — search for the topic directly if URL fetch fails
        try:
            results = client.search(
                query=topic_title,
                search_depth="advanced",
                max_results=3
            )
            # Concatenate snippets from top results
            snippets = [
                r.get("content", "") for r in results.get("results", [])
            ]
            return " ".join(snippets)[:3000]

        except Exception as e2:
            print(f"[Briefing] Fallback search also failed: {e2}")
            return ""  # empty string — LLM will work from snippet only
        
# ------------------------------------------------------------
# Main agent function — LangGraph node
# ------------------------------------------------------------
@handle_agent_error(
    fallback={"daily_brief": None},
    layer="agent"
)
def briefing_agent(state: dict) -> dict:
    """
    Briefing Agent — LangGraph node.

    Reads:  state["chosen_topic"], state["knowledge_snapshot"]
    Writes: state["daily_brief"]
    """
    print("\n[Briefing] Writing personalized brief...")

    chosen_topic = state.get("chosen_topic")
    knowledge_snapshot = state.get("knowledge_snapshot", {})

    if not chosen_topic:
        print("[Briefing] No chosen topic — skipping")
        return {"daily_brief": None}

    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.7   # slightly higher — this is a writing task
    )

    # Build knowledge context for personalization
    known_topics = [
        f"- {e['topic']} (confidence: {e['confidence']})"
        for e in knowledge_snapshot.get("topics", [])
    ]
    knowledge_text = "\n".join(known_topics) if known_topics else "No prior knowledge seeded."

    full_content = _fetch_topic_content(
        chosen_topic.source_url,
        chosen_topic.title
    )
    system_prompt = """You are a world-class personalized AI tutor.
You write daily learning briefs for AI/ML engineers that feel like
they were written specifically for them — because they were.

Your writing style:
- Direct and precise — no fluff, no filler
- Anchored to what the reader already knows
- Makes abstract concepts concrete with analogies
- Treats the reader as intelligent but acknowledges their gaps
- Builds genuine excitement about learning something new

You must respond with a valid JSON object and nothing else."""

    full_content = _fetch_topic_content(
        chosen_topic.source_url,
        chosen_topic.title
    )

    user_prompt = f"""Write a personalized daily learning brief for this topic.
You have been given the FULL SOURCE CONTENT fetched directly from the web.
Use this as your primary source — do not rely on your training data for
the specifics of this topic. If the content is recent, trust it over
anything you think you know.

TOPIC: {chosen_topic.title}
DIFFICULTY FOR THIS USER: {chosen_topic.difficulty}
WHY CHOSEN: {chosen_topic.why_chosen}
WHAT THEY ALREADY KNOW THAT CONNECTS: {chosen_topic.what_you_already_know}
GAPS THIS FILLS: {', '.join(chosen_topic.knowledge_gaps_it_fills)}

FULL SOURCE CONTENT (fetched live):
{full_content if full_content else chosen_topic.summary}

THEIR FULL KNOWLEDGE SNAPSHOT:
{knowledge_text}

Write the brief personalized specifically to this person's knowledge level.
Reference what they already know when explaining new concepts.
Ground every explanation in the source content above — not generic knowledge.
Make the discussion questions genuinely probe their understanding gaps.

Respond with this exact JSON structure:
{{
    "topic_title": "clean topic title",
    "explanation": "3-4 paragraphs explaining the topic anchored to their existing knowledge",
    "connections": [
        "Connection to [known topic]: explanation of how they relate",
        "Connection to [known topic]: explanation of how they relate",
        "Connection to [known topic]: explanation of how they relate"
    ],
    "why_it_matters": "1 paragraph on why this matters for agentic AI systems specifically",
    "discussion_questions": [
        "question 1 — probes a specific gap",
        "question 2 — connects to something they know",
        "question 3 — pushes toward application"
    ],
    "email_hook": "one punchy sentence that makes them want to log in and learn this",
    "tldr": "two sentence summary for the email body",
    "source_url": "{chosen_topic.source_url}"
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        brief = DailyBrief(**data)

        print(f"[Briefing] Brief written for: '{brief.topic_title}'")
        print(f"[Briefing] Email hook: {brief.email_hook}")

        return {"daily_brief": brief}

    except Exception as e:
        print(f"[Briefing] Failed: {e}")
        return {"daily_brief": None}