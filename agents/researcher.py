# agents/researcher.py
# ------------------------------------------------------------
# Retrieval agent — finds new AI/ML topics from the web.
# Runs on schedule as the first node in the daily pipeline.
#
# Input  (from state): knowledge_snapshot
# Output (to state):   candidate_topics
#
# In production: this agent runs as an isolated microservice,
# triggered by an event bus. Multiple instances run in parallel
# for different domains or user segments.
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field
from tavily import TavilyClient
from dotenv import load_dotenv
from errors.handler import handle_agent_error, SearchError
load_dotenv()


# ------------------------------------------------------------
# Output schema — what this agent returns to graph state
#
# In production: Pydantic models are your agent contracts.
# They get validated, logged, and versioned like API schemas.
# Never let agents pass raw dicts between each other.
# ------------------------------------------------------------

class CandidateTopic(BaseModel):
    """
    A single topic candidate found by the Researcher.
    The Curator Agent will score these against the knowledge store.
    """
    title: str = Field(description="Clean topic name, e.g. 'Mixture of Experts'")
    summary: str = Field(description="2-3 sentence plain English summary")
    relevance_score: float = Field(description="0.0 to 1.0 — how relevant to AI/ML")
    source_url: str = Field(description="Where this was found")
    published_date: str = Field(description="When this was published or discovered")
    related_concepts: list[str] = Field(
        description="Concepts this topic connects to — used for knowledge graph linking"
    )


class ResearcherOutput(BaseModel):
    """
    Full output of the Researcher Agent.
    Wraps the candidate list with run metadata.

    In production: metadata fields feed into observability
    dashboards — you track search quality over time.
    """
    candidates: list[CandidateTopic]
    search_queries_used: list[str]
    total_results_found: int
    run_timestamp: str


# ------------------------------------------------------------
# Search query builder
#
# Builds targeted queries based on what the user already knows.
# If you know 'transformers' well, we search for things adjacent
# to transformers — not transformers itself.
#
# In production: this becomes an LLM call that generates
# dynamic queries from the knowledge graph topology.
# ------------------------------------------------------------

def _build_search_queries(knowledge_snapshot: dict) -> list[str]:
    """
    Builds Tavily search queries biased toward knowledge gaps.

    Takes the knowledge snapshot from state and constructs queries
    that look for topics ADJACENT to what the user knows — not
    topics they already have.

    In production: replaced by an LLM that reads the full
    knowledge graph and generates semantically targeted queries.
    """
    base_queries = [
        "latest AI research breakthroughs 2026",
        "new LLM techniques 2026",
        "agentic AI developments 2026",
        "machine learning papers this week",
    ]

    # Build gap-targeted queries from known topics with gaps
    gap_queries = []
    if knowledge_snapshot and "topics" in knowledge_snapshot:
        for entry in knowledge_snapshot["topics"]:
            gaps = entry.get("sub_concepts", {}).get("gaps", [])
            if gaps:
                # Pick the first gap and search for it specifically
                gap_queries.append(
                    f"{gaps[0]} explained AI machine learning"
                )

    # Combine — base queries for discovery, gap queries for depth
    # Cap at 6 total to manage API costs
    all_queries = base_queries + gap_queries[:2]
    return all_queries[:6]


# ------------------------------------------------------------
# Result parser
#
# Converts raw Tavily results into structured CandidateTopic
# objects. This is where we clean and normalize external data
# before it enters our typed system.
#
# In production: this step includes deduplication against a
# results cache (Redis) so you never surface the same article
# twice across runs.
# ------------------------------------------------------------

def _parse_results(results: list[dict], query: str) -> list[CandidateTopic]:
    """
    Parses raw Tavily search results into CandidateTopic objects.
    Filters out low-quality results based on score threshold.
    """
    candidates = []

    for r in results:
        # Tavily returns a relevance score 0-1 — filter weak results
        if r.get("score", 0) < 0.4:
            continue

        # Extract related concepts from the content snippet
        # In production: this is an LLM extraction call
        related = _extract_related_concepts(r.get("content", ""))

        candidate = CandidateTopic(
            title=r.get("title", "Unknown").strip(),
            summary=r.get("content", "")[:300].strip(),
            relevance_score=round(r.get("score", 0.5), 2),
            source_url=r.get("url", ""),
            published_date=r.get("published_date") or datetime.now().isoformat(),
            related_concepts=related
        )
        candidates.append(candidate)

    return candidates


def _extract_related_concepts(text: str) -> list[str]:
    """
    Lightweight keyword extraction to find related AI concepts.
    Scans text for known AI/ML terms.

    In production: replaced by an LLM extraction call or a
    named entity recognition model fine-tuned on AI literature.
    """
    ai_terms = [
        "transformer", "attention", "fine-tuning", "LoRA", "RAG",
        "agent", "LLM", "embedding", "vector", "RLHF", "diffusion",
        "multimodal", "reasoning", "planning", "tool use", "MoE",
        "quantization", "distillation", "inference", "benchmark"
    ]
    found = []
    text_lower = text.lower()
    for term in ai_terms:
        if term.lower() in text_lower and term not in found:
            found.append(term)
    return found[:5]  # cap at 5 related concepts


# ------------------------------------------------------------
# Main agent function
#
# This is what LangGraph calls as a node.
# Signature is always: (state: dict) -> dict
# Returns a PARTIAL state update — only the keys it owns.
#
# In production: wrapped in a retry decorator, emits traces
# to LangSmith, and has a timeout with fallback behavior.
# ------------------------------------------------------------
@handle_agent_error(
    fallback={"candidate_topics": None},
    layer="agent"
)
def researcher_agent(state: dict) -> dict:
    """
    Researcher Agent — LangGraph node.

    Reads:  state["knowledge_snapshot"]
    Writes: state["candidate_topics"]

    This function is intentionally side-effect free except for
    the Tavily API call. It does not write to memory, does not
    send emails, does not modify the knowledge store.
    Pure input → output.
    """
    print("\n[Researcher] Starting topic discovery...")

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    knowledge_snapshot = state.get("knowledge_snapshot", {})

    # Build targeted search queries
    queries = _build_search_queries(knowledge_snapshot)
    print(f"[Researcher] Running {len(queries)} search queries...")

    all_candidates = []
    used_queries = []

    for query in queries:
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_published_date=True
            )
            results = response.get("results", [])
            candidates = _parse_results(results, query)
            all_candidates.extend(candidates)
            used_queries.append(query)
            print(f"[Researcher] '{query}' → {len(candidates)} candidates")

        except Exception as e:
            print(f"[Researcher] Query failed: {query} — {e}")
            continue

    # Deduplicate by title
    seen_titles = set()
    unique_candidates = []
    for c in all_candidates:
        if c.title.lower() not in seen_titles:
            seen_titles.add(c.title.lower())
            unique_candidates.append(c)

    output = ResearcherOutput(
        candidates=unique_candidates,
        search_queries_used=used_queries,
        total_results_found=len(unique_candidates),
        run_timestamp=datetime.now().isoformat()
    )

    print(f"[Researcher] Done — {len(unique_candidates)} unique candidates found")

    # Return ONLY the keys this agent owns
    # LangGraph merges this partial update into the full state
    return {"candidate_topics": output}