# onboarding/seed.py
# ------------------------------------------------------------
# One-time CLI to seed your initial knowledge into the store.
# Run this before your first agent session.
# In production: this becomes a guided onboarding flow in the UI
# or an inference step where an LLM interviews the user.
# ------------------------------------------------------------

import sys
import os

# Make sure imports work from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.knowledge_store import (
    add_topic,
    get_known_topics,
    get_store_summary,
    update_sub_concepts
)

# ------------------------------------------------------------
# Default AI/ML topic tree to guide the seeding process.
# Organized by category so the user can go section by section.
#
# In production: this becomes a dynamic taxonomy pulled from
# a knowledge graph DB — updated as the field evolves.
# ------------------------------------------------------------

TOPIC_TREE = {
    "Foundations": [
        "linear algebra",
        "probability and statistics",
        "calculus",
        "information theory"
    ],
    "Core ML": [
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "gradient descent",
        "backpropagation",
        "regularization"
    ],
    "Deep Learning": [
        "neural networks",
        "CNNs",
        "RNNs",
        "transformers",
        "attention mechanism",
        "embeddings"
    ],
    "LLMs": [
        "pre-training",
        "fine-tuning",
        "RLHF",
        "LoRA",
        "RAG",
        "prompt engineering",
        "context windows",
        "tokenization"
    ],
    "Agentic AI": [
        "AI agents",
        "multi-agent systems",
        "tool use",
        "memory systems",
        "orchestration",
        "LangGraph",
        "LangChain",
        "planning and reasoning"
    ],
    "MLOps": [
        "model evaluation",
        "model deployment",
        "vector databases",
        "model monitoring",
        "experiment tracking"
    ]
}

CONFIDENCE_OPTIONS = {
    "1": "low",      # I've heard of it
    "2": "medium",   # I understand it conceptually
    "3": "high"      # I can explain and apply it
}


# ------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------

def print_header():
    print("\n" + "="*55)
    print("   SAGE — Knowledge Onboarding")
    print("   Let's map out what you already know.")
    print("="*55)
    print("\nFor each topic you'll rate your confidence:")
    print("  1 = I've heard of it")
    print("  2 = I understand it conceptually")
    print("  3 = I can explain and apply it")
    print("  s = Skip this topic entirely")
    print("  q = Quit and save what we have so far")
    print()


def prompt_confidence(topic: str) -> str | None:
    """
    Prompts the user for their confidence on a single topic.
    Returns confidence string or None if skipped.
    """
    while True:
        raw = input(f"  {topic}: ").strip().lower()
        if raw == "q":
            return "quit"
        if raw == "s":
            return None
        if raw in CONFIDENCE_OPTIONS:
            return CONFIDENCE_OPTIONS[raw]
        print("    → Please enter 1, 2, 3, s, or q")


def prompt_sub_concepts(topic: str) -> tuple[list[dict], list[str]]:
    """
    For topics rated medium or high, optionally drill into sub-concepts.
    Returns (known_list, gaps_list) to write into sub_concepts.

    In production: this step is replaced by the Memory Agent
    inferring sub-concept depth from a short probing conversation.
    """
    print(f"\n    Optional: tell me more about what you know in '{topic}'")
    print(f"    (press Enter to skip sub-concept detail)\n")

    known = []
    gaps = []

    raw = input(f"    Sub-concepts you know well (comma separated): ").strip()
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if item:
                known.append({
                    "name": item,
                    "confidence": "medium",
                    "notes": "seeded during onboarding"
                })

    raw = input(f"    Sub-concepts you've heard of but aren't clear on (comma separated): ").strip()
    if raw:
        gaps = [g.strip() for g in raw.split(",") if g.strip()]

    return known, gaps


# ------------------------------------------------------------
# Main seeding flow
# ------------------------------------------------------------

def run_seed():
    print_header()

    # Check if already seeded
    existing = get_known_topics()
    if existing:
        print(f"  ⚠️  You already have {len(existing)} topics seeded.")
        choice = input("  Add more topics? (y/n): ").strip().lower()
        if choice != "y":
            print("\n  Nothing changed. Run the app when ready.\n")
            return

    total_added = 0

    for category, topics in TOPIC_TREE.items():
        print(f"\n── {category} " + "─" * (40 - len(category)))

        for topic in topics:
            # Skip topics already in the store
            if topic in [t.lower() for t in existing]:
                print(f"  {topic}: already seeded ✓")
                continue

            confidence = prompt_confidence(topic)

            if confidence == "quit":
                print(f"\n  Saved {total_added} topics. Exiting.\n")
                _print_summary()
                return

            if confidence is None:
                continue  # user skipped

            # For medium/high confidence, offer sub-concept detail
            known_subs, gap_subs = [], []
            if confidence in ["medium", "high"]:
                known_subs, gap_subs = prompt_sub_concepts(topic)

            # Write to knowledge store
            add_topic(
                topic=topic,
                confidence=confidence,
                connected_to=[],    # connections built up over sessions
                source="seed"
            )

            if known_subs or gap_subs:
                update_sub_concepts(
                    topic=topic,
                    known=known_subs,
                    gaps=gap_subs
                )

            total_added += 1
            print(f"    ✓ saved\n")

    print(f"\n  Done! Added {total_added} topics.")
    _print_summary()


def _print_summary():
    summary = get_store_summary()
    print("\n" + "="*55)
    print("  Your knowledge snapshot:")
    print(f"  Total topics:   {summary['total_topics']}")
    print(f"  High confidence: {summary['topics_by_confidence']['high']}")
    print(f"  Medium:          {summary['topics_by_confidence']['medium']}")
    print(f"  Low:             {summary['topics_by_confidence']['low']}")
    print("="*55 + "\n")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    run_seed()