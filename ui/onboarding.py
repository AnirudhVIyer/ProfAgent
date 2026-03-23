# ui/onboarding.py
# ------------------------------------------------------------
# First-time user onboarding flow.
# Renders inside Streamlit when a user has no topics seeded.
#
# Replaces the CLI seed.py for deployed users.
# Same write path — calls add_topic() and update_sub_concepts()
# exactly like seed.py does.
#
# Flow:
#   1. Welcome screen — explains what Sage does
#   2. Category selection — pick which areas you know
#   3. Topic rating — confidence per topic
#   4. Sub-concept detail — optional depth per topic
#   5. Confirmation — shows knowledge snapshot, launches app
#
# In production: onboarding data feeds into a user profile
# model that personalizes the first week of recommendations
# before enough session data exists.
# ------------------------------------------------------------

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from memory.knowledge_store import (
    add_topic,
    update_sub_concepts,
    get_store_summary
)


# ------------------------------------------------------------
# Topic tree — same as seed.py but structured for UI rendering
# ------------------------------------------------------------

TOPIC_TREE = {
    "Foundations": {
        "icon": "📐",
        "description": "Mathematical and theoretical foundations",
        "topics": [
            "linear algebra",
            "probability and statistics",
            "calculus",
            "information theory"
        ]
    },
    "Core ML": {
        "icon": "🤖",
        "description": "Classical machine learning concepts",
        "topics": [
            "supervised learning",
            "unsupervised learning",
            "reinforcement learning",
            "gradient descent",
            "backpropagation",
            "regularization"
        ]
    },
    "Deep Learning": {
        "icon": "🧬",
        "description": "Neural networks and architectures",
        "topics": [
            "neural networks",
            "CNNs",
            "RNNs",
            "transformers",
            "attention mechanism",
            "embeddings"
        ]
    },
    "LLMs": {
        "icon": "💬",
        "description": "Large language models and techniques",
        "topics": [
            "pre-training",
            "fine-tuning",
            "RLHF",
            "LoRA",
            "RAG",
            "prompt engineering",
            "context windows",
            "tokenization"
        ]
    },
    "Agentic AI": {
        "icon": "🕸️",
        "description": "AI agents and orchestration systems",
        "topics": [
            "AI agents",
            "multi-agent systems",
            "tool use",
            "memory systems",
            "orchestration",
            "LangGraph",
            "LangChain",
            "planning and reasoning"
        ]
    },
    "MLOps": {
        "icon": "⚙️",
        "description": "Deployment, evaluation, and operations",
        "topics": [
            "model evaluation",
            "model deployment",
            "vector databases",
            "model monitoring",
            "experiment tracking"
        ]
    }
}

CONFIDENCE_LABELS = {
    1: ("Never heard of it", "⚪"),
    2: ("Heard of it", "🔴"),
    3: ("Know it conceptually", "🟡"),
    4: ("Can explain it", "🟢"),
    5: ("Can implement it", "⭐")
}

CONFIDENCE_TO_STORE = {
    1: None,      # skip — don't add to store
    2: "low",
    3: "low",
    4: "medium",
    5: "high"
}


# ------------------------------------------------------------
# Onboarding state management
# ------------------------------------------------------------

def init_onboarding_state():
    """Initialize onboarding session state."""
    if "onboarding_step" not in st.session_state:
        st.session_state.onboarding_step = 1

    if "onboarding_ratings" not in st.session_state:
        # topic -> confidence int (1-5)
        st.session_state.onboarding_ratings = {}

    if "onboarding_sub_concepts" not in st.session_state:
        # topic -> {known: [], gaps: []}
        st.session_state.onboarding_sub_concepts = {}

    if "onboarding_selected_categories" not in st.session_state:
        st.session_state.onboarding_selected_categories = []


# ------------------------------------------------------------
# Step renderers
# ------------------------------------------------------------

def render_welcome(user_display_name: str):
    """Step 1 — Welcome and explanation."""

    st.markdown(f"# Welcome to Sage, {user_display_name} 👋")
    st.markdown("#### *Your personal AI learning agent*")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔍 Discovers")
        st.markdown(
            "Every day Sage searches the web for AI/ML topics "
            "you don't know yet — personalized to your gaps."
        )

    with col2:
        st.markdown("### 📬 Briefs")
        st.markdown(
            "You get a daily email with a personalized explanation "
            "anchored to what you already know."
        )

    with col3:
        st.markdown("### 🎓 Teaches")
        st.markdown(
            "Log in to discuss the topic with a Socratic tutor "
            "that knows exactly your knowledge level."
        )

    st.divider()

    st.markdown(
        "**Before we start — let's map out what you already know.** "
        "This takes about 3 minutes and makes every recommendation "
        "immediately relevant to you."
    )
    st.caption(
        "You can always update your knowledge map later "
        "from the Dashboard."
    )

    st.markdown("")

    if st.button(
        "Let's get started →",
        type="primary",
        use_container_width=True
    ):
        st.session_state.onboarding_step = 2
        st.rerun()


def render_category_selection():
    """Step 2 — Pick which categories to rate."""

    st.markdown("# Step 1 of 3 — Your Knowledge Areas")
    st.markdown(
        "Select the areas you have **any** familiarity with. "
        "You'll rate individual topics in the next step."
    )
    st.divider()

    selected = []

    cols = st.columns(2)
    for i, (category, data) in enumerate(TOPIC_TREE.items()):
        with cols[i % 2]:
            checked = st.checkbox(
                f"{data['icon']} **{category}**",
                value=category in st.session_state.onboarding_selected_categories,
                help=data["description"]
            )
            if checked:
                selected.append(category)
            st.caption(data["description"])
            st.markdown("")

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.onboarding_step = 1
            st.rerun()
    with col2:
        if st.button(
            "Rate Topics →",
            type="primary",
            use_container_width=True,
            disabled=len(selected) == 0
        ):
            st.session_state.onboarding_selected_categories = selected
            st.session_state.onboarding_step = 3
            st.rerun()

    if len(selected) == 0:
        st.caption("Select at least one category to continue")


def render_topic_rating():
    """Step 3 — Rate confidence for each topic."""

    selected_categories = st.session_state.onboarding_selected_categories

    st.markdown("# Step 2 of 3 — Rate Your Knowledge")
    st.markdown(
        "For each topic, drag the slider to match your confidence level."
    )

    # Legend
    with st.expander("📖 Confidence scale guide"):
        for level, (label, icon) in CONFIDENCE_LABELS.items():
            st.markdown(f"{icon} **{level} — {label}**")

    st.divider()

    ratings = dict(st.session_state.onboarding_ratings)

    for category in selected_categories:
        data = TOPIC_TREE[category]
        st.markdown(f"### {data['icon']} {category}")

        for topic in data["topics"]:
            current = ratings.get(topic, 1)
            level, (label, icon) = current, CONFIDENCE_LABELS[current]

            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(f"**{topic}**")
                st.caption(f"{icon} {label}")
            with col2:
                rating = st.slider(
                    f"_{topic}_",
                    min_value=1,
                    max_value=5,
                    value=current,
                    key=f"rating_{topic}",
                    label_visibility="collapsed"
                )
                ratings[topic] = rating

        st.markdown("")

    st.divider()

    # Summary of what will be saved
    topics_to_save = {
        t: r for t, r in ratings.items()
        if CONFIDENCE_TO_STORE.get(r) is not None
    }
    st.caption(
        f"{len(topics_to_save)} topics will be added to your "
        f"knowledge graph ({len(ratings) - len(topics_to_save)} skipped "
        f"as unfamiliar)"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.onboarding_ratings = ratings
            st.session_state.onboarding_step = 2
            st.rerun()
    with col2:
        if st.button(
            "Add Detail →",
            type="primary",
            use_container_width=True
        ):
            st.session_state.onboarding_ratings = ratings
            st.session_state.onboarding_step = 4
            st.rerun()


def render_sub_concept_detail():
    """
    Step 4 — Optional sub-concept detail for
    topics rated 4 or 5.
    """

    ratings = st.session_state.onboarding_ratings
    deep_topics = {
        t: r for t, r in ratings.items()
        if r >= 4  # only for topics they know well
    }

    st.markdown("# Step 3 of 3 — Optional Detail")

    if not deep_topics:
        # No topics rated highly — skip to confirmation
        st.session_state.onboarding_step = 5
        st.rerun()
        return

    st.markdown(
        f"You rated **{len(deep_topics)} topics** as topics you can "
        f"explain or implement. "
        f"Optionally tell us more — this makes your Teacher Agent "
        f"significantly smarter from day one."
    )
    st.caption("You can skip any or all of these.")
    st.divider()

    sub_concepts = dict(st.session_state.onboarding_sub_concepts)

    for topic, rating in deep_topics.items():
        icon = CONFIDENCE_LABELS[rating][1]
        with st.expander(f"{icon} {topic}"):
            existing = sub_concepts.get(topic, {"known": "", "gaps": ""})

            known_input = st.text_area(
                "Sub-concepts you know well (comma separated)",
                value=existing.get("known", ""),
                key=f"known_{topic}",
                placeholder="e.g. multi-head attention, positional encoding, QKV mechanism",
                height=80
            )

            gaps_input = st.text_area(
                "Sub-concepts you've heard of but aren't clear on (comma separated)",
                value=existing.get("gaps", ""),
                key=f"gaps_{topic}",
                placeholder="e.g. sparse attention, flash attention, RoPE",
                height=80
            )

            sub_concepts[topic] = {
                "known": known_input,
                "gaps": gaps_input
            }

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.onboarding_sub_concepts = sub_concepts
            st.session_state.onboarding_step = 3
            st.rerun()
    with col2:
        if st.button(
            "Finish Setup →",
            type="primary",
            use_container_width=True
        ):
            st.session_state.onboarding_sub_concepts = sub_concepts
            st.session_state.onboarding_step = 5
            st.rerun()


def render_confirmation(user_id: str):
    """
    Step 5 — Save everything and launch.
    Shows progress while saving.
    """

    st.markdown("# 🎉 Setting Up Your Knowledge Graph")
    st.markdown("Saving your knowledge map — this takes a few seconds...")
    st.divider()

    ratings = st.session_state.onboarding_ratings
    sub_concepts = st.session_state.onboarding_sub_concepts

    topics_to_save = {
        t: r for t, r in ratings.items()
        if CONFIDENCE_TO_STORE.get(r) is not None
    }

    # Progress bar
    progress = st.progress(0)
    status = st.empty()
    total = len(topics_to_save)
    saved = 0
    errors = 0

    for topic, rating in topics_to_save.items():
        confidence = CONFIDENCE_TO_STORE[rating]
        status.caption(f"Saving: {topic}...")

        try:
            # Save topic
            add_topic(
                topic=topic,
                user_id=user_id,
                confidence=confidence,
                source="seed"
            )

            # Save sub-concepts if provided
            if topic in sub_concepts:
                raw = sub_concepts[topic]

                known_list = []
                known_raw = raw.get("known", "")
                if known_raw.strip():
                    for item in known_raw.split(","):
                        item = item.strip()
                        if item:
                            known_list.append({
                                "name": item,
                                "confidence": confidence,
                                "notes": "seeded during onboarding"
                            })

                gaps_list = []
                gaps_raw = raw.get("gaps", "")
                if gaps_raw.strip():
                    gaps_list = [
                        g.strip()
                        for g in gaps_raw.split(",")
                        if g.strip()
                    ]

                if known_list or gaps_list:
                    update_sub_concepts(
                        topic=topic,
                        known=known_list,
                        gaps=gaps_list,
                        user_id=user_id
                    )

            saved += 1

        except Exception as e:
            errors += 1
            print(f"[Onboarding] Failed to save {topic}: {e}")

        progress.progress(saved / total)

    status.empty()
    progress.empty()

    # Show results
    st.success(f"✓ Knowledge graph created — {saved} topics saved!")

    if errors > 0:
        st.warning(f"{errors} topics failed to save — you can add them later")

    # Show summary
    summary = get_store_summary(user_id=user_id)

    st.divider()
    st.markdown("### Your Knowledge Snapshot")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Topics Mapped",
            summary["total_topics"]
        )
    with col2:
        st.metric(
            "Strong Areas",
            summary["topics_by_confidence"]["high"]
        )
    with col3:
        st.metric(
            "Building",
            summary["topics_by_confidence"]["medium"]
        )

    st.divider()
    st.markdown(
        "**You're all set.** Sage will find your first topic gap "
        "and email you when it's ready. You can also run the "
        "pipeline manually from the sidebar anytime."
    )

    st.markdown("")

    if st.button(
        "Launch Sage →",
        type="primary",
        use_container_width=True
    ):
        # Clear onboarding state
        for key in [
            "onboarding_step",
            "onboarding_ratings",
            "onboarding_sub_concepts",
            "onboarding_selected_categories"
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()


# ------------------------------------------------------------
# Main onboarding entry point
# Called by ui/app.py when user has no topics
# ------------------------------------------------------------

def render_onboarding(user):
    """
    Main onboarding renderer.
    Called by app.py on first login.

    Manages the full multi-step flow via session state.
    """
    init_onboarding_state()
    step = st.session_state.onboarding_step

    # Progress indicator
    if step > 1:
        steps = ["Welcome", "Areas", "Rating", "Detail", "Done"]
        progress_val = (step - 1) / (len(steps) - 1)
        st.progress(progress_val)
        st.caption(
            f"Step {step - 1} of {len(steps) - 1} — {steps[step - 1]}"
        )
        st.markdown("")

    if step == 1:
        render_welcome(user.display_name)
    elif step == 2:
        render_category_selection()
    elif step == 3:
        render_topic_rating()
    elif step == 4:
        render_sub_concept_detail()
    elif step == 5:
        render_confirmation(user.user_id)


# ------------------------------------------------------------
# Helper — check if user needs onboarding
# ------------------------------------------------------------

def needs_onboarding(user_id: str) -> bool:
    """
    Returns True if user has no topics seeded yet.
    Called by app.py before rendering dashboard.
    """
    try:
        summary = get_store_summary(user_id=user_id)
        return summary["total_topics"] == 0
    except Exception:
        return False