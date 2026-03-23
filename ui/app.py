# ui/app.py
# ------------------------------------------------------------
# Streamlit UI — production version with auth.
#
# Every page is gated behind require_auth().
# user_id flows through every graph call.
# Rate limits shown in sidebar.
# Logout available from sidebar.
# ------------------------------------------------------------

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage

from auth.middleware import require_auth, logout, UserContext
from auth.rate_limiter import get_remaining, RateLimitExceeded
from graph.daily_pipeline import run_daily_pipeline
from graph.chat_pipeline import (
    build_chat_pipeline,
    run_chat_turn,
    session_loader,
    memory_agent,
    session_memory_writer
)
from errors.handler import log_error, SEVERITY_ERROR
from memory.knowledge_store import (
    load_store,
    get_store_summary,
    get_topic_depth
)
from dotenv import load_dotenv

load_dotenv()

try:
    import main  # noqa — starts scheduler as side effect
except Exception as e:
    print(f"[UI] Scheduler start failed: {e}")
    
# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------

st.set_page_config(
    page_title="Sage — Personal AI Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------

def init_session_state():
    if "chat_active" not in st.session_state:
        st.session_state.chat_active = False
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = None
    if "chat_graph" not in st.session_state:
        st.session_state.chat_graph = build_chat_pipeline()
    if "messages_display" not in st.session_state:
        st.session_state.messages_display = []
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = None
    if "last_pipeline_result" not in st.session_state:
        st.session_state.last_pipeline_result = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"

def init_session_state(user: "UserContext" = None):
    if "chat_active" not in st.session_state:
        st.session_state.chat_active = False
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = None
    if "chat_graph" not in st.session_state:
        st.session_state.chat_graph = build_chat_pipeline()
    if "messages_display" not in st.session_state:
        st.session_state.messages_display = []
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"

    # Load today's brief from Supabase on first render
    # So brief survives page refreshes
    if "last_pipeline_result" not in st.session_state and user:
        from memory.knowledge_store import get_latest_brief
        saved_brief = get_latest_brief(user_id=user.user_id)
        if saved_brief:
            # Reconstruct a brief-like object the UI can render
            st.session_state.last_pipeline_result = {
                "daily_brief": _dict_to_brief(saved_brief),
                "chosen_topic": None
            }
    elif "last_pipeline_result" not in st.session_state:
        st.session_state.last_pipeline_result = None


def _dict_to_brief(data: dict):
    """
    Converts a Supabase daily_briefs row back into
    a DailyBrief-like object the UI can render.
    """
    from agents.briefing import DailyBrief
    try:
        return DailyBrief(
            topic_title=data.get("topic_title", ""),
            explanation=data.get("explanation", ""),
            connections=data.get("connections", []),
            why_it_matters=data.get("why_it_matters", ""),
            discussion_questions=data.get("discussion_questions", []),
            email_hook=data.get("email_hook", ""),
            tldr=data.get("tldr", ""),
            source_url=data.get("source_url", "")
        )
    except Exception as e:
        print(f"[UI] Brief reconstruction failed: {e}")
        return None

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

def render_sidebar(user: UserContext):
    with st.sidebar:
        st.markdown("## 🧠 Sage")
        st.markdown(f"*Welcome, {user.display_name}*")
        st.divider()

        # Knowledge stats
        summary = get_store_summary(user_id=user.user_id)
        st.markdown("### Your Knowledge")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Topics", summary["total_topics"])
            st.metric("Sessions", summary["total_sessions"])
        with col2:
            st.metric("High", summary["topics_by_confidence"]["high"])
            st.metric("Gaps", summary["topics_by_confidence"]["low"])

        st.divider()

        # Daily quota
        st.markdown("### Today's Quota")
        remaining = get_remaining(user.user_id, user.is_admin)
        st.progress(
            remaining["llm_calls_remaining"] /
            (500 if user.is_admin else 50),
            text=f"LLM: {remaining['llm_calls_remaining']} left"
        )
        st.progress(
            remaining["tavily_remaining"] /
            (200 if user.is_admin else 20),
            text=f"Search: {remaining['tavily_remaining']} left"
        )

        st.divider()

        # Navigation
        st.markdown("### Navigation")
        page_options = ["Dashboard", "Chat Session"]
        current_index = page_options.index(
            st.session_state.current_page
        )
        page = st.radio(
            "Go to",
            page_options,
            index=current_index,
            label_visibility="collapsed"
        )
        if page != st.session_state.current_page:
            st.session_state.current_page = page

        st.divider()

        # Daily pipeline
        st.markdown("### Daily Pipeline")
        if st.button("▶ Run Now", use_container_width=True):
            with st.spinner("Running daily pipeline..."):
                try:
                    result = run_daily_pipeline(
                        user_id=user.user_id,
                        is_admin=user.is_admin
                    )
                    st.session_state.last_pipeline_result = result
                    st.success("Pipeline complete!")
                except Exception as e:
                    log_error(
                        error=e,
                        layer="ui",
                        function_name="run_daily_pipeline",
                        user_id=user.user_id
                    )
                    st.error(
                        "Pipeline failed. Check logs for details."
                    )
            st.rerun()

        st.divider()

        # Logout
        if st.button("Sign Out", use_container_width=True):
            logout()

    return st.session_state.current_page


# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------

def render_dashboard(user: UserContext):
    st.markdown(f"# Good morning, {user.display_name} 👋")
    st.markdown("Here's your learning snapshot for today.")

    store = load_store(user_id=user.user_id)
    topics = store.get("topics", [])

    # Today's brief
    result = st.session_state.get("last_pipeline_result")

    if result and result.get("daily_brief"):
        brief = result["daily_brief"]
        st.divider()
        st.markdown("## 📬 Today's Topic")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### {brief.topic_title}")
            st.markdown(f"*{brief.tldr}*")
        with col2:
            chosen = result.get("chosen_topic")
            if chosen:
                difficulty_color = {
                    "beginner": "🟢",
                    "intermediate": "🟡",
                    "advanced": "🔴"
                }.get(chosen.difficulty, "⚪")
                st.markdown(
                    f"**Difficulty:** {difficulty_color} {chosen.difficulty}"
                )
                st.markdown(
                    f"**Fills gaps:** "
                    f"{len(chosen.knowledge_gaps_it_fills)}"
                )

        with st.expander("📖 Full explanation"):
            st.markdown(brief.explanation)
        with st.expander("🔗 Connections to what you know"):
            for c in brief.connections:
                st.markdown(f"- {c}")
        with st.expander("💬 Discussion questions"):
            for i, q in enumerate(brief.discussion_questions, 1):
                st.markdown(f"**Q{i}.** {q}")

        st.divider()

        if not st.session_state.chat_active:
            if st.button(
                f"🎓 Start Learning Session — {brief.topic_title}",
                use_container_width=True,
                type="primary"
            ):
                with st.spinner("Initializing session..."):
                    _initialize_chat_session(
                        topic_title=brief.topic_title,
                        daily_brief=brief.__dict__,
                        user=user
                    )
                st.session_state.current_page = "Chat Session"
                st.rerun()
    else:
        st.info(
            "No brief yet today. "
            "Click **▶ Run Now** in the sidebar."
        )

    # Knowledge map
    if topics:
        st.divider()
        st.markdown("## 🗺️ Your Knowledge Map")

        high = [t for t in topics if t["confidence"] == "high"]
        medium = [t for t in topics if t["confidence"] == "medium"]
        low = [t for t in topics if t["confidence"] == "low"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🟢 Strong")
            for t in high:
                depth = get_topic_depth(
                    t["topic"], user_id=user.user_id
                )
                subs = depth["known_count"] if depth else 0
                st.markdown(f"**{t['topic']}**")
                if subs:
                    st.caption(f"{subs} sub-concepts mapped")

        with col2:
            st.markdown("#### 🟡 Building")
            for t in medium:
                depth = get_topic_depth(
                    t["topic"], user_id=user.user_id
                )
                gaps = depth["gap_count"] if depth else 0
                st.markdown(f"**{t['topic']}**")
                if gaps:
                    st.caption(f"{gaps} gaps to fill")

        with col3:
            st.markdown("#### 🔴 Exposed")
            for t in low:
                st.markdown(f"**{t['topic']}**")
                st.caption("Not yet explored")

    # Recent topics
    if topics:
        st.divider()
        st.markdown("## 📚 Recent Topics")
        recent = sorted(
            topics,
            key=lambda t: t.get("date_learned", ""),
            reverse=True
        )[:5]

        for t in recent:
            with st.expander(
                f"**{t['topic']}** — {t['confidence']} confidence"
            ):
                depth = get_topic_depth(
                    t["topic"], user_id=user.user_id
                )
                if depth and depth["known"]:
                    st.markdown("**Known sub-concepts:**")
                    for k in depth["known"]:
                        st.markdown(
                            f"- {k['name']} ({k['confidence']}) "
                            f"— {k['notes']}"
                        )
                if depth and depth["gaps"]:
                    st.markdown("**Gaps:**")
                    for g in depth["gaps"]:
                        st.markdown(f"- {g}")
                if t.get("connected_to"):
                    st.markdown(
                        f"**Connected to:** "
                        f"{', '.join(t['connected_to'])}"
                    )


# ------------------------------------------------------------
# Chat
# ------------------------------------------------------------

def render_chat(user: UserContext):
    if not st.session_state.chat_active:
        st.markdown("# 🎓 Chat Session")
        st.info(
            "No active session. Run the daily pipeline "
            "from the sidebar, then start a session "
            "from the Dashboard. Or start manually below."
        )

        st.divider()
        st.markdown("#### Start a manual session")
        store = load_store(user_id=user.user_id)
        topics = store.get("topics", [])

        if topics:
            topic_names = [t["topic"] for t in topics]
            selected = st.selectbox("Choose a topic", topic_names)
            if st.button("Start Session", type="primary"):
                with st.spinner("Initializing..."):
                    _initialize_chat_session(
                        topic_title=selected,
                        daily_brief={},
                        user=user
                    )
                st.rerun()
        else:
            st.warning(
                "No topics yet. "
                "Run `python onboarding/seed.py` first."
            )
        return

    # Active session
    topic = st.session_state.current_topic
    st.markdown(f"# 🎓 {topic}")

    # Timeout check
    chat_state = st.session_state.chat_state
    if chat_state and chat_state.get("last_activity"):
        last = datetime.fromisoformat(chat_state["last_activity"])
        if datetime.now() - last > timedelta(minutes=30):
            st.warning("Session timed out — saving progress...")
            _end_session(user)
            st.session_state.current_page = "Dashboard"
            st.rerun()

    # End session button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("⏹ End & Save", type="secondary"):
            with st.spinner("Saving session..."):
                _end_session(user)
            st.success("Session saved!")
            st.session_state.current_page = "Dashboard"
            st.rerun()

    st.divider()

    # Message history
    for msg in st.session_state.messages_display:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(
        "Ask Sage anything about today's topic..."
    ):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages_display.append({
            "role": "user", "content": prompt
        })

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Sage is thinking..."):
                try:
                    response, updated_state = run_chat_turn(
                        st.session_state.chat_graph,
                        st.session_state.chat_state,
                        prompt,
                        user_id=user.user_id,
                        is_admin=user.is_admin
                    )
                    st.session_state.chat_state = updated_state
                    st.markdown(response)
                except RateLimitExceeded as e:
                    st.error(
                        f"Daily limit reached: {e}. "
                        "Your session has been saved."
                    )
                    _end_session(user)
                    st.session_state.current_page = "Dashboard"
                    st.rerun()
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.stop()

        st.session_state.messages_display.append({
            "role": "assistant", "content": response
        })

        exit_phrases = [
            "end session", "done", "quit",
            "exit", "goodbye", "bye"
        ]
        if any(p in prompt.lower() for p in exit_phrases):
            _end_session(user)
            st.success("Session complete — knowledge saved! ✓")
            st.session_state.current_page = "Dashboard"
            st.rerun()

        st.rerun()


# ------------------------------------------------------------
# Session helpers
# ------------------------------------------------------------

def _initialize_chat_session(
    topic_title: str,
    daily_brief: dict,
    user: UserContext
):
    store = load_store(user_id=user.user_id)

    initial_state = {
        "topic_title": topic_title,
        "daily_brief": daily_brief,
        "knowledge_snapshot": store,
        "topic_depth": None,
        "system_prompt": "",
        "last_activity": datetime.now().isoformat(),
        "messages": [],
        "concepts_learned": [],
        "gaps_identified": [],
        "user_id": user.user_id,
        "is_admin": user.is_admin
    }

    loaded_state = session_loader(initial_state)
    loaded_state["topic_title"] = topic_title
    loaded_state["daily_brief"] = daily_brief
    loaded_state["user_id"] = user.user_id
    loaded_state["is_admin"] = user.is_admin

    st.session_state.chat_state = loaded_state
    st.session_state.chat_active = True
    st.session_state.current_topic = topic_title
    st.session_state.messages_display = []

    print(f"[UI] Session initialized: {topic_title} for {user.email}")


def _end_session(user: UserContext):
    state = st.session_state.chat_state
    if not state:
        return

    try:
        memory_result = memory_agent(state)
        state.update(memory_result)
        session_memory_writer(
            state,
            user_id=user.user_id
        )
    except Exception as e:
        print(f"[UI] Session end failed: {e}")

    st.session_state.chat_active = False
    st.session_state.chat_state = None
    st.session_state.messages_display = []
    st.session_state.current_topic = None


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    # Auth gate first
    user = require_auth()
    if not user:
        st.stop()

    # Init session state with user context
    # so brief can be loaded from Supabase on first render
    init_session_state(user)

    page = render_sidebar(user)

    if page == "Dashboard":
        render_dashboard(user)
    elif page == "Chat Session":
        render_chat(user)

if __name__ == "__main__":
    main()