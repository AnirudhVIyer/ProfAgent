# ui/admin_panel.py
# ------------------------------------------------------------
# Admin panel — only visible to users with role = 'admin'.
# Manages access requests, users, and usage stats.
#
# Accessible from the sidebar when logged in as admin.
# ------------------------------------------------------------

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime
from memory.supabase_client import get_admin_client
from auth.admin import invite_user, deactivate_user, activate_user
from dotenv import load_dotenv

load_dotenv()


def _get_pending_requests() -> list:
    """Fetches all pending access requests."""
    client = get_admin_client()
    try:
        response = client.table("access_requests") \
            .select("*") \
            .eq("status", "pending") \
            .order("requested_at") \
            .execute()
        return response.data or []
    except Exception as e:
        print(f"[admin_panel] Failed to fetch requests: {e}")
        return []


def _get_all_requests() -> list:
    """Fetches all access requests."""
    client = get_admin_client()
    try:
        response = client.table("access_requests") \
            .select("*") \
            .order("requested_at", desc=True) \
            .execute()
        return response.data or []
    except Exception as e:
        return []


def _get_all_users() -> list:
    """Fetches all user profiles."""
    client = get_admin_client()
    try:
        response = client.table("profiles") \
            .select("*") \
            .order("created_at", desc=True) \
            .execute()
        return response.data or []
    except Exception as e:
        return []


def _approve_request(request: dict) -> bool:
    """
    Approves an access request:
    1. Sends Supabase invite to their email
    2. Updates request status to approved
    """
    client = get_admin_client()

    try:
        # Send Supabase invite
        success = invite_user(
            email=request["email"],
            display_name=request["name"]
        )

        if not success:
            return False

        # Update request status
        client.table("access_requests") \
            .update({
                "status": "approved",
                "reviewed_at": datetime.now().isoformat()
            }) \
            .eq("id", request["id"]) \
            .execute()

        return True

    except Exception as e:
        print(f"[admin_panel] Approve failed: {e}")
        return False


def _reject_request(request_id: str) -> bool:
    """Rejects an access request."""
    client = get_admin_client()
    try:
        client.table("access_requests") \
            .update({
                "status": "rejected",
                "reviewed_at": datetime.now().isoformat()
            }) \
            .eq("id", request_id) \
            .execute()
        return True
    except Exception as e:
        print(f"[admin_panel] Reject failed: {e}")
        return False


def _get_usage_stats() -> list:
    """Gets today's usage stats for all users."""
    client = get_admin_client()
    try:
        from datetime import date
        response = client.table("rate_limits") \
            .select("*, profiles(email, display_name)") \
            .eq("date", date.today().isoformat()) \
            .execute()
        return response.data or []
    except Exception as e:
        return []


def render_admin_panel():
    """
    Main admin panel renderer.
    Only call this for admin users.
    """
    st.markdown("# ⚙️ Admin Panel")
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📥 Access Requests",
        "👥 Users",
        "📊 Usage Today"
    ])

    # ── Tab 1: Access Requests ──────────────────────────────
    with tab1:
        pending = _get_pending_requests()

        if pending:
            st.markdown(
                f"### {len(pending)} Pending Request"
                f"{'s' if len(pending) > 1 else ''}"
            )

            for req in pending:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.markdown(f"**{req['name']}**")
                        st.caption(req['email'])
                        if req.get('note'):
                            st.markdown(
                                f"*\"{req['note']}\"*"
                            )
                        requested = req.get('requested_at', '')[:10]
                        st.caption(f"Requested: {requested}")

                    with col2:
                        if st.button(
                            "✓ Approve",
                            key=f"approve_{req['id']}",
                            type="primary",
                            use_container_width=True
                        ):
                            with st.spinner("Approving..."):
                                success = _approve_request(req)
                            if success:
                                st.success(
                                    f"✓ Invite sent to {req['email']}"
                                )
                                st.rerun()
                            else:
                                st.error("Failed — check logs")

                    with col3:
                        if st.button(
                            "✗ Reject",
                            key=f"reject_{req['id']}",
                            use_container_width=True
                        ):
                            _reject_request(req['id'])
                            st.rerun()

                    st.divider()

        else:
            st.info("No pending requests")

        # Show all requests history
        with st.expander("View all requests history"):
            all_requests = _get_all_requests()
            if all_requests:
                for req in all_requests:
                    status_icon = {
                        "pending": "⏳",
                        "approved": "✓",
                        "rejected": "✗"
                    }.get(req['status'], "?")

                    st.markdown(
                        f"{status_icon} **{req['name']}** "
                        f"— {req['email']} "
                        f"— {req['status']} "
                        f"— {req.get('requested_at', '')[:10]}"
                    )
            else:
                st.caption("No requests yet")

    # ── Tab 2: Users ────────────────────────────────────────
    with tab2:
        users = _get_all_users()

        if users:
            st.markdown(f"### {len(users)} Users")

            for u in users:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        active_icon = "🟢" if u.get('is_active') else "🔴"
                        st.markdown(
                            f"{active_icon} **{u.get('display_name', '')}**"
                        )
                        st.caption(
                            f"{u.get('email', '')} — "
                            f"{u.get('role', 'user')}"
                        )
                        joined = u.get('created_at', '')[:10]
                        st.caption(f"Joined: {joined}")

                    with col2:
                        if u.get('is_active'):
                            if st.button(
                                "Deactivate",
                                key=f"deactivate_{u['id']}",
                                use_container_width=True
                            ):
                                deactivate_user(u['email'])
                                st.rerun()
                        else:
                            if st.button(
                                "Activate",
                                key=f"activate_{u['id']}",
                                type="primary",
                                use_container_width=True
                            ):
                                activate_user(u['email'])
                                st.rerun()

                    with col3:
                        from auth.rate_limiter import get_remaining
                        remaining = get_remaining(u['id'])
                        st.caption(
                            f"LLM: {remaining.get('llm_calls_remaining', '?')} left"
                        )
                        if st.button(
                            "Reset Limits",
                            key=f"reset_{u['id']}",
                            use_container_width=True
                        ):
                            from auth.admin import reset_limits
                            reset_limits(u['email'])
                            st.success("Reset!")
                            st.rerun()

                    st.divider()
        else:
            st.info("No users yet")

    # ── Tab 3: Usage Today ──────────────────────────────────
    with tab3:
        st.markdown("### API Usage Today")
        usage = _get_usage_stats()

        if usage:
            for u in usage:
                profile = u.get('profiles', {})
                email = profile.get('email', 'unknown')
                name = profile.get('display_name', email)

                with st.expander(f"**{name}** — {email}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "LLM Calls",
                            f"{u['llm_calls']}/{u['max_llm_calls']}"
                        )
                    with col2:
                        tokens_k = u['tokens_used'] // 1000
                        max_k = u['max_tokens'] // 1000
                        st.metric(
                            "Tokens",
                            f"{tokens_k}k/{max_k}k"
                        )
                    with col3:
                        st.metric(
                            "Searches",
                            f"{u['tavily_calls']}/{u['max_tavily_calls']}"
                        )
        else:
            st.info("No usage data today yet")