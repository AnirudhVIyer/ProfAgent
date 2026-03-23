# auth/middleware.py
# ------------------------------------------------------------
# Authentication middleware for Streamlit.
#
# Handles:
#   - Login page rendering
#   - Session persistence via st.session_state
#   - Page protection — redirects to login if not authenticated
#   - Token refresh
#
# Every page in the app calls require_auth() at the top.
# If the user isn't logged in, they see the login page.
# If they are, they get their user context injected.
#
# In production: this becomes a proper OAuth2 flow with
# JWT validation on every request. The Streamlit session
# state is replaced by a server-side session store (Redis).
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from supabase import Client
from memory.supabase_client import get_client, get_admin_client
from dotenv import load_dotenv
from errors.handler import handle_auth_error, log_error, SEVERITY_WARNING
load_dotenv()


# ------------------------------------------------------------
# User context
# Passed around the app after authentication.
# Every agent call includes this context.
# ------------------------------------------------------------

class UserContext:
    """
    Authenticated user context.
    Passed to every graph function and agent call.

    In production: this is derived from a validated JWT —
    never from client-supplied data.
    """
    def __init__(self, user_id: str, email: str, role: str,
                 access_token: str, display_name: str = ""):
        self.user_id = user_id
        self.email = email
        self.role = role
        self.access_token = access_token
        self.display_name = display_name or email.split("@")[0]

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role,
            "display_name": self.display_name
        }


# ------------------------------------------------------------
# Login page
# ------------------------------------------------------------

def render_login_page():
    """
    Renders the login page.
    Called by require_auth() when no valid session exists.
    """
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("## 🧠 Sage")
        st.markdown("*Your personal AI learning agent*")
        st.divider()

        st.markdown("### Sign In")
        st.caption(
            "Access is invite-only. "
            "Request access from the admin to get started."
        )

        with st.form("login_form"):
            email = st.text_input(
                "Email",
                placeholder="your@email.com"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="••••••••"
            )
            submitted = st.form_submit_button(
                "Sign In",
                use_container_width=True,
                type="primary"
            )

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password")
                return

            with st.spinner("Signing in..."):
                success, result = _attempt_login(email, password)

            if success:
                st.success("Welcome back!")
                st.rerun()
            else:
                st.error(f"Login failed: {result}")

        st.divider()
        st.caption(
            "Don't have access? "
            "Contact the admin to request an invite."
        )

@handle_auth_error
def _attempt_login(email: str, password: str) -> tuple[bool, str]:
    """
    Attempts login via Supabase Auth.
    Returns (success, error_message_or_empty).

    On success: stores session in st.session_state.
    On failure: returns error message for display.
    """
    try:
        client = get_client()
        response = client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if not response.user:
            return False, "Invalid credentials"

        # Check if user is active
        admin = get_admin_client()
        profile = admin.table("profiles") \
            .select("role, is_active, display_name") \
            .eq("id", response.user.id) \
            .single() \
            .execute()

        if not profile.data:
            return False, "Profile not found"

        if not profile.data.get("is_active", False):
            return False, "Account is inactive. Contact the admin."

        # Store session in Streamlit state
        st.session_state.user_context = UserContext(
            user_id=response.user.id,
            email=response.user.email,
            role=profile.data.get("role", "user"),
            access_token=response.session.access_token,
            display_name=profile.data.get("display_name", "")
        )
        st.session_state.authenticated = True

        print(f"[Auth] Login: {email} ({profile.data.get('role')})")
        return True, ""

    except Exception as e:
        error_msg = str(e)
        # Clean up Supabase error messages for users
        if "Invalid login credentials" in error_msg:
            return False, "Invalid email or password"
        if "Email not confirmed" in error_msg:
            return False, "Please confirm your email first"
        print(f"[Auth] Login failed for {email}: {e}")
        return False, "Login failed. Please try again."


# ------------------------------------------------------------
# Session management
# ------------------------------------------------------------

def require_auth() -> UserContext | None:
    """
    Main auth gate — call this at the top of every page.

    Returns UserContext if authenticated.
    Renders login page and returns None if not.

    Usage in app.py:
        user = require_auth()
        if not user:
            st.stop()
        # rest of page renders here

    In production: also validates JWT expiry and
    refreshes token if within 5 minutes of expiry.
    """
    # Already authenticated this session
    if st.session_state.get("authenticated") and \
       st.session_state.get("user_context"):
        return st.session_state.user_context

    # Try to restore session from Supabase
    restored = _try_restore_session()
    if restored:
        return restored

    # No valid session — show login
    render_login_page()
    return None


def _try_restore_session() -> UserContext | None:
    """
    Attempts to restore a session from Supabase's
    persisted auth state.

    Handles token refresh automatically.
    Returns UserContext if session is valid, None otherwise.
    """
    try:
        client = get_client()
        session = client.auth.get_session()

        if not session or not session.user:
            return None

        admin = get_admin_client()
        profile = admin.table("profiles") \
            .select("role, is_active, display_name") \
            .eq("id", session.user.id) \
            .single() \
            .execute()

        if not profile.data or not profile.data.get("is_active"):
            return None

        context = UserContext(
            user_id=session.user.id,
            email=session.user.email,
            role=profile.data.get("role", "user"),
            access_token=session.access_token,
            display_name=profile.data.get("display_name", "")
        )

        st.session_state.user_context = context
        st.session_state.authenticated = True
        return context

    except Exception:
        return None


def logout():
    """
    Logs out the current user.
    Clears session state and Supabase session.
    """
    try:
        client = get_client()
        client.auth.sign_out()
    except Exception:
        pass

    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()