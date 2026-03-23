# ui/access_request.py
# ------------------------------------------------------------
# Public access request form.
# Shown on the login page for visitors without an account.
#
# Anyone can submit a request — no auth required.
# Admin reviews and approves from the admin panel.
# Approved users get a Supabase invite email automatically.
# ------------------------------------------------------------

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import re
from memory.supabase_client import get_admin_client
from dotenv import load_dotenv

load_dotenv()


def _is_valid_email(email: str) -> bool:
    """Basic email format validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def _submit_request(
    name: str,
    email: str,
    note: str
) -> tuple[bool, str]:
    """
    Saves access request to Supabase.
    Returns (success, message).
    """
    client = get_admin_client()

    try:
        # Check if already requested
        existing = client.table("access_requests") \
            .select("status") \
            .eq("email", email) \
            .execute()

        if existing.data:
            status = existing.data[0]["status"]
            if status == "pending":
                return False, "You already have a pending request. We'll be in touch soon."
            elif status == "approved":
                return False, "Your request was already approved. Check your email for an invite."
            elif status == "rejected":
                return False, "Your request was not approved. Contact us directly if you think this is a mistake."

        # Save request
        client.table("access_requests").insert({
            "name": name,
            "email": email,
            "note": note
        }).execute()

        # Notify admin via Gmail
        _notify_admin(name, email, note)

        return True, ""

    except Exception as e:
        print(f"[access_request] Submit failed: {e}")
        return False, "Something went wrong. Please try again."


def _notify_admin(name: str, email: str, note: str) -> None:
    """
    Sends admin a notification email when
    a new access request comes in.
    """
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        sender = os.getenv("GMAIL_SENDER")
        password = os.getenv("GMAIL_APP_PASSWORD")
        admin_email = os.getenv("GMAIL_RECIPIENT")

        if not all([sender, password, admin_email]):
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🧠 Sage — New Access Request from {name}"
        msg["From"] = f"Sage <{sender}>"
        msg["To"] = admin_email

        body = f"""
New access request received:

Name:  {name}
Email: {email}
Note:  {note or 'No note provided'}

Log into Sage to approve or reject this request.
        """.strip()

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, admin_email, msg.as_string())

    except Exception as e:
        print(f"[access_request] Admin notification failed: {e}")


def render_access_request_form():
    """
    Renders the access request form.
    Called from the login page when user clicks
    'Request Access'.
    """
    st.markdown("### Request Access")
    st.markdown(
        "Sage is invite-only to manage API costs. "
        "Fill out the form below and I'll review your "
        "request within 48 hours."
    )
    st.markdown("")

    with st.form("access_request_form"):
        name = st.text_input(
            "Your name",
            placeholder="Jane Smith"
        )
        email = st.text_input(
            "Your email",
            placeholder="jane@example.com"
        )
        note = st.text_area(
            "Brief note about your AI/ML background",
            placeholder="e.g. ML engineer at a startup, "
                        "building agentic systems, "
                        "interested in LLMs and RAG...",
            height=100
        )
        submitted = st.form_submit_button(
            "Submit Request",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        # Validate
        if not name.strip():
            st.error("Please enter your name")
            return
        if not email.strip() or not _is_valid_email(email):
            st.error("Please enter a valid email address")
            return

        with st.spinner("Submitting request..."):
            success, error = _submit_request(
                name=name.strip(),
                email=email.strip(),
                note=note.strip()
            )

        if success:
            st.success(
                "✓ Request submitted! "
                "You'll receive an invite email within 48 hours."
            )
            st.balloons()
        else:
            st.error(error)