# notifications/gmail.py
# ------------------------------------------------------------
# Delivery layer — sends the daily brief via Gmail SMTP.
# Called by the memory_writer node at the end of the daily pipeline.
#
# Input:  DailyBrief object from state
# Output: Email in your inbox, returns delivery status
#
# In production: this becomes a proper notification service
# with templates, delivery tracking, open rates, and
# multi-channel support (Slack, push, SMS). The interface
# here stays identical — only the transport layer changes.
# ------------------------------------------------------------

import os
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# ------------------------------------------------------------
# Email template builder
#
# Builds both plain text and HTML versions of the email.
# Email clients use HTML when available, plain text as fallback.
#
# In production: templates live in a template engine (Jinja2)
# and are versioned separately from the delivery code.
# A/B testing happens at the template level.
# ------------------------------------------------------------

def _build_plain_text(brief) -> str:
    """
    Plain text fallback for email clients that don't render HTML.
    """
    connections = "\n".join([f"  • {c}" for c in brief.connections])
    questions = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(brief.discussion_questions)])

    return f"""
SAGE — Your Daily AI Learning Brief
{datetime.now().strftime("%A, %B %d %Y")}
{'='*50}

TODAY'S TOPIC: {brief.topic_title}

{brief.tldr}

EXPLANATION
{brief.explanation}

CONNECTIONS TO WHAT YOU KNOW
{connections}

WHY IT MATTERS
{brief.why_it_matters}

DISCUSSION QUESTIONS FOR YOUR SESSION
{questions}

SOURCE: {brief.source_url}

{'='*50}
Log in to discuss this topic with your Teacher Agent →
http://localhost:8501
""".strip()


def _build_html(brief) -> str:
    """
    HTML email template — clean, readable, minimal.

    In production: this becomes a proper responsive HTML email
    template with your brand, tracking pixels, and CTA buttons
    that deep-link into the app.
    """
    connections_html = "\n".join([
        f"<li style='margin-bottom:8px'>{c}</li>"
        for c in brief.connections
    ])

    questions_html = "\n".join([
        f"""<div style='background:#f8f9fa;border-left:3px solid
        #4A90D9;padding:12px 16px;margin-bottom:10px;
        border-radius:0 6px 6px 0;'>
        <span style='color:#4A90D9;font-weight:600'>Q{i+1}.</span>
        {q}</div>"""
        for i, q in enumerate(brief.discussion_questions)
    ])

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',
  sans-serif;max-width:680px;margin:0 auto;padding:20px;
  background:#ffffff;color:#1a1a1a;">

  <!-- Header -->
  <div style="border-bottom:2px solid #4A90D9;padding-bottom:16px;
    margin-bottom:24px;">
    <div style="color:#4A90D9;font-size:12px;font-weight:600;
      letter-spacing:2px;text-transform:uppercase;">SAGE</div>
    <div style="color:#666;font-size:13px;">
      {datetime.now().strftime("%A, %B %d %Y")}
    </div>
  </div>

  <!-- Topic title -->
  <h1 style="font-size:24px;font-weight:700;margin:0 0 8px 0;
    line-height:1.3;">{brief.topic_title}</h1>

  <!-- TL;DR -->
  <p style="font-size:15px;color:#444;line-height:1.6;
    border-left:3px solid #4A90D9;padding-left:14px;
    margin:16px 0 24px 0;">{brief.tldr}</p>

  <!-- Explanation -->
  <h2 style="font-size:16px;font-weight:600;color:#1a1a1a;
    margin:24px 0 12px 0;">What is it?</h2>
  <p style="font-size:15px;line-height:1.7;color:#333;
    margin:0 0 20px 0;">{brief.explanation}</p>

  <!-- Connections -->
  <h2 style="font-size:16px;font-weight:600;color:#1a1a1a;
    margin:24px 0 12px 0;">How it connects to what you know</h2>
  <ul style="font-size:14px;line-height:1.7;color:#333;
    padding-left:20px;margin:0 0 20px 0;">
    {connections_html}
  </ul>

  <!-- Why it matters -->
  <h2 style="font-size:16px;font-weight:600;color:#1a1a1a;
    margin:24px 0 12px 0;">Why it matters</h2>
  <p style="font-size:15px;line-height:1.7;color:#333;
    margin:0 0 20px 0;">{brief.why_it_matters}</p>

  <!-- Discussion questions -->
  <h2 style="font-size:16px;font-weight:600;color:#1a1a1a;
    margin:24px 0 12px 0;">
    Questions to explore in your session
  </h2>
  {questions_html}

  <!-- CTA -->
  <div style="text-align:center;margin:32px 0 24px 0;">
    <a href="http://localhost:8501"
      style="background:#4A90D9;color:#ffffff;padding:14px 32px;
      border-radius:6px;text-decoration:none;font-weight:600;
      font-size:15px;display:inline-block;">
      Start Learning →
    </a>
  </div>

  <!-- Source -->
  <div style="border-top:1px solid #eee;padding-top:16px;
    margin-top:24px;">
    <p style="font-size:12px;color:#999;margin:0;">
      Source: <a href="{brief.source_url}"
      style="color:#4A90D9;">{brief.source_url}</a>
    </p>
  </div>

</body>
</html>
""".strip()


# ------------------------------------------------------------
# Main delivery function
#
# Called directly by the memory_writer node — not a LangGraph
# node itself. Notifications are side effects of the pipeline,
# not reasoning steps. This distinction matters in production:
# side effects go in delivery services, reasoning goes in agents.
# ------------------------------------------------------------

def send_daily_brief(brief,recipient_email) -> bool:
    """
    Sends the daily brief email via Gmail SMTP.
    Returns True on success, False on failure.

    Called by: memory_writer node after brief is confirmed ready.

    In production: returns a delivery receipt object with
    message_id, timestamp, and delivery status for tracking.
    """
    sender = os.getenv("GMAIL_SENDER")
    password = os.getenv("GMAIL_APP_PASSWORD")
    recipient = recipient_email
    

    if not all([sender, password, recipient]):
        print("[Gmail] Missing credentials in .env — skipping email")
        return False

    try:
        # Build message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🧠 {brief.email_hook}"
        msg["From"] = f"Sage Learning Agent <{sender}>"
        msg["To"] = recipient

        # Attach both versions — client picks best it can render
        msg.attach(MIMEText(_build_plain_text(brief), "plain"))
        msg.attach(MIMEText(_build_html(brief), "html"))

        # Send via Gmail SMTP
        print("[Gmail] Connecting to SMTP...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())

        print(f"[Gmail] ✓ Brief sent to {recipient}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("[Gmail] Auth failed — check GMAIL_APP_PASSWORD in .env")
        return False

    except Exception as e:
        print(f"[Gmail] Send failed: {e}")
        return False