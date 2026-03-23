# auth/admin.py
# ------------------------------------------------------------
# Admin CLI — manage users and access.
#
# Run these functions from terminal when someone requests
# access. Never expose these to the frontend.
#
# Usage:
#   python auth/admin.py invite someone@email.com "Their Name"
#   python auth/admin.py list
#   python auth/admin.py deactivate someone@email.com
#   python auth/admin.py reset_limits someone@email.com
#
# In production: this becomes an admin dashboard with
# audit logs, usage analytics, and bulk operations.
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from memory.supabase_client import get_admin_client
from dotenv import load_dotenv

load_dotenv()


def invite_user(email: str, display_name: str = "") -> bool:
    """
    Invites a new user via Supabase Auth invite flow.

    The user receives an email with a magic link.
    They click it, set a password, and their profile
    is automatically created via the DB trigger.

    Only invited users can access the app — no self-signup.
    """
    client = get_admin_client()

    try:
        # Send invite via Supabase Admin API
        response = client.auth.admin.invite_user_by_email(
            email,
            options={
                "data": {
                    "display_name": display_name or email.split("@")[0]
                }
            }
        )

        if response.user:
            print(f"✓ Invite sent to {email}")
            print(f"  User ID: {response.user.id}")
            print(f"  They will receive an email to set their password")
            return True
        else:
            print(f"✗ Failed to invite {email}")
            return False

    except Exception as e:
        print(f"✗ Invite failed for {email}: {e}")
        return False


def list_users() -> None:
    """Lists all users and their current status."""
    client = get_admin_client()

    try:
        response = client.table("profiles") \
            .select("email, display_name, role, is_active, created_at") \
            .order("created_at") \
            .execute()

        users = response.data or []

        if not users:
            print("No users found")
            return

        print(f"\n{'='*60}")
        print(f"{'Email':<30} {'Role':<10} {'Active':<8} {'Joined'}")
        print(f"{'='*60}")

        for u in users:
            joined = u.get("created_at", "")[:10]
            active = "✓" if u.get("is_active") else "✗"
            print(
                f"{u.get('email', ''):<30} "
                f"{u.get('role', 'user'):<10} "
                f"{active:<8} "
                f"{joined}"
            )

        print(f"{'='*60}")
        print(f"Total: {len(users)} users\n")

    except Exception as e:
        print(f"✗ List users failed: {e}")


def deactivate_user(email: str) -> bool:
    """
    Deactivates a user — they can no longer log in.
    Data is preserved. Can be reactivated later.
    """
    client = get_admin_client()

    try:
        response = client.table("profiles") \
            .update({"is_active": False}) \
            .eq("email", email) \
            .execute()

        if response.data:
            print(f"✓ Deactivated {email}")
            return True
        else:
            print(f"✗ User {email} not found")
            return False

    except Exception as e:
        print(f"✗ Deactivate failed: {e}")
        return False


def activate_user(email: str) -> bool:
    """Reactivates a previously deactivated user."""
    client = get_admin_client()

    try:
        response = client.table("profiles") \
            .update({"is_active": True}) \
            .eq("email", email) \
            .execute()

        if response.data:
            print(f"✓ Activated {email}")
            return True
        else:
            print(f"✗ User {email} not found")
            return False

    except Exception as e:
        print(f"✗ Activate failed: {e}")
        return False


def make_admin(email: str) -> bool:
    """Promotes a user to admin role."""
    client = get_admin_client()

    try:
        response = client.table("profiles") \
            .update({"role": "admin"}) \
            .eq("email", email) \
            .execute()

        if response.data:
            print(f"✓ {email} is now an admin")
            return True
        else:
            print(f"✗ User {email} not found")
            return False

    except Exception as e:
        print(f"✗ Make admin failed: {e}")
        return False


def reset_limits(email: str) -> bool:
    """
    Resets today's rate limits for a user.
    Use when a user hits limits and needs more quota.
    """
    client = get_admin_client()

    try:
        # Get user_id from email
        profile = client.table("profiles") \
            .select("id") \
            .eq("email", email) \
            .single() \
            .execute()

        if not profile.data:
            print(f"✗ User {email} not found")
            return False

        user_id = profile.data["id"]

        client.table("rate_limits") \
            .update({
                "llm_calls": 0,
                "tokens_used": 0,
                "tavily_calls": 0
            }) \
            .eq("user_id", user_id) \
            .eq("date", datetime.now().date().isoformat()) \
            .execute()

        print(f"✓ Rate limits reset for {email}")
        return True

    except Exception as e:
        print(f"✗ Reset limits failed: {e}")
        return False


def show_usage(email: str) -> None:
    """Shows today's usage stats for a specific user."""
    client = get_admin_client()

    try:
        profile = client.table("profiles") \
            .select("id, display_name") \
            .eq("email", email) \
            .single() \
            .execute()

        if not profile.data:
            print(f"✗ User {email} not found")
            return

        user_id = profile.data["id"]

        usage = client.table("rate_limits") \
            .select("*") \
            .eq("user_id", user_id) \
            .eq("date", datetime.now().date().isoformat()) \
            .execute()

        if not usage.data:
            print(f"No usage today for {email}")
            return

        u = usage.data[0]
        print(f"\nUsage today for {email}:")
        print(f"  LLM calls:    {u['llm_calls']}/{u['max_llm_calls']}")
        print(f"  Tokens:       {u['tokens_used']:,}/{u['max_tokens']:,}")
        print(f"  Tavily calls: {u['tavily_calls']}/{u['max_tavily_calls']}\n")

    except Exception as e:
        print(f"✗ Show usage failed: {e}")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sage Admin CLI")
    subparsers = parser.add_subparsers(dest="command")

    # invite
    invite_parser = subparsers.add_parser("invite", help="Invite a new user")
    invite_parser.add_argument("email")
    invite_parser.add_argument("name", nargs="?", default="")

    # list
    subparsers.add_parser("list", help="List all users")

    # deactivate
    deactivate_parser = subparsers.add_parser("deactivate", help="Deactivate a user")
    deactivate_parser.add_argument("email")

    # activate
    activate_parser = subparsers.add_parser("activate", help="Activate a user")
    activate_parser.add_argument("email")

    # make_admin
    admin_parser = subparsers.add_parser("make_admin", help="Make a user admin")
    admin_parser.add_argument("email")

    # reset_limits
    reset_parser = subparsers.add_parser("reset_limits", help="Reset rate limits")
    reset_parser.add_argument("email")

    # usage
    usage_parser = subparsers.add_parser("usage", help="Show usage for a user")
    usage_parser.add_argument("email")

    args = parser.parse_args()

    if args.command == "invite":
        invite_user(args.email, args.name)
    elif args.command == "list":
        list_users()
    elif args.command == "deactivate":
        deactivate_user(args.email)
    elif args.command == "activate":
        activate_user(args.email)
    elif args.command == "make_admin":
        make_admin(args.email)
    elif args.command == "reset_limits":
        reset_limits(args.email)
    elif args.command == "usage":
        show_usage(args.email)
    else:
        parser.print_help()