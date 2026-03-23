# main.py
# ------------------------------------------------------------
# Application entry point.
# Starts APScheduler for daily pipeline before Streamlit.
#
# In production at scale: scheduler runs as a separate
# Railway service so it doesn't share resources with the UI.
# For a personal project, same container is fine.
# ------------------------------------------------------------

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def start_scheduler():
    """
    Starts APScheduler to run daily pipeline at 9am UTC.
    Runs in background thread — doesn't block Streamlit.

    In production: replaced by Railway's cron jobs or
    an external scheduler like Inngest or Trigger.dev.
    These are more reliable than in-process schedulers
    because they survive container restarts.
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from memory.supabase_client import get_admin_client

    scheduler = BackgroundScheduler()

    def run_pipeline_for_all_users():
        """
        Runs daily pipeline for every active user.
        Called by scheduler at 9am UTC daily.
        """
        print(f"\n[Scheduler] Daily run starting — {datetime.now()}")

        try:
            from graph.daily_pipeline import run_daily_pipeline

            # Get all active users
            client = get_admin_client()
            profiles = client.table("profiles") \
                .select("id, email, role") \
                .eq("is_active", True) \
                .execute()

            users = profiles.data or []
            print(f"[Scheduler] Running for {len(users)} users")

            for user in users:
                try:
                    print(f"[Scheduler] Running for {user['email']}...")
                    run_daily_pipeline(
                        user_id=user["id"],
                        is_admin=user.get("role") == "admin"
                    )
                    print(f"[Scheduler] ✓ Done for {user['email']}")

                except Exception as e:
                    print(f"[Scheduler] ✗ Failed for {user['email']}: {e}")
                    continue

            print(f"[Scheduler] Daily run complete — {datetime.now()}\n")

        except Exception as e:
            print(f"[Scheduler] Fatal error in daily run: {e}")

    # Schedule daily at 9am UTC
    scheduler.add_job(
        run_pipeline_for_all_users,
        trigger=CronTrigger(hour=9, minute=0),
        id="daily_pipeline",
        name="Daily Learning Pipeline",
        replace_existing=True
    )

    scheduler.start()
    print(f"[Scheduler] Started — daily pipeline runs at 9am UTC")
    return scheduler


# Start scheduler when app launches
_scheduler = None

try:
    _scheduler = start_scheduler()
except Exception as e:
    print(f"[Scheduler] Failed to start: {e}")