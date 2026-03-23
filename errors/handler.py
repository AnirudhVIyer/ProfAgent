# errors/handler.py
# ------------------------------------------------------------
# Centralized error handling for the entire application.
#
# Three things live here:
#   1. Custom exception types — typed errors for each layer
#   2. Error logger — writes to Supabase + console
#   3. Decorators — wrap agents and functions cleanly
#
# Usage:
#   from errors.handler import handle_agent_error, AgentError
#
#   @handle_agent_error(fallback={"chosen_topic": None})
#   def curator_agent(state):
#       ...
#
# In production: error logs feed into an alerting system
# (PagerDuty, Slack webhook) for critical failures.
# Non-critical errors are batched and reviewed daily.
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
import functools
from datetime import datetime
from typing import Any, Callable, Optional
from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------------------------
# Custom exception types
#
# Typed exceptions let you catch specific failure modes
# without catching everything. Each layer has its own type
# so you always know where the error originated.
#
# In production: each exception type maps to a severity
# level and alerting rule. AgentError is warning-level,
# AuthError is critical, RateLimitError is info-level.
# ------------------------------------------------------------

class SageError(Exception):
    """Base exception for all Sage errors."""
    def __init__(self, message: str, user_id: str = None,
                 context: dict = None):
        super().__init__(message)
        self.user_id = user_id
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()


class AgentError(SageError):
    """
    Raised when an agent fails to complete its task.
    Examples: LLM call fails, Pydantic validation fails,
    unexpected response format.
    """
    pass


class SearchError(SageError):
    """
    Raised when Tavily search fails.
    Non-critical — agents should fall back to training data.
    """
    pass


class MemoryError(SageError):
    """
    Raised when knowledge store read/write fails.
    Critical — data may be lost if not handled.
    """
    pass


class PipelineError(SageError):
    """
    Raised when a graph pipeline fails to complete.
    Examples: node execution failure, state corruption.
    """
    pass


class AuthError(SageError):
    """
    Raised for authentication and authorization failures.
    Critical — security-relevant errors.
    """
    pass


class RateLimitError(SageError):
    """
    Raised when a user exceeds their daily quota.
    Expected — not a bug, just a limit being hit.
    """
    pass


# ------------------------------------------------------------
# Error logger
#
# Writes errors to both console and Supabase errors table.
# Supabase logging is best-effort — if it fails we still
# log to console so nothing is silently swallowed.
#
# In production: also sends to Sentry, Datadog, or
# whatever observability platform you use.
# ------------------------------------------------------------

# Error severity levels
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITY_CRITICAL = "critical"

# Map exception types to severity
SEVERITY_MAP = {
    RateLimitError: SEVERITY_INFO,
    SearchError: SEVERITY_WARNING,
    AgentError: SEVERITY_WARNING,
    MemoryError: SEVERITY_ERROR,
    PipelineError: SEVERITY_ERROR,
    AuthError: SEVERITY_CRITICAL,
    SageError: SEVERITY_WARNING,
    Exception: SEVERITY_ERROR
}


def log_error(
    error: Exception,
    layer: str,
    function_name: str,
    user_id: str = None,
    context: dict = None,
    severity: str = None
) -> None:
    """
    Logs an error to console and Supabase.

    Args:
        error:         The exception that was raised
        layer:         Which layer failed (agent/graph/memory/auth/ui)
        function_name: Which function failed
        user_id:       Which user was affected (if known)
        context:       Additional context dict for debugging
        severity:      Override auto-detected severity

    In production: critical errors trigger immediate Slack
    alert. Errors are grouped by type for daily review.
    """
    # Auto-detect severity from exception type
    if severity is None:
        for exc_type, sev in SEVERITY_MAP.items():
            if isinstance(error, exc_type):
                severity = sev
                break
        severity = severity or SEVERITY_ERROR

    # Build error record
    error_record = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity,
        "layer": layer,
        "function": function_name,
        "error_type": type(error).__name__,
        "message": str(error),
        "user_id": user_id,
        "context": context or {},
        "traceback": traceback.format_exc()
    }

    # Always log to console
    _log_to_console(error_record)

    # Try to log to Supabase — best effort
    _log_to_supabase(error_record)


def _log_to_console(record: dict) -> None:
    """Formats and prints error to console."""
    severity_icons = {
        SEVERITY_INFO: "ℹ️",
        SEVERITY_WARNING: "⚠️",
        SEVERITY_ERROR: "❌",
        SEVERITY_CRITICAL: "🚨"
    }
    icon = severity_icons.get(record["severity"], "❌")

    print(
        f"\n{icon} [{record['severity'].upper()}] "
        f"{record['layer']}.{record['function']}\n"
        f"   Type: {record['error_type']}\n"
        f"   Message: {record['message']}\n"
        f"   User: {record.get('user_id', 'unknown')}\n"
        f"   Time: {record['timestamp']}\n"
    )

    # Only print traceback for errors and critical
    if record["severity"] in [SEVERITY_ERROR, SEVERITY_CRITICAL]:
        print(f"   Traceback:\n{record['traceback']}")


def _log_to_supabase(record: dict) -> None:
    """
    Writes error to Supabase error_logs table.
    Best-effort — never raises, never blocks.
    """
    try:
        from memory.supabase_client import get_admin_client
        client = get_admin_client()

        client.table("error_logs").insert({
            "severity": record["severity"],
            "layer": record["layer"],
            "function_name": record["function"],
            "error_type": record["error_type"],
            "message": record["message"],
            "user_id": record.get("user_id"),
            "context": record.get("context", {}),
            "traceback": record.get("traceback", ""),
            "created_at": record["timestamp"]
        }).execute()

    except Exception:
        # Silently ignore — logging should never crash the app
        pass


# ------------------------------------------------------------
# Decorators
#
# Clean way to add error handling to any function.
# Wrap once — consistent behavior everywhere.
#
# Three decorators for three use cases:
#   @handle_agent_error    — for LangGraph agent nodes
#   @handle_memory_error   — for knowledge store functions
#   @handle_pipeline_error — for graph pipeline functions
# ------------------------------------------------------------

def handle_agent_error(
    fallback: Any = None,
    layer: str = "agent"
) -> Callable:
    """
    Decorator for LangGraph agent nodes.

    On failure:
    - Logs the error with full context
    - Returns fallback dict so pipeline continues
    - Never raises — keeps the graph running

    Usage:
        @handle_agent_error(fallback={"chosen_topic": None})
        def curator_agent(state: dict) -> dict:
            ...

    In production: fallback values are carefully chosen
    so downstream nodes handle None gracefully. Each agent
    documents what its fallback means for the pipeline.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user_id from state if available
            user_id = None
            if args and isinstance(args[0], dict):
                user_id = args[0].get("user_id")

            try:
                return func(*args, **kwargs)

            except Exception as e:
                log_error(
                    error=e,
                    layer=layer,
                    function_name=func.__name__,
                    user_id=user_id,
                    context={
                        "args_type": str(type(args[0]))
                        if args else "none"
                    }
                )
                # Return fallback so pipeline continues
                return fallback if fallback is not None else {}

        return wrapper
    return decorator


def handle_memory_error(
    fallback: Any = None,
    layer: str = "memory"
) -> Callable:
    """
    Decorator for knowledge store functions.

    Same as handle_agent_error but with memory-specific
    severity — memory failures are more critical because
    data may be lost.

    Usage:
        @handle_memory_error(fallback=None)
        def get_topic_depth(topic, user_id):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id")

            try:
                return func(*args, **kwargs)

            except Exception as e:
                log_error(
                    error=MemoryError(str(e)),
                    layer=layer,
                    function_name=func.__name__,
                    user_id=user_id,
                    severity=SEVERITY_ERROR
                )
                return fallback

        return wrapper
    return decorator


def handle_pipeline_error(
    fallback: Any = None,
    layer: str = "pipeline"
) -> Callable:
    """
    Decorator for graph pipeline functions.

    Pipeline errors are critical — they affect the entire
    daily run for a user. Logged at error level.

    Usage:
        @handle_pipeline_error(fallback={})
        def memory_writer(state):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = None
            if args and isinstance(args[0], dict):
                user_id = args[0].get("user_id")

            try:
                return func(*args, **kwargs)

            except Exception as e:
                log_error(
                    error=PipelineError(str(e)),
                    layer=layer,
                    function_name=func.__name__,
                    user_id=user_id,
                    severity=SEVERITY_ERROR
                )
                return fallback if fallback is not None else {}

        return wrapper
    return decorator


def handle_auth_error(func: Callable) -> Callable:
    """
    Decorator for auth functions.
    Auth errors are critical — logged immediately.
    Returns None on failure so UI shows login page.

    Usage:
        @handle_auth_error
        def _attempt_login(email, password):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error(
                error=AuthError(str(e)),
                layer="auth",
                function_name=func.__name__,
                severity=SEVERITY_CRITICAL
            )
            return None
    return wrapper