# memory/supabase_client.py
# ------------------------------------------------------------
# Single Supabase connection point for the entire application.
# Every module imports the clients from here.
# Never create Supabase clients anywhere else.
#
# Two clients:
#   supabase_client  — uses anon key, respects RLS
#                      used for user-facing operations
#   admin_client     — uses service key, bypasses RLS
#                      used for backend/scheduler operations
#
# In production: connection pooling is handled by Supabase's
# PgBouncer automatically. For very high traffic you'd add
# a local connection pool (asyncpg + PgBouncer).
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# Validate environment variables at import time
# Fail fast — better to crash on startup than mid-request
# ------------------------------------------------------------

_required_vars = [
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_KEY"
]

_missing = [v for v in _required_vars if not os.getenv(v)]
if _missing:
    raise EnvironmentError(
        f"Missing required Supabase environment variables: {_missing}\n"
        f"Check your .env file."
    )

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


# ------------------------------------------------------------
# Client factory
# Lazy initialization — clients created on first import
# ------------------------------------------------------------

_anon_client: Client | None = None
_admin_client: Client | None = None


def get_client() -> Client:
    """
    Returns the anon client — respects Row Level Security.

    Use for: any operation tied to a logged-in user.
    The user's JWT token should be set on this client
    after login so RLS policies apply correctly.

    In production: this client is instantiated per-request
    with the user's JWT injected so RLS is enforced properly.
    """
    global _anon_client
    if _anon_client is None:
        _anon_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _anon_client


def get_admin_client() -> Client:
    """
    Returns the service role client — bypasses Row Level Security.

    Use for:
    - Daily pipeline (runs without a logged-in user)
    - Memory writes after sessions
    - Admin operations (inviting users, resetting limits)
    - Any backend operation that needs cross-user access

    NEVER expose this client to the frontend.
    NEVER use this for user-facing queries.

    In production: this client runs only in the backend
    worker process, never in the UI process.
    """
    global _admin_client
    if _admin_client is None:
        _admin_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _admin_client


def get_authenticated_client(access_token: str) -> Client:
    """
    Returns an anon client with a user's JWT injected.
    RLS policies will apply as that specific user.

    Called by: UI layer after successful login.
    Ensures users can only read/write their own data
    even if they somehow bypass the application layer.

    In production: token refresh is handled here too —
    check expiry, refresh if needed, return valid client.
    """
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.auth.set_session(
        access_token=access_token,
        refresh_token=""
    )
    return client


# ------------------------------------------------------------
# Health check — called on startup to verify connection
# ------------------------------------------------------------

def check_connection() -> bool:
    """
    Verifies Supabase connection is working.
    Called once at application startup.

    In production: this feeds into a health check endpoint
    that your load balancer pings to verify the service is up.
    """
    try:
        admin = get_admin_client()
        # Simple query to verify connection
        admin.table("profiles").select("id").limit(1).execute()
        print("[Supabase] ✓ Connection verified")
        return True
    except Exception as e:
        print(f"[Supabase] ✗ Connection failed: {e}")
        return False


# ------------------------------------------------------------
# Embedding helper
# Generates vectors for semantic similarity search
#
# Using Supabase's built-in embedding via OpenAI compatible
# endpoint — or fall back to a simple hash if not configured.
#
# In production: use a dedicated embedding model.
# Cohere embed-english-v3.0 is cost-effective and good.
# Anthropic doesn't have an embedding API yet so we use
# either OpenAI text-embedding-3-small or Cohere.
# ------------------------------------------------------------

def generate_embedding(text: str) -> list[float] | None:
    """
    Generates a 1024-dim embedding for semantic search.

    Called by: knowledge_store.py when adding new topics
    so the Curator Agent can do semantic similarity search.

    Returns None if embedding generation fails —
    caller should handle gracefully (store without embedding,
    fall back to string matching).

    In production: batched embedding calls for efficiency.
    Cache embeddings in Redis to avoid re-computing.
    """
    try:
        # Try OpenAI embeddings if key is available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from openai import OpenAI
            openai_client = OpenAI(api_key=openai_key)
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1024
            )
            return response.data[0].embedding

        # Try Cohere if key is available
        cohere_key = os.getenv("COHERE_API_KEY")
        if cohere_key:
            import cohere
            co = cohere.Client(cohere_key)
            response = co.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings[0]

        # No embedding provider configured
        print("[Supabase] No embedding provider configured — storing without vector")
        return None

    except Exception as e:
        print(f"[Supabase] Embedding generation failed: {e}")
        return None