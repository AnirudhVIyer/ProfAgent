# 🧠 ProfAgent(Sage) — Personal AI Learning Agent

> An autonomous multi-agent system that learns what you know, finds what you don't, and teaches you every day.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)
![Supabase](https://img.shields.io/badge/Supabase-pgvector-orange)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

**[Live Demo](https://your-railway-url.up.railway.app)** • **[Request Access](#access)**

---

## What Is Sage?

Sage is a fully autonomous AI learning agent built for AI/ML engineers who want to stay at the frontier without spending hours manually researching.

Every day at 9am Sage:
1. Reads your personal knowledge graph
2. Searches the web for topics you don't know yet
3. Selects the single most relevant gap to fill today
4. Writes a personalized brief anchored to what you already know
5. Emails you when it's ready

When you log in, a Teacher Agent engages you in a Socratic conversation calibrated exactly to your knowledge level. After the session, a Memory Agent extracts what you learned and updates your knowledge graph. Tomorrow's pipeline wakes up smarter about you.

---

## Demo

### Daily Pipeline
```
[memory_loader]  Loading knowledge snapshot — 24 topics
[Researcher]     Running 6 targeted search queries...
[Curator]        Rule filter: 18 → 11 candidates
[Curator]        LLM selected: 'Mixture of Experts Architecture'
[Curator]        Why: You know transformers deeply but MoE is a gap...
[Briefing]       Fetching full source content...
[Briefing]       Brief written — personalized to your knowledge level
[Gmail]          ✓ Brief sent to your@email.com
[memory_writer]  Topic saved to knowledge graph
```

### Chat Session
```
Sage: You know transformers well — let's connect MoE to that.
      In a standard transformer, all parameters are active for
      every token. What do you think MoE changes about that?

You:  I guess it only activates some of the layers?

Sage: Close — it's not layers but experts. Each MoE layer has N
      expert FFN networks. A router decides which 2 activate per
      token. Why might that be more efficient than your current
      understanding of how FFN layers work?
```

---

## Architecture
```
┌──────────────────────────────────────────────────────────┐
│                    DAILY PIPELINE                         │
│                   (runs 9am UTC)                          │
│                                                           │
│  memory_loader → researcher → curator → briefing →       │
│  notifier → memory_writer                                 │
└──────────────────────────┬───────────────────────────────┘
                           ↕ Supabase pgvector
┌──────────────────────────┴───────────────────────────────┐
│                    CHAT PIPELINE                          │
│                   (runs on demand)                        │
│                                                           │
│  session_loader → [compress?] → teacher ⟲ →             │
│  memory_agent → session_memory_writer                     │
└──────────────────────────────────────────────────────────┘
```

### Agent Roster

| Agent | Job | LLM |
|-------|-----|-----|
| **Researcher** | Searches web via Tavily for new AI/ML topics | No |
| **Curator** | Semantically selects best topic for today | Yes |
| **Briefing** | Writes personalized explanation + questions | Yes |
| **Teacher** | Socratic tutor, decides when to search autonomously | Yes + Tools |
| **Memory** | Extracts structured knowledge from conversation | Yes |

### Key Design Decisions

**Typed shared state over raw dicts**
Every agent reads from and writes to a `TypedDict` state object. Agents never import each other — they communicate only through state. This means any agent can be swapped, tested, or replaced independently.

**Two-tier memory architecture**
LangGraph state is ephemeral — it dies when the graph run ends. A `memory_loader` node at graph start and `memory_writer` at graph end form explicit bookends that bridge ephemeral state and persistent Supabase storage. This is why knowledge survives across daily runs.

**Sub-concept granularity**
Not just "I know transformers" but "I know multi-head attention at high confidence, sparse attention is a gap." The Memory Agent extracts this from conversation transcripts after every session. The Teacher Agent reads it before every session.

**Agentic tool use over fixed routing**
The Teacher Agent decides when to call Tavily using Claude's native tool calling — not keyword matching or hardcoded rules. The tool docstring is the control surface for search frequency.

**Supabase over Pinecone**
Auth, vector storage, rate limiting, and session data in one platform. Row Level Security enforced at the DB level — not application level. One user literally cannot read another's data even if application code has a bug.

---

## Features

### 🔄 Autonomous Daily Pipeline
- Runs every day at 9am UTC for all active users
- Tavily web search biased toward your knowledge gaps
- LLM curator performs semantic gap analysis
- Briefing agent fetches full source content before writing
- Gmail notification with personalized HTML email

### 🎓 Socratic Teacher Agent
- Knows exactly what you know and what your gaps are
- Autonomously decides when to search the web mid-conversation
- System prompt built once per session — never rebuilt per turn
- Context compression for long sessions — nothing lost
- Full transcript saved to Supabase after every session

### 🧠 Compounding Knowledge Graph
- Topic-level confidence: low / medium / high
- Sub-concept granularity within each topic
- Gap tracking across sessions
- Cohere embeddings for semantic similarity search
- Graph grows and deepens with every session

### 🔐 Production Security
- Invite-only access — no self-signup possible
- Supabase Auth with JWT + Row Level Security
- Per-user daily rate limits on LLM + Tavily calls
- Three-layer input guardrails with jailbreak detection
- Centralized error logging to Supabase

### 👤 Per-User Onboarding
- Guided UI on first login — no CLI required
- Category selection → topic rating → sub-concept detail
- Immediately personalizes day-one recommendations

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Orchestration | LangGraph | Typed state, conditional routing, HITL |
| LLM | Claude Sonnet (Anthropic) | Best instruction following |
| Web Search | Tavily API | Clean API, no scraping |
| Embeddings | Cohere embed-english-v3.0 | Free tier, 1024 dims |
| Database | Supabase Postgres + pgvector | Auth + vectors in one |
| Auth | Supabase Auth | Invite-only, JWT, RLS |
| UI | Streamlit | Fast iteration |
| Deployment | Railway | Git push deploys, Docker |
| Email | Gmail SMTP | Simple, reliable |
| Scheduling | APScheduler | In-process, zero ops |

---

## Project Structure
```
sage/
├── agents/
│   ├── researcher.py       # Tavily retrieval + topic scoring
│   ├── curator.py          # LLM gap analysis + selection
│   ├── briefing.py         # Personalized brief writer
│   ├── teacher.py          # Socratic tutor + tool use
│   └── guardrails.py       # Input validation + jailbreak detection
├── graph/
│   ├── daily_pipeline.py   # Scheduled LangGraph DAG
│   └── chat_pipeline.py    # Interactive LangGraph DAG + compression
├── memory/
│   ├── knowledge_store.py  # Supabase pgvector interface
│   └── supabase_client.py  # Connection singleton + embeddings
├── auth/
│   ├── middleware.py        # Login + session management
│   ├── rate_limiter.py      # Per-user quota enforcement
│   └── admin.py            # Invite users, manage access
├── errors/
│   └── handler.py          # Centralized error handling + logging
├── notifications/
│   └── gmail.py            # HTML email delivery
├── onboarding/
│   └── seed.py             # CLI knowledge seeding (local dev)
├── ui/
│   ├── app.py              # Main Streamlit app
│   └── onboarding.py       # First-time user onboarding flow
├── Dockerfile
├── railway.toml
├── requirements.txt
└── .env.example
```

---

## Access

Sage is invite-only to manage API costs responsibly.

**To request access:**
1. Open an issue titled **"Access Request"**
2. Include a brief note on your AI/ML background
3. I'll send a Supabase invite within 48 hours

Once you accept the invite and set your password, you'll be walked through a 3-minute onboarding flow that maps your existing knowledge before your first session.

---

## Running Locally

**1. Clone and install**
```bash
git clone https://github.com/AnirudhVIyer/ProfAgent.git
cd sage
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Environment variables**
```bash
cp .env.example .env
# Add your API keys to .env
```

**3. Supabase setup**
- Create a project at [supabase.com](https://supabase.com)
- Run the full schema from the SQL Editor:
```sql
-- Enable pgvector
create extension if not exists vector;
-- (see full schema in supabase/schema.sql)
```

**4. Run**
```bash
streamlit run ui/app.py
```

The onboarding flow will guide you through seeding your knowledge on first login.

---

## Environment Variables
```bash
# .env.example
ANTHROPIC_API_KEY=          # claude.ai API key
TAVILY_API_KEY=             # tavily.com API key
SUPABASE_URL=               # https://xxxx.supabase.co
SUPABASE_ANON_KEY=          # anon/public key
SUPABASE_SERVICE_KEY=       # service role key (backend only)
COHERE_API_KEY=             # cohere.com API key
GMAIL_SENDER=               # gmail address for sending
GMAIL_APP_PASSWORD=         # gmail app password
GMAIL_RECIPIENT=            # where to send daily briefs
```

---

## Inviting Users (Admin)
```bash
# Invite someone
python auth/admin.py invite someone@email.com "Their Name"

# List all users
python auth/admin.py list

# Deactivate a user
python auth/admin.py deactivate someone@email.com

# Check usage
python auth/admin.py usage someone@email.com

# Reset daily limits
python auth/admin.py reset_limits someone@email.com
```

---

## Roadmap

- [ ] LLM-powered Researcher Agent with dynamic query generation
- [ ] Knowledge graph visualization (D3.js)
- [ ] Spaced repetition — revisit low-confidence topics
- [ ] LangSmith tracing integration
- [ ] Multi-topic chat sessions
- [ ] Brief history browser
- [ ] Admin analytics dashboard
- [ ] Mobile responsive UI

---

## Built With

Architected and designed by me. Implemented with Claude (Anthropic) as a coding assistant. All system design decisions, architectural choices, and technical direction are my own.

---

## License

MIT