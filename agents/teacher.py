# agents/teacher.py
# ------------------------------------------------------------
# Teacher Agent — core of the chat session experience.
#
# Now includes Tavily web search tool.
# When a question requires current or specific information,
# the teacher searches before responding.
#
# Two functions exposed to graph:
#   build_system_prompt() — called once by session_loader
#   teacher_agent()       — called every turn by chat_pipeline
#
# In production: search results are cached by query hash
# (Redis, 1hr TTL) so repeated questions don't burn Tavily
# quota. Search calls are also rate-limited separately from
# LLM calls since they have different cost profiles.
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from typing import Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_core.tools import tool
from errors.handler import handle_agent_error, SearchError
from agents.guardrails import check_message

load_dotenv()




# ------------------------------------------------------------
# Tavily search
# ------------------------------------------------------------
# ------------------------------------------------------------
# Search tool definition
#
# Defined as a LangChain tool so Claude can decide when
# to call it based on conversation context.
#
# In production: tools are versioned and monitored.
# Each tool call is logged with input/output for debugging
# and to track Tavily quota usage per user.
# ------------------------------------------------------------

from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """
    Search the web for current information about an AI/ML topic.

    Use this tool when:
    - The user asks about recent developments, papers, or releases
    - You need specific technical details you are not confident about
    - The user asks to compare specific current models or benchmarks
    - The question involves something that may have changed recently
    - You need to cite a specific source or paper

    Do NOT use this tool when:
    - The question is a conceptual explanation you know well
    - The user is asking you to quiz them or ask questions
    - The answer is fundamental ML knowledge unlikely to have changed
    - You just used it and the results already cover the question

    Args:
        query: A specific, focused search query. Be precise.
               Good: 'sparse attention mechanism transformer 2025'
               Bad: 'tell me about attention'
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    try:
        print(f"[teacher] 🔍 Searching: '{query}'")

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=3,
            include_answer=True
        )

        results = response.get("results", [])
        tavily_answer = response.get("answer", "")

        if not results and not tavily_answer:
            return "No relevant results found."

        formatted = []

        if tavily_answer:
            formatted.append(f"Summary: {tavily_answer}\n")

        for i, r in enumerate(results, 1):
            formatted.append(
                f"Source {i}: {r.get('title', '')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Content: {r.get('content', '')[:500]}\n"
            )

        result_text = "\n".join(formatted)
        print(f"[teacher] Search returned {len(results)} results")
        return result_text

    except Exception as e:
        print(f"[teacher] Search failed: {e}")
        return f"Search failed: {e}"

# ------------------------------------------------------------
# System prompt builder
# Called ONCE by session_loader — never rebuilt per turn
# ------------------------------------------------------------

def build_system_prompt(
    topic_title: str,
    knowledge_snapshot: dict,
    topic_depth: Optional[dict],
    daily_brief: Optional[dict]
) -> str:
    """
    Builds the teacher system prompt from session context.
    Called once at session start by session_loader.
    Stored in state["system_prompt"].
    """
    known_topics = [
        f"- {e['topic']} (confidence: {e['confidence']})"
        for e in knowledge_snapshot.get("topics", [])
    ]
    knowledge_text = (
        "\n".join(known_topics)
        if known_topics
        else "No prior topics seeded yet."
    )

    depth_text = ""
    if topic_depth:
        known_subs = [
            f"  - {k['name']} ({k['confidence']} confidence)"
            for k in topic_depth.get("known", [])
        ]
        gaps = [
            f"  - {g}"
            for g in topic_depth.get("gaps", [])
        ]
        depth_text = f"""
WHAT THEY ALREADY KNOW ABOUT THIS TOPIC:
{chr(10).join(known_subs) if known_subs else "  - Nothing yet — fresh topic"}

KNOWN GAPS TO FILL IN THIS TOPIC:
{chr(10).join(gaps) if gaps else "  - None identified yet"}
"""

    brief_text = ""
    if daily_brief:
        explanation = daily_brief.get("explanation", "")
        if explanation:
            brief_text = f"""
TODAY'S BRIEF (what they read before this session):
{explanation}
"""

    return f"""You are Sage — a world-class personal AI tutor specializing in AI/ML.
You are in a one-on-one teaching session with an AI/ML engineer.

TODAY'S TOPIC: {topic_title}
{brief_text}
YOUR STUDENT'S FULL KNOWLEDGE MAP:
{knowledge_text}
{depth_text}
YOUR TEACHING PHILOSOPHY:
- Socratic first — ask questions before giving answers
- Connect everything to what they already know
- When they demonstrate understanding, explicitly acknowledge it
- When you detect a gap, probe it gently before explaining
- Keep explanations concrete — use code snippets or analogies
- Never talk down to them — they are an engineer, not a student
- Always build on existing knowledge rather than starting fresh

WEB SEARCH RESULTS:
When you see [WEB SEARCH RESULTS] in the conversation, it means
fresh information was retrieved from the web for this question.
Use it to ground your answer in current, accurate information.
Cite the source when you use specific facts from it.
If the results aren't relevant, ignore them and answer from knowledge.

SESSION RULES:
- Stay focused on today's topic and its natural connections
- If they go off-topic, acknowledge briefly and guide back
- End every response with a probing question or invitation to go deeper
- Track what they demonstrate understanding of during the session

ENDING THE SESSION:
When the user says 'end session', 'done', 'quit', 'exit',
or 'goodbye' — give a warm 2-3 sentence summary of what
was covered and what they demonstrated understanding of.
The system handles memory extraction after you say goodbye."""


# ------------------------------------------------------------
# Teacher Agent — LangGraph node
#
# Now with search-before-respond pattern.
# If the message triggers a search, results are injected
# as a system message before the LLM call.
# ------------------------------------------------------------
@handle_agent_error(
    fallback={
        "messages": [],
        "last_activity": ""
    },
    layer="agent"
)
def teacher_agent(state: dict) -> dict:
    """
    Teacher Agent — LangGraph node.

    Uses Claude's native tool calling to decide when
    to search. Claude calls search_web() if it needs
    current information, then responds grounded in results.

    Flow:
    1. Bind search_web tool to LLM
    2. Call LLM — it decides whether to use the tool
    3. If tool called — execute search, feed results back
    4. LLM produces final response
    5. Return AI response

    In production: tool calls are logged separately from
    LLM calls for granular cost tracking. Search results
    are cached by query hash to avoid duplicate Tavily calls.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.7
    )

    # Bind search tool to LLM
    # Claude will call it autonomously when it decides to
    llm_with_tools = llm.bind_tools([search_web])

    system_prompt = state.get(
        "system_prompt",
        "You are a helpful AI tutor. Teach clearly and ask good questions."
    )

    messages = state.get("messages", [])
    topic_title = state.get("topic_title", "")

    turn_num = len([
        m for m in messages if isinstance(m, HumanMessage)
    ])
    print(f"[teacher] Turn {turn_num} — calling LLM...")

    # ── Guardrail check ──────────────────────────────────────
    # Run before any LLM or search call
    # Blocks off-topic queries with a polite redirect
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None
    )
    if last_human:
        user_id = state.get("user_id")
        guard = check_message(last_human.content, user_id=user_id)

        if guard.blocked:
            print(f"[guardrails] Blocked turn {turn_num}")
            return {
                "messages": [AIMessage(content=guard.response)],
                "last_activity": datetime.now().isoformat()
            }

    try:
        # First LLM call — may or may not call search tool
        response = llm_with_tools.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])

        # Check if Claude decided to use the search tool
        tool_calls = getattr(response, "tool_calls", [])

        if tool_calls:
            print(f"[teacher] Claude requested {len(tool_calls)} search(es)")

            # Execute each tool call
            tool_results = []
            for tool_call in tool_calls:
                query = tool_call["args"].get("query", "")
                result = search_web.invoke({"query": query})

                # Format tool result as LangChain expects
                from langchain_core.messages import ToolMessage
                tool_results.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )

            # Second LLM call — now grounded in search results
            # Pass: system + history + first response + tool results
            final_response = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                *messages,
                response,           # Claude's first response with tool call
                *tool_results       # Search results
            ])

            print(f"[teacher] Grounded response — {len(final_response.content)} chars")
            return {
                "messages": [AIMessage(content=final_response.content)],
                "last_activity": datetime.now().isoformat()
            }

        # No tool called — direct response
        print(f"[teacher] Direct response — {len(response.content)} chars")
        return {
            "messages": [AIMessage(content=response.content)],
            "last_activity": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"[teacher] LLM call failed: {e}")
        return {
            "messages": [AIMessage(
                content="I ran into an issue. Could you rephrase that?"
            )],
            "last_activity": datetime.now().isoformat()
        }
