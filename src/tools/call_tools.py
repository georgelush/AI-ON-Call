"""
src/tools/call_tools.py - LangGraph tools for voice call sessions.

Tools used by itsm_agent (and future call agents) to verify identity,
collect information, and push results to n8n at call completion.

Session store (_sessions) is module-level so both tools and the agent
can read/write the same session dict via session_id (= Twilio call_sid).
"""

import os
import json
import logging
import requests
import redis
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis session store
# Session key: "call_session:{call_sid}"
# TTL: 2 hours — no call should last longer than this
# _messages (LangGraph history) stored separately in-memory per process
# ---------------------------------------------------------------------------

_SESSION_TTL = 7200  # 2 hours in seconds
_SESSION_PREFIX = "call_session:"

_redis: redis.Redis = redis.Redis.from_url(
    os.environ.get("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True,
)

# In-memory store for LangGraph message history (not JSON-serializable)
_messages_store: dict[str, list] = {}


def _session_key(session_id: str) -> str:
    return f"{_SESSION_PREFIX}{session_id}"


def get_session(session_id: str) -> dict:
    """Returns the session dict for a call. Raises if not found."""
    raw = _redis.get(_session_key(session_id))
    if raw is None:
        raise KeyError(f"No active session for call_sid={session_id}")
    return json.loads(raw)


def save_session(session_id: str, session: dict) -> None:
    """Persists the session dict back to Redis, refreshing TTL."""
    _redis.setex(_session_key(session_id), _SESSION_TTL, json.dumps(session))


def create_redis_session(session_id: str, data: dict) -> None:
    """Creates a new session in Redis. Called by itsm_agent.create_session()."""
    _redis.setex(_session_key(session_id), _SESSION_TTL, json.dumps(data))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def collect_das_code(code: str, session_id: str) -> str:
    """Records the User Auth Code (identity code) provided by the customer.
    Call this when the customer says their User Auth Code.
    The code is saved and sent to n8n for verification — do not ask the customer
    if the code is correct or incorrect."""
    session = get_session(session_id)
    session["das_code_received"] = code.strip()
    session["das_collected"] = True
    session["_das_ask"] = 0  # reset counter — User Auth Code step is done
    save_session(session_id, session)
    logger.info("User Auth Code collected for session %s", session_id)
    return "User Auth Code recorded. You may proceed."


@tool
def collect_note(note: str, eta: str, session_id: str) -> str:
    """Saves the work note and ETA provided by the customer into the call session.
    Call this once the customer has given their update and estimated resolution time."""
    session = get_session(session_id)
    session["work_note"] = note.strip()
    session["eta"] = eta.strip()
    session["_note_ask"] = 0  # reset counter — note step is done
    save_session(session_id, session)
    logger.info("Note collected for session %s", session_id)
    return "Work note and ETA saved successfully."


@tool
def complete_call(session_id: str) -> str:
    """Marks the call as completed and pushes the result to n8n.
    Call this when the conversation is fully done and all information has been collected."""
    session = get_session(session_id)
    session["status"] = "completed"
    save_session(session_id, session)
    _push_to_n8n(session_id, session)
    return "Call completed. Result sent to processing queue."


@tool
def escalate_to_human(reason: str, session_id: str) -> str:
    """Escalates the call to a human agent and notifies n8n.
    Call this when: the customer explicitly requests a human agent,
    or the conversation cannot continue."""
    session = get_session(session_id)
    session["status"] = "escalated"
    session["escalation_reason"] = reason
    save_session(session_id, session)
    _push_to_n8n(session_id, session)
    logger.info("Call escalated for session %s, reason: %s", session_id, reason)
    return "I'll transfer you to a human agent now. Please hold."


# ---------------------------------------------------------------------------
# Internal helper — not a tool
# ---------------------------------------------------------------------------

def _push_to_n8n(session_id: str, session: dict) -> None:
    """Sends call result payload to the n8n webhook. Fire-and-forget."""
    n8n_base = os.environ.get("N8N_INSTANCE_URL", "").rstrip("/")
    webhook_path = session.get("n8n_webhook_path", "")
    api_key = os.environ.get("N8N_API_KEY", "")

    if not webhook_path:
        logger.warning("n8n not configured — skipping push for session %s", session_id)
        return

    # If n8n_webhook_path is a full URL (e.g. webhook-test), use it directly
    if webhook_path.startswith("http"):
        url = webhook_path
    else:
        if not n8n_base:
            logger.warning("N8N_INSTANCE_URL not set — skipping push for session %s", session_id)
            return
        url = f"{n8n_base}/webhook/{webhook_path}"
    payload = {
        "call_sid": session_id,
        "ticket_id": session.get("ticket_id"),
        "status": session.get("status", "unknown"),
        "das_code_received": session.get("das_code_received"),
        "work_note": session.get("work_note"),
        "eta": session.get("eta"),
        "escalation_reason": session.get("escalation_reason"),
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-N8N-API-KEY"] = api_key

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info("n8n push OK for session %s — HTTP %s", session_id, response.status_code)
    except requests.RequestException as exc:
        logger.error("n8n push FAILED for session %s: %s", session_id, exc)


# Exported list for LangGraph bind_tools / ToolNode
CALL_TOOLS = [collect_das_code, collect_note, complete_call, escalate_to_human]
