"""
src/routers/calls_router.py - Twilio voice call protocol handler.

Manages the full lifecycle of an outbound voice call:
  POST /calls/outbound   — triggered by n8n, dials the customer via Twilio
  POST /calls/voice      — Twilio webhook, returns TwiML with ConversationRelay
  WS   /calls/ws/{sid}  — WebSocket conversation loop (one per active call)
  POST /calls/status     — Twilio status callback (no-answer, busy, failed, etc.)

This router handles protocol only. All business logic lives in servicenow_agent.py.
To add a new agent type (e.g. pizza), add it to _AGENT_REGISTRY below.
"""

import asyncio
import os
import json
import hashlib
import logging
import time
import requests
from xml.etree.ElementTree import Element, SubElement, tostring

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from pydantic import BaseModel

from src.agents import servicenow_agent
from src.agents.servicenow_agent import CALL_TIMEOUT_SIGNAL
from src.config import langfuse_client
from langfuse.types import TraceContext

logger = logging.getLogger(__name__)


def _hangup_via_rest(call_sid: str) -> None:
    """Explicitly end the call via Twilio REST API.
    More reliable than just closing the WebSocket — Twilio sees the call as completed."""
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    if not account_sid or not auth_token:
        return
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json",
            data={"Status": "completed"},
            auth=(account_sid, auth_token),
            timeout=10,
        )
        logger.info("Hangup REST call_sid=%s status=%s", call_sid[-6:], r.status_code)
    except Exception as exc:
        logger.error("Hangup REST failed: %s", exc)

router = APIRouter(prefix="/calls", tags=["calls"])

_bearer = HTTPBearer()


def _verify_bearer(credentials: HTTPAuthorizationCredentials = Security(_bearer)) -> None:
    """Validates the Bearer token on protected endpoints."""
    expected = os.environ.get("CALLS_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="CALLS_API_KEY not configured on server")
    if credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

# ---------------------------------------------------------------------------
# Agent registry — maps agent_type string to agent module
# Add new agent types here without touching any other file
# ---------------------------------------------------------------------------
_AGENT_REGISTRY = {
    "servicenow": servicenow_agent,
}

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class OutboundCallRequest(BaseModel):
    to_number: str
    from_number: str
    agent_type: str
    ticket_id: str
    n8n_webhook_path: str
    short_description: str | None = None
    customer_name: str | None = None
    assignment_group: str | None = None
    priority: str | None = None


# ---------------------------------------------------------------------------
# POST /calls/outbound — n8n triggers this to initiate a call
# ---------------------------------------------------------------------------

@router.post("/outbound")
async def outbound_call(req: OutboundCallRequest, _: None = Security(_verify_bearer)):
    """Initiates an outbound call via Twilio and creates the agent session."""

    agent = _AGENT_REGISTRY.get(req.agent_type)
    if agent is None:
        raise HTTPException(status_code=400, detail=f"Unknown agent_type: {req.agent_type}")

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    public_base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

    if not account_sid or not auth_token:
        raise HTTPException(status_code=500, detail="Twilio credentials not configured")
    if not public_base_url:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL not configured")

    # Build TwiML voice URL and status callback URL
    voice_url = f"{public_base_url}/calls/voice"
    status_url = f"{public_base_url}/calls/status"

    # Place the call via Twilio REST API
    twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json"
    payload = {
        "To": req.to_number,
        "From": req.from_number,
        "Url": voice_url,
        "StatusCallback": status_url,
        "StatusCallbackMethod": "POST",
        "StatusCallbackEvent": "completed",
    }

    try:
        response = requests.post(
            twilio_url,
            data=payload,
            auth=(account_sid, auth_token),
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Twilio call initiation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Twilio error: {exc}")

    call_data = response.json()
    call_sid = call_data.get("sid", "")

    # Pre-create the agent session so it's ready when the WebSocket connects
    agent.create_session(call_sid, {
        "ticket_id": req.ticket_id,
        "n8n_webhook_path": req.n8n_webhook_path,
        "short_description": req.short_description,
        "customer_name": req.customer_name,
        "assignment_group": req.assignment_group,
        "priority": req.priority,
        "stt_mode": "conversation_relay",
    })

    logger.info("Call initiated: call_sid=%s to=%s agent=%s", call_sid, req.to_number, req.agent_type)
    return {"status": "call_initiated", "call_sid": call_sid}


# ---------------------------------------------------------------------------
# POST /calls/voice — Twilio webhook, customer answered, return TwiML
# ---------------------------------------------------------------------------

@router.post("/voice")
async def voice_webhook(request: Request):
    """Returns TwiML that instructs Twilio to open a ConversationRelay WebSocket."""
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")

    public_base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    # wss:// required — ConversationRelay only accepts secure WebSocket
    ws_url = public_base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/calls/ws/{call_sid}"

    # Build TwiML response
    response_el = Element("Response")
    connect_el = SubElement(response_el, "Connect")
    relay_el = SubElement(connect_el, "ConversationRelay")
    relay_el.set("url", ws_url)
    # welcomeGreeting omitted — agent sends first message via WebSocket text event
    relay_el.set("ttsProvider", "google")
    relay_el.set("voice", "en-US-Neural2-F")
    relay_el.set("transcriptionProvider", "google")
    relay_el.set("speechModel", "telephony")
    relay_el.set("interruptByDtmf", "false")
    relay_el.set("speechTimeout", "600")

    twiml = tostring(response_el, encoding="unicode", xml_declaration=False)
    logger.info("TwiML sent for call_sid=%s ws_url=%s", call_sid, ws_url)
    return Response(content=twiml, media_type="application/xml")


# ---------------------------------------------------------------------------
# WS /calls/ws/{call_sid} — ConversationRelay conversation loop
# ---------------------------------------------------------------------------

@router.websocket("/ws/{call_sid}")
async def conversation_websocket(websocket: WebSocket, call_sid: str):
    """Handles the real-time conversation between Twilio and the agent."""
    await websocket.accept()
    logger.info("WebSocket connected for call_sid=%s", call_sid)

    # Determine which agent to use based on session data
    # Default to itsm agent if session not found or agent_type missing
    agent = servicenow_agent

    try:
        # Send the opening greeting — agent generates first message
        try:
            opening = await agent.process_turn(call_sid, "__start__")
            if opening and opening != CALL_TIMEOUT_SIGNAL:
                words = opening.split()
                for i, word in enumerate(words):
                    token = word if i == len(words) - 1 else word + " "
                    await websocket.send_text(json.dumps({
                        "type": "text", "token": token, "last": i == len(words) - 1,
                    }))
        except KeyError:
            logger.warning("No session found for call_sid=%s on WebSocket connect", call_sid)
            await websocket.close()
            return

        # Main conversation loop
        async for raw_message in websocket.iter_text():
            try:
                msg = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from Twilio for call_sid=%s", call_sid)
                continue

            msg_type = msg.get("type", "")

            if msg_type == "prompt":
                # Customer spoke — process the utterance
                customer_text = msg.get("voicePrompt", "").strip()
                if not customer_text:
                    customer_text = "__silence__"

                t_turn_start = time.time()
                reply = await agent.process_turn(call_sid, customer_text)
                turn_latency = time.time() - t_turn_start

                if reply == CALL_TIMEOUT_SIGNAL:
                    # Hard timeout — close immediately without sending text
                    logger.info("Hard timeout signal received for call_sid=%s", call_sid)
                    await websocket.close()
                    agent.flush_session(call_sid)
                    return

                if reply:
                    logger.info("TURN LATENCY [%s] llm_only=%.2fs", call_sid[-6:], turn_latency)
                    # Log turn latency to Langfuse as event on this call's trace
                    try:
                        trace_id = hashlib.md5(call_sid.encode()).hexdigest()
                        langfuse_client.create_event(
                            trace_context=TraceContext(trace_id=trace_id),
                            name="turn-latency",
                            input={"customer": customer_text},
                            output={"agent": reply[:300]},
                            metadata={
                                "total_s": round(turn_latency, 3),
                                "stt_s": 0,  # Twilio handles STT — not measurable
                                "llm_s": round(turn_latency, 3),
                                "mode": "conversation_relay",
                            },
                        )
                    except Exception:
                        pass
                    # Stream word by word — TTS starts on first word, reducing perceived latency
                    words = reply.split()
                    for i, word in enumerate(words):
                        token = word if i == len(words) - 1 else word + " "
                        await websocket.send_text(json.dumps({
                            "type": "text", "token": token, "last": i == len(words) - 1,
                        }))

                # Close call if agent reached a terminal state (completed, escalated, timeout)
                if agent.should_close_call(call_sid):
                    logger.info("Call reached terminal state, hanging up call_sid=%s", call_sid)
                    # Small delay so Twilio finishes speaking the queued goodbye TTS
                    await asyncio.sleep(6)
                    await asyncio.get_event_loop().run_in_executor(
                        None, _hangup_via_rest, call_sid
                    )
                    await websocket.close()
                    agent.flush_session(call_sid)
                    return

            elif msg_type == "disconnect":
                # Customer hung up
                reason = msg.get("reason", "unknown")
                logger.info("Customer disconnected call_sid=%s reason=%s", call_sid, reason)
                await agent.handle_disconnect(call_sid)
                return

            elif msg_type == "dtmf":
                # Ignore DTMF tones (keypad presses)
                pass

            else:
                logger.debug("Unhandled WS message type=%s call_sid=%s", msg_type, call_sid)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for call_sid=%s", call_sid)
        await agent.handle_disconnect(call_sid)


# ---------------------------------------------------------------------------
# POST /calls/status — Twilio status callback (no-answer, busy, failed, etc.)
# ---------------------------------------------------------------------------

@router.post("/status")
async def call_status_callback(request: Request):
    """Receives Twilio call status events and notifies n8n for non-connected calls."""
    form = await request.form()
    call_sid = form.get("CallSid", "")
    call_status = form.get("CallStatus", "")
    to_number = form.get("To", "")

    logger.info("Call status: call_sid=%s status=%s to=%s", call_sid, call_status, to_number)

    # Only handle terminal non-connected statuses here
    # "completed" is handled by handle_disconnect or complete_call tool
    if call_status in ("no-answer", "busy", "failed", "canceled"):
        _notify_n8n_no_connection(call_sid, call_status, to_number)

    return Response(content="", status_code=204)


def _notify_n8n_no_connection(call_sid: str, status: str, to_number: str) -> None:
    """Pushes a no-connection result to n8n when the call was never answered."""
    n8n_base = os.environ.get("N8N_INSTANCE_URL", "").rstrip("/")
    api_key = os.environ.get("N8N_API_KEY", "")

    # Try to get webhook path from session if it exists; fall back to default
    webhook_path = "snow-call-result"
    try:
        from src.tools.call_tools import get_session
        session = get_session(call_sid)
        webhook_path = session.get("n8n_webhook_path", webhook_path)
    except KeyError:
        pass

    if not n8n_base:
        logger.warning("n8n not configured — skipping status push for call_sid=%s", call_sid)
        return

    url = f"{n8n_base}/webhook/{webhook_path}"
    payload = {
        "call_sid": call_sid,
        "status": status,
        "to_number": to_number,
        "das_code_received": None,
        "work_note": None,
        "eta": None,
        "escalation_reason": None,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-N8N-API-KEY"] = api_key

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        logger.info("n8n notified for no-connection call_sid=%s status=%s", call_sid, status)
    except requests.RequestException as exc:
        logger.error("n8n notify failed for call_sid=%s: %s", call_sid, exc)
