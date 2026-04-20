"""
src/routers/whisper_router.py - Whisper STT variant of the call agent.

Uses Twilio Media Streams (inbound audio) + Twilio REST TTS (Say verb) for output:
  - STT: raw mulaw 8kHz -> WAV 16kHz -> Whisper (latency in console logs)
  - LLM: same servicenow_agent
  - TTS: Twilio built-in <Say> via REST call update — no TTS model needed on proxy

Flow per turn:
  1. Customer speaks -> Whisper STT -> LLM reply
  2. REST API: calls.update(twiml=<Say>reply</Say><Connect><Stream.../>)
  3. Twilio speaks reply, then reconnects stream -> next turn continues

Session keyed by call_sid (Redis) — survives stream reconnects.

Endpoints:
  POST /calls-whisper/outbound  — same fields as /calls/outbound
  POST /calls-whisper/voice     — TwiML: <Connect><Stream inbound_track>
  WS   /calls-whisper/ws/{sid} — Whisper STT loop
  POST /calls-whisper/status    — Twilio status callback
"""

import asyncio
import audioop
import base64
import hashlib
import io
import json
import logging
import os
import time
import wave

import requests
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, HTTPException, Security
from fastapi.responses import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from xml.etree.ElementTree import Element, SubElement, tostring
from src.agents import servicenow_agent
from src.agents.servicenow_agent import CALL_TIMEOUT_SIGNAL
from src.config import LLM_API_KEY, LLM_PROXY, langfuse_client
from langfuse.types import TraceContext

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calls-whisper", tags=["calls-whisper"])

_bearer = HTTPBearer()

# Tracks call_sids currently doing TTS injection — stream stop is NOT a real disconnect
_tts_injecting: set[str] = set()

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
_MULAW_RATE = 8000          # Twilio Media Streams: mulaw 8kHz
_WHISPER_RATE = 16000       # Whisper expects 16kHz
_CHUNK_MS = 20              # Twilio sends 20ms chunks
_SILENCE_RMS = 450          # RMS below this = silence
_SILENCE_CHUNKS = 35        # 35 x 20ms = 700ms silence → end of speech
_SPEECH_MIN_CHUNKS = 10     # 10 x 20ms = 200ms minimum speech to process


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _verify_bearer(credentials: HTTPAuthorizationCredentials = Security(_bearer)) -> None:
    expected = os.environ.get("CALLS_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="CALLS_API_KEY not configured on server")
    if credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")


# ---------------------------------------------------------------------------
# Request schema — identical to /calls/outbound
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
# Audio helpers
# ---------------------------------------------------------------------------

def _mulaw_to_wav(chunks: list[bytes]) -> bytes:
    """Convert list of mulaw 8kHz chunks -> WAV 16kHz bytes for Whisper."""
    raw = b"".join(chunks)
    pcm_8k = audioop.ulaw2lin(raw, 2)
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, _MULAW_RATE, _WHISPER_RATE, None)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_WHISPER_RATE)
        wf.writeframes(pcm_16k)
    return buf.getvalue()


# TODO (Whisper streaming upgrade): This function is file-based — it sends the full WAV after VAD detects
# end-of-speech, adding ~1.2–1.8s fixed STT latency per turn.
# When the proxy exposes a real-time transcription endpoint (/v1/audio/transcriptions with streaming),
# replace this function with: _whisper_stream_chunk(chunk) called per audio chunk + _whisper_finalize()
# to commit the transcript. STT latency drops to ~0.1–0.3s, on par with ConversationRelay.
def _whisper_transcribe(wav_bytes: bytes, trace_id: str) -> str:
    """
    Send WAV to Whisper via LiteLLM proxy.
    Records STT latency as a Langfuse span so it appears in the trace timeline.
    """
    t0 = time.time()
    try:
        url = LLM_PROXY.rstrip("/") + "/audio/transcriptions"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            files={"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")},
            data={"model": WHISPER_MODEL},
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        latency = time.time() - t0
        logger.info("WHISPER (%.2fs): %s", latency, text[:100])
        # Record STT span in Langfuse — visible in trace timeline alongside LLM spans
        try:
            langfuse_client.create_event(
                trace_context=TraceContext(trace_id=trace_id),
                name="whisper-stt",
                input={"audio_bytes": len(wav_bytes)},
                output={"transcript": text[:200]},
                metadata={"latency_s": round(latency, 3), "model": WHISPER_MODEL},
            )
        except Exception:
            pass  # Langfuse span is best-effort — never block STT
        return text
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        return ""


def _tts_via_twilio_say(call_sid: str, text: str, ws_url: str, final: bool = False) -> None:
    """Inject TTS into active call via Twilio REST API using <Say> verb.
    final=False: Twilio reconnects to WebSocket after speaking — conversation continues.
    final=True:  Twilio hangs up after speaking — no reconnect, no Error 31901."""
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    # Escape XML special characters in agent reply
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    if final:
        twiml = f'<Response><Say>{safe}</Say><Hangup/></Response>'
    else:
        twiml = (
            f'<Response>'
            f'<Say>{safe}</Say>'
            f'<Connect><Stream url="{ws_url}"/></Connect>'
            f'</Response>'
        )
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json",
            data={"Twiml": twiml},
            auth=(account_sid, auth_token),
            timeout=10,
        )
        if r.status_code not in (200, 202):
            logger.warning("TTS REST update returned %s: %s", r.status_code, r.text[:200])
        else:
            logger.info("TTS injected via <Say> for call_sid=%s final=%s", call_sid[-6:], final)
    except Exception as exc:
        logger.error("TTS REST inject failed: %s", exc)


# ---------------------------------------------------------------------------
# POST /calls-whisper/outbound
# ---------------------------------------------------------------------------

@router.post("/outbound")
async def outbound_call(req: OutboundCallRequest, _: None = Security(_verify_bearer)):
    """Initiates an outbound call via Twilio using Whisper STT."""
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    public_base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

    if not account_sid or not auth_token:
        raise HTTPException(status_code=500, detail="Twilio credentials not configured")
    if not public_base_url:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL not configured")

    voice_url = f"{public_base_url}/calls-whisper/voice"
    status_url = f"{public_base_url}/calls-whisper/status"

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
            twilio_url, data=payload, auth=(account_sid, auth_token), timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Twilio call initiation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Twilio error: {exc}")

    call_sid = response.json().get("sid", "")

    servicenow_agent.create_session(call_sid, {
        "ticket_id": req.ticket_id,
        "n8n_webhook_path": req.n8n_webhook_path,
        "short_description": req.short_description,
        "customer_name": req.customer_name,
        "assignment_group": req.assignment_group,
        "priority": req.priority,
        "stt_mode": "whisper",
    })

    logger.info("Whisper call initiated: call_sid=%s to=%s", call_sid, req.to_number)
    return {"status": "call_initiated", "call_sid": call_sid, "stt_mode": "whisper"}


# ---------------------------------------------------------------------------
# POST /calls-whisper/voice
# ---------------------------------------------------------------------------

@router.post("/voice")
async def voice_webhook(request: Request):
    """Returns TwiML instructing Twilio to open a bidirectional Media Stream."""
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")

    public_base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    ws_url = public_base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/calls-whisper/ws/{call_sid}"

    response_el = Element("Response")
    connect_el = SubElement(response_el, "Connect")
    stream_el = SubElement(connect_el, "Stream")
    stream_el.set("url", ws_url)
    stream_el.set("track", "inbound_track")  # receive customer audio only

    twiml = tostring(response_el, encoding="unicode", xml_declaration=False)
    logger.info("Whisper TwiML sent for call_sid=%s ws_url=%s", call_sid, ws_url)
    return Response(content=twiml, media_type="application/xml")


# ---------------------------------------------------------------------------
# WS /calls-whisper/ws/{call_sid} — Media Streams conversation loop
# ---------------------------------------------------------------------------

@router.websocket("/ws/{call_sid}")
async def whisper_websocket(websocket: WebSocket, call_sid: str):
    """
    Handles Twilio Media Streams audio:
      inbound audio -> buffer -> VAD -> Whisper STT -> LLM -> Twilio <Say> TTS
    Stream reconnects automatically after each <Say> — conversation continues per turn.
    """
    await websocket.accept()
    logger.info("Whisper WebSocket connected for call_sid=%s", call_sid)

    public_base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    ws_url = public_base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/calls-whisper/ws/{call_sid}"

    audio_buffer: list[bytes] = []
    is_speaking = False
    silence_chunks = 0
    speech_chunks = 0

    # On first connection: generate opening greeting and inject via <Say>
    # On reconnection after TTS: session already has messages, skip greeting
    from src.tools.call_tools import _messages_store
    is_new_session = len(_messages_store.get(call_sid, [])) <= 1

    if is_new_session:
        try:
            opening = await servicenow_agent.process_turn(call_sid, "__start__")
        except KeyError:
            logger.warning("No session for call_sid=%s on Whisper WebSocket", call_sid)
            await websocket.close()
            return

        if opening and opening != CALL_TIMEOUT_SIGNAL:
            is_final = servicenow_agent.should_close_call(call_sid)
            _tts_injecting.add(call_sid)
            await asyncio.get_event_loop().run_in_executor(
                None, _tts_via_twilio_say, call_sid, opening, ws_url, is_final
            )
            # Stream will stop and reconnect after <Say> — handler returns here
            await websocket.close()
            return

    try:
        async for raw_msg in websocket.iter_text():
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            event = msg.get("event", "")

            if event == "start":
                logger.info("Media stream started: call_sid=%s", call_sid[-6:])

            elif event == "media":
                if msg.get("media", {}).get("track") == "outbound":
                    continue

                chunk = base64.b64decode(msg["media"]["payload"])
                rms = audioop.rms(audioop.ulaw2lin(chunk, 2), 2)

                if rms > _SILENCE_RMS:
                    if not is_speaking:
                        is_speaking = True
                        speech_chunks = 0
                    silence_chunks = 0
                    speech_chunks += 1
                    audio_buffer.append(chunk)  # TODO (Whisper streaming): replace with _whisper_stream_chunk(chunk) — no buffer needed
                else:
                    if is_speaking:
                        silence_chunks += 1
                        audio_buffer.append(chunk)  # TODO (Whisper streaming): same — stream chunk directly, VAD only needed to trigger _whisper_finalize()

                        if silence_chunks >= _SILENCE_CHUNKS and speech_chunks >= _SPEECH_MIN_CHUNKS:
                            is_speaking = False
                            utterance = audio_buffer.copy()
                            audio_buffer.clear()
                            speech_chunks = 0
                            silence_chunks = 0

                            wav = _mulaw_to_wav(utterance)
                            # Pass MD5 trace_id so Langfuse event links to correct trace
                            stt_trace_id = hashlib.md5(call_sid.encode()).hexdigest()
                            t_turn_start = time.time()
                            transcript = await asyncio.get_event_loop().run_in_executor(
                                None, _whisper_transcribe, wav, stt_trace_id
                            )
                            if not transcript:
                                continue

                            stt_latency = time.time() - t_turn_start
                            logger.info(">>> CUSTOMER [%s]: %s", call_sid[-6:], transcript)
                            t_llm_start = time.time()
                            reply = await servicenow_agent.process_turn(call_sid, transcript)
                            llm_latency = time.time() - t_llm_start
                            total_latency = time.time() - t_turn_start

                            if reply == CALL_TIMEOUT_SIGNAL:
                                await websocket.close()
                                servicenow_agent.flush_session(call_sid)
                                return

                            if reply:
                                logger.info("TURN LATENCY [%s] stt=%.2fs llm=%.2fs total=%.2fs",
                                            call_sid[-6:], stt_latency, llm_latency, total_latency)
                                # Log total turn latency to Langfuse
                                try:
                                    langfuse_client.create_event(
                                        trace_context=TraceContext(trace_id=stt_trace_id),
                                        name="turn-latency",
                                        input={"customer": transcript},
                                        output={"agent": reply[:300]},
                                        metadata={
                                            "total_s": round(total_latency, 3),
                                            "stt_s": round(stt_latency, 3),
                                            "llm_s": round(llm_latency, 3),
                                            "mode": "whisper",
                                        },
                                    )
                                except Exception:
                                    pass
                                # Check if call is done BEFORE injecting TTS so we can
                                # send <Hangup/> instead of <Stream> — avoids Error 31901
                                is_final = servicenow_agent.should_close_call(call_sid)
                                _tts_injecting.add(call_sid)
                                await asyncio.get_event_loop().run_in_executor(
                                    None, _tts_via_twilio_say, call_sid, reply, ws_url, is_final
                                )
                                if is_final:
                                    servicenow_agent.flush_session(call_sid)
                                await websocket.close()
                                return

                            if servicenow_agent.should_close_call(call_sid):
                                await asyncio.sleep(8)
                                await websocket.close()
                                servicenow_agent.flush_session(call_sid)
                                return

            elif event == "stop":
                if call_sid in _tts_injecting:
                    # Normal — stream stopped because we injected TTS, it will reconnect
                    _tts_injecting.discard(call_sid)
                else:
                    # Customer hung up
                    logger.info("Media stream stopped for call_sid=%s", call_sid[-6:])
                    await servicenow_agent.handle_disconnect(call_sid)
                return

    except WebSocketDisconnect:
        if call_sid not in _tts_injecting:
            logger.info("Whisper WebSocket disconnected for call_sid=%s", call_sid[-6:])
            await servicenow_agent.handle_disconnect(call_sid)
        _tts_injecting.discard(call_sid)


# ---------------------------------------------------------------------------
# POST /calls-whisper/status
# ---------------------------------------------------------------------------

@router.post("/status")
async def whisper_status_callback(request: Request):
    """Receives Twilio call status events."""
    form = await request.form()
    call_sid = form.get("CallSid", "")
    call_status = form.get("CallStatus", "")
    logger.info("Whisper call status: call_sid=%s status=%s", call_sid, call_status)
    return Response(content="", status_code=204)
