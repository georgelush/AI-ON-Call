## On-Call Voice Agent

Automates outbound IT support calls triggered by ServiceNow incidents.

### How it works
1. **n8n** detects a new ServiceNow ticket and calls `POST /calls/outbound`
2. The agent dials the customer via **Twilio ConversationRelay**
3. A **LangGraph ReAct agent** conducts the call in 4 steps:
   - Greet and collect the customer's 6-digit DAS identity code
   - Offer a ticket summary (priority, description, assignment group)
   - Collect a work note and ETA from the customer
   - Confirm and close the call
4. All collected data is pushed back to **n8n → ServiceNow**

### STT options
| Mode | Endpoint | Speech-to-text |
|---|---|---|
| ConversationRelay | `/calls/ws/{sid}` | Twilio built-in STT |
| Whisper | `/calls-whisper/ws/{sid}` | OpenAI Whisper via LiteLLM proxy |

### Stack
- **LangGraph** — ReAct agent graph (tool calling loop)
- **Twilio** — outbound dialing, ConversationRelay, Media Streams
- **Redis** — per-call session persistence (TTL 2h)
- **n8n** — orchestration + ServiceNow DAS validation
- **Langfuse** — LLM call tracing and observability
- **FastAPI** — REST API server (port 8080)
- **Gradio Studio** — local debug UI (port 8000)

### Security
- Bearer token auth on all protected endpoints
- Prompt injection detection in customer speech
- System prompt and internal data never disclosed to caller
- Call hard limit: 3 minutes
