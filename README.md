# AI Phone Assistant

> Working name. Final name to be decided.

A personal AI receptionist that answers my calls, holds a real conversation with callers, remembers past interactions, and texts me a summary. Built with FastAPI, Twilio and LLMs.

**Status:** Work in progress. Currently building Stage 1.

---

## What it does

When someone calls, the assistant picks up, introduces itself as an AI, and has a natural conversation to find out:

- who is calling
- what they need
- how urgent it is
- the best way to call them back

It then texts me a short summary of the call.

### Planned features

- **Real conversation:** asks follow-up questions until it understands the caller's needs, instead of just recording a voicemail.
- **Caller memory:** recognizes returning callers and refers back to earlier calls. If a previous call was missed, it acknowledges that and asks how it can help.
- **Notes by SMS:** I can text the assistant notes about a person (for example, "Note for Ahmed: interview moved to Monday") so it picks up where things stand the next time they call.
- **Private and shareable notes:** private notes help the assistant understand context but are never repeated to callers. Shareable notes can be passed on.
- **Calendar actions:** checking availability and booking time through Google Calendar.
- **Live streaming audio:** low-latency, interruptible conversation over WebSockets.

---

## Privacy and safety by design

- **Transparent:** callers are told they are speaking with an AI and that calls may be recorded.
- **Caller ID is not trusted for sensitive info:** since phone numbers can be spoofed, the assistant may greet a known caller by name but never shares sensitive information based on the number alone.
- **Private notes stay private:** private notes inform the assistant's understanding and are never read back to callers.
- **Secrets are never committed:** credentials are loaded from environment variables locally and from AWS Secrets Manager in production.

---

## Tech stack

| Area | Tools |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Telephony | Twilio Voice (TwiML, `<Gather>`, Media Streams), Twilio SMS |
| AI | LLM for conversation and summaries |
| Data | DynamoDB (planned) |
| Local testing | ngrok |
| Deployment | AWS (Lambda for early stages, EC2 or ECS Fargate for streaming), AWS Secrets Manager |

---

## Roadmap

- [ ] **Stage 1: Greeting.** Twilio number calls a FastAPI endpoint that returns a TwiML greeting.
- [ ] **Stage 2: Conversation.** Turn-by-turn speech conversation with an LLM, plus an SMS summary after the call.
- [ ] **Stage 3: Caller memory.** Contacts and call history stored in a database.
- [ ] **Stage 4: Notes.** Add notes by SMS, tagged private or shareable.
- [ ] **Stage 5: Advanced.** Calendar tool calling, real-time audio streaming with barge-in, and AWS deployment.

---

## Architecture (current)

```
Caller ──> Twilio number ──> Voice webhook (HTTP POST)
                                   │
                                   ▼
                          ngrok (local tunnel)
                                   │
                                   ▼
                          FastAPI app ──> returns TwiML
```

My personal number forwards calls to the Twilio number, so the assistant can handle calls when I'm unavailable.

---

## Getting started

### Prerequisites

- Python 3.10+
- A Twilio account with a voice-enabled phone number
- ngrok

### Setup

```bash
git clone https://github.com/<your-username>/ai-phone-assistant.git
cd ai-phone-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values (needed from Stage 2 onward).

### Run locally

```bash
uvicorn main:app --reload
```

In a second terminal:

```bash
ngrok http 8000
```

### Connect Twilio

1. In the Twilio Console, open your phone number's configuration.
2. Under **Voice Configuration**, set "A call comes in" to **Webhook**.
3. Enter your ngrok URL followed by the voice route, for example `https://<your-ngrok-domain>/voice`, with method **HTTP POST**.
4. Save, then call the number.

---

## Why I'm building this

Similar products exist, but building one myself is a hands-on way to learn real-time voice AI, LLM orchestration, and production deployment on AWS. It is also the first module of a larger personal AI system I plan to build.

---

## Author

**Simrat Pal Singh Dhillon**, Kitchener, Ontario
Full-stack developer (Python, Django, AWS) and AI postgraduate.