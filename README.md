# AI Phone Assistant

A private AI receptionist that answers phone calls, holds a natural conversation, remembers callers, and texts its owner a summary. Speech recognition, the language model and the voice all run **locally on your own GPU**: no AI API keys, no per-minute AI costs, and callers' words never leave your machine.

The assistant's name is configurable. The default is **Nova**.

---

## What a call sounds like

> **Nova:** Hi, you've reached the office of Simrat. I'm Nova, his AI assistant, and this call may be recorded. Am I speaking with Sara?
> **Caller:** Yes, you are.
> **Nova:** Simrat asked me to pass on a message: interview scheduled on Monday, 11am. What can I help you with?
> **Caller:** That's what I was calling about, thanks. Nothing else.
> **Nova:** Okay, just to confirm: Sara, about the interview schedule, and Simrat can call you back on this number. Did I get that right?
> **Caller:** Yes.
> **Nova:** Great, I'll pass that on to Simrat. Have a good one, Sara.

A few seconds later, the owner gets a text:

```
Call from Sara (+1548XXXXXXX)
Reason: the interview schedule
Urgency: normal
Callback: +1548XXXXXXX
Passed on: interview scheduled on Monday, 11am
```

---

## Features

- **Natural conversation.** Asks open questions, lets callers speak in any order, infers urgency instead of interrogating, and confirms the details before hanging up.
- **Caller memory.** Recognizes returning callers, refers to their previous calls, and notices when someone called earlier but hung up without leaving a message.
- **Notes by SMS.** The owner texts the assistant, for example `Note for Ahmed: interview moved to Monday. Share`, and the assistant uses it the next time that person calls.
- **Private and shareable notes.** Shareable notes are passed on word for word. Private notes are never shown to the AI at all and only appear in the owner's summary.
- **Interruptions (barge-in).** Callers can talk over the assistant and it stops to listen. Listening sounds like "mhm" don't count as interruptions.
- **Natural voice.** Kokoro text-to-speech, streamed sentence by sentence so replies start playing quickly.
- **SMS summaries** after every call, including missed calls, with urgent calls flagged first.
- **Full call recording** in stereo (caller on the left, assistant on the right) plus a saved transcript.

---

## How it works

```
Caller ──> Twilio number ──> POST /voice ──> TwiML: <Connect><Stream>
                                                       │
                       live audio over WebSocket (8 kHz mu-law, both directions)
                                                       │
                                                       ▼
┌──────────────────────────── FastAPI server (/media-stream) ─────────────────────────────┐
│                                                                                         │
│  audio ─> Silero VAD ─> faster-whisper ─> LLM (Ollama) ─> Kokoro TTS ─> audio back      │
│           (turn end)    (speech to text)  (JSON reply)    (streamed)                    │
│                                               │                                         │
│                      caller memory, notes, read-back, safety rules (code)               │
│                                               │                                         │
│                                SQLite: contacts, calls, transcripts, notes              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                                       │
                                     SMS summary to the owner (Twilio API)
```

Owner's notes arrive the other way: SMS to the Twilio number, `POST /sms`, parsed and saved to the database.

### Typical latency per turn

| Step | Time |
|---|---|
| Detecting the caller has finished (Silero VAD) | 0.8 s |
| Speech to text (Whisper, GPU) | ~0.3 s |
| LLM reply (Qwen 2.5 14B, GPU) | ~1.0 s |
| First audio of the reply (Kokoro, GPU, streamed) | ~0.2 s |

Measured on a laptop RTX 4080 (12 GB).

---

## Design decisions

**Code decides what must happen, the model decides how to say it.** The LLM handles wording and understanding. Anything that must always be right is enforced in code:

- The read-back of the caller's details is written by code, so it always contains every detail exactly.
- A call only counts as "completed" when the caller confirms the read-back, and a reply starting with "no", "wrong" or "actually" is never accepted as a confirmation, whatever the model says.
- Details are locked against mishearings until the read-back (so "soon" misheard as "Owen" can't overwrite a name), and corrections are allowed after it.
- The model returns structured JSON (enforced with a schema). Code validates it and never lets bad output crash a call.

**Caller ID is not trusted.** Phone numbers can be spoofed, so a known number only lets the assistant *ask* "Am I speaking with Sara?". Previous calls and shareable notes are only mentioned after the caller confirms.

**Private notes never reach the AI.** Telling a model "don't repeat this" works most of the time, not every time. So private notes are kept out of the prompt entirely and only appear in the owner's SMS.

**The AI and recording disclosure can't be interrupted.** The greeting always plays in full.

**Barge-in uses Twilio marks.** A named mark is sent after each reply, and Twilio echoes it back when the caller has heard the end. A `clear` message stops playback instantly when the caller cuts in, and the unheard audio is dropped from the recording too.

**Speaking runs in a background task.** Replies are synthesized and sent sentence by sentence while the main loop keeps listening, so an interruption can cancel the rest of a reply mid-way.

**Models load at startup, not at import.** All heavy loading happens in FastAPI's `lifespan`, so importing a module never touches the GPU.

---

## Tech stack

| Area | Tools |
|---|---|
| Server | Python 3.12, FastAPI, Uvicorn, WebSockets |
| Telephony | Twilio Voice (Media Streams), Twilio SMS |
| Voice activity detection | Silero VAD (energy-based detector as a fallback) |
| Speech to text | faster-whisper |
| LLM | Ollama, Qwen 2.5 14B, structured JSON output |
| Text to speech | Kokoro (Piper as a lightweight fallback) |
| Database | SQLite with SQLModel |
| Config | pydantic-settings, `.env` files |
| Local tunnel | ngrok |

---

## Getting started

### Prerequisites

- **Python 3.12**
- **An NVIDIA GPU with about 12 GB of memory** for everything on the GPU. With less, run text-to-speech on the CPU or use a smaller LLM (see Configure).
- **[Ollama](https://ollama.com)**
- **A Twilio account**, upgraded from trial (trial accounts block live audio streams), with a phone number that has Voice and SMS
- **[ngrok](https://ngrok.com)** or another HTTPS tunnel

### 1. Install

```bash
git clone https://github.com/simratdhill0n/ai-phone-assistant.git
cd ai-phone-assistant
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
```

Install the **CUDA build of PyTorch first**. A plain `pip install torch` usually gives you a CPU-only build. Get the exact command for your system from [pytorch.org](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then the rest:

```bash
pip install -r requirements.txt
pip install silero-vad --no-deps     # --no-deps so it can't replace your CUDA PyTorch
```

Check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Models

```bash
ollama pull qwen2.5:14b              # or qwen2.5:7b for smaller GPUs
hf download Systran/faster-whisper-small.en
hf download hexgrad/Kokoro-82M
```

Kokoro also needs **espeak-ng** for pronouncing unusual words. On Windows, install the `.msi` from the espeak-ng GitHub releases page.

Optional: so Ollama keeps the model loaded between calls instead of unloading it after 5 idle minutes:

```bash
setx OLLAMA_KEEP_ALIVE "30m"         # then restart Ollama
```

### 3. Configure

Copy `.env.example` to `.env.dev` and fill it in.

| Variable | What it is |
|---|---|
| `PUBLIC_HOST` | Your tunnel domain, no `https://` (e.g. `abc123.ngrok-free.dev`) |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | From the Twilio Console. Keep the token secret |
| `TWILIO_PHONE_NUMBER` | Your Twilio number, `+1` format |
| `OWNER_NAME`, `OWNER_PHONE` | Who the assistant works for. Only `OWNER_PHONE` can send notes |
| `OWNER_TIMEZONE` | e.g. `America/Toronto` |
| `ASSISTANT_NAME` | Default `Nova` |
| `DATABASE_URL` | Default `sqlite:///assistant.db` |
| `OLLAMA_MODEL` | Any model pulled with Ollama |
| `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE` | e.g. `small.en`, `cuda`, `float16` (CPU: `cpu`, `int8`) |
| `TTS_ENGINE` | `kokoro` or `piper` |
| `KOKORO_VOICE`, `KOKORO_SPEED`, `KOKORO_DEVICE` | e.g. `af_heart`, `1.0`, `cuda` (or `cpu`) |
| `PIPER_VOICE_PATH` | Only for `TTS_ENGINE=piper` |
| `VAD_ENGINE` | `silero` (default) or `energy` |

### 4. Run

```bash
uvicorn main:app --reload
```

In a second terminal:

```bash
ngrok http 8000
```

Wait for `Speech-to-text ready`, `Text-to-speech ready` and `LLM loaded and ready`.

### 5. Connect Twilio

On your number's **Configure** page in the Twilio Console:

- **Voice Configuration**, "A call comes in": Webhook, `https://<PUBLIC_HOST>/voice`, HTTP POST
- **Messaging Configuration**, "A message comes in": Webhook, `https://<PUBLIC_HOST>/sms`, HTTP POST

Every webhook is verified with Twilio's request signature, so `PUBLIC_HOST` must match the URL Twilio calls exactly.

Then call your Twilio number. To have the assistant answer calls to your personal number, set up call forwarding on your phone to the Twilio number.

---

## Leaving notes by SMS

Text the Twilio number from `OWNER_PHONE`:

```
Note for Ahmed: interview moved to Monday. Share
Note for Ahmed: recruiter at Shopify. Private
Note for Ahmed 519-555-0123: met at the conference
```

- Ending with `Share` makes a note shareable. Anything else is **private by default**.
- Include a number if the person hasn't called before, or if two contacts share a name.
- Each shareable note is passed on once, after the caller confirms who they are.

---

## Project structure

```
main.py             FastAPI app: webhooks, the live call loop, barge-in, streaming replies
llm.py              The conversation: prompts, structured output, read-back, confirmation rules
stt.py              Speech to text (faster-whisper)
tts.py              Text to speech (Kokoro or Piper), sentence splitting
vad.py              Voice activity detection (Silero or energy-based)
memory.py           Caller memory: greeting and history for returning callers
notes.py            Parsing the owner's SMS notes
db.py               Database tables and queries (SQLModel)
sms.py              SMS summaries to the owner
call_recorder.py    Stereo WAV recording of each call
twilio_security.py  Twilio webhook signature verification
config.py           Settings loaded from .env files
```

---

## Troubleshooting

- **Calls hang up right after connecting:** trial Twilio accounts don't support live audio streams. Upgrade the account.
- **Every webhook returns 403:** `PUBLIC_HOST` doesn't match the URL Twilio is calling, so the signature check fails.
- **Replies suddenly take 10+ seconds:** the GPU is out of memory and spilling into system RAM. Check `nvidia-smi` for a second Python process or other GPU-heavy apps, or set `KOKORO_DEVICE=cpu`.
- **Hugging Face downloads fail or time out:** retry, or download once and then set `HF_HUB_OFFLINE=1` so startup only uses cached files.
- **`torch.cuda.is_available()` is `False`:** you have the CPU build of PyTorch. Reinstall from the CUDA index (see Install).

---

## Privacy and responsible use

- Callers are always told they're speaking with an AI and that the call may be recorded. Check the recording laws where you operate.
- Recordings, transcripts and the database contain real people's personal information. They're git-ignored and stay on your machine. If you deploy this for anyone else, you need a privacy policy and secure storage (in Canada, PIPEDA applies).
- Never commit `.env` files. Rotate your Twilio Auth Token if it's ever exposed.

---

## Roadmap

- [x] Live call audio over Twilio Media Streams
- [x] Local speech to text, LLM and text to speech
- [x] Natural conversation with structured output and code-enforced confirmation
- [x] Caller memory and call history
- [x] Notes by SMS, private and shareable
- [x] Barge-in, Silero VAD, streamed replies
- [ ] Evaluation set for measuring conversation quality
- [ ] Call transfer with a whisper ("Sara is calling about the interview, press 1 to connect")
- [ ] Calendar integration
- [ ] AWS deployment

---

## Why I built this

Products like this exist, but building one end to end was the best way to learn real-time voice AI: audio streaming, turn-taking, running models on a GPU, keeping an LLM reliable, and handling personal data responsibly. It's the first module of a larger personal AI system.

## Author

**Simrat Pal Singh Dhillon**, Kitchener, Ontario
Full-stack developer (Python, Django, AWS) and AI postgraduate.