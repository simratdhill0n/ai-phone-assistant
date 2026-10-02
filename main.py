import asyncio
import audioop
import base64
import json
import re
import time
from contextlib import asynccontextmanager
from html import escape

from fastapi import Depends, FastAPI, Request, Response, WebSocket, WebSocketDisconnect

from call_recorder import CallRecorder
from config import settings
from db import init_db, save_call, utcnow
from llm import Conversation, warm_up
from memory import build_caller_context
from notes import handle_owner_sms
from sms import notify_owner
from stt import transcribe, warm_up_stt
from tts import synthesize
from twilio_security import verify_twilio
from vad import VoiceActivityDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code before "yield" runs once at startup, code after it at shutdown
    init_db()
    await asyncio.to_thread(warm_up_stt)
    print("Speech-to-text ready.")
    try:
        await warm_up()
        print("LLM loaded and ready.")
    except Exception as e:
        print(f"Could not warm up the LLM (is Ollama running?): {e}")
    yield


app = FastAPI(lifespan=lifespan)

# 20 ms of phone audio: 160 samples * 2 bytes = 320 bytes of 16-bit PCM
FRAME_BYTES_PCM = 320
# 8000 samples/s * 2 bytes per sample
PCM_BYTES_PER_SECOND = 16000

GREETING = (
    f"Hi, you've reached the office of {settings.owner_name}. "
    f"I'm {settings.assistant_name}, his AI assistant, and this call may be recorded. "
    f"{settings.owner_name} is unavailable right now, but I can take a message. "
    "May I have your name and the reason for your call?"
)


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/voice", dependencies=[Depends(verify_twilio)])
async def voice_endpoint(request: Request):
    # Twilio's webhook includes the caller's number in the "From" field
    form = await request.form()
    caller_number = form.get("From", "")

    # First dynamic value inside our XML: escape it, so characters like
    # & or < can never break the TwiML.
    caller_xml = escape(caller_number, quote=True)

    # No <Say> anymore: the assistant greets the caller in its own voice.
    # When our server closes the stream, the TwiML ends and Twilio hangs up.
    # <Parameter> passes values into the stream's "start" event.
    twiml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Connect>
                <Stream statusCallback="https://{settings.public_host}/stream-status" url="wss://{settings.public_host}/media-stream">
                    <Parameter name="caller_number" value="{caller_xml}" />
                </Stream>
            </Connect>
        </Response>
        """
    return Response(content=twiml_content, media_type="application/xml")


async def send_audio(websocket: WebSocket, stream_sid: str, pcm: bytes, recorder: CallRecorder):
    """Send 8 kHz 16-bit PCM audio to the caller: PCM -> mu-law -> base64."""
    recorder.add_assistant_audio(pcm)

    for i in range(0, len(pcm), FRAME_BYTES_PCM):
        chunk = pcm[i:i + FRAME_BYTES_PCM]
        mulaw = audioop.lin2ulaw(chunk, 2)
        payload = base64.b64encode(mulaw).decode("ascii")
        await websocket.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        })


async def say(websocket: WebSocket, stream_sid: str, text: str, recorder: CallRecorder) -> float:
    """Speak text to the caller. Returns how long the audio lasts, in seconds."""
    speech = await asyncio.to_thread(synthesize, text)
    await send_audio(websocket, stream_sid, speech, recorder)
    print(f"Assistant said: {text}")
    return len(speech) / PCM_BYTES_PER_SECOND


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("Twilio Media Stream connected.")

    recorder = None
    vad = None
    conversation = None
    call_sid = "unknown_call"
    stream_sid = None
    caller_number = ""
    started_at = utcnow()
    stt_hints = ""
    speaking_until = 0.0   # time.monotonic() when the assistant's audio ends

    try:
        while True:
            message_text = await websocket.receive_text()
            packet = json.loads(message_text)
            event = packet.get("event")

            if event == "start":
                start_data = packet.get("start", {})
                call_sid = start_data.get("callSid", "stream")
                stream_sid = start_data.get("streamSid")
                # Values we passed with <Parameter> arrive in customParameters
                caller_number = start_data.get("customParameters", {}).get("caller_number", "")
                recorder = CallRecorder(call_sid)
                vad = VoiceActivityDetector()

                # Caller memory: look up this number's history (database = blocking)
                greeting, caller_context, known_name = await asyncio.to_thread(
                    build_caller_context, caller_number, GREETING
                )
                conversation = Conversation(greeting, caller_number, caller_context)

                # Names Whisper should expect on this call
                hint_names = [settings.owner_name, settings.assistant_name]
                if known_name:
                    hint_names.append(known_name)
                stt_hints = ", ".join(hint_names)
                print(f"Call from {caller_number or 'unknown number'}")
                if caller_context:
                    print(f"Known caller.{caller_context}")
                print(f"Recording started. Saving to {recorder.path}")

                duration = await say(websocket, stream_sid, greeting, recorder)
                speaking_until = time.monotonic() + duration

            elif event == "media":
                payload = packet.get("media", {}).get("payload")
                if not (payload and recorder):
                    continue

                mulaw_data = base64.b64decode(payload)
                pcm_data = audioop.ulaw2lin(mulaw_data, 2)
                recorder.add_caller_audio(pcm_data)

                utterance = vad.process(pcm_data)
                if not utterance:
                    continue

                # Did the caller start talking while the assistant was still
                # speaking? (A "mhm" or "yeah" over our audio.) Without
                # barge-in support yet, we ignore those.
                spoke_at = time.monotonic() - len(utterance) / PCM_BYTES_PER_SECOND
                if spoke_at < speaking_until:
                    print("Ignored speech that overlapped the assistant.")
                    continue

                # 1. Speech to text
                t0 = time.perf_counter()
                text = await asyncio.to_thread(transcribe, utterance, stt_hints)
                t1 = time.perf_counter()
                # Whisper "hears" things in noise, like ". . ." or "Thank you."
                # No letters or digits at all means nothing real was said.
                if not re.search(r"[A-Za-z0-9]", text):
                    print(f"Ignored non-speech transcript: {text!r}")
                    continue
                print(f"Caller said: {text}")

                # 2. LLM decides the reply
                reply, end_call = await conversation.reply(text)
                t2 = time.perf_counter()

                # 3. Text to speech, sent to the caller
                duration = await say(websocket, stream_sid, reply, recorder)
                speaking_until = time.monotonic() + duration
                t3 = time.perf_counter()
                print(f"  timing: stt {t1 - t0:.2f}s | llm {t2 - t1:.2f}s | tts+send {t3 - t2:.2f}s")

                if end_call:
                    # Twilio plays our audio in real time, so wait for the
                    # goodbye to finish before hanging up.
                    await asyncio.sleep(duration + 0.5)
                    print("Assistant ended the call.")
                    await websocket.close()
                    break

            elif event == "stop":
                print("Twilio sent stop event.")
                break

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for CallSid: {call_sid}")

    finally:
        if recorder:
            recorder.close()
            print(f"Recording for {call_sid} saved to {recorder.path}")

        # Text the owner, whether the call completed or the caller hung up early
        if conversation:
            call_completed = conversation.completed
            await notify_owner(conversation.details, caller_number, call_completed)

            # Save the call. "missed" = they left without telling us anything.
            details = conversation.details
            if call_completed:
                status = "completed"
            elif details.name or details.reason:
                status = "incomplete"
            else:
                status = "missed"

            try:
                await asyncio.to_thread(
                    save_call,
                    call_sid, caller_number, started_at, status, details,
                    conversation.transcript, str(recorder.path) if recorder else None,
                )
                print(f"Call saved to database as '{status}'.")
            except Exception as e:
                # A database problem must never take the server down
                print(f"Failed to save call: {e}")


@app.post("/stream-status", dependencies=[Depends(verify_twilio)])
async def stream_status_endpoint(request: Request):
    data = await request.form()
    print(f"Received stream status: {dict(data)}")
    return {"status": "ok"}


@app.post("/sms", dependencies=[Depends(verify_twilio)])
async def sms_endpoint(request: Request):
    """Incoming text messages. Only the owner can leave notes."""
    form = await request.form()
    sender = form.get("From", "")
    body = form.get("Body", "").strip()

    if sender != settings.owner_phone:
        # Anyone else texting this number gets no reply, and nothing is saved.
        print(f"Ignored SMS from {sender}")
        return Response(content="<Response/>", media_type="application/xml")

    print(f"Note from owner: {body}")
    reply = await handle_owner_sms(body)
    print(f"Replied: {reply}")

    # Replying is just TwiML again: <Message> sends an SMS back to the sender
    twiml = f"<Response><Message>{escape(reply)}</Message></Response>"
    return Response(content=twiml, media_type="application/xml")