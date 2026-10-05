import asyncio
import audioop
import base64
import json
import re
import io
import time
import wave
from contextlib import asynccontextmanager
from html import escape

from fastapi import Depends, FastAPI, Request, Response, WebSocket, WebSocketDisconnect

from calender_check import call_facts
from call_recorder import CallRecorder
from config import settings
from db import get_notes, init_db, mark_notes_delivered, save_call, utcnow
from llm import Conversation, warm_up
from memory import build_caller_context
from notes import handle_owner_sms
from sms import notify_owner
from stt import load_stt, transcribe, warm_up_stt
from tts import load_tts, split_sentences, synthesize, warm_up_tts
from transfer import finish, get_whisper, mark_accepted, transfer_call, whisper_text
from twilio_security import verify_twilio
from vad import create_vad


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts (before yield) and once when it
    stops (after yield). Everything heavy happens here, not at import time,
    so a plain `import main` never loads a model."""
    init_db()

    # Loading models blocks for seconds, so do it in threads.
    await asyncio.to_thread(load_stt)
    await asyncio.to_thread(warm_up_stt)
    print("Speech-to-text ready.")

    await asyncio.to_thread(load_tts)
    await asyncio.to_thread(warm_up_tts)
    print(f"Text-to-speech ready ({settings.tts_engine}).")

    # Said when a transfer fails. Twilio plays it with <Play>, after our
    # stream has ended, so we make it now in the same voice as the rest.
    pcm = await asyncio.to_thread(synthesize, TRANSFER_FAILED_TEXT)
    AUDIO_FILES["transfer_failed"] = pcm_to_wav(pcm)

    try:
        await warm_up()
        print("LLM loaded and ready.")
    except Exception as e:
        print(f"Could not warm up the LLM (is Ollama running?): {e}")
    yield


app = FastAPI(lifespan=lifespan)

TRANSFER_FAILED_TEXT = (
    f"Sorry, {settings.owner_name} couldn't pick up right now. "
    "Your message has been passed on, and he'll get back to you as soon as he can. Goodbye."
)
AUDIO_FILES: dict[str, bytes] = {}   # name -> WAV bytes, served at /audio/<name>.wav


def pcm_to_wav(pcm: bytes) -> bytes:
    """8 kHz 16-bit mono PCM -> a WAV file in memory (a format Twilio can <Play>)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(pcm)
    return buffer.getvalue()

# 20 ms of phone audio: 160 samples * 2 bytes = 320 bytes of 16-bit PCM
FRAME_BYTES_PCM = 320
# 8000 samples/s * 2 bytes per sample
PCM_BYTES_PER_SECOND = 16000

# Barge-in: how long the caller must keep talking over the assistant before
# it stops. Long enough that a quick "mhm" or a cough doesn't cut it off.
BARGE_IN_MS = 400

# Listening sounds people make while someone else talks. Not an answer,
# so after one of these the assistant repeats what it was saying.
FILLER_ONLY = re.compile(
    r"^\W*(m+|h?m+h?m+|hmm+|uh+|um+|uh[\s-]*huh|ah+|oh+)(\W+(m+|h?m+h?m+|hmm+|uh+|um+|uh[\s-]*huh|ah+|oh+))*\W*$",
    re.IGNORECASE,
)

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


async def send_mark(websocket: WebSocket, stream_sid: str, name: str):
    """Twilio sends this mark back once the caller has HEARD everything we
    sent before it. That's how we know when the assistant stops talking."""
    await websocket.send_json({"event": "mark", "streamSid": stream_sid, "mark": {"name": name}})


async def clear_audio(websocket: WebSocket, stream_sid: str):
    """Tell Twilio to throw away any of our audio it hasn't played yet."""
    await websocket.send_json({"event": "clear", "streamSid": stream_sid})


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
    # Playback state. "playing" is True from the moment the assistant starts
    # a reply until Twilio echoes that reply's mark back (= the caller heard
    # the end of it).
    playing = False
    interruptible = False    # greeting and goodbye can't be interrupted
    mark_count = 0
    barged_in = False        # caller cut in, not yet sure it was a real answer
    last_reply = ""          # what the assistant last said, to repeat after a filler
    hang_up_after_speaking = False
    speak_task: asyncio.Task | None = None
    private_notes: list[str] = []
    note_texts: dict[int, str] = {}

    async def speak(text: str, mark_name: str) -> None:
        """Runs in the BACKGROUND: synthesize sentence by sentence and send
        each one as soon as it's ready. The caller hears the first sentence
        while the rest are still being generated, and the main loop keeps
        listening the whole time (so barge-in can cancel this task)."""
        try:
            started = time.perf_counter()
            first_audio = None
            for sentence in split_sentences(text):
                pcm = await asyncio.to_thread(synthesize, sentence)
                if first_audio is None:
                    first_audio = time.perf_counter() - started
                await send_audio(websocket, stream_sid, pcm, recorder)
            # Sent after the LAST sentence: Twilio echoes it once all of it was heard
            await send_mark(websocket, stream_sid, mark_name)
            print(f"Assistant said: {text}")
            print(f"  tts: first audio after {first_audio or 0:.2f}s, "
                  f"all sent after {time.perf_counter() - started:.2f}s")
        except asyncio.CancelledError:
            raise   # barge-in cancelled us: let the cancellation through
        except Exception as e:
            # A background task's errors are otherwise silent, so log them
            print(f"Speaking failed: {e!r}")

    def start_speaking(text: str, can_interrupt: bool = True) -> None:
        """Begin a reply in the background and return immediately."""
        nonlocal speak_task, playing, interruptible, mark_count, last_reply
        # Reserve this reply's mark name NOW, so a late mark from an older
        # reply can never be mistaken for this one finishing.
        mark_count += 1
        speak_task = asyncio.create_task(speak(text, f"m{mark_count}"))
        playing, interruptible = True, can_interrupt
        last_reply = text

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
                vad = create_vad(settings.vad_engine)

                # Caller memory: look up this number's history (database = blocking)
                greeting, caller_context, known_name = await asyncio.to_thread(
                    build_caller_context, caller_number, GREETING
                )

                # Notes about this number. Shareable ones not yet passed on go
                # to the conversation (spoken by code, never seen by the LLM).
                # Private ones are kept for the owner's SMS only.
                notes = await asyncio.to_thread(get_notes, caller_number) if known_name else []
                shareable = [(n.id, n.text) for n in notes
                             if n.visibility == "shareable" and n.delivered_at is None]
                private_notes = [n.text for n in notes if n.visibility == "private"]

                # Ask the calendar in the background while the greeting plays:
                # the greeting is fixed text, so it doesn't have to wait.
                calendar_task = asyncio.create_task(
                    asyncio.to_thread(call_facts, caller_number, bool(known_name))
                )

                # Names Whisper should expect on this call
                hint_names = [settings.owner_name, settings.assistant_name]
                if known_name:
                    hint_names.append(known_name)
                stt_hints = ", ".join(hint_names)
                print(f"Call from {caller_number or 'unknown number'}")
                if caller_context:
                    print(f"Known caller.{caller_context}")
                print(f"Recording started. Saving to {recorder.path}")

                # The greeting includes the AI and recording disclosure,
                # so the caller can't talk over it.
                start_speaking(greeting, can_interrupt=False)

                # By the time the greeting has started, the calendar has
                # usually answered. call_facts never raises: on any calendar
                # problem it returns (None, None) and the call goes on without it.
                availability, appointment_text = await calendar_task
                if availability:
                    print(f"Calendar: {availability}")
                conversation = Conversation(
                    greeting, caller_number, caller_context, known_name, shareable,
                    availability=availability, appointment_text=appointment_text,
                )
                note_texts = {n.id: n.text for n in notes}

            elif event == "media":
                payload = packet.get("media", {}).get("payload")
                if not (payload and recorder):
                    continue

                mulaw_data = base64.b64decode(payload)
                pcm_data = audioop.ulaw2lin(mulaw_data, 2)
                recorder.add_caller_audio(pcm_data)

                utterance = vad.process(pcm_data)

                # Barge-in: the caller has been talking over the assistant
                # long enough to mean it. Stop talking and listen.
                if playing and interruptible and vad.speech_ms >= BARGE_IN_MS:
                    if speak_task and not speak_task.done():
                        speak_task.cancel()        # stop generating the rest
                    await clear_audio(websocket, stream_sid)   # drop what's queued at Twilio
                    recorder.clear_assistant_audio()
                    playing = False
                    barged_in = True   # decided after transcription: real answer or just "mm"?
                    print("Caller interrupted. Assistant stopped talking.")

                if not utterance:
                    continue

                # Speech that ended while the assistant was STILL talking was
                # short enough not to trigger barge-in: a "mhm", a cough, or
                # speech during the greeting. Not a real turn, so skip it.
                if playing:
                    print("Ignored short speech over the assistant.")
                    continue

                # 1. Speech to text
                t0 = time.perf_counter()
                text = await asyncio.to_thread(transcribe, utterance, stt_hints)
                t1 = time.perf_counter()
                # Whisper "hears" things in noise, like ". . ." or "Thank you."
                # No letters or digits at all means nothing real was said.
                if not re.search(r"[A-Za-z0-9]", text):
                    print(f"Ignored non-speech transcript: {text!r}")
                    if barged_in:
                        # The cut-in was just noise: carry on with what we were saying
                        barged_in = False
                        start_speaking(last_reply)
                    continue

                if barged_in:
                    barged_in = False
                    if FILLER_ONLY.match(text):
                        # "Mm" / "mhm" means "go on", not "stop": repeat the reply
                        print(f"Cut-in was just a listening sound ({text!r}). Repeating.")
                        start_speaking(last_reply)
                        continue
                    # A real answer: now we know the caller didn't hear it all
                    conversation.mark_interrupted()

                print(f"Caller said: {text}")

                # 2. LLM decides the reply
                reply, end_call = await conversation.reply(text)
                t2 = time.perf_counter()
                print(f"  timing: stt {t1 - t0:.2f}s | llm {t2 - t1:.2f}s")

                # 3. Speak it, streaming, in the background. The goodbye can't be
                #    interrupted, and we hang up once Twilio confirms it was heard.
                start_speaking(reply, can_interrupt=not end_call)
                if end_call:
                    hang_up_after_speaking = True

            elif event == "mark":
                # Only the LATEST reply's mark means "finished talking"
                if packet.get("mark", {}).get("name") == f"m{mark_count}":
                    playing = False
                    if hang_up_after_speaking:
                        if conversation.wants_transfer:
                            whisper = whisper_text(conversation.details, caller_number)
                            if await transfer_call(call_sid, whisper):
                                # Twilio now runs the <Dial> instead of our stream.
                                # It will send "stop" and close the socket itself.
                                hang_up_after_speaking = False
                                continue
                            # Twilio refused: just end the call. The SMS still goes out.
                        print("Assistant ended the call.")
                        await websocket.close()
                        break

            elif event == "stop":
                print("Twilio sent stop event.")
                break

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for CallSid: {call_sid}")

    finally:
        # Don't leave a reply generating for a call that has ended
        if speak_task and not speak_task.done():
            speak_task.cancel()

        if recorder:
            recorder.close()
            print(f"Recording for {call_sid} saved to {recorder.path}")

        # Text the owner, whether the call completed or the caller hung up early
        if conversation:
            call_completed = conversation.completed
            delivered = conversation.delivered_note_ids
            await notify_owner(
                conversation.details, caller_number, call_completed,
                private_notes=private_notes,
                delivered_notes=[note_texts[i] for i in delivered],
            )

            # Remember which shareable notes were passed on, so they aren't repeated
            if delivered:
                try:
                    await asyncio.to_thread(mark_notes_delivered, delivered)
                except Exception as e:
                    print(f"Failed to mark notes delivered: {e}")

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


# ---------- Call transfer (see transfer.py for the whole flow) ----------

@app.post("/whisper", dependencies=[Depends(verify_twilio)])
async def whisper_endpoint(request: Request):
    """The owner picked up. Only THEY hear this, the caller hears ringing."""
    form = await request.form()
    text = escape(get_whisper(form.get("ParentCallSid", "")))
    twiml = f"""<Response>
        <Gather numDigits="1" timeout="6" action="https://{settings.public_host}/whisper-answer">
            <Say>{text} Press 1 to take the call.</Say>
        </Gather>
        <Hangup/>
    </Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.post("/whisper-answer", dependencies=[Depends(verify_twilio)])
async def whisper_answer_endpoint(request: Request):
    """The owner pressed a key. An empty <Response> ends the whisper, and
    Twilio joins the two calls. <Hangup/> here hangs up only the owner."""
    form = await request.form()
    if form.get("Digits") == "1":
        mark_accepted(form.get("ParentCallSid", ""))
        print("Owner accepted the transfer.")
        return Response(content="<Response/>", media_type="application/xml")
    return Response(content="<Response><Hangup/></Response>", media_type="application/xml")


@app.post("/transfer-result", dependencies=[Depends(verify_twilio)])
async def transfer_result_endpoint(request: Request):
    """Runs on the CALLER's call once the <Dial> is over: either the owner
    talked to them and hung up, or the owner never took the call."""
    form = await request.form()
    taken = finish(form.get("CallSid", ""))
    print(f"Transfer finished: {'owner took the call' if taken else 'owner did not answer'} "
          f"(DialCallStatus={form.get('DialCallStatus')})")
    if taken:
        return Response(content="<Response><Hangup/></Response>", media_type="application/xml")
    twiml = f"""<Response>
        <Play>https://{settings.public_host}/audio/transfer_failed.wav</Play>
        <Hangup/>
    </Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.get("/audio/{name}.wav")
def audio_file(name: str):
    """Pre-made audio for Twilio's <Play>. Nothing private here, so no signature check."""
    if name not in AUDIO_FILES:
        return Response(status_code=404)
    return Response(content=AUDIO_FILES[name], media_type="audio/wav")