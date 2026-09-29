import asyncio
import audioop
import base64
import json
import time

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect

from call_recorder import CallRecorder
from config import settings
from stt import transcribe
from vad import VoiceActivityDetector
from tts import synthesize

app = FastAPI()

# 20 ms of phone audio: 8000 samples/s * 0.02 s = 160 samples.
# In 16-bit PCM that's 160 * 2 = 320 bytes.
FRAME_BYTES_PCM = 320


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/voice")
async def voice_endpoint():
    twiml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say>Hello, this is Simrat's AI voice assistant. This call may be recorded.</Say>
            <Connect>
                <Stream statusCallback="https://{settings.public_host}/stream-status" url="wss://{settings.public_host}/media-stream" />
            </Connect>
            <Say>Call ended.</Say>
        </Response>
        """
    return Response(content=twiml_content, media_type="application/xml")


async def send_audio(websocket: WebSocket, stream_sid: str, pcm: bytes, recorder: CallRecorder):
    """Send 8 kHz 16-bit PCM audio to the caller through Twilio.

    This is the reverse of what we do with incoming audio:
    incoming:  base64 -> mu-law -> PCM
    outgoing:  PCM -> mu-law -> base64
    """
    # Put the same audio in the recording's right (assistant) channel
    recorder.add_assistant_audio(pcm)

    # Send in 20 ms chunks, the same size Twilio sends to us
    for i in range(0, len(pcm), FRAME_BYTES_PCM):
        chunk = pcm[i:i + FRAME_BYTES_PCM]
        mulaw = audioop.lin2ulaw(chunk, 2)
        payload = base64.b64encode(mulaw).decode("ascii")

        await websocket.send_json({
            "event": "media",
            "streamSid": stream_sid,  # tells Twilio which stream this audio is for
            "media": {"payload": payload},
        })


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("Twilio Media Stream connected.")

    recorder = None
    vad = None
    call_sid = "unknown_call"
    stream_sid = None

    try:
        while True:
            message_text = await websocket.receive_text()
            packet = json.loads(message_text)
            event = packet.get("event")

            if event == "start":
                start_data = packet.get("start", {})
                call_sid = start_data.get("callSid", "stream")
                # We need streamSid to send audio back to this call
                stream_sid = start_data.get("streamSid")
                recorder = CallRecorder(call_sid)
                vad = VoiceActivityDetector()
                print(f"Recording started. Saving to {recorder.path}")

            elif event == "media":
                payload = packet.get("media", {}).get("payload")

                if payload and recorder:
                    mulaw_data = base64.b64decode(payload)
                    pcm_data = audioop.ulaw2lin(mulaw_data, 2)

                    recorder.add_caller_audio(pcm_data)

                    utterance = vad.process(pcm_data)
                    if utterance:
                        started = time.perf_counter()
                        text = await asyncio.to_thread(transcribe, utterance)
                        took = time.perf_counter() - started
                        print(f"Caller said: {text}  (transcribed in {took:.2f}s)")

                        if text:
                            reply = f"I heard you say: {text}"
                            speech = await asyncio.to_thread(synthesize, reply)
                            await send_audio(websocket, stream_sid, speech, recorder)
                            print(f"Assistant said: {reply}")

            elif event == "stop":
                print("Twilio sent stop event.")
                break

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for CallSid: {call_sid}")

    finally:
        if recorder:
            recorder.close()
            print(f"Recording for {call_sid} saved to {recorder.path}")


@app.post("/stream-status")
async def stream_status_endpoint(request: Request):
    data = await request.form()
    print(f"Received stream status: {dict(data)}")
    return {"status": "ok"}