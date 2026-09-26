import json
import base64
import audioop
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect, Request

from call_recorder import CallRecorder
from config import settings

app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/voice")
async def voice_endpoint():
    # Build your raw TwiML string
    twiml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say>Hello, this is Simrat's AI voice assistant.</Say>
            <Connect>
                <Stream statusCallback="https://{settings.public_host}/stream-status" url="wss://{settings.public_host}/media-stream" />
            </Connect>
            <Say>Call ended.</Say>
        </Response>
        """
    
    # Return a custom Response with the application/xml media type
    return Response(content=twiml_content, media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("Twilio Media Stream connected.")

    recorder = None
    call_sid = "unknown_call"

    try:
        while True:
            # 1. Twilio sends data as JSON text strings
            message_text = await websocket.receive_text()
            packet = json.loads(message_text)

            # 2. Start of the stream: create one recorder for the whole call
            if packet.get("event") == "start":
                start_data = packet.get("start", {})
                call_sid = start_data.get("callSid", "stream")
                recorder = CallRecorder(call_sid)
                print(f"Recording started. Saving to {recorder.path}")

            # 3. Process the streaming audio chunks
            elif packet.get("event") == "media":
                media = packet.get("media", {})
                payload = media.get("payload")

                if payload and recorder:
                    # Decode the base64 string back into raw mu-law bytes
                    mulaw_data = base64.b64decode(payload)

                    # Convert Twilio's 8-bit mu-law audio to standard 16-bit linear PCM
                    pcm_data = audioop.ulaw2lin(mulaw_data, 2)

                    # Caller audio goes to the left channel
                    recorder.add_caller_audio(pcm_data)

            # 4. Explicitly stop if Twilio sends the stop event
            elif packet.get("event") == "stop":
                print("Twilio sent stop event.")
                break

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for CallSid: {call_sid}")

    finally:
        # Always finalize the recording, however the call ended
        if recorder:
            recorder.close()
            print(f"Recording for {call_sid} saved to {recorder.path}")
@app.post("/stream-status")
async def stream_status_endpoint(request: Request):
    data = await request.form()
    print(f"Received stream status: {dict(data)}")
    return {"status": "ok"}