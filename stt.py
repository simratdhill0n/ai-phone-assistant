"""Speech-to-text using faster-whisper, running locally on the CPU."""

import audioop

import numpy as np
from faster_whisper import WhisperModel

# Loaded once when the app starts, then reused for every utterance.
# int8 = smaller, faster weights with almost no accuracy loss.
model = WhisperModel("base.en", device="cpu", compute_type="int8")


def transcribe(pcm_8k: bytes) -> str:
    """Turn one utterance (8 kHz, 16-bit PCM bytes) into text.

    This is a normal (not async) function on purpose: it does heavy CPU
    work, so main.py runs it in a separate thread with asyncio.to_thread.
    """
    # 1. Whisper expects 16 kHz audio, phone audio is 8 kHz.
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)

    # 2. Bytes -> numbers (-32768..32767) -> floats (-1.0..1.0)
    samples = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0

    # 3. Transcribe. The work actually happens while looping over segments.
    segments, _ = model.transcribe(samples, beam_size=5)
    return " ".join(segment.text.strip() for segment in segments)