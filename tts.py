"""Text-to-speech using Piper, running locally on the CPU."""

import audioop

from piper import PiperVoice

VOICE_PATH = "models/piper/en_US-hfc_female-medium.onnx"

# Loaded once at startup. Piper finds the matching .onnx.json config
# automatically, as long as it sits next to the .onnx file.
voice = PiperVoice.load(VOICE_PATH)


def synthesize(text: str) -> bytes:
    """Turn text into 8 kHz, 16-bit PCM audio, ready for send_audio().

    A normal (not async) function: it does heavy CPU work, so main.py runs
    it in a separate thread with asyncio.to_thread, like transcribe().
    """
    # 1. Generate speech. Piper outputs 16-bit PCM at the voice's own
    #    sample rate (22,050 Hz for most medium voices).
    #    The piper-tts API changed between versions, so support both.
    if hasattr(voice, "synthesize_stream_raw"):   # piper-tts 1.2.x
        pcm = b"".join(voice.synthesize_stream_raw(text))
    else:                                          # piper-tts 1.3+
        pcm = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))

    # 2. Phone audio is 8 kHz, so downsample. This is the reverse of what
    #    stt.py does (8 kHz up to 16 kHz for Whisper).
    source_rate = voice.config.sample_rate
    pcm_8k, _ = audioop.ratecv(pcm, 2, 1, source_rate, 8000, None)
    return pcm_8k