"""Text-to-speech. Two engines, picked with the TTS_ENGINE setting:

- kokoro: natural, human-like voice. Runs on the GPU if available.
- piper:  older, more robotic, but very light on CPU. Kept as a fallback.

Both return the same thing: 8 kHz, 16-bit PCM bytes, ready for send_audio().
"""

import audioop
import re

import numpy as np

from config import settings

# ---------- Text normalization (shared by both engines) ----------

# Something that looks like a phone number: 7+ digits, optionally starting
# with +, possibly broken up by spaces, dashes, dots or brackets.
PHONE_PATTERN = re.compile(r"\+?\(?\d[\d\s().-]{5,}\d")


def _say_digits(match: re.Match) -> str:
    """Turn '+15485771772' into '5 4 8, 5 7 7, 1 7 7 2'."""
    digits = re.sub(r"\D", "", match.group())   # keep only the digits

    # North American numbers: drop the leading country code 1
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]

    # Group 10-digit numbers like people say them: 3, 3, 4.
    # The commas make the voice pause between groups.
    if len(digits) == 10:
        groups = [digits[:3], digits[3:6], digits[6:]]
    else:
        groups = [digits]

    return ", ".join(" ".join(group) for group in groups)


def normalize_for_speech(text: str) -> str:
    """Rewrite text so it sounds right when spoken aloud.

    TTS voices read '5485771772' as one huge number. This spells phone
    numbers out digit by digit instead.
    """
    return PHONE_PATTERN.sub(_say_digits, text)


# ---------- Engines ----------
# Only the chosen engine is imported and loaded, so you don't need the
# other one installed.

if settings.tts_engine == "kokoro":
    from kokoro import KPipeline

    # lang_code "a" = American English ("b" = British).
    # Loaded once at startup. KOKORO_DEVICE picks "cuda" (GPU) or "cpu".
    # On a GPU shared with a big LLM, CPU can leave the LLM more room.
    _pipeline = KPipeline(lang_code="a", device=settings.kokoro_device)
    _SOURCE_RATE = 24000   # Kokoro always outputs 24 kHz

    def _generate(text: str) -> bytes:
        chunks = []
        # Kokoro splits long text into pieces and yields one audio chunk per
        # piece: (graphemes, phonemes, audio). We only need the audio.
        for _, _, audio in _pipeline(
            text, voice=settings.kokoro_voice, speed=settings.kokoro_speed
        ):
            if audio is None:
                continue
            # audio is a float tensor from -1.0 to 1.0. Turn it into
            # 16-bit integers, the format the rest of our pipeline uses.
            samples = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            samples = np.clip(samples, -1.0, 1.0)
            chunks.append((samples * 32767).astype("<i2").tobytes())
        return b"".join(chunks)

else:
    from piper import PiperVoice

    # Piper finds the matching .onnx.json config automatically, as long as
    # it sits next to the .onnx file.
    _voice = PiperVoice.load(settings.piper_voice_path)
    _SOURCE_RATE = _voice.config.sample_rate   # usually 22,050 Hz

    def _generate(text: str) -> bytes:
        # The piper-tts API changed between versions, so support both.
        if hasattr(_voice, "synthesize_stream_raw"):   # piper-tts 1.2.x
            return b"".join(_voice.synthesize_stream_raw(text))
        return b"".join(chunk.audio_int16_bytes for chunk in _voice.synthesize(text))


def synthesize(text: str) -> bytes:
    """Turn text into 8 kHz, 16-bit PCM audio, ready for send_audio().

    A normal (not async) function: it does heavy work, so main.py runs it in
    a separate thread with asyncio.to_thread, like transcribe().
    """
    pcm = _generate(normalize_for_speech(text))

    # Phone audio is 8 kHz, so downsample from the engine's own rate.
    pcm_8k, _ = audioop.ratecv(pcm, 2, 1, _SOURCE_RATE, 8000, None)
    return pcm_8k


def split_sentences(text: str) -> list[str]:
    """Split a reply into sentences, so the first can play while the rest
    are still being generated. 'Got it. What's your name?' -> two pieces."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Glue tiny fragments (like a stray "Ok.") onto the previous sentence
        if sentences and len(part) < 4:
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences


def warm_up_tts() -> None:
    """Synthesize one short phrase at startup. The first run on the GPU is
    slow (setup), so do it before any caller is waiting."""
    synthesize("Hello.")