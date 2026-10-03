"""Speech-to-text using faster-whisper, on the GPU (or CPU as a fallback)."""

import audioop
import os
import sys
from pathlib import Path

import numpy as np

from config import settings


def _add_nvidia_dll_dirs() -> None:
    """Windows only: make NVIDIA's CUDA libraries findable.

    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip packages put their DLLs
    in site-packages/nvidia/<lib>/bin, which Windows doesn't search by
    default. Linux finds them without help.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return  # packages not installed: CPU mode, or CUDA installed system-wide
    for base in nvidia.__path__:
        for bin_dir in Path(base).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


# The model starts empty. load_stt() fills it, called once from the server's
# startup (lifespan). Importing this file stays cheap: no GPU memory used,
# no files read, so tests and tools can import it freely.
_model = None


def load_stt() -> None:
    """Load Whisper. Called once at server startup."""
    global _model
    if _model is not None:
        return   # already loaded
    _add_nvidia_dll_dirs()
    from faster_whisper import WhisperModel   # imported here, after the DLL setup
    #   GPU: device="cuda", compute_type="float16"
    #   CPU: device="cpu",  compute_type="int8"
    _model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def _require_model():
    if _model is None:
        raise RuntimeError("Speech-to-text model not loaded. Call load_stt() at startup.")
    return _model


def transcribe(pcm_8k: bytes, hints: str = "") -> str:
    """Turn one utterance (8 kHz, 16-bit PCM bytes) into text.

    hints: words Whisper should expect, e.g. "Simrat, Nova, Ahmed".
    Whisper treats this as text that came just before the audio, so it
    strongly prefers these spellings when it hears something similar.
    """
    # 1. Whisper expects 16 kHz audio, phone audio is 8 kHz.
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)

    # 2. Bytes -> numbers (-32768..32767) -> floats (-1.0..1.0)
    samples = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0

    # 3. Transcribe. The work actually happens while looping over segments.
    segments, _ = _require_model().transcribe(
        samples,
        beam_size=5,
        language="en",                      # skip language detection, saves time
        initial_prompt=hints or None,
    )
    # Whisper rates how likely each segment is to be silence or noise.
    # Drop the ones it thinks probably weren't speech.
    text = " ".join(
        segment.text.strip() for segment in segments if segment.no_speech_prob < 0.6
    ).strip()

    # Known Whisper quirk: given noise and an initial_prompt, it sometimes
    # just repeats the prompt back. That's not something the caller said.
    if hints and _simplify(text) == _simplify(hints):
        return ""
    return text


def _simplify(text: str) -> str:
    """Lowercase letters and digits only, for loose comparisons."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def warm_up_stt() -> None:
    """Run one tiny transcription at startup. The first GPU run is slow
    (CUDA setup), so do it before any caller is waiting."""
    segments, _ = _require_model().transcribe(np.zeros(16000, dtype=np.float32), language="en")
    list(segments)   # segments is lazy: consume it so the work actually runs