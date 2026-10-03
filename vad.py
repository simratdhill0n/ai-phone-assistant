"""Energy-based voice activity detection (VAD).

Feed it one 16-bit PCM frame at a time. It tracks whether the caller is
silent or speaking, collects the audio of each utterance, and returns that
audio when the caller finishes speaking.

States:
    SILENT   -> SPEAKING  when RMS stays above THRESHOLD for START_FRAMES
    SPEAKING -> SILENT    when RMS stays below THRESHOLD for END_FRAMES
"""

import math
import sys
from array import array

# Tuning values. Twilio frames are 20 ms each.
THRESHOLD = 30      # RMS above this counts as speech
START_FRAMES = 3    # 60 ms of speech to start an utterance
END_FRAMES = 50     # 1 second of silence to end an utterance

SILENT = "silent"
SPEAKING = "speaking"


def frame_rms(pcm: bytes) -> float:
    """Loudness of one 16-bit PCM frame (root mean square)."""
    samples = array("h", pcm)
    if sys.byteorder == "big":
        samples.byteswap()  # PCM from Twilio/WAV is little-endian
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class VoiceActivityDetector:
    def __init__(
        self,
        threshold: float = THRESHOLD,
        start_frames: int = START_FRAMES,
        end_frames: int = END_FRAMES,
    ):
        self.threshold = threshold
        self.start_frames = start_frames
        self.end_frames = end_frames
        self._reset()

    def _reset(self) -> None:
        self.state = SILENT
        self._loud_count = 0        # consecutive loud frames while silent
        self._quiet_count = 0       # consecutive quiet frames while speaking
        self._pending = []          # loud frames before speech is confirmed
        self._utterance = bytearray()
        self._trailing_bytes = 0    # silence at the end of the utterance

    @property
    def speech_ms(self) -> int:
        """How long the current utterance has lasted so far, in milliseconds.
        0 when the caller isn't speaking. Lets main.py react to speech while
        it's still happening (barge-in), not only once it ends."""
        if self.state != SPEAKING:
            return 0
        return len(self._utterance) // 320 * 20   # 320 bytes = one 20 ms frame

    def process(self, pcm: bytes) -> bytes | None:
        """Feed one frame. Returns the utterance audio when speech ends,
        otherwise None."""
        is_loud = frame_rms(pcm) > self.threshold

        if self.state == SILENT:
            if is_loud:
                self._loud_count += 1
                self._pending.append(pcm)
                if self._loud_count >= self.start_frames:
                    # Speech confirmed. Keep the frames that confirmed it,
                    # so the start of the first word isn't cut off.
                    self.state = SPEAKING
                    self._utterance = bytearray(b"".join(self._pending))
                    self._pending.clear()
                    self._quiet_count = 0
                    self._trailing_bytes = 0
            else:
                # A short noise, not speech. Start over.
                self._loud_count = 0
                self._pending.clear()
            return None

        # SPEAKING
        self._utterance.extend(pcm)
        if is_loud:
            self._quiet_count = 0
            self._trailing_bytes = 0
        else:
            self._quiet_count += 1
            self._trailing_bytes += len(pcm)
            if self._quiet_count >= self.end_frames:
                # Caller finished. Drop the trailing silence and hand
                # back just the speech.
                end = len(self._utterance) - self._trailing_bytes
                audio = bytes(self._utterance[:end])
                self._reset()
                return audio
        return None


# =====================================================================
# Silero VAD: a small neural network that estimates the PROBABILITY that
# a slice of audio is human speech. Unlike the energy detector above, a
# loud fan or a passing car scores low, and a quiet "um..." scores high.
# =====================================================================

SILERO_WINDOW = 256          # Silero needs exactly 256 samples per call at 8 kHz (32 ms)
SILERO_WINDOW_BYTES = SILERO_WINDOW * 2
WINDOW_MS = 32

START_PROB = 0.5             # above this, a window counts as speech
END_PROB = 0.35              # below this, a window counts as silence. The gap
                             # between the two (hysteresis) stops flickering
                             # when the probability hovers around one value.
START_MS = 96                # 3 windows of speech to start an utterance
END_MS = 800                 # this much silence ends an utterance
PREROLL_MS = 192             # audio kept from just BEFORE speech started, so
                             # the first syllable isn't clipped


class SileroVoiceActivityDetector:
    """Same interface as VoiceActivityDetector: process() takes one 20 ms
    Twilio frame and returns the utterance audio when speech ends."""

    def __init__(self):
        import numpy as np
        import torch
        from silero_vad import load_silero_vad

        self._np, self._torch = np, torch
        # The model remembers context between calls to it (internal state),
        # so each phone call needs its OWN copy. Shared, two callers' audio
        # would mix. It's tiny and loads from the installed package, no download.
        self._model = load_silero_vad()

        self._buffer = bytearray()   # Twilio sends 160-sample frames; Silero wants 256
        self._preroll: list[bytes] = []
        self._reset()

    def _reset(self) -> None:
        self.state = SILENT
        self._speech_run = 0         # consecutive speech windows while silent
        self._silence_run = 0        # consecutive silent windows while speaking
        self._utterance = bytearray()
        self._trailing_bytes = 0

    @property
    def speech_ms(self) -> int:
        """How long the current utterance has lasted so far (for barge-in)."""
        if self.state != SPEAKING:
            return 0
        return len(self._utterance) // 16   # 16 bytes per ms at 8 kHz 16-bit

    def _speech_probability(self, window: bytes) -> float:
        samples = self._np.frombuffer(window, dtype="<i2").astype(self._np.float32) / 32768.0
        with self._torch.no_grad():
            return self._model(self._torch.from_numpy(samples), 8000).item()

    def process(self, pcm: bytes) -> bytes | None:
        """Feed one frame. Returns the utterance audio when speech ends."""
        self._buffer.extend(pcm)
        result = None
        # Run Silero on every complete 256-sample window we've collected
        while len(self._buffer) >= SILERO_WINDOW_BYTES:
            window = bytes(self._buffer[:SILERO_WINDOW_BYTES])
            del self._buffer[:SILERO_WINDOW_BYTES]
            done = self._process_window(window)
            if done is not None:
                result = done
        return result

    def _process_window(self, window: bytes) -> bytes | None:
        prob = self._speech_probability(window)

        if self.state == SILENT:
            # Keep a short rolling history of audio before speech starts
            self._preroll.append(window)
            if len(self._preroll) > PREROLL_MS // WINDOW_MS:
                self._preroll.pop(0)

            self._speech_run = self._speech_run + 1 if prob > START_PROB else 0
            if self._speech_run * WINDOW_MS >= START_MS:
                self.state = SPEAKING
                self._utterance = bytearray(b"".join(self._preroll))
                self._preroll.clear()
                self._silence_run = 0
                self._trailing_bytes = 0
            return None

        # SPEAKING
        self._utterance.extend(window)
        if prob < END_PROB:
            self._silence_run += 1
            self._trailing_bytes += len(window)
            if self._silence_run * WINDOW_MS >= END_MS:
                end = len(self._utterance) - self._trailing_bytes
                audio = bytes(self._utterance[:end])
                self._reset()
                self._model.reset_states()   # fresh context for the next utterance
                return audio
        else:
            self._silence_run = 0
            self._trailing_bytes = 0
        return None


def create_vad(engine: str = "silero"):
    """Pick the detector by name. 'silero' (default) or 'energy'."""
    if engine == "energy":
        return VoiceActivityDetector()
    return SileroVoiceActivityDetector()