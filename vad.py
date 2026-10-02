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