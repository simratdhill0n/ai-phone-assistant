"""Records a whole phone call into one stereo WAV file.

Left channel: caller. Right channel: assistant.

The caller's audio is the clock. Twilio sends a caller frame every 20 ms,
even during silence, so every time a caller frame arrives we write one frame
to each channel. Assistant audio is queued and written alongside the caller
frames as they arrive. When the assistant isn't speaking, its channel gets
silence.
"""

import sys
import wave
from array import array
from pathlib import Path

SAMPLE_RATE = 8000   # Twilio Media Streams use 8 kHz
SAMPLE_WIDTH = 2     # 16-bit PCM = 2 bytes per sample
CHANNELS = 2         # stereo: left = caller, right = assistant


class CallRecorder:
    def __init__(self, call_sid: str, directory: str = "recordings"):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f"{call_sid}.wav"

        self._wav = wave.open(str(self.path), "wb")
        self._wav.setnchannels(CHANNELS)
        self._wav.setsampwidth(SAMPLE_WIDTH)
        self._wav.setframerate(SAMPLE_RATE)

        # Assistant audio waiting to be written next to future caller frames
        self._assistant_pending = bytearray()
        self._closed = False

    def add_assistant_audio(self, pcm: bytes) -> None:
        """Queue 16-bit PCM audio that was just sent to the caller."""
        if not self._closed:
            self._assistant_pending.extend(pcm)

    def add_caller_audio(self, pcm: bytes) -> None:
        """Write one caller frame (16-bit PCM) plus matching assistant audio."""
        if self._closed:
            return

        # Take the same number of bytes from the assistant queue,
        # padding with silence (zero bytes) if there isn't enough.
        n = len(pcm)
        assistant = bytes(self._assistant_pending[:n])
        del self._assistant_pending[:n]
        if len(assistant) < n:
            assistant += bytes(n - len(assistant))

        self._wav.writeframes(self._interleave(pcm, assistant))

    def close(self) -> None:
        """Finalize the WAV header. Safe to call more than once."""
        if not self._closed:
            self._closed = True
            self._wav.close()

    @staticmethod
    def _interleave(left: bytes, right: bytes) -> bytes:
        """Combine two mono 16-bit streams into stereo: L, R, L, R, ..."""
        left_samples = array("h", left)
        right_samples = array("h", right)

        stereo = array("h", bytes(len(left) * 2))
        stereo[0::2] = left_samples   # even positions: left channel
        stereo[1::2] = right_samples  # odd positions: right channel

        # WAV files are little-endian. array uses the machine's native order,
        # which is little-endian on almost every modern computer, but be safe.
        if sys.byteorder == "big":
            stereo.byteswap()

        return stereo.tobytes()
