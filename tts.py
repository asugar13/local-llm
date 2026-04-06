"""Text-to-speech output using macOS `say` — fully local, no model download.
Swap speak() for Coqui TTS later without changing any call sites.
"""
import subprocess
import threading

_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def speak(text: str, voice: str = "Samantha") -> None:
    """Synthesise text and play it. Blocks until playback finishes."""
    global _proc
    with _lock:
        _proc = subprocess.Popen(["say", "-v", voice, text])
    _proc.wait()
    with _lock:
        _proc = None


def stop() -> None:
    """Interrupt any ongoing speech immediately."""
    global _proc
    with _lock:
        if _proc is not None:
            _proc.terminate()
            _proc = None
