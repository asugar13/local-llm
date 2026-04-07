"""Text-to-speech output — three backends selectable at runtime.

Backends
--------
macos   macOS built-in `say` command. Zero setup, instant start. Voice quality
        is functional but synthetic-sounding. Best for quick testing.
kokoro  Kokoro TTS. Lightweight (~80 MB), fast, high-quality English voice.
        Works on Python 3.12+. Requires: pip install kokoro && brew install espeak-ng
coqui   Coqui TTS (tacotron2-DDC). Heavier but best multilingual support.
        Requires Python <3.12 and: pip install TTS soundfile
"""
import os
import subprocess
import tempfile
import threading

# ── shared stop state ─────────────────────────────────────────────────────────
_lock = threading.Lock()
_proc: subprocess.Popen | None = None   # macOS say process
_coqui_model = None                     # lazy-loaded Coqui TTS instance
_kokoro_pipeline = None                 # lazy-loaded Kokoro pipeline


# ── macOS backend ─────────────────────────────────────────────────────────────
def _speak_macos(text: str) -> None:
    global _proc
    with _lock:
        _proc = subprocess.Popen(["say", "-v", "Samantha", text])
    _proc.wait()
    with _lock:
        _proc = None


# ── Kokoro backend ────────────────────────────────────────────────────────────
def _load_kokoro():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        try:
            from kokoro import KPipeline
        except ImportError:
            raise RuntimeError("Kokoro not installed. Run: pip install kokoro && brew install espeak-ng")
        _kokoro_pipeline = KPipeline(lang_code="a")  # "a" = American English
    return _kokoro_pipeline


def _speak_kokoro(text: str) -> None:
    import sounddevice as sd
    pipeline = _load_kokoro()
    for _, _, audio in pipeline(text, voice="af_heart", speed=1):
        sd.play(audio, 24000, device=1)
        sd.wait()


# ── Coqui backend ─────────────────────────────────────────────────────────────
def _load_coqui():
    global _coqui_model
    if _coqui_model is None:
        try:
            from TTS.api import TTS
        except ImportError:
            raise RuntimeError("Coqui TTS not installed. Run: pip install TTS soundfile")
        _coqui_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    return _coqui_model


def _speak_coqui(text: str) -> None:
    import sounddevice as sd
    import soundfile as sf
    model = _load_coqui()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        model.tts_to_file(text=text, file_path=path)
        data, samplerate = sf.read(path)
        sd.play(data, samplerate)
        sd.wait()
    finally:
        os.unlink(path)


# ── public API ────────────────────────────────────────────────────────────────
BACKENDS = {
    "macos":  "macOS Samantha — instant, no setup",
    "kokoro": "Kokoro — lightweight, fast, strong English quality",
    "coqui":  "Coqui TTS — heavier, best for multilingual",
}


def speak(text: str, backend: str = "macos") -> None:
    """Synthesise text and play it. Blocks until playback finishes."""
    if backend == "kokoro":
        _speak_kokoro(text)
    elif backend == "coqui":
        _speak_coqui(text)
    else:
        _speak_macos(text)


def stop() -> None:
    """Interrupt any ongoing speech immediately."""
    global _proc
    with _lock:
        if _proc is not None:
            _proc.terminate()
            _proc = None
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass
