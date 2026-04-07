"""Text-to-speech output — three backends selectable at runtime.

Backends
--------
macos   macOS built-in `say` command. Zero setup, instant start.
kokoro  Kokoro TTS. Lightweight (~80 MB), fast, high-quality English voice.
        Requires: pip install kokoro soundfile && brew install espeak-ng
coqui   Coqui TTS. English via xtts_v2 (voice cloning), Spanish via vits.
        Requires Python <3.12 and: pip install TTS soundfile
"""
import os
import subprocess
import tempfile
import threading

# ── shared stop state ─────────────────────────────────────────────────────────
_lock = threading.Lock()
_proc: subprocess.Popen | None = None   # macOS say process
_coqui_model = None                     # lazy-loaded xtts_v2 instance
_coqui_es_model = None                  # lazy-loaded Spanish VITS instance
_kokoro_pipeline = None                 # lazy-loaded Kokoro pipeline


# ── macOS backend ─────────────────────────────────────────────────────────────
_MACOS_VOICES = {
    "en": "Samantha",
    "es": "Monica",
}


def _speak_macos(text: str, language: str = "en") -> None:
    global _proc
    voice = _MACOS_VOICES.get(language, "Samantha")
    with _lock:
        _proc = subprocess.Popen(["say", "-v", voice, text])
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
    import soundfile as sf
    pipeline = _load_kokoro()
    text = " ".join(text.splitlines())  # flatten newlines so pipeline reads the full text
    for _, _, audio in pipeline(text, voice="af_heart", speed=1):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            sf.write(path, audio, 24000)
            subprocess.run(["afplay", path], check=True)
        finally:
            os.unlink(path)


# ── Coqui backend ─────────────────────────────────────────────────────────────
_XTTS_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
}

_XTTS_DEFAULT_SPEAKER = "Claribel Dervla"


def _load_coqui_es():
    global _coqui_es_model
    if _coqui_es_model is None:
        try:
            from TTS.api import TTS
        except ImportError:
            raise RuntimeError("Coqui TTS not installed.")
        _coqui_es_model = TTS(model_name="tts_models/es/css10/vits", progress_bar=False)
    return _coqui_es_model


def _load_coqui():
    global _coqui_model
    if _coqui_model is None:
        try:
            from TTS.api import TTS
        except ImportError:
            raise RuntimeError("Coqui TTS not installed. Run: pip install TTS soundfile")
        os.environ["COQUI_TOS_AGREED"] = "1"  # auto-accept non-commercial CPML license
        _coqui_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    return _coqui_model


def _speak_coqui(text: str, language: str = "en", speaker_wav: str | None = None) -> None:
    if not speaker_wav:
        if language == "es":
            model = _load_coqui_es()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            try:
                model.tts_to_file(text=text, file_path=path)
                subprocess.run(["afplay", path], check=True)
            finally:
                os.unlink(path)
            return
        _speak_macos(text)
        return
    model = _load_coqui()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        model.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=path)
        subprocess.run(["afplay", path], check=True)
    finally:
        os.unlink(path)


# ── public API ────────────────────────────────────────────────────────────────
BACKENDS = {
    "macos":  "macOS Samantha - instant, no setup",
    "kokoro": "Kokoro - lightweight, fast, strong English quality",
    "coqui":  "Coqui TTS - heavier, best for multilingual",
}


def speak(text: str, backend: str = "macos",
          language: str = "en", speaker_wav: str | None = None) -> None:
    """Synthesise text and play it. Blocks until playback finishes."""
    if backend == "kokoro":
        _speak_kokoro(text)
    elif backend == "coqui":
        _speak_coqui(text, language=language, speaker_wav=speaker_wav)
    else:
        _speak_macos(text, language=language)


def stop() -> None:
    """Interrupt any ongoing speech immediately."""
    global _proc
    with _lock:
        if _proc is not None:
            _proc.terminate()
            _proc = None
