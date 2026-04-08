"""Speech-to-text helper using local Whisper — no audio ever leaves the machine."""
import numpy as np
import sounddevice as sd
import whisper

_model = None  # lazy-loaded on first transcription


def _get_model():
    global _model
    if _model is None:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = whisper.load_model("large-v3", device=device)
    return _model

# Module-level stream state (persists across Streamlit reruns in the same process)
_buffer: list[np.ndarray] = []
_stream: sd.InputStream | None = None


def start_recording(sample_rate: int = 16000) -> None:
    """Begin recording from the default microphone using a non-blocking callback stream."""
    global _buffer, _stream
    _buffer = []

    def _callback(indata: np.ndarray, frames: int, time, status) -> None:
        _buffer.append(indata.flatten().copy())

    _stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=_callback,
    )
    _stream.start()


def stop_recording() -> np.ndarray:
    """Stop recording and return the captured audio as a float32 numpy array."""
    global _stream
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None
    return np.concatenate(_buffer) if _buffer else np.array([], dtype="float32")


def transcribe(audio: np.ndarray, language: str = "en") -> str:
    """Transcribe a float32 numpy array directly — no ffmpeg required."""
    if len(audio) == 0:
        return ""
    result = _get_model().transcribe(audio, fp16=True, language=language)
    return result["text"].strip()
