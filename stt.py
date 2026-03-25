"""Speech-to-text helper using local Whisper — no audio ever leaves the machine."""
import numpy as np
import sounddevice as sd
import whisper

# Load once at startup; keep in memory between turns.
_model = whisper.load_model("base")

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


def transcribe(audio: np.ndarray) -> str:
    """Transcribe a float32 numpy array directly — no ffmpeg required."""
    if len(audio) == 0:
        return ""
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=_model.dims.n_mels).to(_model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(_model, mel, options)
    return result.text.strip()
