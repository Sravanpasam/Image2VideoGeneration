"""Audio helpers: normalisation to the format SadTalker expects.

SadTalker's audio front-end (``src/utils/audio.py``) reads a 16 kHz mono WAV file
with ``librosa.core.load(path, sr=16000)``.  Whatever the TTS provider produces
(MP3 at 24 kHz, FLAC at 22.05 kHz, ...) has to be converted first.

Conversion strategy (first one that works wins):

1. ``ffmpeg`` binary - the reference path, present on Colab and most Linux hosts.
2. ``soundfile`` + ``scipy.signal.resample_poly`` - pure Python fallback.
3. ``soundfile`` + numpy linear interpolation - last resort.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

from .config import TARGET_CHANNELS, TARGET_SAMPLE_RATE

__all__ = [
    "AudioInfo",
    "concat_wavs",
    "ffmpeg_path",
    "probe_audio",
    "to_wav_16k_mono",
    "write_silence",
]


def ffmpeg_path() -> str | None:
    """Return the path to the ``ffmpeg`` executable, or ``None``."""
    return shutil.which("ffmpeg")


def probe_audio(path: str | Path) -> "AudioInfo":
    """Inspect an audio file (duration, sample rate, channels)."""
    path = Path(path)
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return AudioInfo(
            path=str(path),
            duration=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            backend="soundfile",
        )
    except Exception:
        pass

    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate() or 1
        return AudioInfo(
            path=str(path),
            duration=frames / float(rate),
            sample_rate=int(rate),
            channels=int(handle.getnchannels()),
            backend="wave",
        )


class AudioInfo(dict):
    """Small dict-like record describing an audio file."""

    def __init__(self, path: str, duration: float, sample_rate: int, channels: int, backend: str):
        super().__init__(
            path=path,
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            backend=backend,
        )

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


def _resample_linear(data, src_rate: int, dst_rate: int):
    import numpy as np

    if src_rate == dst_rate:
        return data
    duration = data.shape[0] / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_idx = np.linspace(0.0, data.shape[0], num=data.shape[0], endpoint=False)
    dst_idx = np.linspace(0.0, data.shape[0], num=dst_len, endpoint=False)
    if data.ndim == 1:
        return np.interp(dst_idx, src_idx, data).astype(np.float32)
    out = np.empty((dst_len, data.shape[1]), dtype=np.float32)
    for channel in range(data.shape[1]):
        out[:, channel] = np.interp(dst_idx, src_idx, data[:, channel])
    return out


def to_wav_16k_mono(src: str | Path, dst: str | Path) -> Path:
    """Convert any audio file to a 16 kHz mono WAV file at ``dst``.

    Returns the path of the written file.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.suffix.lower() != ".wav":
        dst = dst.with_suffix(".wav")

    ffmpeg = ffmpeg_path()
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-ac", str(TARGET_CHANNELS),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and dst.is_file() and dst.stat().st_size > 44:
            return dst
        ffmpeg_error = (proc.stderr or "").strip()[-500:]
    else:
        ffmpeg_error = "ffmpeg not found on PATH"

    # --- pure Python fallback -------------------------------------------------
    try:
        import numpy as np
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - depends on the host
        raise RuntimeError(
            "Cannot convert audio to 16 kHz mono WAV. Install ffmpeg or "
            f"`pip install soundfile numpy scipy`. ffmpeg said: {ffmpeg_error}"
        ) from exc

    data, src_rate = sf.read(str(src), always_2d=True)
    data = np.asarray(data, dtype=np.float32)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    if src_rate != TARGET_SAMPLE_RATE:
        try:
            from math import gcd

            from scipy.signal import resample_poly

            data = resample_poly(
                data[:, 0], TARGET_SAMPLE_RATE // gcd(src_rate, TARGET_SAMPLE_RATE),
                src_rate // gcd(src_rate, TARGET_SAMPLE_RATE),
            )[:, None]
        except Exception:
            data = _resample_linear(data, src_rate, TARGET_SAMPLE_RATE)
    sf.write(str(dst), data, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return dst


def concat_wavs(paths: list[str | Path], dst: str | Path) -> Path:
    """Concatenate PCM WAV files that share the same sample rate / channels."""
    if not paths:
        raise ValueError("concat_wavs() needs at least one input file")
    if len(paths) == 1:
        shutil.copyfile(paths[0], dst)
        return Path(dst)

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
    with wave.open(str(dst), "wb") as out:
        out.setparams(params)
        for item in paths:
            with wave.open(str(item), "rb") as handle:
                if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (
                    params.nchannels, params.sampwidth, params.framerate
                ):
                    raise ValueError(
                        f"Cannot concatenate {item}: sample rate / channel layout differs"
                    )
                out.writeframes(handle.readframes(handle.getnframes()))
    return dst


def write_silence(dst: str | Path, seconds: float, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    """Write a silent mono WAV file (used for idle-mode tests)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames = int(max(0.0, seconds) * sample_rate)
    with wave.open(str(dst), "wb") as handle:
        handle.setnchannels(TARGET_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00" * (frames * 2))
    return dst


def dump_json(obj) -> str:
    """Small helper so manifests are written with stable key ordering."""
    return json.dumps(obj, indent=2, sort_keys=True, default=str)
