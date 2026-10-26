"""Text-to-speech for Script2Presenter.

Three providers, all free and all usable on Google Colab without an API key:

``edge``
    Microsoft's neural voices (Azure Edge read-aloud endpoint).  Highest quality of
    the three and the default.  Requires network access at synthesis time.
    *Verified locally on 2026-10-22 with edge-tts 7.3.1 (322 voices listed).*

``piper``
    Piper / rhasspy - a fully **offline** ONNX neural TTS (VITS) model that runs on
    the CPU in real time.  Voice models are MIT licensed and downloaded once
    (~63 MB) from the ``rhasspy/piper-voices`` Hugging Face repository.  This is the
    provider to use when the demo must work without any network access.
    *Verified locally on 2026-10-22 with piper-tts 1.8.0 (16 kHz mono output).*

``gtts``
    Google Translate TTS.  Pure-python fallback, no model download, but it is
    occasionally rate limited from datacentre IPs.

Notes
-----
* No provider needs a paid plan, an account or a credential of any kind.
* ``kokoro-onnx`` was evaluated as a fourth option but rejected: it requires
  ``numpy>=2.0.2`` while SadTalker needs ``numpy<1.24`` (see
  :func:`s2p.patches.required_numpy_spec`), so the two cannot coexist in one
  environment.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .audio_utils import concat_wavs, probe_audio, to_wav_16k_mono
from .config import DEFAULT_VOICE, MAX_SCRIPT_CHARS, PROJECT_ROOT

__all__ = [
    "PROVIDER_ORDER",
    "PIPER_VOICES",
    "TTSError",
    "TTSResult",
    "available_providers",
    "list_voices",
    "synthesize",
]

#: Order in which the ``auto`` provider tries the backends.
PROVIDER_ORDER = ("edge", "piper", "gtts")

#: Piper voices that are known to work well for narration.  ``path`` is relative to
#: ``rhasspy/piper-voices`` on Hugging Face.
PIPER_VOICES: dict[str, str] = {
    "en_US-amy-low": "en/en_US/amy/low",
    "en_US-amy-medium": "en/en_US/amy/medium",
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_US-ryan-high": "en/en_US/ryan/high",
    "en_GB-alba-medium": "en/en_GB/alba/medium",
    "en_GB-alan-medium": "en/en_GB/alan/medium",
}

PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

#: Languages accepted by the gTTS fallback.
GTTS_LANGS = ("en", "en-uk", "en-au", "en-us", "fr", "de", "es", "it")

#: Where Piper voice models are cached.
PIPER_MODEL_DIR = PROJECT_ROOT / "models" / "piper"

#: Cache for the (network dependent) Edge voice list.
_EDGE_VOICE_CACHE = PROJECT_ROOT / "models" / "edge_voices.json"

_MAX_CHUNK_CHARS = 1800


class TTSError(RuntimeError):
    """Raised when every configured TTS provider failed."""


@dataclass
class TTSResult:
    """Outcome of a successful synthesis."""

    path: Path
    provider: str
    voice: str
    duration: float
    raw_path: Path | None = None
    attempts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "provider": self.provider,
            "voice": self.voice,
            "duration_s": round(self.duration, 3),
            "raw_path": str(self.raw_path) if self.raw_path else None,
            "attempts": list(self.attempts),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _report(progress, fraction: float, message: str) -> None:
    if progress is not None:
        try:
            progress(fraction, message)
        except TypeError:
            progress(message)


def _run_async(factory):
    """Run ``factory()`` (which returns a coroutine) whether or not a loop is running.

    ``edge-tts`` is async.  Gradio and Jupyter sometimes call our synchronous API
    from inside a running event loop, where ``asyncio.run`` would raise
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``.
    In that case the coroutine is executed on a private thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    import concurrent.futures

    def _worker():
        return asyncio.run(factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result()


def split_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split a script into chunks of at most ``max_chars`` characters.

    Splits happen at sentence boundaries when possible so that the prosody of each
    chunk stays natural.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        while len(sentence) > max_chars:
            # Hard-split a single very long sentence on whitespace.
            head, sentence = sentence[:max_chars], sentence[max_chars:]
            space = head.rfind(" ")
            if space > 0:
                head, sentence = head[:space], head[space:] + sentence
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head.strip())
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence.strip()
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def available_providers() -> dict[str, bool]:
    """Report which TTS providers can be used in the current environment."""
    status: dict[str, bool] = {}
    try:
        import edge_tts  # noqa: F401

        status["edge"] = True
    except Exception:
        status["edge"] = False
    try:
        import piper  # noqa: F401

        status["piper"] = True
    except Exception:
        status["piper"] = False
    try:
        import gtts  # noqa: F401

        status["gtts"] = True
    except Exception:
        status["gtts"] = False
    return status


def list_voices(provider: str = "edge") -> list[str]:
    """Return the voice identifiers available for ``provider``."""
    provider = provider.lower()
    if provider == "edge":
        return _edge_voices()
    if provider == "piper":
        return sorted(PIPER_VOICES)
    if provider == "gtts":
        return list(GTTS_LANGS)
    raise ValueError(f"Unknown TTS provider: {provider!r}")


def _edge_voices() -> list[str]:
    cached = _load_edge_voice_cache()
    if cached:
        return cached
    try:
        import edge_tts
    except Exception as exc:
        raise TTSError(
            "edge-tts is not installed. Run `pip install edge-tts` in this environment."
        ) from exc

    async def _fetch() -> list[str]:
        voices = await edge_tts.list_voices()
        return [v["ShortName"] for v in voices]

    try:
        names = _run_async(_fetch)
    except Exception as exc:
        raise TTSError(f"Could not list Edge TTS voices: {exc}") from exc
    try:
        _EDGE_VOICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _EDGE_VOICE_CACHE.write_text(json.dumps(names, indent=2), encoding="utf-8")
    except Exception:
        pass
    return names


def _load_edge_voice_cache() -> list[str]:
    try:
        if _EDGE_VOICE_CACHE.is_file():
            return json.loads(_EDGE_VOICE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return []
    return []


# --------------------------------------------------------------------------- #
# Provider implementations
# --------------------------------------------------------------------------- #

def _synthesize_edge(text: str, voice: str, out_file: Path, rate: str | None) -> None:
    import edge_tts

    async def _run() -> None:
        kwargs = {}
        if rate:
            kwargs["rate"] = rate
        communicate = edge_tts.Communicate(text, voice, **kwargs)
        await communicate.save(str(out_file))

    try:
        _run_async(_run)
    except TypeError:
        # Older edge-tts releases do not accept the keyword arguments above.
        async def _run_legacy() -> None:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(out_file))

        _run_async(_run_legacy)


def _piper_model_paths(voice: str, model_dir: Path) -> tuple[Path, Path]:
    if voice not in PIPER_VOICES:
        raise TTSError(
            f"Unknown Piper voice {voice!r}. Known voices: {sorted(PIPER_VOICES)}"
        )
    folder = PIPER_VOICES[voice]
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / f"{voice}.onnx"
    config = model_dir / f"{voice}.onnx.json"
    base = f"{PIPER_BASE_URL}/{folder}"
    for remote, local in ((f"{voice}.onnx", model), (f"{voice}.onnx.json", config)):
        if not local.is_file() or local.stat().st_size == 0:
            try:
                urllib.request.urlretrieve(f"{base}/{remote}", local)
            except Exception as exc:
                raise TTSError(f"Could not download Piper voice file {remote}: {exc}") from exc
    return model, config


def _synthesize_piper(text: str, voice: str, out_file: Path, model_dir: Path) -> None:
    try:
        from piper import PiperVoice
    except Exception as exc:
        raise TTSError(
            "piper-tts is not installed. Run `pip install piper-tts` in this environment."
        ) from exc

    model, config = _piper_model_paths(voice, model_dir)
    try:
        piper_voice = PiperVoice.load(str(model), config_path=str(config))
    except Exception as exc:
        raise TTSError(f"Could not load Piper voice {voice!r}: {exc}") from exc

    import wave

    with wave.open(str(out_file), "wb") as handle:
        piper_voice.synthesize_wav(text, handle)


def _synthesize_gtts(text: str, voice: str, out_file: Path) -> None:
    try:
        from gtts import gTTS
    except Exception as exc:
        raise TTSError(
            "gTTS is not installed. Run `pip install gTTS` in this environment."
        ) from exc

    lang = voice if voice in GTTS_LANGS else "en"
    try:
        gTTS(text=text, lang=lang).save(str(out_file))
    except Exception as exc:
        raise TTSError(f"gTTS synthesis failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def synthesize(
    text: str,
    provider: str = "auto",
    voice: str | None = None,
    out_dir: str | Path | None = None,
    rate: str | None = None,
    progress=None,
    piper_model_dir: str | Path | None = None,
) -> TTSResult:
    """Turn ``text`` into a 16 kHz mono WAV file.

    Parameters
    ----------
    text:
        The script to speak.
    provider:
        ``"edge"``, ``"piper"``, ``"gtts"`` or ``"auto"`` (try all in order).
    voice:
        Provider specific voice identifier.  Defaults to
        :data:`s2p.config.DEFAULT_VOICE` for Edge, to ``en_US-amy-medium`` for
        Piper and to ``en`` for gTTS.
    out_dir:
        Where to write the WAV file.  Defaults to ``<project>/work/tts``.
    rate:
        Optional speaking-rate modifier understood by Edge TTS (e.g. ``"+10%"``).
    progress:
        Optional ``callable(fraction: float, message: str)`` used by the UI.

    Returns
    -------
    TTSResult
        The path of the generated WAV file plus metadata.
    """
    text = (text or "").strip()
    if not text:
        raise TTSError("The script is empty - there is nothing to speak.")
    if len(text) > MAX_SCRIPT_CHARS:
        raise TTSError(
            f"The script is too long ({len(text)} characters, limit {MAX_SCRIPT_CHARS}). "
            "Shorten it or split it into several videos."
        )

    out_dir = Path(out_dir) if out_dir else (PROJECT_ROOT / "work" / "tts")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(piper_model_dir) if piper_model_dir else PIPER_MODEL_DIR

    provider = (provider or "auto").lower()
    if provider == "auto":
        chain = list(PROVIDER_ORDER)
    elif provider in PROVIDER_ORDER:
        chain = [provider]
    else:
        raise TTSError(
            f"Unknown TTS provider {provider!r}. Choose one of {PROVIDER_ORDER} or 'auto'."
        )

    attempts: list[str] = []
    chunks = split_text(text)

    for candidate in chain:
        selected_voice = _default_voice(candidate, voice)
        try:
            _report(progress, 0.05, f"Synthesising speech with '{candidate}' ({selected_voice}) ...")
            raw_files: list[Path] = []
            for index, chunk in enumerate(chunks):
                suffix = ".wav" if candidate == "piper" else ".mp3"
                raw = out_dir / f"tts_{candidate}_{index:02d}{suffix}"
                if candidate == "edge":
                    _synthesize_edge(chunk, selected_voice, raw, rate)
                elif candidate == "piper":
                    _synthesize_piper(chunk, selected_voice, raw, model_dir)
                else:
                    _synthesize_gtts(chunk, selected_voice, raw)
                if not raw.is_file() or raw.stat().st_size == 0:
                    raise TTSError(f"{candidate} produced an empty file for chunk {index}.")
                raw_files.append(raw)

            _report(progress, 0.6, "Converting speech to 16 kHz mono WAV ...")
            wav_files = [
                to_wav_16k_mono(raw, out_dir / f"{raw.stem}_16k.wav") for raw in raw_files
            ]
            final = out_dir / "speech_16k_mono.wav"
            if len(wav_files) == 1:
                shutil.copyfile(wav_files[0], final)
            else:
                concat_wavs(wav_files, final)

            info = probe_audio(final)
            _report(progress, 1.0, f"Speech ready ({info.duration:.1f}s).")
            return TTSResult(
                path=final,
                provider=candidate,
                voice=selected_voice,
                duration=float(info.duration),
                raw_path=raw_files[0],
                attempts=attempts,
            )
        except Exception as exc:
            attempts.append(f"{candidate}: {exc}")
            continue

    raise TTSError(
        "All text-to-speech providers failed:\n  - " + "\n  - ".join(attempts)
    )


def _default_voice(provider: str, voice: str | None) -> str:
    """Pick a voice that actually belongs to ``provider``.

    The UI passes a single voice selection while the ``auto`` chain may move on to
    another provider, so a voice id from a different engine must be replaced
    instead of failing the whole synthesis.
    """
    if provider == "edge":
        return voice or DEFAULT_VOICE
    if provider == "piper":
        return voice if voice in PIPER_VOICES else "en_US-amy-medium"
    return voice if voice in GTTS_LANGS else "en"
