"""End-to-end orchestration: portrait + script -> talking-head MP4.

    portrait ──► [1] TTS ──► 16 kHz mono WAV ──┐
                                               ├──► [2] SadTalker ──► MP4
    (uploaded) ────────────────────────────────┘

The pipeline is deliberately split into two independent stages so that each of them
can be validated (and profiled) on its own, and so that a failed speech synthesis
never wastes GPU time.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .audio_utils import probe_audio
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EXPRESSION_SCALE,
    DEFAULT_PREPROCESS,
    DEFAULT_PROVIDER,
    DEFAULT_SIZE,
    MAX_SCRIPT_CHARS,
    Settings,
    ensure_directories,
    load_settings,
)
from .sadtalker_backend import GenerationResult, SadTalkerBackend, SadTalkerError
from .tts import TTSError, TTSResult, synthesize

__all__ = ["PipelineError", "PipelineResult", "Script2Presenter", "validate_portrait"]

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class PipelineError(RuntimeError):
    """Raised for user-facing failures (bad input, missing model, ...)."""


@dataclass
class PipelineResult:
    """Everything that happened during one run."""

    video_path: Path
    audio_path: Path
    image_path: Path
    run_dir: Path
    manifest_path: Path
    tts: TTSResult
    generation: GenerationResult
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "video_path": str(self.video_path),
            "audio_path": str(self.audio_path),
            "image_path": str(self.image_path),
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path),
            "tts": self.tts.to_dict(),
            "generation": self.generation.to_dict(),
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }


def validate_portrait(image_path: str | Path) -> Path:
    """Validate that ``image_path`` is a usable portrait.

    Raises :class:`PipelineError` with an actionable message when it is not.
    """
    path = Path(image_path)
    if not path.is_file():
        raise PipelineError(f"Portrait not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise PipelineError(
            f"Unsupported image type {path.suffix!r}. Please upload a JPG or PNG portrait."
        )
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise PipelineError(f"The uploaded file is not a readable image: {exc}") from exc

    if width < 64 or height < 64:
        raise PipelineError(
            f"The portrait is too small ({width}x{height}px). Use at least 256x256px."
        )
    return path


class Script2Presenter:
    """High level API used by the Gradio app, the CLI and the Colab notebook."""

    def __init__(self, settings: Settings | None = None, device: str = "auto"):
        self.settings = ensure_directories(settings or load_settings())
        self.backend = SadTalkerBackend(self.settings, device=device)

    # -- helpers ----------------------------------------------------------- #

    def preflight(self, size: int = DEFAULT_SIZE, preprocess: str = DEFAULT_PREPROCESS):
        """Return the preflight checks of the SadTalker backend."""
        from .sadtalker_backend import preflight as _preflight

        return _preflight(self.settings, size=size, preprocess=preprocess)

    def warmup(self) -> None:
        """Load the models so the first generation is not slowed down by loading."""
        self.backend.warmup()

    # -- main entry point -------------------------------------------------- #

    def run(
        self,
        image_path: str | Path,
        script: str,
        provider: str = DEFAULT_PROVIDER,
        voice: str | None = None,
        size: int = DEFAULT_SIZE,
        preprocess: str = DEFAULT_PREPROCESS,
        still: bool = False,
        pose_style: int = 0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enhancer: str | None = None,
        expression_scale: float = DEFAULT_EXPRESSION_SCALE,
        progress=None,
    ) -> PipelineResult:
        """Generate a talking-head MP4 from a portrait and a script."""
        def _report(fraction: float, message: str) -> None:
            if progress is not None:
                try:
                    progress(fraction, message)
                except TypeError:
                    progress(message)

        script = (script or "").strip()
        if not script:
            raise PipelineError("Please enter the script that should be spoken.")
        if len(script) > MAX_SCRIPT_CHARS:
            raise PipelineError(
                f"The script is too long ({len(script)} characters, limit {MAX_SCRIPT_CHARS})."
            )

        portrait = validate_portrait(image_path)

        run_dir = self.settings.output_dir / (
            time.strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        local_image = run_dir / f"portrait{portrait.suffix.lower()}"
        shutil.copyfile(portrait, local_image)

        # --- stage 1: text to speech ------------------------------------- #
        _report(0.02, "Generating speech from the script ...")
        t0 = time.time()
        try:
            tts_result = synthesize(
                script,
                provider=provider,
                voice=voice,
                out_dir=run_dir / "speech",
                progress=lambda f, m: _report(0.02 + 0.13 * f, m),
            )
        except TTSError as exc:
            raise PipelineError(str(exc)) from exc
        tts_seconds = time.time() - t0

        speech_path = Path(tts_result.path)
        audio_info = probe_audio(speech_path)

        # --- stage 2: SadTalker ------------------------------------------ #
        _report(0.16, "Preparing the SadTalker models ...")
        try:
            generation = self.backend.generate(
                image_path=local_image,
                audio_path=speech_path,
                out_dir=run_dir / "sadtalker",
                size=size,
                preprocess=preprocess,
                still=still,
                pose_style=pose_style,
                batch_size=batch_size,
                enhancer=enhancer,
                expression_scale=expression_scale,
                progress=lambda f, m: _report(0.16 + 0.84 * f, m),
            )
        except SadTalkerError as exc:
            raise PipelineError(str(exc)) from exc

        timings = {
            "tts_s": tts_seconds,
            "sadtalker_s": generation.elapsed_s,
            "total_s": tts_seconds + generation.elapsed_s,
        }

        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": self.settings.describe(),
            "inputs": {
                "portrait": str(local_image),
                "script": script,
                "script_chars": len(script),
            },
            "tts": tts_result.to_dict(),
            "audio": {
                "path": str(speech_path),
                "duration_s": round(float(audio_info.duration), 3),
                "sample_rate": audio_info.sample_rate,
                "channels": audio_info.channels,
            },
            "generation": generation.to_dict(),
            "timings": timings,
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        _report(1.0, f"Done in {timings['total_s']:.1f}s.")
        return PipelineResult(
            video_path=generation.video_path,
            audio_path=speech_path,
            image_path=local_image,
            run_dir=run_dir,
            manifest_path=manifest_path,
            tts=tts_result,
            generation=generation,
            timings=timings,
        )
