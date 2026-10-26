"""Tests for :mod:`s2p.pipeline` (orchestration, validation, manifest).

The SadTalker backend is replaced by a stub so that the orchestration logic can be
verified on a machine without a GPU.  Real GPU inference is verified in Colab.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from s2p.pipeline import PipelineError, Script2Presenter, validate_portrait
from s2p.sadtalker_backend import GenerationResult
from s2p.tts import TTSResult


class StubBackend:
    """Mimics :class:`s2p.sadtalker_backend.SadTalkerBackend`."""

    def __init__(self, settings, device: str = "auto"):
        self.settings = settings
        self.device_request = device
        self.calls: list[dict] = []

    def warmup(self):  # pragma: no cover - not used by the pipeline
        pass

    def generate(self, image_path, audio_path, out_dir, size=256, preprocess="crop",
                 still=False, pose_style=0, batch_size=2, enhancer=None,
                 expression_scale=1.0, progress=None):
        self.calls.append(
            dict(image_path=str(image_path), audio_path=str(audio_path), size=size,
                 preprocess=preprocess, still=still, pose_style=pose_style,
                 batch_size=batch_size, enhancer=enhancer,
                 expression_scale=expression_scale)
        )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video = out_dir / "stub_video.mp4"
        video.write_bytes(b"\x00" * 64)

        from s2p.audio_utils import probe_audio

        frames = int(probe_audio(audio_path).duration * 25)
        if progress:
            progress(1.0, "done")
        return GenerationResult(
            video_path=video,
            save_dir=out_dir,
            duration_s=frames / 25.0,
            frames=frames,
            size=size,
            preprocess=preprocess,
            enhancer=enhancer,
            elapsed_s=1.25,
            device="cpu",
        )


@pytest.fixture()
def app(settings, monkeypatch):
    """A Script2Presenter whose TTS and SadTalker stages are stubbed out."""
    instance = Script2Presenter(settings=settings)
    instance.backend = StubBackend(settings)

    def fake_synthesize(text, provider="auto", voice=None, out_dir=None, progress=None,
                        piper_model_dir=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "speech_16k_mono.wav"
        sr = 16000
        seconds = max(1.0, len(text) / 15.0)
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        sf.write(str(path), 0.1 * np.sin(2 * np.pi * 200 * t), sr, subtype="PCM_16")
        if progress:
            progress(1.0, "stub tts")
        return TTSResult(path=path, provider="stub", voice="stub", duration=seconds)

    monkeypatch.setattr("s2p.pipeline.synthesize", fake_synthesize)
    return instance


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_validate_portrait_accepts_png(portrait: Path):
    assert validate_portrait(portrait) == portrait


def test_validate_portrait_rejects_missing_file(tmp_path: Path):
    with pytest.raises(PipelineError, match="not found"):
        validate_portrait(tmp_path / "nope.png")


def test_validate_portrait_rejects_wrong_extension(tmp_path: Path):
    path = tmp_path / "portrait.bmp"
    path.write_bytes(b"BM")
    with pytest.raises(PipelineError, match="Unsupported image type"):
        validate_portrait(path)


def test_validate_portrait_rejects_broken_image(tmp_path: Path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not a png")
    with pytest.raises(PipelineError, match="not a readable image"):
        validate_portrait(path)


def test_validate_portrait_rejects_tiny_image(tmp_path: Path):
    from PIL import Image

    path = tmp_path / "tiny.png"
    Image.new("RGB", (16, 16)).save(path)
    with pytest.raises(PipelineError, match="too small"):
        validate_portrait(path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def test_run_requires_a_script(app, portrait: Path):
    with pytest.raises(PipelineError, match="script"):
        app.run(portrait, "   ")


def test_run_rejects_missing_portrait(app, tmp_path: Path):
    with pytest.raises(PipelineError):
        app.run(tmp_path / "missing.png", "hello world")


def test_run_produces_video_and_manifest(app, portrait: Path):
    result = app.run(
        portrait,
        "Hello professor, this is my Script2Presenter demo.",
        size=256,
        preprocess="crop",
        enhancer="gfpgan",
    )

    assert result.video_path.is_file()
    assert result.audio_path.is_file()
    assert result.manifest_path.is_file()
    assert result.generation.frames > 0
    assert result.timings["total_s"] >= result.timings["tts_s"]

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation"]["enhancer"] == "gfpgan"
    assert manifest["inputs"]["script_chars"] > 0
    assert manifest["audio"]["sample_rate"] == 16000
    assert manifest["timings"]["total_s"] > 0


def test_run_passes_settings_to_the_backend(app, portrait: Path):
    app.run(portrait, "hello", size=512, preprocess="full", still=True, pose_style=7)
    call = app.backend.calls[-1]
    assert call["size"] == 512
    assert call["preprocess"] == "full"
    assert call["still"] is True
    assert call["pose_style"] == 7


def test_run_keeps_the_original_portrait(app, portrait: Path):
    before = portrait.read_bytes()
    app.run(portrait, "hello")
    assert portrait.read_bytes() == before

# --------------------------------------------------------------------------- #
# Working directory
# --------------------------------------------------------------------------- #

def test_backend_runs_inside_the_sadtalker_root(settings):
    """gfpgan passes ``model_rootpath='gfpgan/weights'`` (relative) to facexlib."""
    from s2p.sadtalker_backend import SadTalkerBackend

    backend = SadTalkerBackend(settings)
    before = Path.cwd()

    with backend._inside_sadtalker_root():
        assert Path.cwd().resolve() == settings.sadtalker_root.resolve()

    assert Path.cwd() == before


def test_backend_restores_the_cwd_after_an_error(settings):
    from s2p.sadtalker_backend import SadTalkerBackend

    backend = SadTalkerBackend(settings)
    before = Path.cwd()

    with pytest.raises(RuntimeError):
        with backend._inside_sadtalker_root():
            raise RuntimeError("boom")

    assert Path.cwd() == before


def test_notebook_matplotlib_backend_is_replaced(monkeypatch):
    """Colab exports MPLBACKEND=module://matplotlib_inline..., which the venv lacks."""
    from s2p.sadtalker_backend import _force_headless_matplotlib

    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    _force_headless_matplotlib()
    assert os.environ["MPLBACKEND"] == "Agg"


def test_explicit_matplotlib_backend_is_kept(monkeypatch):
    from s2p.sadtalker_backend import _force_headless_matplotlib

    monkeypatch.setenv("MPLBACKEND", "svg")
    _force_headless_matplotlib()
    assert os.environ["MPLBACKEND"] == "svg"
