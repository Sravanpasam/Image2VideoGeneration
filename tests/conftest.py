"""Shared pytest fixtures for the Script2Presenter test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def settings(tmp_path: Path):
    """A :class:`s2p.config.Settings` that points at a fake SadTalker checkout."""
    from s2p.config import Settings

    fake_root = tmp_path / "SadTalker"
    (fake_root / "src" / "config").mkdir(parents=True)
    (fake_root / "src" / "utils").mkdir(parents=True)
    (fake_root / "src" / "utils" / "init_path.py").write_text("# stub\n", encoding="utf-8")
    (fake_root / "checkpoints").mkdir(parents=True)
    (fake_root / "gfpgan" / "weights").mkdir(parents=True)

    return Settings(
        sadtalker_root=fake_root,
        checkpoint_dir=fake_root / "checkpoints",
        output_dir=tmp_path / "outputs",
        work_dir=tmp_path / "work",
    )


@pytest.fixture()
def portrait(tmp_path: Path) -> Path:
    """A small but valid PNG portrait."""
    from PIL import Image

    path = tmp_path / "portrait.png"
    image = Image.new("RGB", (256, 256), color=(120, 140, 160))
    image.save(path)
    return path


@pytest.fixture()
def wav_16k(tmp_path: Path) -> Path:
    """A 2 second 16 kHz mono sine wave."""
    import numpy as np
    import soundfile as sf

    path = tmp_path / "tone.wav"
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    sf.write(str(path), 0.2 * np.sin(2 * np.pi * 220 * t), sr, subtype="PCM_16")
    return path
