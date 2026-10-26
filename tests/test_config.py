"""Tests for :mod:`s2p.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from s2p.config import (
    DEFAULT_SIZE,
    FPS,
    TARGET_SAMPLE_RATE,
    Settings,
    load_settings,
    resolve_sadtalker_root,
)


def test_constants_match_sadtalker_expectations():
    # SadTalker's audio front-end is trained on 16 kHz audio and renders at 25 fps.
    assert TARGET_SAMPLE_RATE == 16000
    assert FPS == 25
    assert DEFAULT_SIZE in (256, 512)


def test_resolve_sadtalker_root_prefers_explicit(tmp_path: Path):
    root = resolve_sadtalker_root(tmp_path)
    assert root == tmp_path.resolve()


def test_resolve_sadtalker_root_uses_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("S2P_SADTALKER_ROOT", str(tmp_path))
    assert resolve_sadtalker_root() == tmp_path.resolve()


def test_resolve_sadtalker_root_detects_checkout(monkeypatch, tmp_path: Path):
    checkout = tmp_path / "SadTalker"
    (checkout / "src" / "utils").mkdir(parents=True)
    (checkout / "src" / "utils" / "init_path.py").write_text("# stub\n", encoding="utf-8")
    monkeypatch.setenv("S2P_SADTALKER_ROOT", str(checkout))
    assert resolve_sadtalker_root() == checkout.resolve()


def test_resolve_sadtalker_root_raises_when_missing(monkeypatch):
    monkeypatch.delenv("S2P_SADTALKER_ROOT", raising=False)
    monkeypatch.setattr("s2p.config._SADTALKER_CANDIDATES", ())
    with pytest.raises(FileNotFoundError):
        resolve_sadtalker_root()


def test_load_settings_creates_expected_paths(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("S2P_SADTALKER_ROOT", str(tmp_path))
    monkeypatch.delenv("S2P_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("S2P_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("S2P_WORK_DIR", raising=False)

    settings = load_settings()
    assert settings.sadtalker_root == tmp_path.resolve()
    assert settings.checkpoint_dir == (tmp_path / "checkpoints").resolve()
    assert settings.config_dir == (tmp_path / "src" / "config").resolve()
    assert settings.face_weights_dir == (tmp_path / "gfpgan" / "weights").resolve()
    assert isinstance(settings.describe(), dict)
