"""Tests for :mod:`s2p.audio_utils`."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from s2p.audio_utils import (
    concat_wavs,
    probe_audio,
    to_wav_16k_mono,
    write_silence,
)


def test_probe_audio_reads_wav(wav_16k: Path):
    info = probe_audio(wav_16k)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert 1.5 < info.duration < 2.5


def test_to_wav_16k_mono_is_idempotent_for_16k_wav(wav_16k: Path, tmp_path: Path):
    out = to_wav_16k_mono(wav_16k, tmp_path / "out.wav")
    assert out.is_file()
    info = probe_audio(out)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.duration == pytest.approx(probe_audio(wav_16k).duration, abs=0.05)


def test_to_wav_16k_mono_downsamples_and_monoises(tmp_path: Path):
    import soundfile as sf

    src = tmp_path / "stereo44k.wav"
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    sf.write(str(src), stereo, sr, subtype="PCM_16")

    out = to_wav_16k_mono(src, tmp_path / "converted.wav")
    info = probe_audio(out)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert 0.9 < info.duration < 1.1


def test_to_wav_16k_mono_changes_extension(tmp_path: Path, wav_16k: Path):
    out = to_wav_16k_mono(wav_16k, tmp_path / "no_extension")
    assert out.suffix == ".wav"
    assert out.is_file()


def test_concat_wavs(tmp_path: Path, wav_16k: Path):
    out = concat_wavs([wav_16k, wav_16k], tmp_path / "joined.wav")
    with wave.open(str(out), "rb") as handle:
        assert handle.getnframes() == 2 * 16000 * 2
        assert handle.getframerate() == 16000


def test_concat_wavs_single_file(tmp_path: Path, wav_16k: Path):
    out = concat_wavs([wav_16k], tmp_path / "copy.wav")
    assert out.read_bytes() == wav_16k.read_bytes()


def test_write_silence(tmp_path: Path):
    out = write_silence(tmp_path / "silence.wav", 0.5)
    info = probe_audio(out)
    assert info.duration == pytest.approx(0.5, abs=0.01)
    assert info.sample_rate == 16000


def test_concat_rejects_mismatched_formats(tmp_path: Path, wav_16k: Path):
    import soundfile as sf

    other = tmp_path / "other.wav"
    sf.write(str(other), np.zeros(1000), 8000, subtype="PCM_16")
    with pytest.raises(ValueError):
        concat_wavs([wav_16k, other], tmp_path / "bad.wav")
