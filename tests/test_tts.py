"""Tests for :mod:`s2p.tts`.

The network dependent tests are marked ``network`` and are skipped unless
``S2P_RUN_NETWORK_TESTS=1`` is set (see ``pytest.ini``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from s2p.tts import (
    PROVIDER_ORDER,
    TTSError,
    available_providers,
    list_voices,
    split_text,
    synthesize,
)

RUN_NETWORK = os.environ.get("S2P_RUN_NETWORK_TESTS") == "1"
network_only = pytest.mark.skipif(
    not RUN_NETWORK, reason="network test - set S2P_RUN_NETWORK_TESTS=1 to enable"
)


# --------------------------------------------------------------------------- #
# Pure logic
# --------------------------------------------------------------------------- #

def test_split_text_keeps_short_text_whole():
    assert split_text("Hello there, this is a short script.") == [
        "Hello there, this is a short script."
    ]


def test_split_text_splits_on_sentences():
    text = " ".join(f"This is sentence number {i}." for i in range(200))
    chunks = split_text(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    # No words may be lost.
    assert " ".join(chunks).split() == text.split()


def test_split_text_handles_empty_input():
    assert split_text("   ") == []


def test_provider_order_is_stable():
    assert PROVIDER_ORDER == ("edge", "piper", "gtts")


def test_available_providers_reports_booleans():
    status = available_providers()
    assert set(status) == {"edge", "piper", "gtts"}
    assert all(isinstance(value, bool) for value in status.values())


def test_list_voices_for_piper_is_offline():
    voices = list_voices("piper")
    assert "en_US-amy-medium" in voices


def test_list_voices_rejects_unknown_provider():
    with pytest.raises(ValueError):
        list_voices("does-not-exist")


def test_synthesize_rejects_empty_script(tmp_path: Path):
    with pytest.raises(TTSError):
        synthesize("", provider="auto", out_dir=tmp_path)


def test_synthesize_rejects_too_long_script(tmp_path: Path):
    with pytest.raises(TTSError):
        synthesize("a" * 6000, provider="auto", out_dir=tmp_path)


def test_synthesize_rejects_unknown_provider(tmp_path: Path):
    with pytest.raises(TTSError):
        synthesize("hello", provider="not-a-provider", out_dir=tmp_path)


# --------------------------------------------------------------------------- #
# Network dependent
# --------------------------------------------------------------------------- #

@network_only
def test_edge_tts_produces_playable_audio(tmp_path: Path):
    result = synthesize(
        "Hello, this is Script2Presenter speaking.",
        provider="edge",
        voice="en-US-AriaNeural",
        out_dir=tmp_path,
    )
    assert result.path.is_file()
    assert result.provider == "edge"
    assert result.duration > 0.5
    from s2p.audio_utils import probe_audio

    info = probe_audio(result.path)
    assert info.sample_rate == 16000
    assert info.channels == 1


@network_only
def test_piper_tts_produces_playable_audio(tmp_path: Path):
    result = synthesize(
        "Hello, this is Script2Presenter speaking.",
        provider="piper",
        voice="en_US-amy-low",
        out_dir=tmp_path,
    )
    assert result.path.is_file()
    assert result.provider == "piper"
    assert result.duration > 0.5
