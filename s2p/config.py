"""Central configuration for Script2Presenter.

Everything that can change between the local (CPU, no-GPU) development machine and
the Google Colab T4 runtime is resolved here, so the rest of the code never has to
guess where the SadTalker checkout or its checkpoints live.

No credentials, tokens or API keys are read or stored by this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: SadTalker's audio front-end expects 16 kHz mono PCM.
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

#: SadTalker renders at a fixed frame rate.
FPS = 25

#: Default neural voice (free Microsoft Edge TTS voice, no API key required).
DEFAULT_VOICE = "en-US-AriaNeural"

#: ``auto`` walks the provider chain in :mod:`s2p.tts`.
DEFAULT_PROVIDER = "auto"

#: SadTalker face-renderer resolutions that ship as pretrained checkpoints.
SUPPORTED_SIZES = (256, 512)
DEFAULT_SIZE = 256

#: ``crop`` keeps the animated face only, ``full`` pastes it back on the portrait.
DEFAULT_PREPROCESS = "crop"
SUPPORTED_PREPROCESS = ("crop", "extcrop", "resize", "full", "extfull")

DEFAULT_POSE_STYLE = 0
DEFAULT_BATCH_SIZE = 2
DEFAULT_EXPRESSION_SCALE = 1.0
DEFAULT_ENHANCER = None  # None | "gfpgan" | "RestoreFormer"

#: Upper bound on a single script, keeps the demo snappy and avoids abuse.
MAX_SCRIPT_CHARS = 5000

#: Candidate locations for the official SadTalker checkout, in priority order.
_SADTALKER_CANDIDATES = (
    "/content/SadTalker",
    str(PROJECT_ROOT / "third_party" / "SadTalker"),
    str(PROJECT_ROOT / "SadTalker"),
    str(PROJECT_ROOT.parent / "SadTalker"),
)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings."""

    sadtalker_root: Path
    checkpoint_dir: Path
    output_dir: Path
    work_dir: Path
    device: str = "auto"

    @property
    def config_dir(self) -> Path:
        """SadTalker's ``src/config`` directory (holds the YAML + 3DMM prior)."""
        return self.sadtalker_root / "src" / "config"

    @property
    def face_weights_dir(self) -> Path:
        """Where facexlib looks for the alignment / detection checkpoints."""
        return self.sadtalker_root / "gfpgan" / "weights"

    def describe(self) -> dict:
        return {
            "sadtalker_root": str(self.sadtalker_root),
            "checkpoint_dir": str(self.checkpoint_dir),
            "output_dir": str(self.output_dir),
            "work_dir": str(self.work_dir),
            "device": self.device,
        }


def resolve_sadtalker_root(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the official SadTalker checkout.

    Order of precedence: explicit argument, ``S2P_SADTALKER_ROOT`` env var, then a
    list of well-known locations (Colab's ``/content/SadTalker`` first).
    """
    if explicit:
        return Path(explicit).expanduser().resolve()

    env_root = _env_path("S2P_SADTALKER_ROOT")
    if env_root:
        return env_root

    for candidate in _SADTALKER_CANDIDATES:
        path = Path(candidate)
        if (path / "src" / "utils" / "init_path.py").is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Could not find the SadTalker checkout. Clone it with\n"
        "    git clone https://github.com/OpenTalker/SadTalker.git /content/SadTalker\n"
        "or set the S2P_SADTALKER_ROOT environment variable to its path."
    )


def load_settings(
    sadtalker_root: str | os.PathLike | None = None,
    output_dir: str | os.PathLike | None = None,
    work_dir: str | os.PathLike | None = None,
    device: str = "auto",
) -> Settings:
    """Build a :class:`Settings` instance from arguments + environment."""
    root = resolve_sadtalker_root(sadtalker_root)
    checkpoints = _env_path("S2P_CHECKPOINT_DIR") or (root / "checkpoints")
    outputs = Path(output_dir).expanduser().resolve() if output_dir else (
        _env_path("S2P_OUTPUT_DIR") or (PROJECT_ROOT / "outputs")
    )
    work = Path(work_dir).expanduser().resolve() if work_dir else (
        _env_path("S2P_WORK_DIR") or (PROJECT_ROOT / "work")
    )
    return Settings(
        sadtalker_root=root,
        checkpoint_dir=checkpoints,
        output_dir=outputs,
        work_dir=work,
        device=device,
    )


def ensure_directories(settings: Settings) -> Settings:
    """Create the output/work/checkpoint directories if they do not exist yet."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # facexlib resolves its weights relative to the current working directory
    # (``<cwd>/gfpgan/weights``) and creates the folder on demand only sometimes.
    try:
        settings.face_weights_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return settings
