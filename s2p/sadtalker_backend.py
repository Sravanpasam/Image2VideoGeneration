"""Thin, defensive wrapper around the official SadTalker inference code.

The upstream project ships ``inference.py`` (CLI) and ``app_sadtalker.py``
(Gradio).  We import the same ``src`` modules directly so that the TTS step and
the video generation can live in a single process, which is what the Gradio app
and the Colab notebook need.

Nothing in this module imports torch at import time, so the rest of the package
stays importable on a machine without a GPU (or without torch at all).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Ensure headless matplotlib before any plotting backends load (e.g. matplotlib-inline)
os.environ.setdefault("MPLBACKEND", "Agg")


def _force_headless_matplotlib() -> None:
    """Make matplotlib work in a subprocess that inherits a notebook's backend.

    Colab exports ``MPLBACKEND=module://matplotlib_inline.backend_inline``.  The
    project runs in its own virtualenv, which does not contain matplotlib-inline,
    so matplotlib rejects that value while importing pyplot:

        KeyError: backend: 'module://matplotlib_inline...' is not a valid value

    Any ``module://`` backend is therefore replaced with the headless ``Agg``
    backend.  An explicit, usable backend chosen by the caller is left alone.
    """
    backend = os.environ.get("MPLBACKEND", "")
    if not backend or backend.startswith("module://"):
        os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib

        if str(matplotlib.get_backend()).startswith("module://"):
            matplotlib.use("Agg", force=True)
    except Exception:  # matplotlib is optional at import time
        pass


_force_headless_matplotlib()

from .config import FPS, Settings
from .patches import apply_all_patches, format_report

__all__ = ["SadTalkerBackend", "SadTalkerError", "GenerationResult", "preflight"]


class SadTalkerError(RuntimeError):
    """Raised when SadTalker cannot produce a video."""


@dataclass
class GenerationResult:
    """Result of one SadTalker run."""

    video_path: Path
    save_dir: Path
    duration_s: float
    frames: int
    size: int
    preprocess: str
    enhancer: str | None
    elapsed_s: float
    device: str

    def to_dict(self) -> dict:
        return {
            "video_path": str(self.video_path),
            "save_dir": str(self.save_dir),
            "duration_s": round(self.duration_s, 3),
            "frames": self.frames,
            "size": self.size,
            "preprocess": self.preprocess,
            "enhancer": self.enhancer,
            "elapsed_s": round(self.elapsed_s, 2),
            "device": self.device,
        }


# --------------------------------------------------------------------------- #
# Preflight checks
# --------------------------------------------------------------------------- #

def preflight(settings: Settings, size: int = 256, preprocess: str = "crop") -> list[tuple[bool, str]]:
    """Check that everything SadTalker needs is present.

    Returns a list of ``(ok, message)`` tuples so the caller can render them.
    """
    from .audio_utils import ffmpeg_path

    checks: list[tuple[bool, str]] = []

    root = settings.sadtalker_root
    checks.append(
        (root.is_dir() and (root / "src" / "utils" / "init_path.py").is_file(),
         f"SadTalker checkout: {root}")
    )
    checks.append(
        ((settings.config_dir / "facerender.yaml").is_file(),
         f"SadTalker config: {settings.config_dir}")
    )

    checkpoint = settings.checkpoint_dir / f"SadTalker_V0.0.2_{size}.safetensors"
    checks.append(
        (checkpoint.is_file() and checkpoint.stat().st_size > 0,
         f"Face model ({size}px): {checkpoint}")
    )

    mapping = settings.checkpoint_dir / (
        "mapping_00109-model.pth.tar" if "full" in preprocess else "mapping_00229-model.pth.tar"
    )
    checks.append((mapping.is_file(), f"Mapping network: {mapping}"))

    for name in ("alignment_WFLW_4HG.pth", "detection_Resnet50_Final.pth"):
        path = settings.face_weights_dir / name
        checks.append((path.is_file(), f"Face model weights: {path}"))

    ffmpeg = ffmpeg_path()
    checks.append((bool(ffmpeg), f"ffmpeg executable: {ffmpeg or 'NOT FOUND'}"))

    # SadTalker writes the final MP4 with imageio, which needs the
    # imageio-ffmpeg backend. Checking it here turns a missing/broken backend
    # into a preflight failure instead of a crash after the whole GPU render.
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        checks.append((True, f"imageio-ffmpeg backend: {exe}"))
    except Exception as exc:
        checks.append((False, f"imageio-ffmpeg writer is not usable: {exc}"))

    try:
        import torch

        cuda = torch.cuda.is_available()
        checks.append(
            (True, f"torch {torch.__version__} - CUDA available: {cuda}"
                   + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))
        )
    except Exception as exc:
        checks.append((False, f"torch is not importable: {exc}"))

    return checks


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #

class SadTalkerBackend:
    """Runs the official SadTalker models on a portrait + a 16 kHz mono WAV."""

    def __init__(self, settings: Settings, device: str = "auto"):
        self.settings = settings
        self.device_request = device
        self._device: str | None = None
        self._modules: dict | None = None
        self._crop_and_extract = None
        self._audio_to_coeff = None
        self._animate_cache: dict[tuple[int, str], object] = {}
        self.patch_report: list[str] = []
        self._lock = threading.RLock()

    # -- setup ------------------------------------------------------------- #

    @contextlib.contextmanager
    def _inside_sadtalker_root(self):
        """Temporarily make the SadTalker checkout the working directory.

        ``gfpgan/utils.py`` passes ``model_rootpath='gfpgan/weights'`` - a
        *relative* path - to facexlib, so the face-restoration weights are only
        found when the process runs from the SadTalker root (which is also how
        upstream's ``inference.py`` is launched).  Without this the enhancer would
        silently re-download ~500 MB into the current directory or fail offline.
        """
        root = self.settings.sadtalker_root
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield
        finally:
            os.chdir(previous)

    def prepare(self) -> list[str]:
        """Apply the compatibility patches and import the SadTalker modules."""
        if self._modules is not None:
            return self.patch_report

        # Ensure headless matplotlib before importing anything that might use it
        _force_headless_matplotlib()
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
        except Exception:
            pass

        results = apply_all_patches(self.settings.sadtalker_root)
        self.patch_report = format_report(results).splitlines()

        root = str(self.settings.sadtalker_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            import torch

            from src.facerender.animate import AnimateFromCoeff
            from src.generate_batch import get_data
            from src.generate_facerender_batch import get_facerender_data
            from src.test_audio2coeff import Audio2Coeff
            from src.utils.init_path import init_path
            from src.utils.preprocess import CropAndExtract
        except Exception as exc:  # pragma: no cover - depends on the environment
            # Give more context for common matplotlib backend issues in Colab
            err = str(exc)
            if "matplotlib" in err.lower() and "backend" in err.lower():
                raise SadTalkerError(
                    "Matplotlib backend misconfigured in this environment. Try setting "
                    "MPLBACKEND=Agg before launching. Original error: " + err
                ) from exc
            raise SadTalkerError(
                "Could not import the SadTalker modules. Make sure the repository is "
                f"cloned to {self.settings.sadtalker_root} and that its dependencies "
                f"are installed. Original error: {exc}"
            ) from exc

        if self.device_request == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device_request

        self._modules = {
            "torch": torch,
            "AnimateFromCoeff": AnimateFromCoeff,
            "get_data": get_data,
            "get_facerender_data": get_facerender_data,
            "Audio2Coeff": Audio2Coeff,
            "init_path": init_path,
            "CropAndExtract": CropAndExtract,
        }
        return self.patch_report

    @property
    def device(self) -> str:
        if self._device is None:
            self.prepare()
        return self._device or "cpu"

    def _paths(self, size: int, preprocess: str) -> dict:
        modules = self._modules or self._prepare_modules()
        return modules["init_path"](
            str(self.settings.checkpoint_dir),
            str(self.settings.config_dir),
            size,
            False,
            preprocess,
        )

    def _prepare_modules(self) -> dict:
        self.prepare()
        assert self._modules is not None
        return self._modules

    def _get_crop_and_extract(self):
        if self._crop_and_extract is None:
            modules = self._prepare_modules()
            paths = self._paths(256, "crop")
            self._crop_and_extract = modules["CropAndExtract"](paths, self.device)
        return self._crop_and_extract

    def _get_audio_to_coeff(self):
        if self._audio_to_coeff is None:
            modules = self._prepare_modules()
            paths = self._paths(256, "crop")
            self._audio_to_coeff = modules["Audio2Coeff"](paths, self.device)
        return self._audio_to_coeff

    def _get_animator(self, size: int, preprocess: str):
        key = (size, preprocess)
        if key not in self._animate_cache:
            modules = self._prepare_modules()
            paths = self._paths(size, preprocess)
            self._animate_cache[key] = modules["AnimateFromCoeff"](paths, self.device)
        return self._animate_cache[key]

    # -- generation -------------------------------------------------------- #

    def generate(
        self,
        image_path: str | Path,
        audio_path: str | Path,
        out_dir: str | Path,
        size: int = 256,
        preprocess: str = "crop",
        still: bool = False,
        pose_style: int = 0,
        batch_size: int = 2,
        enhancer: str | None = None,
        expression_scale: float = 1.0,
        progress=None,
    ) -> GenerationResult:
        """Animate ``image_path`` with ``audio_path`` and return the MP4 path."""
        image_path = Path(image_path).expanduser().resolve()
        audio_path = Path(audio_path).expanduser().resolve()
        out_dir = Path(out_dir).expanduser().resolve()

        with self._lock, self._inside_sadtalker_root():
            return self._generate(
                image_path,
                audio_path,
                out_dir,
                size=size,
                preprocess=preprocess,
                still=still,
                pose_style=pose_style,
                batch_size=batch_size,
                enhancer=enhancer,
                expression_scale=expression_scale,
                progress=progress,
            )

    def _generate(
        self,
        image_path: Path,
        audio_path: Path,
        out_dir: Path,
        size: int,
        preprocess: str,
        still: bool,
        pose_style: int,
        batch_size: int,
        enhancer: str | None,
        expression_scale: float,
        progress,
    ) -> GenerationResult:
        modules = self._prepare_modules()
        torch = modules["torch"]

        out_dir.mkdir(parents=True, exist_ok=True)

        if not image_path.is_file():
            raise SadTalkerError(f"Portrait not found: {image_path}")
        if not audio_path.is_file():
            raise SadTalkerError(f"Audio file not found: {audio_path}")

        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
        save_dir = out_dir / run_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Copy (never move) the inputs so the original files stay usable.
        local_image = save_dir / f"source{image_path.suffix.lower()}"
        local_audio = save_dir / "driven_audio.wav"
        shutil.copyfile(image_path, local_image)
        shutil.copyfile(audio_path, local_audio)

        started = time.time()

        def _report(fraction: float, message: str) -> None:
            if progress is not None:
                try:
                    progress(fraction, message)
                except TypeError:
                    progress(message)

        _report(0.05, "Extracting 3DMM coefficients and landmarks from the portrait ...")
        first_frame_dir = save_dir / "first_frame_dir"
        first_frame_dir.mkdir(parents=True, exist_ok=True)
        crop_and_extract = self._get_crop_and_extract()
        try:
            extracted = crop_and_extract.generate(
                str(local_image), str(first_frame_dir), preprocess, True, size
            )
        except Exception as exc:
            raise SadTalkerError(
                f"Could not process the portrait: {exc}. A clear, frontal face works best."
            ) from exc

        # Upstream returns ``None, None`` (two values) when it finds no face.
        if not extracted or len(extracted) < 3:
            raise SadTalkerError(
                "No face was detected in the portrait. Use a JPG/PNG image with a "
                "single, clearly visible face."
            )

        first_coeff_path, crop_pic_path, crop_info = extracted

        if first_coeff_path is None or crop_pic_path is None:
            raise SadTalkerError(
                "No face was detected in the portrait. Use a JPG/PNG image with a "
                "single, clearly visible face."
            )

        _report(0.25, "Predicting lip motion and head pose from the audio ...")
        audio_to_coeff = self._get_audio_to_coeff()
        batch = modules["get_data"](
            first_coeff_path,
            str(local_audio),
            self.device,
            ref_eyeblink_coeff_path=None,
            still=still,
            idlemode=False,
            length_of_audio=False,
            use_blink=True,
        )
        coeff_path = audio_to_coeff.generate(batch, str(save_dir), pose_style, None)
        frames = int(batch["num_frames"])

        _report(0.45, f"Rendering {frames} frames ...")
        data = modules["get_facerender_data"](
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            str(local_audio),
            batch_size,
            still_mode=still,
            preprocess=preprocess,
            size=size,
            expression_scale=expression_scale,
        )
        animator = self._get_animator(size, preprocess)
        video_path = animator.generate(
            data,
            str(save_dir),
            str(local_image),
            crop_info,
            enhancer=enhancer,
            preprocess=preprocess,
            img_size=size,
        )

        video_path = Path(video_path)
        if not video_path.is_file():
            raise SadTalkerError(f"SadTalker did not produce a video file at {video_path}")

        elapsed = time.time() - started
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _report(1.0, "Video ready.")
        return GenerationResult(
            video_path=video_path,
            save_dir=save_dir,
            duration_s=frames / float(FPS),
            frames=frames,
            size=size,
            preprocess=preprocess,
            enhancer=enhancer,
            elapsed_s=elapsed,
            device=self.device,
        )

    # -- housekeeping ------------------------------------------------------ #

    def warmup(self) -> None:
        """Load all models once so the first real generation is fast."""
        self.prepare()
        with self._lock, self._inside_sadtalker_root():
            self._get_crop_and_extract()
            self._get_audio_to_coeff()
            self._get_animator(256, "crop")


def check_ffmpeg() -> bool:
    """True when an ``ffmpeg`` binary is reachable on PATH."""
    from .audio_utils import ffmpeg_path

    return ffmpeg_path() is not None


def which() -> str:  # pragma: no cover - convenience for the CLI
    return os.environ.get("S2P_SADTALKER_ROOT", "<auto>")
