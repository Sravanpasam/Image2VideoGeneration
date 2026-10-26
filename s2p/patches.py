"""Idempotent compatibility patches for SadTalker and its dependency stack.

SadTalker was released in 2023 against the 2023 PyTorch / NumPy / Pillow stack.
The Colab runtime of 2026 ships NumPy 2.x, Pillow >= 10 and PyTorch >= 2.6, which
breaks four separate things in the upstream code:

=========================================  ===========================================
Upstream breakage                          Fix applied here
=========================================  ===========================================
``np.VisibleDeprecationWarning`` removed   Avoided entirely: we pin ``numpy<1.24``
in NumPy 2.x                              (see :func:`required_numpy_spec`).
``np.float`` removed in NumPy 1.24         Same as above.
``Image.ANTIALIAS`` removed in Pillow 10   ``patch_pillow_antialias``
``torch.load`` defaults to                 ``patch_torch_load_weights_only``
``weights_only=True`` in PyTorch 2.6+
``basicsr`` imports the removed             ``patch_basicsr_functional_tensor``
``torchvision.transforms.functional_tensor``
``from gfpgan import GFPGANer`` is a hard   ``patch_gfpgan_lazy_import``
import of the optional enhancer
=========================================  ===========================================

Every patch writes a ``<file>.orig`` backup once and is safe to run repeatedly.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PatchResult",
    "apply_all_patches",
    "patch_basicsr_functional_tensor",
    "patch_gfpgan_lazy_import",
    "patch_pillow_antialias",
    "patch_torch_load_weights_only",
    "required_numpy_spec",
]


@dataclass
class PatchResult:
    """Outcome of a single patch."""

    name: str
    applied: bool
    detail: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        state = "PATCHED" if self.applied else "OK      "
        return f"[{state}] {self.name}: {self.detail}"


def _backup(path: Path, backup_suffix: str = ".orig") -> Path | None:
    """Copy ``path`` to its ``.orig`` backup once. Returns the backup path."""
    backup = path.with_suffix(path.suffix + backup_suffix)
    if backup.exists():
        return None
    shutil.copy2(path, backup)
    return backup


def _replace_in_file(path: Path, old: str, new: str, backup_suffix: str = ".orig") -> tuple[bool, str]:
    """Replace ``old`` with ``new`` inside ``path``.

    Returns ``(changed, detail)``. Never overwrites an existing backup, and never
    writes a Python file that no longer compiles.
    """
    if not path.is_file():
        return False, f"file not found: {path}"
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False, f"already applied: {path}"
    if old not in text:
        return False, f"pattern not found in {path}"
    updated = text.replace(old, new)
    if path.suffix == ".py":
        try:
            compile(updated, str(path), "exec")
        except SyntaxError as exc:
            return False, f"refusing to break {path.name}: {exc}"
    backup = _backup(path, backup_suffix)
    path.write_text(updated, encoding="utf-8")
    name = backup.name if backup else path.with_suffix(path.suffix + backup_suffix).name
    return True, f"{path} (backup: {name})"


def patch_pillow_antialias(sadtalker_root: Path) -> PatchResult:
    """Replace the removed ``PIL.Image.ANTIALIAS`` alias with ``Image.LANCZOS``.

    ``Image.ANTIALIAS`` was an alias of ``Image.LANCZOS`` before Pillow 10 removed
    it, so the substitution is behaviour-preserving.

    The replacement must stay a bare attribute: SadTalker calls it inline, e.g.
    ``img.resize((256, 256), Image.ANTIALIAS)``, so any trailing comment would
    swallow the closing parenthesis and make the module unimportable.  Files
    damaged by that earlier revision are repaired here.
    """
    target = Path(sadtalker_root) / "src" / "utils" / "croper.py"
    broken = "Image.LANCZOS  # LANCZOS replaced the Pillow <10 ANTIALIAS alias"

    repaired = False
    if target.is_file():
        text = target.read_text(encoding="utf-8")
        if broken in text:
            _backup(target)
            target.write_text(text.replace(broken, "Image.LANCZOS"), encoding="utf-8")
            repaired = True

    changed, detail = _replace_in_file(target, "Image.ANTIALIAS", "Image.LANCZOS")
    if repaired:
        detail = f"repaired the earlier broken replacement; {detail}"
        changed = True
    return PatchResult("pillow-antialias", changed, detail)


def patch_gfpgan_lazy_import(sadtalker_root: Path) -> PatchResult:
    """Make the optional GFPGAN enhancer import non-fatal.

    ``src/utils/face_enhancer.py`` is imported at module level by
    ``src/facerender/animate.py``. Without this patch the whole inference stack
    needs ``gfpgan`` + ``basicsr`` installed even when the enhancer is disabled.
    """
    target = Path(sadtalker_root) / "src" / "utils" / "face_enhancer.py"
    old_import = "from gfpgan import GFPGANer\n"
    new_import = (
        "try:  # optional dependency: only required for the GFPGAN face enhancer\n"
        "    from gfpgan import GFPGANer\n"
        "except Exception:  # pragma: no cover\n"
        "    GFPGANer = None\n"
    )
    changed, detail = _replace_in_file(target, old_import, new_import)

    # Also raise a clear error if somebody enables the enhancer without gfpgan.
    guard_old = "    restorer = GFPGANer("
    guard_new = (
        "    if GFPGANer is None:\n"
        "        raise RuntimeError(\n"
        "            'The GFPGAN enhancer was requested but gfpgan is not installed. '\n"
        "            'Run: pip install gfpgan'\n"
        "        )\n"
        "    restorer = GFPGANer("
    )
    guarded, _ = _replace_in_file(target, guard_old, guard_new)
    if changed or guarded:
        detail += " (+ runtime guard)"
    return PatchResult("gfpgan-lazy-import", changed or guarded, detail)


def patch_basicsr_functional_tensor() -> PatchResult:
    """Patch ``basicsr`` to import ``rgb_to_grayscale`` from a valid location.

    ``basicsr==1.4.2`` imports ``torchvision.transforms.functional_tensor`` which
    was removed in torchvision 0.17.  The function lives in
    ``torchvision.transforms.functional`` in every modern version.
    """
    try:
        import basicsr  # noqa: F401
    except Exception:
        return PatchResult("basicsr-functional-tensor", False, "basicsr not installed")

    basicsr_dir = Path(basicsr.__file__).resolve().parent
    target = basicsr_dir / "data" / "degradations.py"
    changed, detail = _replace_in_file(
        target,
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale",
    )
    return PatchResult("basicsr-functional-tensor", changed, detail)


def _torch_version(torch_module) -> tuple[int, ...]:
    """Parse ``torch.__version__`` (``"2.0.1+cu117"``) into a comparable tuple."""
    raw = str(getattr(torch_module, "__version__", "0")).split("+")[0]
    parts: list[int] = []
    for piece in raw.split("."):
        digits = "".join(character for character in piece if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def patch_torch_load_weights_only() -> PatchResult:
    """Restore the pre-2.6 ``torch.load`` default (``weights_only=False``).

    PyTorch 2.6 flipped ``torch.load`` to ``weights_only=True``.  Several SadTalker
    checkpoints are ``.pth.tar`` archives that contain optimiser metadata, which the
    restricted unpickler rejects.  We keep the upstream behaviour explicitly.

    The patch is only installed on torch >= 2.6: older releases either ignore the
    argument or do not accept it at all, and the notebook pins torch 2.0.1.
    """
    try:
        import torch
    except Exception:
        return PatchResult("torch-load-weights-only", False, "torch not installed")

    if getattr(torch.load, "_s2p_patched", False):
        return PatchResult("torch-load-weights-only", False, "already patched")

    if _torch_version(torch) < (2, 6):
        return PatchResult(
            "torch-load-weights-only",
            False,
            f"not needed on torch {torch.__version__} (the default changed in 2.6)",
        )

    original = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    _load._s2p_patched = True
    _load.__doc__ = original.__doc__
    torch.load = _load
    return PatchResult("torch-load-weights-only", True, "torch.load now defaults to weights_only=False")


def required_numpy_spec() -> str:
    """Return the NumPy pin that SadTalker's source code requires.

    ``src/face3d/util/preprocess.py`` uses ``np.VisibleDeprecationWarning``
    (removed in NumPy 2) and ``src/face3d/util/my_awing_arch.py`` uses ``np.float``
    (removed in NumPy 1.24).  ``numpy==1.23.5`` satisfies both and has wheels for
    CPython 3.8 - 3.11.
    """
    return "numpy==1.23.5"


def apply_all_patches(sadtalker_root: Path, include_torch: bool = True) -> list[PatchResult]:
    """Apply every compatibility patch and return a human readable report."""
    results = [
        patch_pillow_antialias(sadtalker_root),
        patch_gfpgan_lazy_import(sadtalker_root),
        patch_basicsr_functional_tensor(),
    ]
    if include_torch:
        results.append(patch_torch_load_weights_only())
    return results


def format_report(results: list[PatchResult]) -> str:
    return "\n".join(str(r) for r in results)
