"""Tests for the compatibility patches in :mod:`s2p.patches`."""

from __future__ import annotations

from pathlib import Path

from s2p.patches import (
    apply_all_patches,
    patch_basicsr_functional_tensor,
    patch_gfpgan_lazy_import,
    patch_pillow_antialias,
    required_numpy_spec,
)


def _make_fake_sadtalker(root: Path) -> Path:
    (root / "src" / "utils").mkdir(parents=True)
    (root / "src" / "utils" / "croper.py").write_text(
        "from PIL import Image\nimg = img.resize(rsize, Image.ANTIALIAS)\n", encoding="utf-8"
    )
    (root / "src" / "utils" / "face_enhancer.py").write_text(
        "import os\nimport torch\n\nfrom gfpgan import GFPGANer\n\n"
        "def enhancer_generator_no_len(images, method='gfpgan'):\n"
        "    restorer = GFPGANer(model_path='x')\n"
        "    yield restorer\n",
        encoding="utf-8",
    )
    return root


def test_pillow_patch_replaces_antialias(tmp_path: Path):
    root = _make_fake_sadtalker(tmp_path / "SadTalker")
    result = patch_pillow_antialias(root)
    assert result.applied is True
    text = (root / "src" / "utils" / "croper.py").read_text(encoding="utf-8")
    assert "Image.ANTIALIAS" not in text
    assert "Image.LANCZOS" in text
    # A backup must exist and keep the original content.
    backup = root / "src" / "utils" / "croper.py.orig"
    assert backup.is_file()
    assert "Image.ANTIALIAS" in backup.read_text(encoding="utf-8")


def test_pillow_patch_is_idempotent(tmp_path: Path):
    root = _make_fake_sadtalker(tmp_path / "SadTalker")
    patch_pillow_antialias(root)
    second = patch_pillow_antialias(root)
    assert second.applied is False
    assert "already applied" in second.detail


def test_pillow_patch_keeps_the_file_compilable(tmp_path: Path):
    """A trailing comment would swallow the ``)`` of the inline resize() call."""
    root = _make_fake_sadtalker(tmp_path / "SadTalker")
    patch_pillow_antialias(root)
    source = (root / "src" / "utils" / "croper.py").read_text(encoding="utf-8")
    compile(source, "croper.py", "exec")
    assert "img.resize(rsize, Image.LANCZOS)" in source


def test_pillow_patch_repairs_an_earlier_broken_replacement(tmp_path: Path):
    root = _make_fake_sadtalker(tmp_path / "SadTalker")
    croper = root / "src" / "utils" / "croper.py"
    croper.write_text(
        "from PIL import Image\n"
        "img = img.resize(rsize, Image.LANCZOS  "
        "# LANCZOS replaced the Pillow <10 ANTIALIAS alias)\n",
        encoding="utf-8",
    )
    result = patch_pillow_antialias(root)
    assert result.applied is True
    assert "repaired" in result.detail
    source = croper.read_text(encoding="utf-8")
    compile(source, "croper.py", "exec")
    assert "img.resize(rsize, Image.LANCZOS)" in source


def test_gfpgan_patch_makes_import_optional(tmp_path: Path):
    root = _make_fake_sadtalker(tmp_path / "SadTalker")
    result = patch_gfpgan_lazy_import(root)
    assert result.applied is True
    text = (root / "src" / "utils" / "face_enhancer.py").read_text(encoding="utf-8")
    assert "try:  # optional dependency" in text
    assert "except Exception:  # pragma: no cover" in text
    assert "GFPGANer = None" in text
    # The enhancer must fail loudly instead of crashing with a NameError.
    assert "raise RuntimeError(" in text


def test_gfpgan_patch_is_idempotent(tmp_path: Path):
    root = _make_fake_sadtalker(tmp_path / "SadTalker")
    patch_gfpgan_lazy_import(root)
    assert patch_gfpgan_lazy_import(root).applied is False


def test_patches_do_not_crash_without_basicsr():
    result = patch_basicsr_functional_tensor()
    assert result.applied in (True, False)
    assert isinstance(result.detail, str)


def test_apply_all_patches_returns_report(tmp_path: Path):
    root = _make_fake_sadtalker(tmp_path / "SadTalker")
    results = apply_all_patches(root, include_torch=False)
    names = {r.name for r in results}
    assert {"pillow-antialias", "gfpgan-lazy-import", "basicsr-functional-tensor"} <= names
    assert all(isinstance(r.detail, str) for r in results)


def test_required_numpy_spec_is_below_124():
    spec = required_numpy_spec()
    assert "numpy" in spec
    version = spec.split("==")[1]
    major, minor = (int(part) for part in version.split(".")[:2])
    assert (major, minor) < (1, 24)

def test_torch_version_parser_handles_cuda_suffix():
    from s2p.patches import _torch_version

    class FakeTorch:
        __version__ = "2.0.1+cu117"

    assert _torch_version(FakeTorch) == (2, 0, 1)
    assert _torch_version(FakeTorch) < (2, 6)


def test_torch_load_patch_is_a_noop_before_torch_26(monkeypatch):
    """torch 2.0.1 does not accept ``weights_only`` on every call path."""
    import sys
    import types

    from s2p.patches import patch_torch_load_weights_only

    def fake_load(*args, **kwargs):  # pragma: no cover - never called
        return "loaded"

    fake_torch = types.SimpleNamespace(__version__="2.0.1+cu117", load=fake_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = patch_torch_load_weights_only()

    assert result.applied is False
    assert "not needed" in result.detail
    assert fake_torch.load is fake_load


def test_torch_load_patch_forces_weights_only_off_from_torch_26(monkeypatch):
    import sys
    import types

    from s2p.patches import patch_torch_load_weights_only

    calls = {}

    def fake_load(*args, **kwargs):
        calls.update(kwargs)
        return "loaded"

    fake_torch = types.SimpleNamespace(__version__="2.6.0+cu124", load=fake_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = patch_torch_load_weights_only()

    assert result.applied is True
    assert fake_torch.load("model.pth") == "loaded"
    assert calls["weights_only"] is False
