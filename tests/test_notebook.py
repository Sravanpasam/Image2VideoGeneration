"""Guards the generated Colab notebook (``Script2Presenter_Colab.ipynb``).

The notebook embeds the application sources in ``%%writefile`` cells.  These tests
make sure the notebook is valid JSON, matches the current sources and never
contains stale outputs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "Script2Presenter_Colab.ipynb"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_notebook", ROOT / "tools" / "build_notebook.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_exists():
    assert NOTEBOOK.is_file(), "run `python tools/build_notebook.py`"


def test_notebook_is_valid_nbformat():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) > 20


def test_notebook_has_no_stored_outputs():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert not cell.get("outputs"), f"cell {index} contains stale outputs"
            assert cell.get("execution_count") is None, f"cell {index} was executed"


def test_notebook_targets_a_gpu():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert nb.metadata.get("accelerator") == "GPU"
    assert nb.metadata.get("colab", {}).get("gpuType") == "T4"


def test_notebook_is_in_sync_with_sources():
    builder = _load_builder()
    assert builder.main(["--check"]) == 0


def test_every_embedded_file_matches_disk():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    embedded = {}
    for cell in nb.cells:
        if cell.cell_type != "code" or not cell.source.startswith("%%writefile "):
            continue
        header, body = cell.source.split("\n", 1)
        path = Path(header.split(" ", 1)[1].strip())
        embedded[path.name] = (path, body)

    assert embedded, "the notebook must embed the application sources"
    for name, (path, body) in embedded.items():
        if name == "script.txt":
            continue
        relative = path.relative_to("/content/Script2Presenter")
        on_disk = (ROOT / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert body == on_disk, f"{name} in the notebook is stale"


def test_notebook_mentions_the_hard_constraints():
    text = NOTEBOOK.read_text(encoding="utf-8")
    # The two facts that make or break the demo must be documented in place.
    assert "numpy==1.23.5" in text
    assert "T4" in text
    assert "Apache-2.0" in text


def test_notebook_json_is_parseable():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4

def test_setup_runs_after_the_sources_are_written():
    """setup_colab.sh imports s2p for its smoke test, so the package must exist."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    setup_index = None
    module_indexes = []

    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if cell.source.startswith("#@title 2. Run the setup"):
            setup_index = index
        if cell.source.startswith("%%writefile /content/Script2Presenter/s2p/"):
            module_indexes.append(index)

    assert setup_index is not None, "the setup cell is missing"
    assert module_indexes, "the notebook does not embed the s2p package"
    assert max(module_indexes) < setup_index, (
        "the sources must be written before setup_colab.sh runs"
    )
    names = {nb.cells[i].source.splitlines()[0].rsplit("/", 1)[-1] for i in module_indexes}
    assert "sadtalker_backend.py" in names


def test_folders_are_created_before_the_first_writefile():
    """``%%writefile`` cannot create directories, and section 1 may be re-run alone."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    first_writefile = next(
        index
        for index, cell in enumerate(nb.cells)
        if cell.cell_type == "code" and cell.source.startswith("%%writefile ")
    )
    before = nb.cells[:first_writefile]
    assert any("os.makedirs(" in cell.source for cell in before), (
        "a cell must create /content/Script2Presenter before the first %%writefile"
    )


def test_setup_script_avoids_the_basicsr_build_isolation_trap():
    """basicsr's setup.py imports torch, so it needs --no-build-isolation."""
    script = (ROOT / "scripts" / "setup_colab.sh").read_text(encoding="utf-8")
    assert "--no-build-isolation" in script
    assert "basicsr==1.4.2" in script


def _pinned(requirements: str, name: str) -> tuple[int, ...]:
    prefix = f"{name}=="
    for line in requirements.splitlines():
        if line.startswith(prefix):
            return tuple(int(part) for part in line[len(prefix):].strip().split("."))
    raise AssertionError(f"{name} is not pinned in requirements-colab.txt")


def test_imageio_ffmpeg_pin_does_not_need_pkg_resources():
    """0.4.x imports pkg_resources, which setuptools >= 81 removed.

    The failure only shows up when SadTalker writes the finished MP4, i.e. after
    the whole GPU render, so it must be caught here instead.
    """
    requirements = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")
    assert _pinned(requirements, "imageio-ffmpeg") >= (0, 5, 1)
    assert "setuptools<81" in requirements


def test_setup_script_keeps_setuptools_below_81():
    script = (ROOT / "scripts" / "setup_colab.sh").read_text(encoding="utf-8")
    install = next(
        line for line in script.splitlines() if "uv pip install" in line and "cython" in line
    )
    assert '"setuptools<81"' in install


def test_setup_script_exercises_the_ffmpeg_writer():
    """The verify step must write a real MP4, not just import the modules."""
    script = (ROOT / "scripts" / "setup_colab.sh").read_text(encoding="utf-8")
    assert "imageio.mimsave(" in script
    assert "imageio_ffmpeg.get_ffmpeg_exe()" in script

