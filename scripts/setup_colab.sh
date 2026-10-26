#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Prepare a Google Colab (or any Ubuntu) GPU machine for Script2Presenter.
#
#   bash scripts/setup_colab.sh [/content/Script2Presenter]
#
# What it does
#   1. installs `uv` (fast, no Anaconda licence questions)
#   2. downloads a standalone CPython 3.10 (numpy 1.23.5 has no 3.12+ wheels)
#   3. creates /content/.s2p-venv and installs the pinned dependency set
#   4. clones the official SadTalker checkout at a pinned commit
#   5. downloads the checkpoints
#   6. verifies the environment (torch + CUDA + numpy)
#
# Safe to re-run: every step is skipped when it is already done.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="${1:-/content/Script2Presenter}"
VENV="${S2P_VENV:-/content/.s2p-venv}"
SADTALKER_DIR="${SADTALKER_DIR:-/content/SadTalker}"
SADTALKER_COMMIT="cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8"
PYTHON_VERSION="${S2P_PYTHON:-3.10}"
SIZE="${S2P_SIZE:-256}"

log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }

log "project:  $PROJECT_DIR"
log "venv:     $VENV (python $PYTHON_VERSION)"

# --- 1. uv ----------------------------------------------------------------
if ! python -m uv --version >/dev/null 2>&1; then
  log "installing uv ..."
  python -m pip install -q uv
fi
python -m uv --version

# --- 2/3. interpreter + dependencies --------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  log "creating the python $PYTHON_VERSION environment ..."
  python -m uv venv --python "$PYTHON_VERSION" "$VENV"
fi
PY="$VENV/bin/python"
"$PY" --version

if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  log "installing torch (CUDA build, ~2 GB) ..."
  python -m uv pip install --python "$PY" torch==2.0.1 torchvision==0.15.2
fi

log "installing the pinned runtime dependencies ..."
python -m uv pip install --python "$PY" \
  -c "$PROJECT_DIR/constraints.txt" \
  -r "$PROJECT_DIR/requirements-colab.txt"

log "installing basicsr / facexlib / gfpgan (without their stale metadata) ..."
# These three only ship sdists and their setup.py declares
# setup_requires=['cython', 'numpy', 'torch']. With build isolation pip would try
# to download a whole second torch (~600 MB) into a throw-away build env, and
# basicsr's setup.py imports torch at build time - hence --no-build-isolation
# with torch already present.  --no-deps keeps their 2023 pins (tb-nightly, av,
# face_alignment) out of the environment.
# setuptools must stay below 81: it is what builds basicsr here (via
# --no-build-isolation) and 81 dropped pkg_resources, which several of these
# 2023-era packages still import at runtime. Must match requirements-colab.txt.
python -m uv pip install --python "$PY" -q pip "setuptools<81" wheel cython
"$PY" -m pip install -q --no-build-isolation --no-deps \
  basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8
"$PY" -c "import basicsr, facexlib, gfpgan; print('basicsr/facexlib/gfpgan OK')"

log "guarding the numpy pin ..."
"$PY" - <<'PYEOF'
import sys
import numpy
if numpy.__version__ != "1.23.5":
    sys.exit(
        f"numpy drifted to {numpy.__version__}; SadTalker requires 1.23.5. "
        "Re-run: pip install -c constraints.txt numpy==1.23.5"
    )
print("numpy", numpy.__version__, "OK")
PYEOF

# --- 4. SadTalker --------------------------------------------------------
if [ ! -d "$SADTALKER_DIR/.git" ]; then
  log "cloning SadTalker ..."
  git clone -q https://github.com/OpenTalker/SadTalker.git "$SADTALKER_DIR"
fi
log "pinning SadTalker to $SADTALKER_COMMIT ..."
git -C "$SADTALKER_DIR" fetch -q --depth 1 origin "$SADTALKER_COMMIT" 2>/dev/null || \
  git -C "$SADTALKER_DIR" fetch -q origin
git -C "$SADTALKER_DIR" checkout -q "$SADTALKER_COMMIT" 2>/dev/null || \
  git -C "$SADTALKER_DIR" reset -q --hard "$SADTALKER_COMMIT"

# --- 5. checkpoints ------------------------------------------------------
log "downloading the checkpoints (size $SIZE) ..."
"$PY" "$PROJECT_DIR/scripts/download_checkpoints.py" --dest "$SADTALKER_DIR" --size "$SIZE"

# --- 6. verify -----------------------------------------------------------
log "verifying the environment ..."
cd "$PROJECT_DIR"
S2P_SADTALKER_ROOT="$SADTALKER_DIR" "$PY" - <<'PYEOF'
import os, sys

import torch, numpy, cv2, librosa, soundfile, pydub, edge_tts, piper, gradio, safetensors
import kornia, skimage

print(f"  python      : {sys.version.split()[0]}")
print(f"  torch       : {torch.__version__}")
print(f"  cuda        : {torch.cuda.is_available()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only'})")
print(f"  numpy       : {numpy.__version__}")
print(f"  gradio      : {gradio.__version__}")

# The render finishes with imageio.mimsave(path, frames, fps=25). Exercise that
# exact path now - a broken ffmpeg backend (e.g. a missing pkg_resources) would
# otherwise only surface after minutes of GPU work.
import tempfile
import numpy as np
import imageio
import imageio_ffmpeg

smoke_path = os.path.join(tempfile.mkdtemp(), "smoke.mp4")
imageio.mimsave(
    smoke_path, [np.zeros((64, 64, 3), dtype=np.uint8)] * 4, fps=25
)
print(f"  imageio     : {imageio.__version__} ("
      f"{os.path.basename(imageio_ffmpeg.get_ffmpeg_exe())}, "
      f"wrote {os.path.getsize(smoke_path)} bytes)")

if os.path.isfile("s2p/__init__.py"):
    import s2p
    print(f"  s2p         : {s2p.__version__}")
else:
    print("  s2p         : skipped (run the notebook's source cells first)")
PYEOF

log "done. Run the demo with:"
echo "  cd $PROJECT_DIR"
echo "  $PY -m s2p.cli --image portrait.png --text 'Hello world' --out out.mp4"
echo "  $PY -m s2p.ui --share        # optional web UI"
