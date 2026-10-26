# Troubleshooting

## Setup

### `numpy drifted to 2.x` / `np.float` / `np.VisibleDeprecationWarning`

SadTalker's `face3d` code needs **NumPy 1.23.x**:
`src/face3d/util/my_awing_arch.py` uses `np.float` (removed in 1.24) and
`src/face3d/util/preprocess.py` uses `np.VisibleDeprecationWarning` (removed in 2.0).

```bash
$PY -m pip install -c constraints.txt --force-reinstall numpy==1.23.5
```

The setup script aborts if the pin is violated, and `constraints.txt` is applied to
every install, so a normal re-run fixes it.

### `No module named 'audioop'` (from pydub)

You are running on Python 3.13. `audioop` was removed from the standard library.
Use the pinned Python 3.10 environment (`scripts/setup_colab.sh`) instead of the
Colab kernel interpreter.

### `Unable to install torch` / CUDA not available

* The wheel needs ~2 GB of free disk: `df -h /content`.
* Check `nvidia-smi` first; without a GPU torch still installs but `cuda` is `False`.
* Force the CPU path for a quick smoke test: `--device cpu` (very slow).

### `uv python install 3.10` fails (no network / proxy)

Fallback that uses Miniconda instead:

```bash
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/m.sh
bash /tmp/m.sh -b -p /opt/conda
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
/opt/conda/bin/conda create -y -n s2p -c conda-forge python=3.10
PY=/opt/conda/envs/s2p/bin/python
$PY -m pip install torch==2.0.1 torchvision==0.15.2
$PY -m pip install -c constraints.txt -r requirements-colab.txt
$PY -m pip install --no-deps basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8
```

## SadTalker / inference

| error | cause | fix |
|---|---|---|
| `Image.ANTIALIAS` | Pillow ≥ 10 removed the alias | `apply_all_patches()` rewrites `src/utils/croper.py` (keeps `.orig`); the replacement is a bare attribute (never with an inline comment that would swallow `)` ) |
| `'(' was never closed` in `croper.py` | a previous patch appended a trailing comment that broke the resize call | patched to repair any broken replacement and to refuse writing un-compilable Python files |
| `cannot import name 'rgb_to_grayscale' from torchvision.transforms.functional_tensor` | basicsr 1.4.2 imports a removed module | patched in `basicsr/data/degradations.py` |
| `UnpicklingError ... weights_only` | PyTorch ≥ 2.6 flips the `torch.load` default | `patch_torch_load_weights_only()` sets `weights_only=False` for the `.pth.tar` archives |
| `ModuleNotFoundError: gfpgan` | optional enhancer | patched import; only fails if you enable `--enhancer`. Install with `pip install --no-deps gfpgan==1.3.8` |
| `No face was detected` | detection failed | frontal face, ≥ 256×256, no heavy occlusion, one person |
| `cv2` fails with `libGL.so.1` | GUI OpenCV on a headless box | the pinned `opencv-python-headless` covers it; never install `opencv-python` on top |
| `ffmpeg not found` | SadTalker muxes audio with the `ffmpeg` binary | `apt-get install -y ffmpeg` (Colab has it) |
| CUDA out of memory at 512 px | ~10 GB VRAM needed | use `--size 256`, or lower `--batch-size 1` |
| `imageio` warns about `fps` | imageio ≥ 2.28 deprecates it | pinned to 2.19.3 |
| `No module named 'pkg_resources'` after the render finishes | setuptools >= 81 removed `pkg_resources`, and imageio-ffmpeg 0.4.x imports it | pinned `imageio-ffmpeg==0.5.1` + `setuptools<81`; quick fix: `$PY -m pip install "imageio-ffmpeg==0.5.1" "setuptools<81"` |
| `KeyError: backend: 'module://matplotlib_inline.backend_inline'` | Colab exports `MPLBACKEND` for the notebook kernel; that backend is not installed in the project venv | `s2p.sadtalker_backend` replaces any `module://` backend with `Agg`; alternatively `MPLBACKEND=Agg $PY -m s2p.cli ...` |

Patches are applied automatically the first time the backend is used. To inspect
what was applied:

```bash
python -c "from s2p.patches import apply_all_patches, format_report; from s2p.config import load_settings; s=load_settings(); print(format_report(apply_all_patches(s.sadtalker_root)))"
```

## Speech

| error | fix |
|---|---|
| `edge-tts` returns 403/`Invalid response status` | the endpoint throttles datacentre IPs; retry, or switch to Piper |
| Piper download blocked | pre-download `en_US-amy-low.onnx` + `.json` into `models/piper/` (they are also mirrored on Hugging Face) |
| gTTS `429` | rate limited; it is the last resort, use Piper |
| Audio sounds slow/fast | the file must be 16 kHz mono — `s2p.audio_utils.probe_audio()` reports what the file really is |

```bash
python -m s2p.cli --image face.jpg --text "Hello" --provider piper --voice en_US-amy-low -o out.mp4
python -m s2p.cli --list-voices
```

## UI

| symptom | fix |
|---|---|
| `--share` link never appears | run `python -m s2p.ui` in a terminal; inside Colab check `/content/output/gradio.log` |
| Blank page behind a proxy | `python -m s2p.ui --host 0.0.0.0 --port 7860` and open the proxied URL |
| Buttons do nothing | open the browser console; a stale Gradio version is the usual cause (`pip show gradio`, the code supports 4.x and 5.x) |
| No audio in the preview | the MP4 has an AAC track; some browsers mute autoplay — use the download button |

## Colab

* `/content` is wiped when the session ends. The generated MP4 and the manifests live
  there — download them before closing the tab.
* Re-running the setup cell after a disconnect is fast (idempotent) unless `/content`
  was reset, in which case the model downloads happen again.
* The `*.gradio.live` link is valid for 72 hours and only while the cell's process runs.
