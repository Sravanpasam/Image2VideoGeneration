# Script2Presenter

**A portrait photograph + a script → a downloadable talking-head video.**

Script2Presenter is a small, auditable pipeline that turns *one* JPG/PNG portrait
and *one* block of text into an MP4 of that person speaking the text. It is built
on two free, open pretrained models and runs end-to-end on a free Google Colab T4
GPU — no API key, no account, no paid service.

```
 script.txt ──► neural TTS ──► 16 kHz mono WAV ─┐
                                                 ├──► SadTalker ──► MP4 + manifest.json
 portrait.jpg ──────────────────────────────────┘
```

| | |
|---|---|
| Speech | **edge-tts** (Microsoft neural voices) → **Piper** (offline VITS) → **gTTS** |
| Animation | **SadTalker** V0.0.2 checkpoints (CVPR 2023, Apache-2.0) |
| UI | Gradio (`python -m s2p.ui`) |
| Batch / scripted use | `python -m s2p.cli` |
| Colab | [`Script2Presenter_Colab.ipynb`](Script2Presenter_Colab.ipynb) |
| Tests | `pytest` — 49 offline tests + 2 network tests |

---

## Quick start (Google Colab, ~10 min)

1. Open **`Script2Presenter_Colab.ipynb`** in [Colab](https://colab.research.google.com/).
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`.

The notebook installs an isolated Python 3.10 environment in `/content/.s2p-venv`,
downloads 1.2 GB of official checkpoints, generates a ~30 s video from the bundled
example portrait, plays it inline and offers it for download. It also starts the
Gradio UI with a public `*.gradio.live` link.

The notebook is **generated from this repository** (`python tools/build_notebook.py`),
so it always contains the exact sources below and a test fails if they drift.

## Local usage

```bash
python -m venv .venv
. .venv/Scripts/activate            # Windows
pip install -r requirements-local.txt

# environment report (no GPU needed)
python -m s2p.cli --check

# web UI
python -m s2p.ui                    # http://localhost:7860

# scripted generation (needs the SadTalker environment, see below)
python -m s2p.cli -i portrait.jpg -t "Hello, this is my demo." -o out.mp4 --json
```

Everything except the animation stage (TTS, UI, validation, manifest) runs on any
machine. Animation needs the pinned stack from `requirements-colab.txt`
(NumPy 1.23.5 + PyTorch + the SadTalker checkout); `scripts/setup_colab.sh` builds
it and works on any Ubuntu box with a GPU.

```bash
S2P_SADTALKER_ROOT=/path/to/SadTalker python -m s2p.cli -i face.jpg -t "Hi" -o out.mp4
```

## How it works

| Stage | Module | What happens |
|---|---|---|
| 1. Input validation | `s2p/pipeline.py` | JPG/PNG, readable, ≥ 64×64, script 1–5000 chars; the portrait is **copied**, never moved |
| 2. Speech | `s2p/tts.py` | sentence chunking (1800 chars) → first working provider → concat → **16 kHz mono WAV** |
| 3. Face analysis | `s2p/sadtalker_backend.py` | `CropAndExtract`: facexlib detection/alignment, 3DMM reconstruction |
| 4. Motion | same | `Audio2Coeff`: audio → pose + expression coefficients at 25 fps |
| 5. Render | same | `AnimateFromCoeff`: frame generation, optional GFPGAN, ffmpeg mux |
| 6. Record | `s2p/pipeline.py` | `outputs/<timestamp>_<id>/manifest.json` with inputs, timings, device, paths |

### Four engineering decisions that make it work

**1. NumPy is pinned to 1.23.5 — and that forces Python ≤ 3.11.**
SadTalker's `face3d` code uses `np.float` (removed in NumPy 1.24) and
`np.VisibleDeprecationWarning` (removed in NumPy 2.0). The Colab runtime of 2026
ships Python 3.13, for which NumPy 1.23.5 has no wheels. The notebook therefore
creates an isolated Python 3.10 environment instead of patching NumPy's ABI.
`constraints.txt` is applied to every `pip` call and a guard aborts the setup if
NumPy ever drifts.

**2. The code is patched, not forked.**
`/s2p/patches.py` applies four idempotent fixes to the *official* checkout at load
time (`Image.ANTIALIAS` → `Image.LANCZOS`, lazy `gfpgan` import with a clear runtime
error, `basicsr`'s removed `functional_tensor` import, `torch.load(weights_only=False)`
for the `.pth.tar` checkpoints). Each patch keeps a `.orig` backup and can be run
twice safely. No file of SadTalker is permanently modified.

**3. Speech is a three-provider chain, not a single dependency.**
`edge` gives the best quality but needs network; `piper` is a 63 MB ONNX model that
runs fully offline on CPU; `gtts` is the last resort. `auto` walks the chain and
reports what it used in the manifest. `kokoro-onnx` was rejected because it needs
NumPy ≥ 2.0.2, which is incompatible with SadTalker.

**4. Every run is measured, not guessed.**
`manifest.json` stores TTS time, render time, device, resolution, framing, voice,
frame count and output paths. The evaluation in [`docs/EVALUATION.md`](docs/EVALUATION.md)
is built from those files.

## Evaluation (summary)

Measured on the development machine (Windows, CPU, no GPU) and in the Colab T4
notebook. Full protocol and tables: [`docs/EVALUATION.md`](docs/EVALUATION.md).

| Metric | Result |
|---|---|
| Unit tests | 49 passed, 2 network-gated (`S2P_RUN_NETWORK_TESTS=1` → 51 passed) |
| edge-tts | 4.5 s of 24 kHz audio in ~1.5 s (≈ 3× real time) |
| piper-tts | 5.9 s of 16 kHz audio in ~3.7 s (≈ 1.6× real time, offline) |
| Pipeline orchestration | verified with a stub backend: manifest, validation, copy-not-move |
| GPU render | *fill in from a Colab run — see the notebook's timing table* |

## Repo layout

```
Script2Presenter/
├─ s2p/                      # the application (importable package)
│  ├─ config.py              # paths, constants, settings resolution
│  ├─ patches.py             # idempotent compatibility patches for SadTalker
│  ├─ audio_utils.py         # everything → 16 kHz mono WAV, concat, probe
│  ├─ tts.py                 # edge-tts → piper → gTTS chain
│  ├─ sadtalker_backend.py   # thin wrapper around the official inference modules
│  ├─ pipeline.py            # orchestration + manifest
│  ├─ ui.py                  # Gradio app
│  └─ cli.py                 # python -m s2p.cli
├─ tests/                    # 51 tests (2 network-gated)
├─ tools/build_notebook.py   # generates the Colab notebook from the sources
├─ scripts/
│  ├─ setup_colab.sh         # python 3.10 + pinned deps + SadTalker + checkpoints
│  └─ download_checkpoints.py# resumable, size-verified, GitHub→HF fallback
├─ requirements-colab.txt    # pinned runtime (Python 3.10)
├─ requirements-local.txt    # UI/TTS/tests only
├─ constraints.txt           # numpy==1.23.5 for every pip call
├─ Script2Presenter_Colab.ipynb  # generated, self-contained
└─ docs/                     # evaluation, demo plan, troubleshooting
```

## Limitations

* **One frontal face per image.** Profile views, heavy occlusion or multiple faces
  are rejected or produce artefacts. SadTalker animates the face region, not the body.
* **Lip sync is approximate.** It is driven by mel-spectrogram features, not a
  phoneme aligner; fast speech and non-English text can drift.
* **Head motion is synthesised**, so long clips show repeated motion patterns.
  `--still` reduces it.
* **Not real-time on CPU.** A 30 s clip needs roughly a minute on a T4 and much
  longer on CPU.
* **No emotion control from text.** Only pitch/pose style and expression scale.

## Ethical use

Only use images of people who consented to be animated, and say clearly that the
video is synthetic when you publish it. This project is a demonstration of
open-source research models; it includes no impersonation features and no
watermark removal.

## Credits and licence

* **SadTalker** — Zhang et al., CVPR 2023, [OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker), **Apache-2.0**.
* **GFPGAN / facexlib** — Tencent ARC (Apache-2.0 / MIT).
* **Piper** — rhasspy (MIT); voices from `rhasspy/piper-voices`.
* **edge-tts** — unofficial client for the free Microsoft Edge read-aloud endpoint.
* Application code in this repository: MIT.
