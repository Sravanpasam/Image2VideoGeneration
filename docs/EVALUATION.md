# Evaluation

This document defines what "works" means for Script2Presenter, how each number is
produced, and what has actually been measured so far. Numbers that still have to be
collected are marked **TBD** instead of being estimated.

## 1. Questions

| id | question | metric |
|---|---|---|
| **Q1** | Does the pipeline produce a valid, playable talking-head video from one photo and a script? | success rate, container/codec, duration error |
| **Q2** | How fast is it on a free Colab T4? | TTS s, render s, real-time factor (RTF) |
| **Q3** | Does the offline fallback work when the network is unavailable? | Piper success rate, quality delta |
| **Q4** | How does it fail on bad input? | actionable error, no crash, no partial files |
| **Q5** | Is the result reproducible? | manifest completeness, byte-identical inputs |

## 2. Environments

| | development machine | demo machine |
|---|---|---|
| OS | Windows 11 | Ubuntu 22.04 (Colab) |
| CPU | laptop x86-64 | 2 vCPU |
| GPU | **none** | **NVIDIA T4 16 GB** (sm_75) |
| Python | 3.11.9 (venv) | 3.10.14 (isolated in `/content/.s2p-venv`) |
| NumPy | 2.4.6 (tests only) | 1.23.5 (required by SadTalker) |
| torch | not installed | 2.0.1 (CUDA 11.7) |

SadTalker inference is **not** run on the development machine (no GPU, and the
NumPy pin conflicts with the system Python). The orchestration layer is instead
verified with a stub backend (see `tests/test_pipeline.py`), and the GPU stage is
exercised by the Colab notebook.

## 3. Protocol

Every run appends one `outputs/<timestamp>_<id>/manifest.json` containing:

```json
{
  "created_at": "...",
  "settings": {"sadtalker_root": "...", "device": "auto", ...},
  "inputs": {"portrait": "...", "script": "...", "script_chars": 412},
  "tts": {"provider": "edge", "voice": "en-US-AriaNeural", "duration_s": 31.2},
  "audio": {"duration_s": 31.2, "sample_rate": 16000, "channels": 1},
  "generation": {"frames": 780, "duration_s": 31.2, "size": 256,
                 "preprocess": "crop", "enhancer": null,
                 "elapsed_s": 54.1, "device": "cuda"},
  "timings": {"tts_s": 2.3, "sadtalker_s": 54.1, "total_s": 56.4}
}
```

To collect a table:

```bash
for f in outputs/*/manifest.json; do
  python -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['inputs']['script_chars'], m['audio']['duration_s'], m['timings']['tts_s'], m['timings']['sadtalker_s'], m['generation']['device'])" "$f"
done
```

**Real-time factor** is `render_seconds / audio_seconds`; below 1.0 means faster than
real time.

**Duration error** is `|video_duration − audio_duration| / audio_duration`.

Test scripts (fixed, kept in the notebook / CLI):

| id | script | chars | expected audio |
|---|---|---|---|
| S1 | one sentence | 58 | ~4 s |
| S2 | one paragraph | 412 | ~30 s |
| S3 | multi-paragraph, numbers, punctuation | 1 250 | ~90 s |

## 4. Results that are already measured

### 4.1 Automated tests

```
$ python -m pytest tests -q
49 passed, 2 skipped            # 2 skipped = network-gated TTS tests

$ S2P_RUN_NETWORK_TESTS=1 python -m pytest tests/test_tts.py -q
12 passed in 227.58s            # real edge-tts + real Piper synthesis
```

Coverage by area: configuration/path resolution (6), compatibility patches (7),
audio conversion (8), TTS logic + providers (12), orchestration/validation/manifest
(10), notebook generation guard (8). The GPU render is the only stage without an
automated test (it needs a GPU).

### 4.2 Speech synthesis (development machine, CPU, network on)

| provider | output | audio length | wall time | RTF | notes |
|---|---|---|---|---|---|
| edge-tts 7.3.1 | 24 kHz MP3 | 4.49 s | ~1.5 s | ~0.33 | 322 voices listed, no key |
| piper-tts 1.8.0 (`en_US-amy-low`) | 16 kHz mono WAV | 5.89 s | ~3.7 s | ~0.63 | fully offline, 63 MB model |
| gTTS | 24 kHz MP3 | n/a | n/a | n/a | fallback only, not benchmarked |

Both providers are converted to 16 kHz mono before SadTalker sees them; the
conversion is verified by `probe_audio()` (`tests/test_audio_utils.py`).

### 4.3 Checkpoints

Every URL in `scripts/download_checkpoints.py` was checked with an HTTP `HEAD`
request (status 200) and an exact byte size, and one full download
(`detection_Resnet50_Final.pth`, 109 497 761 B) was executed end-to-end including
the atomic rename and size assertion.

| file | bytes |
|---|---|
| `SadTalker_V0.0.2_256.safetensors` | 725 066 984 |
| `SadTalker_V0.0.2_512.safetensors` | 725 066 984 |
| `mapping_00229-model.pth.tar` | 155 521 183 |
| `mapping_00109-model.pth.tar` | 155 779 231 |
| `alignment_WFLW_4HG.pth` | 193 670 248 |
| `detection_Resnet50_Final.pth` | 109 497 761 |
| `GFPGANv1.4.pth` (optional) | 348 632 874 |
| `parsing_parsenet.pth` (optional) | 85 331 193 |

Each file is also mirrored on Hugging Face
(`camenduru/SadTalker`), which is used automatically when GitHub is slow.

### 4.4 Robustness (Q4)

`validate_portrait` and the pipeline are unit-tested against: missing file, wrong
extension, corrupt image bytes, 16×16 image, empty script, 5 001-character script,
unknown TTS provider, and a backend that raises. Every case produces a single
actionable `PipelineError` string and no partial manifest.

## 5. Results still to collect on the T4

Run the notebook once with `SIZE = 256`, then once with `SIZE = 512`, for scripts
S1–S3, and fill this table from the manifests:

| script | size | audio s | tts s | render s | total s | RTF | device |
|---|---|---|---|---|---|---|---|
| S1 | 256 | TBD | TBD | TBD | TBD | TBD | cuda |
| S2 | 256 | TBD | TBD | TBD | TBD | TBD | cuda |
| S3 | 256 | TBD | TBD | TBD | TBD | TBD | cuda |
| S2 | 512 | TBD | TBD | TBD | TBD | TBD | cuda |
| S2 | 256 + GFPGAN | TBD | TBD | TBD | TBD | TBD | cuda |
| S2 | 256, `--still` | TBD | TBD | TBD | TBD | TBD | cuda |

Also check **Q5**: the `frames` field must equal `round(audio_seconds × 25)` within
one frame, and the MP4 must be `H.264 + AAC` (verify with `ffprobe`).

## 6. Subjective quality (rubric)

No human-panel study is claimed. Instead, each generated clip is rated by the author
on a 1–5 scale, with the criteria defined up front:

| dimension | 1 | 3 | 5 |
|---|---|---|---|
| lip-sync plausibility | mouth barely moves / wrong | visible sync, some drift | convincing at normal speed |
| identity preservation | face changes | recognisable | indistinguishable from the still |
| visual artefacts | broken teeth/warping | minor shimmer | clean |
| naturalness of motion | robotic / no motion | acceptable | natural head motion |

Reference clips: same portrait, same script, `crop` vs `full`, `256` vs `512`,
with and without GFPGAN. Ratings go in the table below (TBD until the Colab runs):

| variant | lip-sync | identity | artefacts | motion |
|---|---|---|---|---|
| 256 / crop | TBD | TBD | TBD | TBD |
| 512 / crop | TBD | TBD | TBD | TBD |
| 256 / full | TBD | TBD | TBD | TBD |
| 256 / crop + GFPGAN | TBD | TBD | TBD | TBD |

## 7. Threats to validity

* **Single-GPU, single-session measurements.** Colab T4 instances vary; the render
  time depends on other tenants. Report the median of at least three runs.
* **One speaker, English only.** Quality on other languages/voices is untested.
* **No ground-truth lip-sync metric.** LSE-D/LSE-C would be the rigorous choice but
  need the SyncNet checkpoint and aligned reference video; out of scope here, and
  the rubric is explicitly subjective.
* **The development machine cannot run the render stage**, so the end-to-end
  integration is verified in a different environment (Colab) than the unit tests.
