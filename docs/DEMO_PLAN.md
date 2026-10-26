# Demo plan — 5 minutes

## Goal

Show, in five minutes, that **one photo + one script becomes one video**, explain
*why* it works, and be honest about what it cannot do.

## Pre-demo checklist (do this 30 minutes before)

- [ ] Colab notebook already run **once** in this session (models cached, ≈ 2 min warm).
  Running the setup cold takes ~8 minutes — never do that in front of an audience.
- [ ] One **pre-generated MP4** downloaded to the laptop (`/content/output/script2presenter_demo.mp4`)
  as a fallback for a network/GPU failure.
- [ ] Portrait prepared: frontal, well lit, ≥ 512×512, single face, neutral expression.
- [ ] Script prepared (≈ 600 characters ≈ 45 s of speech) — see the template below.
- [ ] Tab 1: the notebook (setup + generate cells). Tab 2: `docs/EVALUATION.md`.
  Tab 3: this plan.
- [ ] Volume on: the demo is audio-first.
- [ ] If presenting online: test screen share **with audio**.

## Timing

| time | what | screen |
|---|---|---|
| 0:00 | The problem in one sentence: "turning a script into a presenter video normally means a studio, a camera and a person. This needs one photo." | title slide |
| 0:30 | Show the two inputs side by side (portrait + script). Point out: only two inputs. | inputs cell |
| 1:00 | Run **Generate**. While it runs (~60 s), explain the pipeline diagram: TTS → 16 kHz WAV → SadTalker (3DMM + motion) → MP4. | generate cell |
| 2:30 | Play the result. Let it play with sound, uninterrupted. | result cell |
| 3:00 | Open the manifest: "this is not a black box — every run records device, timings, voice, resolution." Compare TTS vs render time. | manifest table |
| 3:40 | Reliability story: three TTS providers (edge → Piper offline → gTTS), four idempotent patches to the official checkout, 51 tests, `numpy==1.23.5` guard. | `docs/EVALUATION.md` |
| 4:20 | Limitations slide: one frontal face, approximate lip sync, no emotion from text, not real time on CPU, consent/ethics. | limitations |
| 4:45 | Close: "code, notebook, tests and evaluation are in the repo; the notebook is generated from the code so it cannot drift." | repo tree |

## Script template (paste into `/content/input/script.txt`)

```
Hello, and thank you for watching this demonstration.

This video was generated from a single photograph and the text you are hearing right now.
First, a neural text to speech model turns this script into audio. Then SadTalker predicts
three dee facial coefficients and head motion from that audio, and renders twenty five frames
per second.

Everything runs on a free graphics card in the cloud, without an api key, and without a
paid service. The whole pipeline is open source and reproducible.
```

Numbers are spelled out on purpose: it makes the TTS sound natural and it shows the
audience that the *text* is the only control signal.

## If something fails

| failure | say this | do this |
|---|---|---|
| GPU not available | "the free tier is busy, here is a recording from earlier" | play the pre-generated MP4 |
| setup cell runs long | "while it installs, let me explain the architecture" | keep talking over the progress bars |
| `No face was detected` | "this is a real limitation — one clear frontal face" | switch to the prepared portrait |
| edge-tts rate limited | "the network voice is throttled, so it falls back to the offline model" | set `PROVIDER="piper"` (offline) |
| Colab disconnects | "good moment to talk about reproducibility" | reload, the setup is idempotent and cached |
| video has no sound | check browser autoplay/mute; the MP4 has an AAC track | download and play locally |

## Anticipated questions

**"Is this a deepfake tool?"** It animates a still portrait with a script you provide.
Consent and disclosure are non-negotiable; the project includes no impersonation
features and no watermark removal.

**"Why not just use a commercial API?"** No key, no cost, no data leaves the machine
except the TTS request (and Piper works fully offline). The trade-off is quality and
setup time.

**"Why does setup take 8 minutes?"** PyTorch (2 GB) + 1.2 GB of checkpoints. It is a
one-time cost per Colab session and the script is idempotent.

**"Why Python 3.10 and NumPy 1.23?"** SadTalker's research code uses `np.float` and
`np.VisibleDeprecationWarning`. Pinning is cheaper and safer than rewriting the model
code, and the constraint is enforced so it cannot silently drift.

**"How good is the lip sync?"** Approximate: it is driven by mel-spectrograms, not
phonemes. Good at normal speed, drifts on very fast speech. Documented in
`docs/EVALUATION.md` §7.

**"What would you do with more time?"** LSE-D/LSE-C lip-sync metrics, a human MOS
panel, batching for long scripts, and a Wav2Lip post-pass for tighter sync.
