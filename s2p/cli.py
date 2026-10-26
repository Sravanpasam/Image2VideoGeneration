"""Command line interface: ``python -m s2p.cli --image portrait.jpg --text "..."``.

Useful for scripted batch runs and for the Colab notebook (which does not need a
Gradio server).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_POSE_STYLE,
    DEFAULT_PREPROCESS,
    DEFAULT_PROVIDER,
    DEFAULT_SIZE,
)
from .pipeline import PipelineError, Script2Presenter
from .tts import TTSError, available_providers, list_voices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s2p",
        description="Turn a portrait and a script into a talking-head MP4 (SadTalker).",
    )
    parser.add_argument("--image", "-i", help="portrait JPG/PNG")
    script_group = parser.add_mutually_exclusive_group()
    script_group.add_argument("--text", "-t", help="script to speak")
    script_group.add_argument(
        "--text-file",
        help="read the script from a UTF-8 text file (handy in notebooks)",
    )
    parser.add_argument("--out", "-o", help="path of the resulting MP4")
    parser.add_argument(
        "--sadtalker-root",
        default=None,
        help="path of the official SadTalker checkout (default: auto-detect)",
    )
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        choices=["auto", "edge", "piper", "gtts"])
    parser.add_argument("--voice", default=None, help="provider specific voice id")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, choices=[256, 512])
    parser.add_argument("--preprocess", default=DEFAULT_PREPROCESS, choices=["crop", "full"])
    parser.add_argument("--still", action="store_true", help="reduce head movement")
    parser.add_argument("--enhancer", action="store_true", help="enable the GFPGAN enhancer")
    parser.add_argument("--pose-style", type=int, default=DEFAULT_POSE_STYLE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--check", action="store_true", help="only run the environment checks")
    parser.add_argument("--list-voices", action="store_true", help="print the available voices")
    parser.add_argument("--json", action="store_true", help="print the run manifest as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.list_voices and not args.check and not args.image:
        parser.error("--image is required (unless --check or --list-voices is used)")
    if args.image and not args.check and not args.text and not args.text_file:
        parser.error("pass --text or --text-file, or use --check to only inspect the setup")

    if args.list_voices:
        providers = (
            (args.provider,) if args.provider != "auto" else ("edge", "piper", "gtts")
        )
        for provider in providers:
            try:
                voices = list_voices(provider)
            except Exception as exc:
                print(f"{provider}: unavailable ({exc})")
                continue
            print(f"{provider} ({len(voices)} voices):")
            for voice in voices:
                print(f"  - {voice}")
        return 0

    try:
        if args.sadtalker_root:
            from .config import load_settings

            app = Script2Presenter(
                settings=load_settings(sadtalker_root=args.sadtalker_root), device=args.device
            )
        else:
            app = Script2Presenter(device=args.device)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.text_file:
        try:
            args.text = Path(args.text_file).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not read --text-file: {exc}", file=sys.stderr)
            return 1

    print("Environment:")
    checks = app.preflight(size=args.size, preprocess=args.preprocess)
    for ok, message in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {message}")
    print(f"  TTS providers available: {available_providers()}")

    if args.check:
        failed = [message for ok, message in checks if not ok]
        if failed:
            print(f"\n{len(failed)} check(s) failed - generation would not work yet.")
            return 1
        print("\nAll checks passed.")
        return 0

    if not args.text:
        print("Nothing to do: pass --text / --text-file (or --check).", file=sys.stderr)
        return 2

    try:
        result = app.run(
            image_path=args.image,
            script=args.text,
            provider=args.provider,
            voice=args.voice,
            size=args.size,
            preprocess=args.preprocess,
            still=args.still,
            pose_style=args.pose_style,
            batch_size=args.batch_size,
            enhancer="gfpgan" if args.enhancer else None,
            progress=lambda f, m: print(f"[{f*100:5.1f}%] {m}", flush=True),
        )
    except (PipelineError, TTSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(result.video_path).read_bytes())
        print(f"Video written to {target}")

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(f"Video: {result.video_path}")
        print(f"Speech: {result.audio_path}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Timings: {result.timings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
