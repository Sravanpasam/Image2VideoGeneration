#!/usr/bin/env python3
"""Download the official SadTalker checkpoints.

Only public, official release assets are used:

* https://github.com/OpenTalker/SadTalker/releases/tag/v0.0.2-rc
* https://github.com/xinntao/facexlib/releases  (face detection / alignment)
* https://github.com/TencentARC/GFPGAN/releases  (optional face enhancer)

A Hugging Face mirror is used automatically when GitHub is slow or unreachable.

Usage::

    python scripts/download_checkpoints.py --dest third_party/SadTalker
    python scripts/download_checkpoints.py --size 512 --with-enhancer

The script is resumable-ish: a partial file is kept as ``<name>.part`` and
re-downloaded on the next run; complete files are skipped after a size check.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GH_RELEASE = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
HF_MIRROR = "https://huggingface.co/camenduru/SadTalker/resolve/main/new"

FACEXLIB = "https://github.com/xinntao/facexlib/releases/download"
GFPGAN = "https://github.com/TencentARC/GFPGAN/releases/download"

# (relative path, [urls], exact size in bytes or None, tags)
FILES: list[tuple[str, list[str], int | None, set[str]]] = [
    (
        "checkpoints/SadTalker_V0.0.2_256.safetensors",
        [f"{GH_RELEASE}/SadTalker_V0.0.2_256.safetensors",
         f"{HF_MIRROR}/checkpoints/SadTalker_V0.0.2_256.safetensors"],
        725_066_984,
        {"256"},
    ),
    (
        "checkpoints/SadTalker_V0.0.2_512.safetensors",
        [f"{GH_RELEASE}/SadTalker_V0.0.2_512.safetensors",
         f"{HF_MIRROR}/checkpoints/SadTalker_V0.0.2_512.safetensors"],
        725_066_984,
        {"512"},
    ),
    (
        "checkpoints/mapping_00229-model.pth.tar",
        [f"{GH_RELEASE}/mapping_00229-model.pth.tar",
         f"{HF_MIRROR}/checkpoints/mapping_00229-model.pth.tar"],
        155_521_183,
        {"crop"},
    ),
    (
        "checkpoints/mapping_00109-model.pth.tar",
        [f"{GH_RELEASE}/mapping_00109-model.pth.tar",
         f"{HF_MIRROR}/checkpoints/mapping_00109-model.pth.tar"],
        155_779_231,
        {"full"},
    ),
    (
        "gfpgan/weights/alignment_WFLW_4HG.pth",
        [f"{FACEXLIB}/v0.1.0/alignment_WFLW_4HG.pth",
         f"{HF_MIRROR}/gfpgan/weights/alignment_WFLW_4HG.pth"],
        193_670_248,
        {"core"},
    ),
    (
        "gfpgan/weights/detection_Resnet50_Final.pth",
        [f"{FACEXLIB}/v0.1.0/detection_Resnet50_Final.pth",
         f"{HF_MIRROR}/gfpgan/weights/detection_Resnet50_Final.pth"],
        109_497_761,
        {"core"},
    ),
    (
        "gfpgan/weights/GFPGANv1.4.pth",
        [f"{GFPGAN}/v1.3.0/GFPGANv1.4.pth",
         f"{HF_MIRROR}/gfpgan/weights/GFPGANv1.4.pth"],
        348_632_874,
        {"enhancer"},
    ),
    (
        "gfpgan/weights/parsing_parsenet.pth",
        [f"{FACEXLIB}/v0.2.2/parsing_parsenet.pth",
         f"{HF_MIRROR}/gfpgan/weights/parsing_parsenet.pth"],
        85_331_193,
        {"enhancer"},
    ),
]

MIN_BYTES = 1_000_000


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"


def _download_one(url: str, target: Path, expected: int | None) -> None:
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Script2Presenter/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        if expected and total and total != expected:
            raise OSError(f"size mismatch: server says {total}, expected {expected}")
        received = 0
        last_report = 0.0
        with part.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                now = time.time()
                if now - last_report > 2.0:
                    last_report = now
                    pct = f"{100 * received / total:5.1f}%" if total else "  ?  "
                    print(f"    {pct}  {_human(received)}", end="\r", flush=True)
    print(f"    {_human(received)} downloaded" + " " * 20)

    size = part.stat().st_size
    if expected and size != expected:
        part.unlink(missing_ok=True)
        raise OSError(f"size mismatch after download: {size} != {expected}")
    if size < MIN_BYTES:
        part.unlink(missing_ok=True)
        raise OSError(f"downloaded file is suspiciously small ({size} bytes)")
    os.replace(part, target)


def download(target: Path, urls: list[str], expected: int | None, retries: int = 2) -> bool:
    if target.is_file() and target.stat().st_size >= (expected or MIN_BYTES):
        print(f"  = {target} (already present, {_human(target.stat().st_size)})")
        return True
    for url in urls:
        for attempt in range(1, retries + 1):
            print(f"  > {target}  <-  {url} (attempt {attempt})")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                _download_one(url, target, expected)
                print(f"  + {target} ({_human(target.stat().st_size)})")
                return True
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                print(f"    failed: {exc}")
                time.sleep(2 * attempt)
    print(f"  ! could not download {target}")
    return False


def selected_tags(size: str, with_enhancer: bool, with_full: bool) -> set[str]:
    tags = {"core"}
    if size in ("256", "both"):
        tags.add("256")
    if size in ("512", "both"):
        tags.add("512")
    tags.add("crop")
    if with_full:
        tags.add("full")
    if with_enhancer:
        tags.add("enhancer")
    return tags


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=str(root / "third_party" / "SadTalker"),
        help="SadTalker checkout (default: %(default)s)",
    )
    parser.add_argument("--size", default="256", choices=["256", "512", "both"])
    parser.add_argument("--with-enhancer", action="store_true",
                        help="also fetch the GFPGAN enhancer weights (+430 MB)")
    parser.add_argument("--with-full", action="store_true",
                        help="also fetch the mapping network used by --preprocess full")
    parser.add_argument("--check", action="store_true",
                        help="only report which files are missing")
    args = parser.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()
    tags = selected_tags(args.size, args.with_enhancer, args.with_full)

    wanted = [entry for entry in FILES if entry[3] & tags]
    total = sum((entry[2] or MIN_BYTES) for entry in wanted)

    print(f"SadTalker root : {dest}")
    print(f"Files          : {len(wanted)}  (~{_human(total)})")
    print(f"Selection      : {', '.join(sorted(tags))}\n")

    missing = [e for e in wanted
               if not (dest / e[0]).is_file()
               or (dest / e[0]).stat().st_size < (e[2] or MIN_BYTES)]
    if args.check:
        if not missing:
            print("All requested checkpoints are present.")
            return 0
        print("Missing:")
        for relative, _, size, _ in missing:
            print(f"  - {relative} ({_human(size or MIN_BYTES)})")
        return 1

    failures = 0
    for relative, urls, size, _ in wanted:
        if not download(dest / relative, urls, size):
            failures += 1

    print()
    if failures:
        print(f"{failures} file(s) could not be downloaded.")
        return 1
    print("All requested checkpoints are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
