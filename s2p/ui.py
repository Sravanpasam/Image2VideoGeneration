"""Gradio front-end for Script2Presenter.

The UI is intentionally small: upload a portrait, type the script, press
*Generate*, watch the progress, preview the video, download the MP4.

The code is written against the Gradio 3 / 4 / 5 common subset so that the same
file works in the Colab environment and in a plain local venv.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_POSE_STYLE,
    DEFAULT_PREPROCESS,
    DEFAULT_PROVIDER,
    DEFAULT_SIZE,
    DEFAULT_VOICE,
)
from .pipeline import PipelineError, Script2Presenter

__all__ = ["build_demo", "launch"]

FALLBACK_EDGE_VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-DavisNeural",
    "en-US-SaraNeural",
    "en-US-TonyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-IN-NeerjaNeural",
]


def _gradio_major() -> int:
    try:
        import gradio as gr

        return int(str(gr.__version__).split(".")[0])
    except Exception:
        return 4


_GRADIO_MAJOR = _gradio_major()


def _progress(fraction: float, message: str) -> None:
    """Forward our internal progress callback to Gradio."""
    try:
        import gradio as gr

        value = fraction * 100.0 if _GRADIO_MAJOR <= 3 else min(1.0, max(0.0, fraction))
        gr.Progress()(value, message)
    except Exception:
        pass


def _edge_voice_choices() -> list[str]:
    from .tts import list_voices

    try:
        voices = list_voices("edge")
        return voices or list(FALLBACK_EDGE_VOICES)
    except Exception:
        return list(FALLBACK_EDGE_VOICES)


def _piper_voice_choices() -> list[str]:
    from .tts import list_voices

    try:
        return list_voices("piper")
    except Exception:
        return ["en_US-amy-medium"]


def build_demo(app: Script2Presenter | None = None, title: str = "Script2Presenter"):
    """Build the Gradio Blocks application."""
    import gradio as gr

    app = app or Script2Presenter()

    edge_voices = _edge_voice_choices()
    piper_voices = _piper_voice_choices()
    default_edge = DEFAULT_VOICE if DEFAULT_VOICE in edge_voices else edge_voices[0]

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            """
# Script2Presenter

Turn **one portrait + one script** into a downloadable talking-head MP4.

1. Upload a frontal JPG/PNG portrait.
2. Type the script that should be spoken.
3. Press **Generate video**.

The speech is produced by a free neural TTS model and the portrait is animated
with the official [SadTalker](https://github.com/OpenTalker/SadTalker) checkpoints
(CVPR 2023). No API key, no account, no paid service.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                portrait = gr.Image(
                    label="Portrait (JPG or PNG, single frontal face)",
                    type="filepath",
                    height=320,
                )
                script = gr.Textbox(
                    label="Script",
                    lines=8,
                    placeholder="Good morning! Today I am going to explain how a single photograph can be turned into a talking head video ...",
                )
                with gr.Row():
                    provider = gr.Dropdown(
                        choices=["auto", "edge", "piper", "gtts"],
                        value=DEFAULT_PROVIDER,
                        label="Speech engine",
                        info="auto = edge -> piper -> gtts (first one that works)",
                    )
                    edge_voice = gr.Dropdown(
                        choices=edge_voices, value=default_edge, label="Edge voice"
                    )
                with gr.Row():
                    piper_voice = gr.Dropdown(
                        choices=piper_voices,
                        value=piper_voices[0],
                        label="Piper voice (fully offline)",
                    )
                with gr.Accordion("Advanced settings", open=False):
                    size = gr.Radio(
                        choices=[256, 512],
                        value=DEFAULT_SIZE,
                        label="Face resolution",
                        info="256 is fast, 512 looks sharper (needs the 512 checkpoint)",
                    )
                    preprocess = gr.Radio(
                        choices=["crop", "full"],
                        value=DEFAULT_PREPROCESS,
                        label="Framing",
                        info="crop = face only, full = portrait with the animated face",
                    )
                    still = gr.Checkbox(
                        value=False, label="Still mode (less head movement)"
                    )
                    enhancer = gr.Checkbox(
                        value=False,
                        label="GFPGAN face enhancer (slower, sharper face)",
                    )
                    pose_style = gr.Slider(
                        minimum=0, maximum=46, step=1, value=DEFAULT_POSE_STYLE,
                        label="Pose style",
                    )
                    batch_size = gr.Slider(
                        minimum=1, maximum=8, step=1, value=DEFAULT_BATCH_SIZE,
                        label="Render batch size",
                    )
                with gr.Row():
                    generate = gr.Button("Generate video", variant="primary")
                    doctor = gr.Button("Check environment")

            with gr.Column(scale=1):
                status = gr.Markdown("Ready. Upload a portrait and write a script.")
                audio_out = gr.Audio(label="Generated speech", type="filepath")
                video_out = gr.Video(label="Generated video", format="mp4")
                download = gr.File(label="Download MP4")

        def _on_generate(image, text, provider_value, edge_v, piper_v, size_v,
                         preprocess_v, still_v, enhancer_v, pose_v, batch_v):
            voice = piper_v if provider_value == "piper" else edge_v
            if provider_value == "auto" and not _edge_available():
                voice = piper_v
            try:
                result = app.run(
                    image_path=image,
                    script=text,
                    provider=provider_value,
                    voice=voice,
                    size=int(size_v),
                    preprocess=preprocess_v,
                    still=bool(still_v),
                    pose_style=int(pose_v),
                    batch_size=int(batch_v),
                    enhancer="gfpgan" if enhancer_v else None,
                    progress=_progress,
                )
            except PipelineError as exc:
                traceback.print_exc()
                return (
                    None,
                    None,
                    None,
                    f"### ❌ Failed\n\n```\n{exc}\n```",
                )
            except Exception as exc:  # pragma: no cover - unexpected failures
                traceback.print_exc()
                return (None, None, None, f"### ❌ Unexpected error\n\n```\n{exc}\n```")

            timings = result.timings
            message = (
                f"### ✅ Done\n"
                f"- video: `{result.video_path.name}`\n"
                f"- duration: **{result.generation.duration_s:.1f} s** "
                f"({result.generation.frames} frames @ 25 fps)\n"
                f"- speech engine: `{result.tts.provider}` / `{result.tts.voice}`\n"
                f"- time: TTS {timings['tts_s']:.1f}s + render {timings['sadtalker_s']:.1f}s"
                f" = **{timings['total_s']:.1f}s** on `{result.generation.device}`\n"
                f"- manifest: `{result.manifest_path.name}`"
            )
            return (
                str(result.video_path),
                str(result.audio_path),
                str(result.video_path),
                message,
            )

        def _on_doctor(size_v, preprocess_v):
            checks = app.preflight(size=int(size_v), preprocess=preprocess_v)
            lines = ["### Environment check", ""]
            for ok, message in checks:
                lines.append(f"- {'✅' if ok else '❌'} {message}")
            return "\n".join(lines)

        def _edge_available() -> bool:
            from .tts import available_providers

            return available_providers().get("edge", False)

        inputs = [
            portrait, script, provider, edge_voice, piper_voice,
            size, preprocess, still, enhancer, pose_style, batch_size,
        ]
        outputs = [video_out, audio_out, download, status]

        generate.click(fn=_on_generate, inputs=inputs, outputs=outputs)
        doctor.click(
            fn=_on_doctor,
            inputs=[size, preprocess],
            outputs=[status],
        )

    return demo


def launch(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    app: Script2Presenter | None = None,
    block: bool = True,
):
    """Build and launch the Gradio app."""
    demo = build_demo(app)
    demo.queue()
    return demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        block=block,
        show_error=True,
    )


def _parse_args(argv: list[str] | None = None):  # pragma: no cover - thin wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Script2Presenter web UI.")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    launch(share=args.share, server_name=args.host, server_port=args.port)
