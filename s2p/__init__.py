"""Script2Presenter - portrait + script -> talking-head MP4.

Public entry points::

    from s2p import Script2Presenter
    app = Script2Presenter()
    result = app.run("portrait.jpg", "Hello world, this is my presentation.")

or from the command line::

    python -m s2p.cli --image portrait.jpg --text "Hello world" --out out.mp4
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Settings, load_settings
from .pipeline import PipelineError, PipelineResult, Script2Presenter
from .sadtalker_backend import SadTalkerBackend, SadTalkerError
from .tts import TTSError, TTSResult, list_voices, synthesize

__all__ = [
    "PipelineError",
    "PipelineResult",
    "SadTalkerBackend",
    "SadTalkerError",
    "Script2Presenter",
    "Settings",
    "TTSError",
    "TTSResult",
    "__version__",
    "list_voices",
    "load_settings",
    "synthesize",
]
