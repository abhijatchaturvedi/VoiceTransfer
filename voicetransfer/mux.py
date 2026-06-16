"""Remux converted audio onto a video file via ffmpeg (bundled or system)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_ffmpeg(ffmpeg_path: str) -> str:
    """Return the path to the ffmpeg binary to use."""
    if ffmpeg_path == "auto":
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        logger.debug("Using bundled ffmpeg: %s", path)
        return path
    return ffmpeg_path


def remux(
    input_video: str,
    audio_path: str,
    output_video: str,
    ffmpeg_path: str = "auto",
    copy_video: bool = True,
) -> None:
    """Replace the audio track of input_video with audio_path.

    The video stream is copied without re-encoding (copy_video=True).
    Raises RuntimeError if ffmpeg exits non-zero.
    """
    ff = _resolve_ffmpeg(ffmpeg_path)
    out = Path(output_video)
    out.parent.mkdir(parents=True, exist_ok=True)

    video_codec = "copy" if copy_video else "libx264"
    cmd = [
        ff,
        "-y",                  # overwrite output without prompting
        "-i", str(input_video),
        "-i", str(audio_path),
        "-c:v", video_codec,
        "-c:a", "aac",
        "-map", "0:v:0",       # video stream from first input
        "-map", "1:a:0",       # audio stream from second input (converted)
        "-shortest",           # end when the shorter stream ends
        str(out),
    ]
    logger.info("Muxing command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mux failed (exit {result.returncode}):\n{result.stderr}"
        )
    logger.info("Muxed video saved: %s", out)
