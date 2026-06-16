"""Load and validate config.yaml into typed dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    models_dir: str = "./models"
    content_audio: str = "./input/content.wav"
    target_refs: List[str] = field(default_factory=lambda: ["./input/target.wav"])
    output_audio: str = "./output/converted.wav"
    input_video: str = ""
    output_video: str = "./output/converted.mp4"


@dataclass
class DeviceConfig:
    type: str = "cpu"
    num_threads: int = 4


@dataclass
class BackendConfig:
    name: str = "knn_vc"


@dataclass
class KnnVcConfig:
    prematched: bool = True
    topk: int = 4
    wavlm_layer: int = 6


@dataclass
class AudioConfig:
    output_sample_rate: int = 0
    normalize_loudness: bool = True
    target_lufs: float = -23.0


@dataclass
class LengthConfig:
    enforce_exact: bool = True
    pad_mode: str = "silence"
    warn_if_drift_ms: float = 50.0


@dataclass
class MuxConfig:
    enabled: bool = False
    copy_video_codec: bool = True
    ffmpeg_path: str = "auto"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    knn_vc: KnnVcConfig = field(default_factory=KnnVcConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    length: LengthConfig = field(default_factory=LengthConfig)
    mux: MuxConfig = field(default_factory=MuxConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _g(d: dict, key: str, default):
    """Safe dict get with default."""
    return d.get(key, default) if isinstance(d, dict) else default


def _parse_raw(raw: dict) -> AppConfig:
    """Convert a raw YAML dict into AppConfig, applying defaults for missing keys."""
    p  = raw.get("paths",   {})
    dv = raw.get("device",  {})
    b  = raw.get("backend", {})
    k  = raw.get("knn_vc",  {})
    a  = raw.get("audio",   {})
    le = raw.get("length",  {})
    m  = raw.get("mux",     {})
    lg = raw.get("logging", {})

    return AppConfig(
        paths=PathsConfig(
            models_dir=_g(p, "models_dir", "./models"),
            content_audio=_g(p, "content_audio", "./input/content.wav"),
            target_refs=_g(p, "target_refs", ["./input/target.wav"]),
            output_audio=_g(p, "output_audio", "./output/converted.wav"),
            input_video=_g(p, "input_video", ""),
            output_video=_g(p, "output_video", "./output/converted.mp4"),
        ),
        device=DeviceConfig(
            type=_g(dv, "type", "cpu"),
            num_threads=_g(dv, "num_threads", 4),
        ),
        backend=BackendConfig(name=_g(b, "name", "knn_vc")),
        knn_vc=KnnVcConfig(
            prematched=_g(k, "prematched", True),
            topk=_g(k, "topk", 4),
            wavlm_layer=_g(k, "wavlm_layer", 6),
        ),
        audio=AudioConfig(
            output_sample_rate=_g(a, "output_sample_rate", 0),
            normalize_loudness=_g(a, "normalize_loudness", True),
            target_lufs=_g(a, "target_lufs", -23.0),
        ),
        length=LengthConfig(
            enforce_exact=_g(le, "enforce_exact", True),
            pad_mode=_g(le, "pad_mode", "silence"),
            warn_if_drift_ms=float(_g(le, "warn_if_drift_ms", 50.0)),
        ),
        mux=MuxConfig(
            enabled=_g(m, "enabled", False),
            copy_video_codec=_g(m, "copy_video_codec", True),
            ffmpeg_path=_g(m, "ffmpeg_path", "auto"),
        ),
        logging=LoggingConfig(level=_g(lg, "level", "INFO")),
    )


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load config.yaml and return a fully-defaulted AppConfig."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _parse_raw(raw)
    logger.debug("Config loaded from %s", path)
    return cfg


def validate_for_conversion(config: AppConfig) -> None:
    """Raise clear FileNotFoundError / ValueError for invalid conversion inputs."""
    content = Path(config.paths.content_audio)
    if not content.exists():
        raise FileNotFoundError(
            f"Content audio not found: {content}\n"
            f"  → Set paths.content_audio in config.yaml or pass --content <path>"
        )

    if not config.paths.target_refs:
        raise ValueError("paths.target_refs must contain at least one audio file.")

    for ref in config.paths.target_refs:
        rp = Path(ref)
        if not rp.exists():
            raise FileNotFoundError(
                f"Target reference not found: {rp}\n"
                f"  → Set paths.target_refs in config.yaml or pass --target <path>"
            )

    if config.device.type != "cpu":
        raise ValueError(
            f"Only device.type='cpu' is supported; got '{config.device.type}'. "
            f"This tool is CPU-only by design."
        )

    if config.mux.enabled and config.paths.input_video:
        vp = Path(config.paths.input_video)
        if not vp.exists():
            raise FileNotFoundError(f"Input video not found: {vp}")
