"""Orchestrate the full voice conversion pipeline.

Steps
-----
1. Load content audio at native rate; record exact duration.
2. Load target reference(s) at native rate.
3. Resample both to the backend's required rate (16 kHz for kNN-VC).
4. Run the backend converter.
5. Resample output to the final output sample rate.
6. Length-align: trim or pad to exactly round(content_duration * out_sr) samples.
7. Optional loudness normalization.
8. Save WAV. Optionally mux onto input video.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np

from voicetransfer.audio import (
    load_audio,
    normalize_loudness,
    numpy_to_tensor,
    resample,
    save_audio,
    tensor_to_numpy,
)
from voicetransfer.converters.base import BaseConverter

if TYPE_CHECKING:
    from voicetransfer.config import AppConfig

logger = logging.getLogger(__name__)


# ── Stats dataclasses ─────────────────────────────────────────────────────────

@dataclass
class StepStat:
    name: str
    duration_s: float
    ram_mb: float       # RSS at end of step


@dataclass
class PipelineStats:
    steps: List[StepStat] = field(default_factory=list)
    total_s: float = 0.0
    peak_ram_mb: float = 0.0
    baseline_ram_mb: float = 0.0
    content_duration_s: float = 0.0
    output_duration_s: float = 0.0
    output_sr: int = 0
    drift_corrected_ms: float = 0.0
    output_size_mb: float = 0.0

    @property
    def realtime_factor(self) -> float:
        """Seconds of compute per second of audio (lower = faster)."""
        return self.total_s / self.content_duration_s if self.content_duration_s else 0.0

    @property
    def net_ram_mb(self) -> float:
        """Peak RAM above the pre-pipeline baseline."""
        return max(0.0, self.peak_ram_mb - self.baseline_ram_mb)


# ── RAM helper ────────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    """Current process RSS in MB (0 if psutil not available)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


class _Timer:
    """Context manager that records wall time and RAM at exit."""

    def __init__(self, stats: PipelineStats, name: str):
        self._stats = stats
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        dur = time.perf_counter() - self._t0
        ram = _rss_mb()
        self._stats.steps.append(StepStat(self._name, dur, ram))
        if ram > self._stats.peak_ram_mb:
            self._stats.peak_ram_mb = ram


# ── Length alignment ──────────────────────────────────────────────────────────

def _length_align(
    wav: np.ndarray,
    expected_samples: int,
    pad_mode: str,
    warn_threshold_ms: float,
    out_sr: int,
) -> np.ndarray:
    """Trim or pad wav so its length equals expected_samples exactly.

    Pure function with no side-effects — directly testable.
    """
    actual = len(wav)
    drift_samples = actual - expected_samples
    drift_ms = drift_samples / out_sr * 1000.0

    logger.info(
        "Length alignment: expected=%d  actual=%d  drift=%.1f ms",
        expected_samples, actual, drift_ms,
    )

    if abs(drift_ms) > warn_threshold_ms:
        logger.warning(
            "Pre-correction drift %.1f ms exceeds threshold %.1f ms.",
            drift_ms, warn_threshold_ms,
        )

    if actual == expected_samples:
        return wav
    if actual > expected_samples:
        return wav[:expected_samples]

    pad = np.zeros(expected_samples - actual, dtype=np.float32)
    return np.concatenate([wav, pad])


# ── Converter factory ─────────────────────────────────────────────────────────

def _build_converter(config: "AppConfig", model) -> BaseConverter:
    """Instantiate the backend specified in config.backend.name."""
    name = config.backend.name
    if name == "knn_vc":
        from voicetransfer.converters.knn_vc import KnnVcConverter
        return KnnVcConverter(
            model=model,
            topk=config.knn_vc.topk,
            wavlm_layer=config.knn_vc.wavlm_layer,
            target_vad_level=config.knn_vc.target_vad_level,
        )
    raise ValueError(
        f"Unknown backend '{name}'. "
        f"Add voicetransfer/converters/{name}.py, implement BaseConverter, "
        f"and register it in pipeline._build_converter()."
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(config: "AppConfig", model) -> PipelineStats:
    """Execute the full voice conversion pipeline end-to-end.

    Returns a PipelineStats with per-step timing and RAM usage.
    """
    stats = PipelineStats()
    stats.baseline_ram_mb = _rss_mb()
    wall_start = time.perf_counter()

    # ── Step 1: Load content ─────────────────────────────────────────────────
    with _Timer(stats, "Load content"):
        logger.info("Loading content audio: %s", config.paths.content_audio)
        content_wav, native_sr = load_audio(config.paths.content_audio)
        content_duration_sec = len(content_wav) / native_sr
        stats.content_duration_s = content_duration_sec
        logger.info("Content: %.3f s | %d Hz | %d samples",
                    content_duration_sec, native_sr, len(content_wav))

    # ── Step 2: Load target references ──────────────────────────────────────
    with _Timer(stats, "Load targets"):
        logger.info("Loading %d target reference(s)...", len(config.paths.target_refs))
        target_pairs = [load_audio(p) for p in config.paths.target_refs]

    # ── Step 3: Resample to model rate ───────────────────────────────────────
    with _Timer(stats, "Resample inputs"):
        converter = _build_converter(config, model)
        model_sr = converter.input_sample_rate
        logger.info("Resampling to model rate (%d Hz)...", model_sr)
        content_model = resample(content_wav, native_sr, model_sr)
        target_model = [resample(wav, sr, model_sr) for wav, sr in target_pairs]

    # ── Step 4: Convert ──────────────────────────────────────────────────────
    with _Timer(stats, "Voice conversion (WavLM + kNN + HiFiGAN)"):
        logger.info("Running backend '%s'...", config.backend.name)
        content_tensor = numpy_to_tensor(content_model)
        target_tensors = [numpy_to_tensor(t) for t in target_model]
        out_tensor = converter.convert(content_tensor, target_tensors)
        out_model_wav = tensor_to_numpy(out_tensor)
        logger.info("Backend: %d samples @ %d Hz (%.3f s)",
                    len(out_model_wav), model_sr, len(out_model_wav) / model_sr)

    # ── Step 5: Resample output ──────────────────────────────────────────────
    with _Timer(stats, "Resample output"):
        out_sr = (
            config.audio.output_sample_rate
            if config.audio.output_sample_rate != 0
            else native_sr
        )
        out_wav = resample(out_model_wav, model_sr, out_sr)

    # ── Step 6: Length alignment ─────────────────────────────────────────────
    with _Timer(stats, "Length align"):
        expected_samples = round(content_duration_sec * out_sr)
        if config.length.enforce_exact:
            out_wav = _length_align(
                wav=out_wav,
                expected_samples=expected_samples,
                pad_mode=config.length.pad_mode,
                warn_threshold_ms=config.length.warn_if_drift_ms,
                out_sr=out_sr,
            )
        stats.drift_corrected_ms = abs(len(out_wav) - expected_samples) / out_sr * 1000.0

    # ── Step 7: Loudness normalization ───────────────────────────────────────
    with _Timer(stats, "Loudness normalize"):
        if config.audio.normalize_loudness:
            logger.info("Normalizing to %.1f LUFS...", config.audio.target_lufs)
            out_wav = normalize_loudness(out_wav, out_sr, config.audio.target_lufs)

    # ── Step 8: Save ─────────────────────────────────────────────────────────
    with _Timer(stats, "Save WAV"):
        out_path = Path(config.paths.output_audio)
        logger.info("Saving: %s", out_path)
        save_audio(out_wav, out_sr, out_path)
        stats.output_size_mb = out_path.stat().st_size / 1024 / 1024

    if config.mux.enabled and config.paths.input_video:
        with _Timer(stats, "Mux video"):
            from voicetransfer.mux import remux
            remux(
                input_video=config.paths.input_video,
                audio_path=str(out_path),
                output_video=config.paths.output_video,
                ffmpeg_path=config.mux.ffmpeg_path,
                copy_video=config.mux.copy_video_codec,
            )
    elif config.mux.enabled and not config.paths.input_video:
        logger.warning("mux.enabled=true but paths.input_video is empty — skipping mux.")

    # ── Finalise stats ────────────────────────────────────────────────────────
    stats.total_s = time.perf_counter() - wall_start
    stats.output_duration_s = len(out_wav) / out_sr
    stats.output_sr = out_sr

    # ── CLI summary ───────────────────────────────────────────────────────────
    sep = "─" * 56
    step_lines = "\n".join(
        f"  {s.name:<40} {s.duration_s:>6.1f}s   {s.ram_mb:>7.0f} MB"
        for s in stats.steps
    )
    print(
        f"\n{sep}\n"
        f"  VoiceTransfer complete\n"
        f"{sep}\n"
        f"  Input          : {content_duration_sec:.2f}s  ({native_sr} Hz)\n"
        f"  Output         : {stats.output_duration_s:.2f}s  ({out_sr} Hz)\n"
        f"  Drift corrected: {stats.drift_corrected_ms:.1f} ms\n"
        f"  Total time     : {stats.total_s:.1f}s  "
        f"({stats.realtime_factor:.1f}x realtime)\n"
        f"  Peak RAM       : {stats.peak_ram_mb:.0f} MB  "
        f"(+{stats.net_ram_mb:.0f} MB above baseline)\n"
        f"  Output size    : {stats.output_size_mb:.1f} MB\n"
        f"  Saved          : {out_path.resolve()}\n"
        f"{sep}\n"
        f"  {'Step':<40} {'Time':>7}   {'RAM':>8}\n"
        f"  {'─'*40} {'─'*7}   {'─'*8}\n"
        f"{step_lines}\n"
        f"{sep}"
    )

    return stats
