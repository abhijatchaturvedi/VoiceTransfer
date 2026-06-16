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
from pathlib import Path
from typing import TYPE_CHECKING

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


# ── Length alignment ──────────────────────────────────────────────────────────

def _length_align(
    wav: np.ndarray,
    expected_samples: int,
    pad_mode: str,
    warn_threshold_ms: float,
    out_sr: int,
) -> np.ndarray:
    """Trim or pad wav so its length equals expected_samples exactly.

    Logs a WARNING if the pre-correction drift exceeds warn_threshold_ms.
    This is a pure function with no side-effects, making it directly testable.
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
            "Pre-correction drift %.1f ms exceeds threshold %.1f ms — "
            "check backend output length consistency.",
            drift_ms, warn_threshold_ms,
        )

    if actual == expected_samples:
        return wav

    if actual > expected_samples:
        logger.debug("Trimming %d surplus samples.", drift_samples)
        return wav[:expected_samples]

    # actual < expected_samples
    pad_len = expected_samples - actual
    logger.debug("Padding %d samples (%s).", pad_len, pad_mode)
    pad = np.zeros(pad_len, dtype=np.float32)   # silence regardless of pad_mode
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

def run_pipeline(config: "AppConfig", model) -> None:
    """Execute the full voice conversion pipeline end-to-end."""

    # ── Step 1: Load content ─────────────────────────────────────────────────
    logger.info("Loading content audio: %s", config.paths.content_audio)
    content_wav, native_sr = load_audio(config.paths.content_audio)
    content_duration_sec = len(content_wav) / native_sr
    logger.info(
        "Content: %.4f s | %d Hz | %d samples",
        content_duration_sec, native_sr, len(content_wav),
    )

    # ── Step 2: Load target references ──────────────────────────────────────
    logger.info(
        "Loading %d target reference file(s)...", len(config.paths.target_refs)
    )
    target_pairs = [load_audio(p) for p in config.paths.target_refs]

    # ── Step 3: Build converter and resample to model rate ───────────────────
    converter = _build_converter(config, model)
    model_sr = converter.input_sample_rate

    logger.info("Resampling content to model rate (%d Hz)...", model_sr)
    content_model = resample(content_wav, native_sr, model_sr)

    logger.info("Resampling %d target clip(s) to %d Hz...", len(target_pairs), model_sr)
    target_model = [resample(wav, sr, model_sr) for wav, sr in target_pairs]

    # ── Step 4: Convert ──────────────────────────────────────────────────────
    logger.info("Running backend '%s'...", config.backend.name)
    content_tensor = numpy_to_tensor(content_model)
    target_tensors = [numpy_to_tensor(t) for t in target_model]
    out_tensor = converter.convert(content_tensor, target_tensors)
    out_model_wav = tensor_to_numpy(out_tensor)
    logger.info(
        "Backend produced %d samples @ %d Hz (%.4f s)",
        len(out_model_wav), model_sr, len(out_model_wav) / model_sr,
    )

    # ── Step 5: Resample to final output rate ────────────────────────────────
    out_sr = (
        config.audio.output_sample_rate
        if config.audio.output_sample_rate != 0
        else native_sr
    )
    out_wav = resample(out_model_wav, model_sr, out_sr)

    # ── Step 6: Length alignment ─────────────────────────────────────────────
    expected_samples = round(content_duration_sec * out_sr)
    if config.length.enforce_exact:
        out_wav = _length_align(
            wav=out_wav,
            expected_samples=expected_samples,
            pad_mode=config.length.pad_mode,
            warn_threshold_ms=config.length.warn_if_drift_ms,
            out_sr=out_sr,
        )
    pre_drift_ms = abs(len(out_wav) - expected_samples) / out_sr * 1000.0

    # ── Step 7: Loudness normalization ───────────────────────────────────────
    if config.audio.normalize_loudness:
        logger.info(
            "Applying loudness normalization → %.1f LUFS...", config.audio.target_lufs
        )
        out_wav = normalize_loudness(out_wav, out_sr, config.audio.target_lufs)

    # ── Step 8: Save and optionally mux ─────────────────────────────────────
    out_path = Path(config.paths.output_audio)
    logger.info("Saving WAV: %s", out_path)
    save_audio(out_wav, out_sr, out_path)

    if config.mux.enabled and config.paths.input_video:
        from voicetransfer.mux import remux
        logger.info("Muxing audio onto video: %s", config.paths.input_video)
        remux(
            input_video=config.paths.input_video,
            audio_path=str(out_path),
            output_video=config.paths.output_video,
            ffmpeg_path=config.mux.ffmpeg_path,
            copy_video=config.mux.copy_video_codec,
        )
    elif config.mux.enabled and not config.paths.input_video:
        logger.warning("mux.enabled=true but paths.input_video is empty — skipping mux.")

    # ── Summary ──────────────────────────────────────────────────────────────
    out_duration = len(out_wav) / out_sr
    print(
        f"\n{'─' * 54}\n"
        f"  VoiceTransfer complete\n"
        f"  Input   : {content_duration_sec:.4f}s  "
        f"({len(content_wav)} samples @ {native_sr} Hz)\n"
        f"  Output  : {out_duration:.4f}s  "
        f"({len(out_wav)} samples @ {out_sr} Hz)\n"
        f"  Drift   : {pre_drift_ms:.1f} ms corrected\n"
        f"  Saved   : {out_path.resolve()}\n"
        f"{'─' * 54}"
    )
