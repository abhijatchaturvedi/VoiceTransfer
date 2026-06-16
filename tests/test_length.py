"""Tests for the length-alignment guarantee.

Hard constraint: output audio must contain exactly round(content_duration * out_sr)
samples after the pipeline runs, regardless of what the backend returns.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch
from pathlib import Path
from unittest.mock import MagicMock


# ── Unit tests for _length_align ──────────────────────────────────────────────

from voicetransfer.pipeline import _length_align


def test_length_align_trim():
    """Output longer than expected is trimmed to exactly expected_samples."""
    sr = 22050
    duration = 2.7
    expected = round(duration * sr)

    wav = np.zeros(expected + 500, dtype=np.float32)
    result = _length_align(wav, expected, pad_mode="silence", warn_threshold_ms=50, out_sr=sr)

    assert len(result) == expected, f"Expected {expected}, got {len(result)}"


def test_length_align_pad():
    """Output shorter than expected is padded with silence to exactly expected_samples."""
    sr = 22050
    duration = 2.7
    expected = round(duration * sr)

    wav = np.zeros(expected - 300, dtype=np.float32)
    result = _length_align(wav, expected, pad_mode="silence", warn_threshold_ms=50, out_sr=sr)

    assert len(result) == expected, f"Expected {expected}, got {len(result)}"
    assert np.all(result[expected - 300:] == 0.0), "Padded region must be silence."


def test_length_align_exact_no_change():
    """Output already at exact length is returned unchanged."""
    sr = 16000
    expected = round(1.5 * sr)
    wav = np.ones(expected, dtype=np.float32)
    result = _length_align(wav, expected, pad_mode="silence", warn_threshold_ms=50, out_sr=sr)
    assert len(result) == expected
    assert np.array_equal(result, wav)


def test_length_align_warn_threshold(caplog):
    """A WARNING is emitted when drift exceeds warn_threshold_ms."""
    import logging
    sr = 16000
    duration = 1.0
    expected = round(duration * sr)
    # 200 ms of drift — exceeds the 50 ms threshold
    drift_samples = round(0.200 * sr)
    wav = np.zeros(expected + drift_samples, dtype=np.float32)

    with caplog.at_level(logging.WARNING, logger="voicetransfer.pipeline"):
        _length_align(wav, expected, pad_mode="silence", warn_threshold_ms=50.0, out_sr=sr)

    assert any("drift" in r.message.lower() for r in caplog.records), (
        "Expected a WARNING about drift, but none was logged."
    )


# ── Integration test: full pipeline with a mocked model ───────────────────────

def _make_config(tmp_path: Path, content_path: str, target_path: str, output_path: str):
    """Build an AppConfig pointed at tmp test files without loading config.yaml."""
    from voicetransfer.config import (
        AppConfig, PathsConfig, DeviceConfig, BackendConfig,
        KnnVcConfig, AudioConfig, LengthConfig, MuxConfig, LoggingConfig,
    )
    return AppConfig(
        paths=PathsConfig(
            content_audio=content_path,
            target_refs=[target_path],
            output_audio=output_path,
            models_dir=str(tmp_path / "models"),
        ),
        device=DeviceConfig(type="cpu", num_threads=1),
        backend=BackendConfig(name="knn_vc"),
        knn_vc=KnnVcConfig(prematched=True, topk=4, wavlm_layer=6),
        audio=AudioConfig(output_sample_rate=0, normalize_loudness=False),
        length=LengthConfig(enforce_exact=True, pad_mode="silence", warn_if_drift_ms=50.0),
        mux=MuxConfig(enabled=False),
        logging=LoggingConfig(level="WARNING"),
    )


def test_pipeline_output_length_exact(tmp_path: Path):
    """Pipeline output sample count must equal round(content_duration * out_sr).

    The mock backend returns audio that is intentionally longer than the source
    (simulating real-world kNN-VC drift). The length-alignment step must correct it.
    """
    NATIVE_SR = 22050
    MODEL_SR  = 16000
    DURATION  = 2.7          # seconds — chosen to produce a non-round sample count

    # Create synthetic content: 2.7 s sine wave at 22050 Hz
    content_samples = round(DURATION * NATIVE_SR)
    t = np.linspace(0.0, DURATION, content_samples, dtype=np.float32)
    content_audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)

    content_path = tmp_path / "content.wav"
    sf.write(str(content_path), content_audio, NATIVE_SR)

    # Create synthetic target: 1 s of low-level noise
    target_samples = MODEL_SR  # 1 second
    target_audio = (np.random.default_rng(42).standard_normal(target_samples) * 0.05
                    ).astype(np.float32)
    target_path = tmp_path / "target.wav"
    sf.write(str(target_path), target_audio, NATIVE_SR)

    output_path = tmp_path / "output.wav"

    cfg = _make_config(
        tmp_path,
        str(content_path),
        str(target_path),
        str(output_path),
    )

    # Mock kNN-VC model: return a tensor that is ~200 samples longer than expected
    # at 16 kHz, simulating the real-world drift the backend can introduce.
    model_out_samples = round(DURATION * MODEL_SR) + 200
    mock_out = torch.zeros(model_out_samples)

    mock_model = MagicMock()
    mock_model.get_features.return_value = torch.zeros(50, 1024)
    mock_model.get_matching_set.return_value = torch.zeros(30, 1024)
    mock_model.match.return_value = mock_out

    from voicetransfer.pipeline import run_pipeline
    run_pipeline(cfg, mock_model)

    # Read back the output WAV and verify exact sample count.
    out_data, out_sr = sf.read(str(output_path))
    expected = round(DURATION * out_sr)   # out_sr == NATIVE_SR because output_sample_rate=0

    assert len(out_data) == expected, (
        f"Length guarantee violated: expected {expected} samples, "
        f"got {len(out_data)} (drift = {len(out_data) - expected} samples "
        f"= {(len(out_data) - expected) / out_sr * 1000:.1f} ms)"
    )


def test_pipeline_output_length_short_backend(tmp_path: Path):
    """Length guarantee holds even when the backend returns SHORTER audio than expected."""
    NATIVE_SR = 16000
    MODEL_SR  = 16000
    DURATION  = 1.3

    content_samples = round(DURATION * NATIVE_SR)
    content_audio = np.zeros(content_samples, dtype=np.float32)
    content_path = tmp_path / "content.wav"
    sf.write(str(content_path), content_audio, NATIVE_SR)

    target_path = tmp_path / "target.wav"
    sf.write(str(target_path), np.zeros(MODEL_SR, dtype=np.float32), NATIVE_SR)

    output_path = tmp_path / "output.wav"
    cfg = _make_config(tmp_path, str(content_path), str(target_path), str(output_path))

    # Backend returns 150 fewer samples than expected
    model_out_samples = round(DURATION * MODEL_SR) - 150
    mock_model = MagicMock()
    mock_model.get_features.return_value = torch.zeros(30, 1024)
    mock_model.get_matching_set.return_value = torch.zeros(20, 1024)
    mock_model.match.return_value = torch.zeros(model_out_samples)

    from voicetransfer.pipeline import run_pipeline
    run_pipeline(cfg, mock_model)

    out_data, out_sr = sf.read(str(output_path))
    expected = round(DURATION * out_sr)
    assert len(out_data) == expected, (
        f"Pad path failed: expected {expected}, got {len(out_data)}"
    )
