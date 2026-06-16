"""Audio I/O, resampling, and loudness normalization helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import torch

logger = logging.getLogger(__name__)


def load_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load an audio file as a float32 mono array.

    Returns (waveform, sample_rate). Multi-channel files are mixed down to mono.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    logger.debug("Loaded %s: %d samples @ %d Hz (%.3f s)", path, len(data), sr, len(data) / sr)
    return data, sr


def resample(wav: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample a float32 mono array from from_sr to to_sr. No-op if rates match."""
    if from_sr == to_sr:
        return wav
    out = librosa.resample(wav, orig_sr=from_sr, target_sr=to_sr)
    logger.debug("Resampled %d Hz → %d Hz (%d → %d samples)", from_sr, to_sr, len(wav), len(out))
    return out


def normalize_loudness(wav: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """Normalize integrated loudness to target_lufs (EBU R128) using pyloudnorm.

    Silent or near-silent audio is returned unchanged to avoid amplifying noise.
    """
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    if np.isinf(loudness) or np.isnan(loudness):
        logger.debug("Skipping loudness normalization: audio is silent.")
        return wav
    normalized = pyln.normalize.loudness(wav, loudness, target_lufs)
    logger.debug("Loudness normalized: %.1f LUFS → %.1f LUFS", loudness, target_lufs)
    return normalized.astype(np.float32)


def save_audio(wav: np.ndarray, sr: int, path: str | Path) -> None:
    """Write a float32 mono array to a WAV file, creating parent dirs as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav.astype(np.float32), sr)
    logger.debug("Saved %s (%d samples @ %d Hz)", out, len(wav), sr)


def numpy_to_tensor(wav: np.ndarray) -> torch.Tensor:
    """Convert float32 numpy array (T,) to a CPU float32 torch tensor."""
    return torch.from_numpy(wav.astype(np.float32))


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a float32 numpy array."""
    return t.detach().cpu().numpy().astype(np.float32)
