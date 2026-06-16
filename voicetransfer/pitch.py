"""Pitch analysis and shifting for voice conversion pre-processing.

kNN-VC swaps timbre but never touches fundamental frequency (F0).
If content and target speakers have very different F0 ranges (e.g. male
vs. female), the converted voice sounds unnatural.  This module estimates
each speaker's median F0 and shifts the content audio to match before
passing it to the kNN encoder.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def estimate_median_f0(wav: np.ndarray, sr: int) -> float:
    """Return median voiced F0 in Hz via librosa pyin; 0.0 if undetectable."""
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        wav.astype(np.float32),
        fmin=float(librosa.note_to_hz("C2")),   # ~65 Hz  — below bass
        fmax=float(librosa.note_to_hz("C7")),   # ~2093 Hz — above soprano
        sr=sr,
    )
    if f0 is None or voiced_flag is None:
        return 0.0
    voiced = f0[voiced_flag & ~np.isnan(f0)]
    return float(np.median(voiced)) if len(voiced) > 0 else 0.0


def compute_shift(content_f0: float, target_f0: float, max_semitones: float = 12.0) -> float:
    """Semitones to shift content so its median F0 aligns with target F0."""
    if content_f0 <= 0 or target_f0 <= 0:
        logger.warning(
            "F0 undetectable (content=%.1f Hz, target=%.1f Hz) — skipping pitch shift.",
            content_f0, target_f0,
        )
        return 0.0
    raw = 12.0 * float(np.log2(target_f0 / content_f0))
    if abs(raw) > max_semitones:
        logger.warning("Computed pitch shift %.1f st capped at ±%.1f st.", raw, max_semitones)
    return float(np.clip(raw, -max_semitones, max_semitones))


def apply_shift(wav: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift wav by n_steps semitones via librosa phase vocoder.

    Skips processing if |n_steps| < 0.5 (inaudible).
    """
    import librosa

    if abs(n_steps) < 0.5:
        return wav
    logger.info("Applying pitch shift: %.2f semitones @ %d Hz", n_steps, sr)
    return librosa.effects.pitch_shift(wav.astype(np.float32), sr=sr, n_steps=n_steps)
