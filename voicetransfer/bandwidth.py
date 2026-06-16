"""High-frequency bandwidth extension for voice conversion output.

kNN-VC / HiFiGAN is hard-limited to 8 kHz of bandwidth (16 kHz sample
rate, Nyquist at 8 kHz).  After resampling the converted output to the
native sample rate the energy above 8 kHz is zero — the audio sounds
muffled and "telephone-quality" compared to the original.

This module blends the high-frequency content extracted from the
*original content audio* back into the converted output.  The high
frequencies come from the content speaker, not the target, but for
speech naturalness (air, sibilance, consonant clarity) this is a
significant perceptual improvement over dead silence above 8 kHz.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def blend_hf(
    converted: np.ndarray,
    content: np.ndarray,
    sr: int,
    cutoff_hz: float = 7500.0,
    gain: float = 0.8,
) -> np.ndarray:
    """Mix high-frequency content from original audio into converted output.

    Parameters
    ----------
    converted:  converted audio array at *sr* (output of pipeline step 5)
    content:    original content audio at the same *sr*
    sr:         sample rate of both arrays
    cutoff_hz:  high-pass cutoff — only content above this frequency is blended
    gain:       mix level for the blended HF (0.0 = off, 1.0 = full level)
    """
    if sr <= 16000:
        # Nothing above the HiFiGAN ceiling to blend; skip silently.
        return converted

    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    norm_cutoff = min(cutoff_hz / nyq, 0.999)
    sos = butter(4, norm_cutoff, btype="high", output="sos")

    min_len = min(len(converted), len(content))
    hf = sosfiltfilt(sos, content[:min_len].astype(np.float64)).astype(np.float32)

    out = converted.copy()
    out[:min_len] += gain * hf
    np.clip(out, -1.0, 1.0, out=out)

    logger.info(
        "Bandwidth extension: blended %.0f+ Hz from content (gain=%.2f, sr=%d Hz)",
        cutoff_hz, gain, sr,
    )
    return out
