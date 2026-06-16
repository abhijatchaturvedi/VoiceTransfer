"""Abstract base class that every voice conversion backend must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class BaseConverter(ABC):
    """Interface for voice conversion backends.

    All audio tensors are float32 mono at the backend's input_sample_rate.
    The returned tensor is also float32 mono at input_sample_rate; the
    pipeline resamples it to the final output rate.
    """

    @property
    @abstractmethod
    def input_sample_rate(self) -> int:
        """Sample rate (Hz) that this backend expects for all inputs."""
        ...

    @abstractmethod
    def convert(
        self,
        content_wav: torch.Tensor,
        target_refs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Convert content_wav to the target speaker's voice.

        Args:
            content_wav: Float32 mono waveform at input_sample_rate. Shape (T,).
            target_refs: One or more float32 mono reference clips at input_sample_rate.
                         More clips give the kNN matcher a richer target distribution.

        Returns:
            Float32 mono waveform at input_sample_rate. Shape (T',) where T' ≈ T
            (exact duration alignment is handled by the pipeline, not the backend).
        """
        ...
