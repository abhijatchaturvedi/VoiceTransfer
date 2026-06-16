"""kNN-VC backend: WavLM-Large encoder + prematched HiFiGAN vocoder.

Loaded via torch.hub from bshall/knn-vc. The hub model exposes:
  - get_features(path_or_tensor, weights=None)  -> Tensor (seq_len, dim)
  - get_matching_set([path_or_tensor, ...], weights=None) -> Tensor (total_seq_len, dim)
  - match(query_seq, matching_set, topk=4, tgt_loudness_db=None) -> Tensor (T,)
All audio must be 16 kHz mono float32.

Note on wavlm_layer: the weights kwarg to get_features/get_matching_set requires the
model to return all-layer hidden states (num_layers, seq_len, dim). When tensors are
passed directly this path is not available, so we use the model's default weighting
(equivalent to the layer selected during training). wavlm_layer in config.yaml is
reserved for a future path-based or patched call.
"""

from __future__ import annotations

import logging
from typing import List

import torch

from voicetransfer.converters.base import BaseConverter

logger = logging.getLogger(__name__)

_MODEL_SR = 16_000   # WavLM-Large / HiFiGAN operate at 16 kHz


class KnnVcConverter(BaseConverter):
    """kNN voice conversion: zero-shot, training-free, frame-synchronous."""

    def __init__(self, model, topk: int = 4, wavlm_layer: int = 6) -> None:
        self._model = model
        self._topk = topk
        if wavlm_layer != 6:
            logger.warning(
                "wavlm_layer=%d requested but layer selection via tensor input is not "
                "supported by this hub version; using model default weighting.",
                wavlm_layer,
            )

    @property
    def input_sample_rate(self) -> int:
        return _MODEL_SR

    def convert(
        self,
        content_wav: torch.Tensor,
        target_refs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Run kNN-VC voice conversion.

        Args:
            content_wav: Float32 mono tensor at 16 kHz. Shape (T,).
            target_refs: List of float32 mono tensors at 16 kHz (target speaker).

        Returns:
            Float32 mono waveform at 16 kHz. Shape (T',) where T' ~= T.
        """
        logger.info("Extracting source WavLM features...")
        query_seq = self._model.get_features(content_wav)

        logger.info(
            "Building target matching set from %d reference clip(s)...", len(target_refs)
        )
        matching_set = self._model.get_matching_set(target_refs)

        logger.info(
            "kNN match: query=%s  matching_set=%s  topk=%d",
            tuple(query_seq.shape),
            tuple(matching_set.shape),
            self._topk,
        )
        # tgt_loudness_db=None: skip built-in loudness normalization;
        # the pipeline applies pyloudnorm independently so it is not applied twice.
        out_wav = self._model.match(
            query_seq,
            matching_set,
            topk=self._topk,
            tgt_loudness_db=None,
        )
        return out_wav
