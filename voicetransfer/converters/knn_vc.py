"""kNN-VC backend: WavLM-Large encoder + prematched HiFiGAN vocoder.

Loaded via torch.hub from bshall/knn-vc. The hub model exposes:
  - get_features(path_or_tensor, weights=None)  → Tensor (seq_len, dim)
  - get_matching_set([path_or_tensor, ...], weights=None) → Tensor (total_seq_len, dim)
  - match(query_seq, matching_set, topk=4, tgt_loudness_db=None) → Tensor (T,)
All audio must be 16 kHz mono float32.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from voicetransfer.converters.base import BaseConverter

logger = logging.getLogger(__name__)

_MODEL_SR = 16_000       # WavLM-Large / HiFiGAN operate at 16 kHz
_WAVLM_LAYERS = 25       # 24 transformer layers + 1 CNN feature extractor (layer 0)


class KnnVcConverter(BaseConverter):
    """kNN voice conversion: zero-shot, training-free, frame-synchronous."""

    def __init__(self, model, topk: int = 4, wavlm_layer: int = 6) -> None:
        self._model = model
        self._topk = topk
        self._weights: Optional[torch.Tensor] = self._make_layer_weights(wavlm_layer)

    @staticmethod
    def _make_layer_weights(layer_idx: int) -> Optional[torch.Tensor]:
        """One-hot weight tensor selecting a single WavLM transformer layer.

        WavLM-Large has 24 transformer layers (1-24 in 1-indexed terms).
        Index 0 in the weights tensor corresponds to the CNN feature extractor output.
        We offset by 1 so that wavlm_layer=6 selects transformer layer 6.
        """
        if not (0 <= layer_idx < _WAVLM_LAYERS - 1):
            logger.warning(
                "wavlm_layer=%d is outside [0, 23]; falling back to default weighting.",
                layer_idx,
            )
            return None
        weights = torch.zeros(_WAVLM_LAYERS)
        weights[layer_idx + 1] = 1.0   # +1 to skip the CNN extractor slot
        return weights

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
            Float32 mono waveform at 16 kHz. Shape (T',) where T' ≈ T.
        """
        logger.info("Extracting source WavLM features...")
        try:
            query_seq = self._model.get_features(content_wav, weights=self._weights)
        except TypeError:
            # Older hub versions may not accept the weights kwarg.
            logger.debug("get_features() rejected 'weights' kwarg; retrying without it.")
            query_seq = self._model.get_features(content_wav)

        logger.info(
            "Building target matching set from %d reference clip(s)...", len(target_refs)
        )
        try:
            matching_set = self._model.get_matching_set(target_refs, weights=self._weights)
        except TypeError:
            logger.debug("get_matching_set() rejected 'weights' kwarg; retrying without it.")
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
