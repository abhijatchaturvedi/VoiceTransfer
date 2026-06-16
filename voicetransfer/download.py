"""Model weight management: check cache, download via torch.hub if needed."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicetransfer.config import AppConfig

logger = logging.getLogger(__name__)

# WavLM-Large is the largest / slowest checkpoint (~600 MB).
# If it exists, we assume the full download completed on a prior run.
_WAVLM_SENTINEL = "hub/checkpoints/wavlm_large_finetune.pt"


def _wavlm_cached(models_dir: Path) -> bool:
    return (models_dir / _WAVLM_SENTINEL).exists()


def ensure_models(config: "AppConfig"):
    """Return a loaded kNN-VC model, downloading weights to models_dir if absent.

    Sets TORCH_HOME and HF_HOME to models_dir BEFORE any hub call so weights
    land in the configured cache, not the system default.
    """
    import torch

    models_dir = Path(config.paths.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    # Redirect ALL hub/HF downloads to the configured cache directory.
    os.environ["TORCH_HOME"] = str(models_dir)
    os.environ["HF_HOME"] = str(models_dir)
    logger.debug("TORCH_HOME set to %s", models_dir)

    torch.set_num_threads(config.device.num_threads)
    logger.debug("torch num_threads = %d", config.device.num_threads)

    if _wavlm_cached(models_dir):
        logger.info(
            "kNN-VC weights found in %s — skipping download.", models_dir / "hub/checkpoints"
        )
    else:
        logger.info(
            "kNN-VC weights not found in %s.", models_dir / "hub/checkpoints"
        )
        logger.info(
            "Downloading WavLM-Large + %s HiFiGAN via torch.hub "
            "(one-time, ~600–700 MB)...",
            "prematched" if config.knn_vc.prematched else "standard",
        )

    knn_vc = torch.hub.load(
        "bshall/knn-vc",
        "knn_vc",
        prematched=config.knn_vc.prematched,
        trust_repo=True,
        pretrained=True,
        device=config.device.type,
    )

    if _wavlm_cached(models_dir):
        logger.info("Model ready (weights at %s).", models_dir / "hub/checkpoints")
    else:
        logger.info("Download complete. Weights cached at %s.", models_dir / "hub/checkpoints")

    return knn_vc
