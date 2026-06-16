#!/usr/bin/env python3
"""Pre-fetch all model weights without running a conversion.

Usage
-----
    python download_models.py
    python download_models.py path/to/config.yaml
"""

from __future__ import annotations

import logging
import sys


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    from voicetransfer.config import load_config
    from voicetransfer.download import ensure_models

    # validate_inputs=False: input audio files need not exist to warm the cache.
    cfg = load_config(config_path)

    logger = logging.getLogger(__name__)
    logger.info("Pre-fetching model weights to: %s", cfg.paths.models_dir)

    ensure_models(cfg)

    print(f"\nModel weights are ready in: {cfg.paths.models_dir}")


if __name__ == "__main__":
    main()
