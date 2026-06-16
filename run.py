#!/usr/bin/env python3
"""VoiceTransfer CLI entry point.

Usage
-----
    python run.py --config config.yaml
    python run.py --content speech.wav --target voice.wav --output out.wav
    python run.py --config config.yaml --content speech.wav  # override one field
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voicetransfer",
        description="CPU-only zero-shot voice conversion via kNN-VC.",
    )
    p.add_argument(
        "--config", default="config.yaml", metavar="PATH",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    p.add_argument(
        "--content", metavar="PATH",
        help="Override paths.content_audio",
    )
    p.add_argument(
        "--target", nargs="+", metavar="PATH",
        help="Override paths.target_refs (one or more WAV files)",
    )
    p.add_argument(
        "--output", metavar="PATH",
        help="Override paths.output_audio",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    from voicetransfer.config import load_config, validate_for_conversion

    cfg = load_config(args.config)

    # Apply CLI overrides before validation so error messages use the final paths.
    if args.content:
        cfg.paths.content_audio = args.content
    if args.target:
        cfg.paths.target_refs = args.target
    if args.output:
        cfg.paths.output_audio = args.output

    # Configure logging early so download progress is visible.
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    validate_for_conversion(cfg)

    from voicetransfer.download import ensure_models
    from voicetransfer.pipeline import run_pipeline

    model = ensure_models(cfg)
    run_pipeline(cfg, model)


if __name__ == "__main__":
    main()
