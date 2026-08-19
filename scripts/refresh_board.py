#!/usr/bin/env python3
"""Refresh the hosted board overlay (live team, injuries, remaining SOS).

Same board — names move in place. Does not retrain.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffmodel.config import PREDICT_SEASON  # noqa: E402
from ffmodel.publish import rebuild_embed_board  # noqa: E402


def main() -> None:
    path = rebuild_embed_board(PREDICT_SEASON)
    print(f"Board: {path}")


if __name__ == "__main__":
    main()
