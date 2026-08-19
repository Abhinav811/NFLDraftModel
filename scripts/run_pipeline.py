#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffmodel.config import PREDICT_SEASON, PROCESSED_DIR  # noqa: E402
from ffmodel.correlations import edge_correlation_table, edge_cross_correlation  # noqa: E402
from ffmodel.features.assemble import apply_vegas_overlay, build_panel  # noqa: E402
from ffmodel.features.context import enrich_derived_features  # noqa: E402
from ffmodel.model import predict_season, run_backtest  # noqa: E402
from ffmodel.publish import write_article  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fantasy ranking model end to end.")
    parser.add_argument("--season", type=int, default=PREDICT_SEASON)
    parser.add_argument("--skip-pbp", action="store_true", help="Skip play-by-play (faster, weaker RZ/PROE/SOS).")
    parser.add_argument("--reuse-panel", action="store_true", help="Reuse data/processed/player_panel.parquet.")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = PROCESSED_DIR / "player_panel.parquet"
    if args.reuse_panel and panel_path.exists():
        import pandas as pd

        print(f"Reusing {panel_path}")
        panel = pd.read_parquet(panel_path)
    else:
        panel = build_panel(predict_season=args.season, use_pbp=not args.skip_pbp)

    panel = apply_vegas_overlay(panel)
    print("Enriching derived edges (player vacated, luck, breakouts)...")
    panel = enrich_derived_features(panel)
    panel.to_parquet(PROCESSED_DIR / "player_panel.parquet", index=False)

    print("Backtesting...")
    bt = run_backtest(panel)
    print(bt.by_season.to_string(index=False) if not bt.by_season.empty else "no backtest rows")
    print("Steal eval:", bt.steal_eval)

    print(f"Predicting {args.season}...")
    rankings = predict_season(panel, season=args.season)
    print("Edge correlations (extras vs test actuals and vs 2026 projection)...")
    edges = edge_correlation_table(panel, rankings)
    edge_cross_correlation(panel, rankings)
    print(edges.sort_values("corr_test_actual", key=lambda s: s.abs(), ascending=False).head(12).to_string(index=False))
    path = write_article(rankings, bt.by_season, bt.extra_corrs, bt.steal_eval, season=args.season, panel=panel)
    print(f"Article: {path}")
    print(f"Rankings: {PROCESSED_DIR / f'rankings_{args.season}.csv'}")
    print(rankings.groupby("position").head(3)[["position", "model_rank_pos", "player_name", "adp", "model_fp", "steal_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
