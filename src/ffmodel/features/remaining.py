"""Rest-of-season SOS overlay. Does not retrain.

ros_fp = model_fp
        × (games left / scheduled)
        × (1 + 0.08 × SOS_remaining) / (1 + 0.08 × SOS_full)
        × indoor remaining/full ratio for WR/QB

Opponent defense is last season's EPA allowed. After ~6 weeks, blend in
current-year YTD if that season is in the defense table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, LAST_COMPLETED_SEASON, PBP_COLUMNS, PREDICT_SEASON, PROCESSED_DIR
from ..ingest.nflverse import load_pbp, load_rosters, load_schedules, load_weekly_rosters
from .context import build_schedule_features
from .pbp import _offensive_plays, aggregate_defense_pbp

# Same coefficient as situation_proxy.
SOS_BLEND = 0.08
INDOOR_BASE = 0.97
INDOOR_SPAN = 0.06
YTD_BLEND_AFTER_GAMES = 96  # ~6 weeks × 16 games


def _build_defense_season(season: int) -> pd.DataFrame:
    raw = load_pbp(season, columns=PBP_COLUMNS)
    plays = _offensive_plays(raw)
    out = aggregate_defense_pbp(plays)
    out["season"] = season
    keep = [c for c in ["season", "team", "def_rush_epa", "def_pass_epa", "def_rec_epa"] if c in out.columns]
    return out[keep]


def load_defense_epa(seasons: list[int] | None = None) -> pd.DataFrame:
    """Tiny team-defense table. Built from PBP once, then reused from CSV."""
    path = EXTERNAL_DIR / "defense_epa.csv"
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    want = seasons or [LAST_COMPLETED_SEASON]
    have = pd.DataFrame()
    if path.exists() and path.stat().st_size > 0:
        have = pd.read_csv(path)
    have_seasons = set(int(s) for s in have["season"].unique()) if not have.empty else set()
    missing = [s for s in want if s not in have_seasons]
    frames = [have] if not have.empty else []
    for season in missing:
        try:
            print(f"  Building defense EPA {season} from PBP...")
            frames.append(_build_defense_season(season))
        except Exception as exc:
            print(f"  Defense EPA {season} unavailable ({exc})")
    if not frames:
        return pd.DataFrame(columns=["season", "team", "def_rush_epa", "def_pass_epa", "def_rec_epa"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "team"], keep="last")
    out.to_csv(path, index=False)
    proc = PROCESSED_DIR / "defense_epa.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(proc, index=False)
    return out


def _blend_defense(defense: pd.DataFrame, season: int, completed_games: int) -> pd.DataFrame:
    prior = defense.loc[defense["season"] == season - 1].copy()
    ytd = defense.loc[defense["season"] == season].copy()
    if prior.empty:
        return ytd if not ytd.empty else defense
    if ytd.empty or completed_games < YTD_BLEND_AFTER_GAMES:
        return prior
    w = min(0.5, 0.35 + 0.02 * max(completed_games - YTD_BLEND_AFTER_GAMES, 0) / 16)
    both = prior.merge(ytd, on="team", how="inner", suffixes=("_p", "_y"))
    out = both[["team"]].copy()
    for col in ["def_rush_epa", "def_pass_epa", "def_rec_epa"]:
        out[col] = (1 - w) * both[f"{col}_p"] + w * both[f"{col}_y"]
    out["season"] = season - 1
    return out


def _current_teams(season: int) -> pd.DataFrame:
    weekly = load_weekly_rosters([season])
    src = weekly if not weekly.empty else load_rosters([season])
    if src.empty or "gsis_id" not in src.columns:
        return pd.DataFrame(columns=["player_id", "team"])
    if "week" in src.columns:
        src = src.loc[src["week"] == src["week"].max()]
    out = src.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id", keep="last")
    return pd.DataFrame({"player_id": out["gsis_id"].astype(str).str.strip(), "live_team": out["team"]})


def _pos_sos(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    rush = pd.to_numeric(df.get(f"{prefix}sos_def_rush_epa"), errors="coerce").fillna(0)
    pas = pd.to_numeric(df.get(f"{prefix}sos_def_pass_epa"), errors="coerce").fillna(0)
    rec = pd.to_numeric(df.get(f"{prefix}sos_def_rec_epa"), errors="coerce").fillna(0)
    pos = df["position"]
    sos = -rec
    sos = np.where(pos.eq("QB"), -pas, sos)
    sos = np.where(pos.eq("RB"), -rush, sos)
    return pd.Series(sos, index=df.index)


def attach_remaining_sos(df: pd.DataFrame, season: int = PREDICT_SEASON) -> pd.DataFrame:
    """Attach ros_fp. Steal flags stay frozen. Preseason factor is ~1."""
    out = df.copy()
    try:
        schedules = load_schedules()
        completed = int(
            ((schedules["season"] == season) & (schedules["game_type"] == "REG") & schedules["home_score"].notna()).sum()
        )
        defense = load_defense_epa([season - 1] + ([season] if completed else []))
        defense = _blend_defense(defense, season, completed)
        if defense.empty:
            out["ros_fp"] = pd.to_numeric(out["model_fp"], errors="coerce")
            return out
        full = build_schedule_features(schedules, defense, season, remaining_only=False)
        rem = build_schedule_features(schedules, defense, season, remaining_only=True)
    except Exception as exc:
        print(f"  Remaining SOS skipped ({exc})")
        out["ros_fp"] = pd.to_numeric(out["model_fp"], errors="coerce")
        return out

    teams = _current_teams(season)
    if not teams.empty and "player_id" in out.columns:
        out["player_id"] = out["player_id"].astype(str).str.strip()
        out = out.merge(teams, on="player_id", how="left")
        out["team"] = out["live_team"].fillna(out["team"])
        out = out.drop(columns=["live_team"])

    full = full.add_prefix("full_").rename(columns={"full_team": "team"})
    rem = rem.add_prefix("rem_").rename(columns={"rem_team": "team"})
    out = out.merge(full, on="team", how="left").merge(rem, on="team", how="left")

    sos_full = _pos_sos(out, "full_")
    sos_rem = _pos_sos(out, "rem_")
    games_left = pd.to_numeric(out.get("rem_games_left"), errors="coerce")
    games_sched = pd.to_numeric(out.get("full_games_sched"), errors="coerce").replace(0, np.nan)
    games_factor = (games_left / games_sched).fillna(1.0).clip(lower=0, upper=1)

    sos_factor = (1 + SOS_BLEND * sos_rem) / (1 + SOS_BLEND * sos_full)
    sos_factor = sos_factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    indoor_full = pd.to_numeric(out.get("full_pct_indoor"), errors="coerce").fillna(0.3)
    indoor_rem = pd.to_numeric(out.get("rem_pct_indoor"), errors="coerce").fillna(indoor_full)
    indoor_factor = (INDOOR_BASE + INDOOR_SPAN * indoor_rem) / (INDOOR_BASE + INDOOR_SPAN * indoor_full)
    indoor_factor = indoor_factor.where(out["position"].isin(["WR", "QB"]), 1.0)

    model_fp = pd.to_numeric(out["model_fp"], errors="coerce")
    out["sos_factor"] = sos_factor * indoor_factor
    out["games_left"] = games_left
    out["ros_fp"] = model_fp * games_factor * out["sos_factor"]
    moved = (out["ros_fp"] - model_fp).abs()
    print(
        f"  Remaining SOS: {completed} games played, "
        f"mean factor {out['sos_factor'].mean():.4f}, "
        f"max |Δpts| {moved.max():.1f}"
    )
    drop = [c for c in out.columns if c.startswith("full_") or c.startswith("rem_")]
    return out.drop(columns=drop, errors="ignore")
