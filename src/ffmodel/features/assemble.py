from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    FIRST_SEASON,
    GAMES_PER_SEASON,
    LAST_COMPLETED_SEASON,
    PREDICT_SEASON,
    PROCESSED_DIR,
    SKILL_POSITIONS,
)
from ..ingest.markets import load_adp_history, load_season_props
from ..ingest.nflverse import (
    load_combine,
    load_depth_charts,
    load_draft_picks,
    load_injuries,
    load_player_stats,
    load_players,
    load_rosters,
    load_schedules,
    load_snap_counts,
)
from ..names import canon_team, normalize_name
from .edges import (
    attach_new_oc,
    attach_qb_change,
    build_h2_features,
    build_ngs_lag,
    build_team_qb,
    depth_snapshot,
    load_ecr,
    load_oc_table,
    qb_from_depth,
)
from .context import (
    adp_implied_points,
    age_alpha,
    age_on_sept1,
    blend_projection,
    build_injury_features,
    build_physical_features,
    build_schedule_features,
    build_snap_features,
    enrich_derived_features,
    estimate_age_lambdas,
    oline_index,
    physical_proxy,
    production_proxy,
    props_to_vfp,
    team_coaches,
    tenure_bucket,
    vacated_opportunity,
    situation_proxy,
)
from .pbp import build_pbp_tables

STAT_KEEP = [
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "recent_team",
    "season",
    "games",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "passing_epa",
    "passing_cpoe",
    "pacr",
    "passing_20",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_epa",
    "rushing_first_downs",
    "rushing_20",
    "rushing_40",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_epa",
    "receiving_first_downs",
    "receiving_20",
    "receiving_40",
    "racr",
    "target_share",
    "air_yards_share",
    "wopr",
    "fantasy_points",
    "fantasy_points_ppr",
]


def _skill_stats(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    df["position"] = df["position"].replace({"FB": "RB", "HB": "RB"})
    df = df.loc[df["position"].isin(SKILL_POSITIONS)].copy()
    keep = [c for c in STAT_KEEP if c in df.columns]
    df = df[keep]
    df["ppr"] = df["fantasy_points_ppr"]
    df["ppr_ppg"] = np.where(df["games"] > 0, df["ppr"] / df["games"], np.nan)
    df["name_norm"] = df["player_display_name"].fillna(df["player_name"]).map(normalize_name)
    return df


def build_panel(predict_season: int = PREDICT_SEASON, use_pbp: bool = True) -> pd.DataFrame:
    seasons = list(range(FIRST_SEASON, predict_season + 1))
    lag_seasons = list(range(FIRST_SEASON, predict_season))
    print("Loading nflverse core tables...")
    stats = _skill_stats(load_player_stats(lag_seasons))
    rosters = load_rosters(seasons)
    players = load_players()
    combine = load_combine()
    draft = load_draft_picks()
    injuries = load_injuries(lag_seasons)
    snaps = load_snap_counts(lag_seasons)
    schedules = load_schedules()
    print("Loading markets (ADP + props)...")
    adp = load_adp_history(list(range(FIRST_SEASON, predict_season + 1)))
    props = load_season_props()
    vfp = props_to_vfp(props)

    pbp_player = pbp_team = pbp_def = pd.DataFrame()
    if use_pbp:
        print("Aggregating play-by-play edges...")
        pbp_player, pbp_team, pbp_def = build_pbp_tables(lag_seasons)
        pbp_team = oline_index(pbp_team)
    else:
        print("Skipping PBP (fast mode).")

    print("Loading H2 / NGS / ECR / OC / primary QB...")
    h2 = build_h2_features(lag_seasons)
    ngs = build_ngs_lag()
    ecr = load_ecr()
    oc = load_oc_table()
    qb = build_team_qb(seasons)
    try:
        qb_cur = qb_from_depth(load_depth_charts(predict_season), predict_season)
        if not qb_cur.empty:
            qb = pd.concat(
                [qb.loc[pd.to_numeric(qb["season"], errors="coerce").ne(predict_season)], qb_cur],
                ignore_index=True,
            )
            print(f"  {predict_season} starting QBs from depth: {len(qb_cur)}")
    except Exception as exc:
        print(f"  Current-season QB from depth skipped ({exc})")

    print("Building context features...")
    inj = build_injury_features(injuries)
    snap_f = build_snap_features(snaps, rosters)
    phys = build_physical_features(combine, players, draft)
    coaches = team_coaches(schedules)
    market_fp, _curves = adp_implied_points(adp, stats) if not adp.empty else (pd.DataFrame(), {})

    # Injury trailing 3-year rollups.
    if not inj.empty:
        inj = inj.sort_values(["player_id", "season"])
        inj["major_injury_3yr"] = (
            inj.groupby("player_id")["major_injury"].rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
        )
        inj["injury_weeks_3yr"] = (
            inj.groupby("player_id")["injury_weeks"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
        )

    frames = []
    for season in range(FIRST_SEASON + 1, predict_season + 1):
        print(f"  panel season {season}")
        rost = rosters.loc[rosters["season"] == season].copy()
        rost["position"] = rost["position"].replace({"FB": "RB", "HB": "RB"})
        rost = rost.loc[rost["position"].isin(SKILL_POSITIONS)]
        base = rost.rename(columns={"gsis_id": "player_id", "full_name": "player_name", "team": "team"})[
            ["player_id", "player_name", "position", "team", "season", "years_exp", "birth_date", "height", "weight", "pfr_id"]
        ].drop_duplicates("player_id")
        lag = stats.loc[stats["season"] == season - 1].copy()
        rename_lag = {c: f"{c}_lag" for c in lag.columns if c not in {"player_id", "name_norm"}}
        lag = lag.rename(columns=rename_lag)
        # keep a clean ppr_lag
        if "fantasy_points_ppr_lag" in lag.columns:
            lag["ppr_lag"] = lag["fantasy_points_ppr_lag"]
            lag["ppr_ppg_lag"] = np.where(lag["games_lag"] > 0, lag["ppr_lag"] / lag["games_lag"], np.nan)
        panel = base.merge(lag, on="player_id", how="left")
        panel["name_norm"] = panel["player_name"].map(normalize_name)
        if "name_norm_lag" in panel.columns:
            panel["name_norm"] = panel["name_norm"].fillna(panel["name_norm_lag"])

        if not pbp_player.empty:
            pp = pbp_player.loc[pbp_player["season"] == season - 1].drop(columns=["season"])
            panel = panel.merge(pp, on="player_id", how="left")
        if not pbp_team.empty:
            tm = pbp_team.loc[pbp_team["season"] == season - 1].copy()
            # Situation uses the CURRENT team's prior-year line/scheme, not the player's old team.
            panel = panel.merge(tm.add_suffix("_tm").rename(columns={"team_tm": "team", "season_tm": "tm_season"}), on="team", how="left")
            panel["oline_index"] = panel.get("oline_index_tm")
            panel["coaching_C"] = panel.get("coaching_C_tm")
            panel["proe"] = panel.get("proe_tm")
            panel["pace_neutral"] = panel.get("pace_neutral_tm")
            panel["sack_rate"] = panel.get("sack_rate_tm")
            panel["pass_rate"] = panel.get("pass_rate_tm")

        coach_now = coaches.loc[coaches["season"] == season, ["team", "coach"]]
        coach_prev = coaches.loc[coaches["season"] == season - 1, ["team", "coach"]].rename(columns={"coach": "coach_prev"})
        panel = panel.merge(coach_now, on="team", how="left").merge(coach_prev, on="team", how="left")
        panel["new_hc"] = (panel["coach"] != panel["coach_prev"]).astype(float)

        vac = vacated_opportunity(stats, rosters, season)
        panel = panel.merge(vac.drop(columns=["season"]), on="team", how="left")

        if not pbp_def.empty:
            sched_f = build_schedule_features(schedules, pbp_def, season)
            panel = panel.merge(sched_f.drop(columns=["season"]), on="team", how="left")

        inj_lag = inj.loc[inj["season"] == season - 1].drop(columns=["season"]) if not inj.empty else pd.DataFrame()
        if not inj_lag.empty:
            panel = panel.merge(inj_lag, on="player_id", how="left")
        snap_lag = snap_f.loc[snap_f["season"] == season - 1] if not snap_f.empty else pd.DataFrame()
        if not snap_lag.empty and "player_id" in snap_lag.columns:
            cols = [c for c in ["player_id", "off_snap_pct", "off_snaps"] if c in snap_lag.columns]
            panel = panel.merge(snap_lag[cols], on="player_id", how="left")

        panel = panel.merge(phys, on="player_id", how="left", suffixes=("", "_phys"))
        try:
            depth = depth_snapshot(
                load_depth_charts(season), season, current=(season >= predict_season)
            )
            if not depth.empty:
                panel["player_id"] = panel["player_id"].astype(str)
                panel = panel.merge(depth, on="player_id", how="left")
            else:
                panel["depth_rank"] = np.nan
        except Exception:
            panel["depth_rank"] = np.nan

        mkt = market_fp.loc[market_fp["season"] == season] if not market_fp.empty else pd.DataFrame()
        if not mkt.empty:
            panel = panel.merge(
                mkt[["adp_name_norm", "position", "adp", "market_fp"]].rename(columns={"adp_name_norm": "name_norm"}),
                on=["name_norm", "position"],
                how="left",
            )

        if season == PREDICT_SEASON and not vfp.empty:
            panel = panel.merge(vfp, on="name_norm", how="left")
        else:
            panel["vfp"] = np.nan
            panel["vfp_markets"] = 0

        realized = stats.loc[stats["season"] == season, ["player_id", "ppr", "games", "ppr_ppg"]].rename(
            columns={"ppr": "ppr_actual", "games": "games_actual", "ppr_ppg": "ppg_actual"}
        ) if season <= LAST_COMPLETED_SEASON else pd.DataFrame()
        if not realized.empty:
            panel = panel.merge(realized, on="player_id", how="left")
        else:
            panel["ppr_actual"] = np.nan
            panel["games_actual"] = np.nan
            panel["ppg_actual"] = np.nan

        panel["age"] = age_on_sept1(panel["birth_date"], panel["season"])
        panel["years_exp"] = pd.to_numeric(panel["years_exp"], errors="coerce").fillna(0)
        panel["tenure_bucket"] = panel["years_exp"].map(tenure_bucket)
        frames.append(panel.drop_duplicates("player_id"))

    panel = pd.concat(frames, ignore_index=True)
    panel["player_id"] = panel["player_id"].astype(str)
    panel["team"] = panel["team"].map(canon_team)
    if not h2.empty:
        h2 = h2.copy()
        h2["player_id"] = h2["player_id"].astype(str)
        panel = panel.merge(h2, on=["player_id", "season"], how="left")
    if not ngs.empty:
        ngs = ngs.copy()
        ngs["player_id"] = ngs["player_id"].astype(str)
        panel = panel.merge(ngs, on=["player_id", "season"], how="left")
    if not ecr.empty:
        panel = panel.merge(ecr, on=["season", "name_norm", "position"], how="left")
    if "ecr" not in panel.columns:
        panel["ecr"] = np.nan
    panel["ecr_minus_adp"] = pd.to_numeric(panel["ecr"], errors="coerce") - pd.to_numeric(
        panel["adp"], errors="coerce"
    )
    panel = attach_qb_change(panel, qb)
    panel = attach_new_oc(panel, oc)
    filled = int(panel["depth_rank"].notna().sum()) if "depth_rank" in panel.columns else 0
    print(f"  depth_rank filled {filled}/{len(panel)} ({filled / max(len(panel), 1):.0%})")
    lambdas = estimate_age_lambdas(panel.loc[panel["ppr_actual"].notna()].assign(ppr=lambda d: d["ppr_actual"]))
    panel["age_alpha"] = age_alpha(panel["age"], panel["position"], lambdas)
    panel["production_proxy"] = panel.apply(production_proxy, axis=1)
    panel["situation_proxy"] = panel.apply(situation_proxy, axis=1)
    panel["physical_proxy"] = panel.apply(physical_proxy, axis=1)
    panel["blend_proj"] = panel.apply(blend_projection, axis=1)
    panel["lambda_used"] = panel["position"].map(lambdas)
    panel = apply_vegas_overlay(panel)
    panel = enrich_derived_features(panel)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED_DIR / "player_panel.parquet", index=False)
    print(f"Panel rows={len(panel):,} seasons={sorted(panel['season'].unique())}")
    return panel


def apply_vegas_overlay(panel: pd.DataFrame) -> pd.DataFrame:
    """Refresh 2026 VFP from the latest season-long prop board without rebuilding PBP."""
    props = load_season_props()
    vfp = props_to_vfp(props)
    if vfp.empty:
        return panel
    extra_cols = [c for c in vfp.columns if c != "name_norm"]
    out = panel.copy()
    for col in extra_cols:
        if col not in out.columns:
            out[col] = np.nan
    idx = out["season"] == PREDICT_SEASON
    lookup = vfp.drop_duplicates("name_norm").set_index("name_norm")
    keys = out.loc[idx, "name_norm"]
    for col in extra_cols:
        out.loc[idx, col] = keys.map(lookup[col]) if col in lookup.columns else np.nan
    out.loc[idx, "situation_proxy"] = out.loc[idx].apply(situation_proxy, axis=1)
    out.loc[idx, "blend_proj"] = out.loc[idx].apply(blend_projection, axis=1)
    matched = int(out.loc[idx, "vfp"].notna().sum())
    print(f"  VFP overlay: {matched} / {int(idx.sum())} {PREDICT_SEASON} skill players")
    return out
