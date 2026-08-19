from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PBP_COLUMNS, RZ_WEIGHTS
from ..ingest.nflverse import load_pbp


def _offensive_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    if "penalty" in df.columns:
        df = df.loc[df["penalty"].fillna(0).eq(0)]
    if "two_point_attempt" in df.columns:
        df = df.loc[df["two_point_attempt"].fillna(0).eq(0)]
    if "qb_kneel" in df.columns:
        df = df.loc[df["qb_kneel"].fillna(0).eq(0)]
    if "qb_spike" in df.columns:
        df = df.loc[df["qb_spike"].fillna(0).eq(0)]
    play_ok = df["play_type"].isin(["pass", "run"])
    return df.loc[play_ok].copy()


def _player_from_role(plays: pd.DataFrame, id_col: str, kind: str) -> pd.DataFrame:
    part = plays.dropna(subset=[id_col]).copy()
    part["player_id"] = part[id_col]
    part["role"] = kind
    return part


def aggregate_player_pbp(plays: pd.DataFrame) -> pd.DataFrame:
    rush = _player_from_role(plays.loc[plays["rush_attempt"].fillna(0).eq(1)], "rusher_player_id", "rush")
    rec = _player_from_role(plays.loc[plays["receiver_player_id"].notna()], "receiver_player_id", "rec")
    pas = _player_from_role(plays.loc[plays["pass_attempt"].fillna(0).eq(1)], "passer_player_id", "pass")

    rush["exp_rush"] = rush["rushing_yards"].fillna(0).ge(15).astype(int)
    rush["inside5"] = rush["yardline_100"].fillna(99).le(5).astype(int)
    rush["rz_carry"] = rush["yardline_100"].fillna(99).le(20).astype(int)
    rush["exp_rush_epa"] = np.where(rush["exp_rush"].eq(1), rush["epa"], 0.0)
    rec["exp_rec"] = rec["receiving_yards"].fillna(0).ge(20).astype(int)
    rec["inside10_tgt"] = rec["yardline_100"].fillna(99).le(10).astype(int)
    rec["ez_tgt"] = (
        rec["air_yards"].fillna(-99) >= (rec["yardline_100"].fillna(99) - 0.5)
    ) & rec["yardline_100"].fillna(99).le(20)
    rec["ez_tgt"] = rec["ez_tgt"].astype(int)
    rec["rz_tgt"] = rec["yardline_100"].fillna(99).le(20).astype(int)
    rec["exp_rec_epa"] = np.where(rec["exp_rec"].eq(1), rec["epa"], 0.0)

    rush_g = rush.groupby(["season", "player_id"], as_index=False).agg(
        pbp_carries=("player_id", "size"),
        exp_rushes=("exp_rush", "sum"),
        inside5_carries=("inside5", "sum"),
        rz_carries=("rz_carry", "sum"),
        rush_epa=("epa", "sum"),
        exp_rush_epa=("exp_rush_epa", "sum"),
    )
    rec_g = rec.groupby(["season", "player_id"], as_index=False).agg(
        pbp_targets=("player_id", "size"),
        exp_receptions=("exp_rec", "sum"),
        inside10_targets=("inside10_tgt", "sum"),
        ez_targets=("ez_tgt", "sum"),
        rz_targets=("rz_tgt", "sum"),
        rec_epa=("epa", "sum"),
        rec_air_yards=("air_yards", "sum"),
        rec_yac=("yards_after_catch", "sum"),
        exp_rec_epa=("exp_rec_epa", "sum"),
    )
    pas_g = pas.groupby(["season", "player_id"], as_index=False).agg(
        pbp_pass_attempts=("player_id", "size"),
        pass_epa=("epa", "sum"),
        pass_air_yards=("air_yards", "sum"),
    )

    out = rush_g.merge(rec_g, on=["season", "player_id"], how="outer").merge(
        pas_g, on=["season", "player_id"], how="outer"
    )
    for col in out.columns:
        if col not in {"season", "player_id"}:
            out[col] = out[col].fillna(0)
    out["touches"] = out["pbp_carries"] + out["pbp_targets"]
    out["explosive_rate"] = np.where(
        out["touches"] > 0,
        (out["exp_rushes"] + out["exp_receptions"]) / out["touches"],
        0.0,
    )
    out["explosive_epa"] = out["exp_rush_epa"] + out["exp_rec_epa"]
    out["chunk_rating"] = out["explosive_rate"] * out["explosive_epa"]
    out["hv_rz"] = (
        RZ_WEIGHTS["inside5_carries"] * out["inside5_carries"]
        + RZ_WEIGHTS["inside10_targets"] * out["inside10_targets"]
        + RZ_WEIGHTS["ez_targets"] * out["ez_targets"]
    )
    out["hv_rz_per_touch"] = np.where(out["touches"] > 0, out["hv_rz"] / out["touches"], 0.0)
    return out


def aggregate_team_pbp(plays: pd.DataFrame) -> pd.DataFrame:
    df = plays.dropna(subset=["posteam"]).copy()
    df["is_dropback"] = df["qb_dropback"].fillna(0).eq(1) | df["pass_attempt"].fillna(0).eq(1)
    df["is_sack"] = df["sack"].fillna(0).eq(1)
    df["is_rush"] = df["rush_attempt"].fillna(0).eq(1)
    df["stuff"] = df["is_rush"] & df["yards_gained"].fillna(0).le(0)
    df["neutral"] = df["wp"].between(0.20, 0.80) & df["score_differential"].abs().le(8)
    df["pass_play"] = df["pass_attempt"].fillna(0).eq(1)
    df["rush_epa_val"] = np.where(df["is_rush"], df["epa"], np.nan)
    df["pass_epa_val"] = np.where(df["pass_play"], df["epa"], np.nan)
    df["exp_rush_flag"] = df["is_rush"] & df["rushing_yards"].fillna(0).ge(15)

    team = df.groupby(["season", "posteam"], as_index=False).agg(
        team_plays=("game_id", "size"),
        team_games=("game_id", "nunique"),
        dropbacks=("is_dropback", "sum"),
        sacks=("is_sack", "sum"),
        qb_hits=("qb_hit", "sum"),
        rushes=("is_rush", "sum"),
        stuffs=("stuff", "sum"),
        pass_oe=("pass_oe", "mean"),
        xpass=("xpass", "mean"),
        pass_rate=("pass_play", "mean"),
        rush_epa=("rush_epa_val", "mean"),
        pass_epa=("pass_epa_val", "mean"),
        team_epa=("epa", "mean"),
        explosive_rushes=("exp_rush_flag", "sum"),
    )
    neut = (
        df.loc[df["neutral"]]
        .groupby(["season", "posteam"], as_index=False)
        .agg(neutral_plays=("game_id", "size"), neutral_games=("game_id", "nunique"))
    )
    team = team.merge(neut, on=["season", "posteam"], how="left")
    team["sack_rate"] = np.where(team["dropbacks"] > 0, team["sacks"] / team["dropbacks"], np.nan)
    team["stuff_rate"] = np.where(team["rushes"] > 0, team["stuffs"] / team["rushes"], np.nan)
    team["pace_neutral"] = np.where(
        team["neutral_games"].fillna(0) > 0,
        team["neutral_plays"] / team["neutral_games"],
        team["team_plays"] / team["team_games"],
    )
    team["proe"] = team["pass_oe"] / 100.0  # nflfastR pass_oe is in percentage points
    return team.rename(columns={"posteam": "team"})


def aggregate_defense_pbp(plays: pd.DataFrame) -> pd.DataFrame:
    df = plays.dropna(subset=["defteam"]).copy()
    rush = df.loc[df["rush_attempt"].fillna(0).eq(1)]
    rec = df.loc[df["receiver_player_id"].notna()]
    pas = df.loc[df["pass_attempt"].fillna(0).eq(1)]
    def_rush = rush.groupby(["season", "defteam"], as_index=False).agg(
        def_rush_epa=("epa", "mean"),
        def_rush_yards=("rushing_yards", "mean"),
        def_exp_rush_rate=("rushing_yards", lambda s: s.fillna(0).ge(15).mean()),
    )
    def_pass = pas.groupby(["season", "defteam"], as_index=False).agg(
        def_pass_epa=("epa", "mean"),
        def_pass_yards=("passing_yards", "mean"),
    )
    def_rec = rec.groupby(["season", "defteam"], as_index=False).agg(
        def_rec_epa=("epa", "mean"),
        def_exp_rec_rate=("receiving_yards", lambda s: s.fillna(0).ge(20).mean()),
    )
    out = def_rush.merge(def_pass, on=["season", "defteam"], how="outer").merge(
        def_rec, on=["season", "defteam"], how="outer"
    )
    return out.rename(columns={"defteam": "team"})


def build_pbp_tables(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_parts, team_parts, def_parts = [], [], []
    for season in seasons:
        print(f"  PBP {season}")
        raw = load_pbp(season, columns=PBP_COLUMNS)
        plays = _offensive_plays(raw)
        player_parts.append(aggregate_player_pbp(plays))
        team_parts.append(aggregate_team_pbp(plays))
        def_parts.append(aggregate_defense_pbp(plays))
    return (
        pd.concat(player_parts, ignore_index=True),
        pd.concat(team_parts, ignore_index=True),
        pd.concat(def_parts, ignore_index=True),
    )
