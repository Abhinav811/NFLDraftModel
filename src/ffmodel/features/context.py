from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    AGE_LAMBDA_DEFAULT,
    AGE_PEAKS,
    GAMES_PER_SEASON,
    PPR,
    PREDICT_SEASON,
    SKILL_POSITIONS,
    TD_RATES,
    TENURE_WEIGHTS,
)
from ..names import normalize_name


def tenure_bucket(years_exp: float | int | None) -> str:
    if pd.isna(years_exp):
        return "developing"
    years = int(years_exp)
    if years <= 0:
        return "rookie"
    if years == 1:
        return "sophomore"
    if years <= 3:
        return "developing"
    if years <= 6:
        return "prime"
    return "veteran"


def age_on_sept1(birth: pd.Series, season: pd.Series) -> pd.Series:
    birth = pd.to_datetime(birth, errors="coerce")
    sept = pd.to_datetime(season.astype("Int64").astype(str) + "-09-01")
    return (sept - birth).dt.days / 365.25


def age_alpha(age: pd.Series, position: pd.Series, lambdas: dict[str, float] | None = None) -> pd.Series:
    lambdas = lambdas or AGE_LAMBDA_DEFAULT
    peaks = position.map(AGE_PEAKS).fillna(28.0)
    lam = position.map(lambdas).fillna(0.02)
    over = (age - peaks).clip(lower=0)
    return (1.0 - lam * over**2).clip(lower=0.35, upper=1.05)


def estimate_age_lambdas(panel: pd.DataFrame) -> dict[str, float]:
    """Fit lambda from YoY PPR change for players past the positional peak."""
    out = dict(AGE_LAMBDA_DEFAULT)
    tmp = panel.dropna(subset=["age", "ppr", "ppr_lag"]).copy()
    tmp["yoy"] = tmp["ppr"] / tmp["ppr_lag"].clip(lower=1) - 1
    for pos, peak in AGE_PEAKS.items():
        sub = tmp.loc[(tmp["position"] == pos) & (tmp["age"] > peak) & tmp["ppr_lag"].gt(50)]
        if len(sub) < 40:
            continue
        x = (sub["age"] - peak) ** 2
        y = 1 - (sub["ppr"] / sub["ppr_lag"].clip(lower=1))
        denom = (x**2).sum()
        if denom <= 0:
            continue
        lam = float((x * y).sum() / denom)
        out[pos] = float(np.clip(lam, 0.004, 0.08))
    return out


def american_to_prob(odds: float) -> float:
    if pd.isna(odds):
        return 0.5
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def vig_stripped_expectation(line: float, over_odds: float, under_odds: float, sigma: float) -> float:
    """Shift the posted total toward the no-vig mean using a normal approximation."""
    p_over = american_to_prob(over_odds)
    p_under = american_to_prob(under_odds)
    total = p_over + p_under
    if total <= 0:
        return line
    p_fair = p_over / total
    # Line is treated as a median; shift by inverse-normal of fair over probability.
    from scipy.stats import norm

    z = float(np.clip(norm.ppf(p_fair), -1.5, 1.5))
    return line + z * sigma * 0.35


def props_to_vfp(props: pd.DataFrame) -> pd.DataFrame:
    if props is None or props.empty:
        return pd.DataFrame(columns=["name_norm", "vfp", "vfp_markets"])
    sigma = {
        "pass_yd": 450,
        "pass_td": 4.5,
        "rush_yd": 180,
        "rush_td": 2.8,
        "rec_yd": 160,
        "rec": 12,
        "rec_td": 2.4,
        "int": 2.2,
        "fum": 1.2,
    }
    rows = []
    for name, grp in props.groupby("name_norm"):
        stats: dict[str, float] = {}
        for rec in grp.itertuples(index=False):
            expected = vig_stripped_expectation(
                float(rec.line),
                getattr(rec, "over_odds", np.nan),
                getattr(rec, "under_odds", np.nan),
                sigma.get(rec.market, 50),
            )
            stats[rec.market] = expected
        vfp = (
            stats.get("pass_yd", 0) * PPR["pass_yd"]
            + stats.get("pass_td", 0) * PPR["pass_td"]
            + stats.get("int", 0) * PPR["int"]
            + stats.get("rush_yd", 0) * PPR["rush_yd"]
            + stats.get("rush_td", 0) * PPR["rush_td"]
            + stats.get("rec", 0) * PPR["rec"]
            + stats.get("rec_yd", 0) * PPR["rec_yd"]
            + stats.get("rec_td", 0) * PPR["rec_td"]
            + stats.get("fum", 0) * PPR["fum"]
        )
        rows.append({"name_norm": name, "vfp": vfp, "vfp_markets": len(stats), **{f"v_{k}": v for k, v in stats.items()}})
    return pd.DataFrame(rows)


def build_injury_features(injuries: pd.DataFrame) -> pd.DataFrame:
    if injuries.empty:
        return pd.DataFrame(columns=["season", "player_id"])
    df = injuries.copy()
    status = df["report_status"].fillna("").str.lower()
    practice = df["practice_status"].fillna("").str.lower()
    text = (
        df["report_primary_injury"].fillna("").astype(str)
        + " "
        + df["report_secondary_injury"].fillna("").astype(str)
        + " "
        + df["practice_primary_injury"].fillna("").astype(str)
    ).str.lower()
    major = text.str.contains(
        r"acl|achilles|achilles|rupture|fracture|broken|surgery|torn|lisfranc|patella|achilles",
        regex=True,
    )
    missed = status.isin(["out", "injured reserve", "ir", "pup", "physically unable to perform"]) | practice.str.contains(
        "out|injured reserve|pup", regex=True
    )
    df["missed_week"] = missed.astype(int)
    df["major"] = major.astype(int)
    agg = df.groupby(["season", "gsis_id"], as_index=False).agg(
        injury_weeks=("missed_week", "sum"),
        major_injury=("major", "max"),
        injury_reports=("gsis_id", "size"),
    )
    return agg.rename(columns={"gsis_id": "player_id"})


def build_snap_features(snaps: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty:
        return pd.DataFrame(columns=["season", "player_id"])
    weekly = snaps.loc[snaps["game_type"].fillna("REG").eq("REG")].copy()
    snap = weekly.groupby(["season", "pfr_player_id"], as_index=False).agg(
        off_snap_pct=("offense_pct", "mean"),
        off_snaps=("offense_snaps", "sum"),
        games_snaps=("week", "nunique"),
    )
    roster_key = rosters.dropna(subset=["pfr_id", "gsis_id"])[["season", "pfr_id", "gsis_id"]].drop_duplicates()
    snap = snap.merge(
        roster_key,
        left_on=["season", "pfr_player_id"],
        right_on=["season", "pfr_id"],
        how="left",
    )
    return snap.rename(columns={"gsis_id": "player_id"})


def _to_inches(val) -> float:
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    text = str(val).strip()
    if not text:
        return np.nan
    if "-" in text:
        feet, inches = text.split("-", 1)
        try:
            return float(feet) * 12 + float(inches)
        except ValueError:
            return np.nan
    return pd.to_numeric(text, errors="coerce")


def build_physical_features(combine: pd.DataFrame, players: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    comb = combine.copy()
    comb["name_norm"] = comb["player_name"].map(normalize_name)
    comb["pos_simple"] = comb["pos"].replace({"FB": "RB", "HB": "RB", "TB": "RB", "WR": "WR", "TE": "TE", "QB": "QB"})
    comb = comb.loc[comb["pos_simple"].isin(SKILL_POSITIONS)]
    comb["ht"] = comb["ht"].map(_to_inches)
    comb["wt"] = pd.to_numeric(comb["wt"], errors="coerce")
    for col in ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]:
        comb[col] = pd.to_numeric(comb[col], errors="coerce")
    comb["speed_score"] = np.where(
        comb["forty"].gt(0) & comb["wt"].gt(0),
        comb["wt"] * 200.0 / (comb["forty"] ** 4),
        np.nan,
    )
    comb["burst"] = comb["vertical"].fillna(0) + comb["broad_jump"].fillna(0) / 12.0
    comb["bmi"] = np.where(
        comb["ht"].gt(0) & comb["wt"].gt(0),
        703 * comb["wt"] / (comb["ht"] ** 2),
        np.nan,
    )
    # Prefer pfr_id join onto players.
    ply = players.copy()
    ply["name_norm"] = ply["display_name"].map(normalize_name)
    merged = ply.merge(
        comb.rename(columns={"pfr_id": "pfr_id_c"}),
        left_on="pfr_id",
        right_on="pfr_id_c",
        how="left",
        suffixes=("", "_c"),
    )
    missing = merged["forty"].isna()
    fallback = comb.sort_values("season").drop_duplicates("name_norm", keep="last")
    fb_cols = ["name_norm", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle", "speed_score", "burst", "bmi", "ht", "wt"]
    merged = merged.merge(fallback[fb_cols], on="name_norm", how="left", suffixes=("", "_fb"))
    for col in ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle", "speed_score", "burst", "bmi"]:
        merged[col] = merged[col].fillna(merged.get(f"{col}_fb"))
    d = draft.copy()
    if "pfr_player_id" in d.columns:
        d = d.rename(columns={"pfr_player_id": "pfr_id", "pick": "draft_ovr", "round": "draft_round_d"})
    keep = [c for c in ["pfr_id", "season", "round", "pick", "gsis_id"] if c in d.columns]
    if keep:
        d = d[keep].drop_duplicates()
        if "gsis_id" in d.columns:
            merged = merged.merge(d.rename(columns={"gsis_id": "gsis_id_d"}), left_on="gsis_id", right_on="gsis_id_d", how="left")
    merged["draft_pick_num"] = pd.to_numeric(merged.get("draft_pick"), errors="coerce")
    if "pick" in merged.columns:
        merged["draft_pick_num"] = merged["draft_pick_num"].fillna(pd.to_numeric(merged["pick"], errors="coerce"))
    merged["undrafted"] = merged["draft_pick_num"].isna().astype(int)
    merged["draft_pick_num"] = merged["draft_pick_num"].fillna(250)
    merged["log_draft_pick"] = np.log(merged["draft_pick_num"].clip(lower=1))
    cols = [
        "gsis_id",
        "forty",
        "bench",
        "vertical",
        "broad_jump",
        "cone",
        "shuttle",
        "speed_score",
        "burst",
        "bmi",
        "draft_pick_num",
        "undrafted",
        "log_draft_pick",
        "height",
        "weight",
    ]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].drop_duplicates("gsis_id").rename(columns={"gsis_id": "player_id"})


def team_coaches(schedules: pd.DataFrame) -> pd.DataFrame:
    games = schedules.loc[schedules["game_type"].fillna("REG").isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    home = games[["season", "home_team", "home_coach"]].rename(columns={"home_team": "team", "home_coach": "coach"})
    away = games[["season", "away_team", "away_coach"]].rename(columns={"away_team": "team", "away_coach": "coach"})
    both = pd.concat([home, away], ignore_index=True)
    both = both.dropna(subset=["team", "coach"])
    mode = (
        both.groupby(["season", "team"])["coach"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    return mode


def _schedule_team_games(schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    games = schedules.loc[(schedules["season"] == season) & (schedules["game_type"] == "REG")].copy()
    played = games["home_score"].notna() | games["result"].notna()
    games = games.assign(played=played.astype(bool))
    home = games.copy()
    home["team"] = home["home_team"]
    home["opp"] = home["away_team"]
    home["spread_for"] = home["spread_line"]
    home["indoor"] = home["roof"].isin(["dome", "closed"]).astype(int)
    away = games.copy()
    away["team"] = away["away_team"]
    away["opp"] = away["home_team"]
    away["spread_for"] = -pd.to_numeric(away["spread_line"], errors="coerce")
    away["indoor"] = away["roof"].isin(["dome", "closed"]).astype(int)
    rows = pd.concat([home, away], ignore_index=True)
    rows["total_line"] = pd.to_numeric(rows["total_line"], errors="coerce")
    return rows


def build_schedule_features(
    schedules: pd.DataFrame,
    defense: pd.DataFrame,
    season: int,
    remaining_only: bool = False,
) -> pd.DataFrame:
    rows = _schedule_team_games(schedules, season)
    full_n = rows.groupby("team")["game_id"].nunique().rename("games_sched")
    if remaining_only:
        rows = rows.loc[~rows["played"]].copy()
    prior = defense.copy()
    if "season" in prior.columns:
        use = prior.loc[prior["season"] == season - 1]
        if use.empty:
            use = prior.loc[prior["season"] == prior["season"].max()]
        prior = use
    prior = prior.drop(columns=["season"], errors="ignore")
    rows = rows.merge(prior.add_prefix("opp_").rename(columns={"opp_team": "opp"}), on="opp", how="left")
    feat = rows.groupby("team", as_index=False).agg(
        sos_def_rush_epa=("opp_def_rush_epa", "mean"),
        sos_def_pass_epa=("opp_def_pass_epa", "mean"),
        sos_def_rec_epa=("opp_def_rec_epa", "mean"),
        pct_indoor=("indoor", "mean"),
        avg_total=("total_line", "mean"),
        avg_spread_for=("spread_for", "mean"),
        games_left=("game_id", "nunique"),
    )
    feat = feat.merge(full_n.reset_index(), on="team", how="left")
    feat["games_sched"] = feat["games_sched"].fillna(feat["games_left"])
    feat["season"] = season
    return feat


def vacated_opportunity(stats: pd.DataFrame, rosters: pd.DataFrame, season: int) -> pd.DataFrame:
    prior = stats.loc[stats["season"] == season - 1, ["player_id", "recent_team", "targets", "carries", "attempts"]].copy()
    prior = prior.rename(columns={"recent_team": "team"})
    current_ids = set(rosters.loc[rosters["season"] == season, "gsis_id"].dropna())
    prior["returning"] = prior["player_id"].isin(current_ids).astype(int)
    ret = prior.loc[prior["returning"].eq(1)].groupby("team", as_index=False).agg(
        returning_targets=("targets", "sum"),
        returning_carries=("carries", "sum"),
        returning_attempts=("attempts", "sum"),
    )
    team = prior.groupby("team", as_index=False).agg(
        team_targets=("targets", "sum"),
        team_carries=("carries", "sum"),
        team_attempts=("attempts", "sum"),
    ).merge(ret, on="team", how="left")
    for col in ["returning_targets", "returning_carries", "returning_attempts"]:
        team[col] = team[col].fillna(0)
    team["vacated_target_share"] = np.where(
        team["team_targets"] > 0, 1 - team["returning_targets"] / team["team_targets"], 0
    )
    team["vacated_carry_share"] = np.where(
        team["team_carries"] > 0, 1 - team["returning_carries"] / team["team_carries"], 0
    )
    team["season"] = season
    return team


def adp_implied_points(adp_hist: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Fit a log-ADP curve by position on historical seasons, then score every row."""
    merged = adp_hist.merge(
        stats[["season", "player_id", "position", "fantasy_points_ppr", "player_display_name"]].assign(
            name_norm=lambda d: d["player_display_name"].map(normalize_name)
        ),
        left_on=["season", "adp_name_norm", "position"],
        right_on=["season", "name_norm", "position"],
        how="left",
    )
    curves = {}
    for pos, grp in merged.dropna(subset=["fantasy_points_ppr", "adp"]).groupby("position"):
        if len(grp) < 30:
            continue
        x = np.log(grp["adp"].clip(lower=1))
        y = grp["fantasy_points_ppr"]
        coef = np.polyfit(x, y, 1)
        curves[pos] = coef
    rows = []
    for rec in adp_hist.itertuples(index=False):
        coef = curves.get(rec.position, np.array([-55.0, 280.0]))
        market_fp = float(np.polyval(coef, np.log(max(rec.adp, 1))))
        rows.append(
            {
                "season": rec.season,
                "adp": rec.adp,
                "adp_name_norm": rec.adp_name_norm,
                "position": rec.position,
                "team": rec.team,
                "market_fp": market_fp,
                "player_name": rec.name,
            }
        )
    return pd.DataFrame(rows), curves


def oline_index(team_pbp: pd.DataFrame) -> pd.DataFrame:
    df = team_pbp.copy()
    df["oline_index"] = 0.0

    def _z(series: pd.Series, invert: bool = False) -> pd.Series:
        mu = series.mean()
        sd = series.std(ddof=0)
        if not sd or pd.isna(sd):
            sd = 1.0
        z = (series - mu) / sd
        return -z if invert else z

    for _, grp in df.groupby("season"):
        sack_z = _z(grp["sack_rate"], invert=True)
        stuff_z = _z(grp["stuff_rate"], invert=True)
        rush_z = _z(grp["rush_epa"], invert=False)
        df.loc[grp.index, "oline_index"] = (0.4 * sack_z + 0.3 * stuff_z + 0.3 * rush_z).clip(-3, 3)
    league_pace = df.groupby("season")["pace_neutral"].transform("mean")
    df["coaching_C"] = ((df["proe"].fillna(0) + 1) / 2) * (df["pace_neutral"] / league_pace.replace(0, np.nan))
    return df


def production_proxy(row: pd.Series) -> float:
    ppr = row.get("ppr_lag", np.nan)
    if pd.isna(ppr):
        ppr = 0.0
    alpha = row.get("age_alpha", 1.0)
    missed = row.get("injury_weeks", 0) or 0
    games = GAMES_PER_SEASON.get(int(row.get("season", PREDICT_SEASON) - 1), 17)
    avail = 1 - min(missed, games) / max(games, 1)
    # If they missed time, per-game rate is more informative.
    ppg = row.get("ppr_ppg_lag", np.nan)
    if pd.notna(ppg) and missed >= 3:
        ppr = ppg * GAMES_PER_SEASON.get(int(row.get("season", PREDICT_SEASON)), 17)
    return float(ppr) * float(alpha) * (0.85 + 0.15 * avail)


def situation_proxy(row: pd.Series) -> float:
    base = row.get("vfp") if pd.notna(row.get("vfp")) else row.get("market_fp", np.nan)
    if pd.isna(base):
        base = row.get("ppr_lag", 80) or 80
    oline = row.get("oline_index", 0) or 0
    c = row.get("coaching_C", 1) or 1
    vacated = 0.0
    if row.get("position") in {"WR", "TE"}:
        vacated = row.get("vacated_target_share", 0) or 0
    elif row.get("position") == "RB":
        vacated = row.get("vacated_carry_share", 0) or 0
    indoor = row.get("pct_indoor", 0.3) or 0.3
    sos = 0.0
    if row.get("position") == "QB":
        sos = -(row.get("sos_def_pass_epa", 0) or 0)
    elif row.get("position") == "RB":
        sos = -(row.get("sos_def_rush_epa", 0) or 0)
    else:
        sos = -(row.get("sos_def_rec_epa", 0) or 0)
    depth = row.get("depth_rank", 2) or 2
    depth_mult = {1: 1.08, 2: 0.92, 3: 0.72}.get(int(depth) if pd.notna(depth) else 2, 0.55)
    adj = (1 + 0.03 * oline) * (0.92 + 0.08 * c) * (1 + 0.08 * sos) * depth_mult
    player_vac = row.get("player_vacated_boost", np.nan)
    if pd.notna(player_vac):
        adj *= 1 + 0.0025 * float(player_vac)
    else:
        adj *= 1 + 0.12 * vacated
    if row.get("workload_cliff"):
        adj *= 0.90
    if row.get("breakout_window"):
        adj *= 1.06
    if row.get("team_change"):
        adj *= 0.97
    if row.get("role_expand"):
        adj *= 1.04
    if row.get("chronic_injury"):
        adj *= 0.96
    adj *= 0.97 + 0.06 * indoor if row.get("position") in {"WR", "QB"} else 1.0
    return float(base) * float(adj)


def physical_proxy(row: pd.Series) -> float:
    """Rookie/young-player prior from draft capital and measurables."""
    pick = row.get("draft_pick_num", 250) or 250
    capital = max(0.15, 1.15 - np.log(max(pick, 1)) / np.log(250))
    pos = row.get("position")
    athletic = 1.0
    if pos == "RB" and pd.notna(row.get("speed_score")):
        athletic = 0.85 + 0.15 * np.clip((row["speed_score"] - 90) / 40, -1, 1.5)
    if pos in {"WR", "TE"} and pd.notna(row.get("burst")):
        athletic = 0.9 + 0.1 * np.clip((row["burst"] - 36) / 10, -1, 1.5)
    pos_base = {"QB": 220, "RB": 160, "WR": 140, "TE": 110}.get(pos, 100)
    years = row.get("years_exp", 1) or 1
    if years >= 3:
        return float(row.get("ppr_lag", pos_base) or pos_base)
    return float(pos_base * capital * athletic)


def blend_projection(row: pd.Series) -> float:
    bucket = row.get("tenure_bucket") or tenure_bucket(row.get("years_exp"))
    w_m, w_p, w_s, w_ph, w_a = TENURE_WEIGHTS.get(bucket, TENURE_WEIGHTS["developing"])
    market = row.get("vfp") if pd.notna(row.get("vfp")) and row.get("vfp_markets", 0) >= 1 else row.get("market_fp")
    if pd.isna(market):
        market = row.get("production_proxy", 0) or 0
        w_m = 0.0
        scale = w_p + w_s + w_ph + w_a
        if scale:
            w_p, w_s, w_ph, w_a = [w / scale for w in (w_p, w_s, w_ph, w_a)]
    prod = row.get("production_proxy", market) or 0
    sit = row.get("situation_proxy", market) or 0
    phys = row.get("physical_proxy", market) or 0
    aging = (row.get("age_alpha", 1) or 1) * (prod if prod else market)
    return float(w_m * market + w_p * prod + w_s * sit + w_ph * phys + w_a * aging)


def _z_within(df: pd.DataFrame, col: str, keys: list[str]) -> pd.Series:
    work = df.loc[:, [k for k in keys if k in df.columns]].copy()
    for k in list(work.columns):
        work[k] = work[k].astype("object").where(work[k].notna(), "_na_")
    work["_x"] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    gkeys = list(work.columns.drop("_x"))
    if not gkeys:
        x = work["_x"]
        sd = x.std(ddof=0)
        return (x - x.mean()) / sd if sd else x * 0
    mu = work.groupby(gkeys, dropna=False)["_x"].transform("mean")
    sd = work.groupby(gkeys, dropna=False)["_x"].transform(lambda s: s.std(ddof=0))
    z = (work["_x"] - mu) / sd.replace(0, np.nan)
    return z.fillna(0)


def enrich_derived_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Player-level vacated usage, luck/regression, breakouts, competition — no PBP needed."""
    df = panel.copy()
    games_prev = df["season"].map(lambda s: GAMES_PER_SEASON.get(int(s) - 1, 17) if pd.notna(s) else 17)

    depth = pd.to_numeric(df.get("depth_rank"), errors="coerce")
    df["depth_weight"] = depth.map(lambda r: {1: 0.52, 2: 0.28, 3: 0.13}.get(int(r), 0.04) if pd.notna(r) else 0.10)
    grp = df.groupby(["season", "team", "position"], dropna=False)
    wsum = grp["depth_weight"].transform("sum").replace(0, np.nan)
    df["depth_alloc"] = df["depth_weight"] / wsum

    prior_tgt = pd.to_numeric(df.get("targets_lag"), errors="coerce").fillna(0)
    prior_car = pd.to_numeric(df.get("carries_lag"), errors="coerce").fillna(0)
    ret_tgt = pd.to_numeric(df.get("returning_targets"), errors="coerce").replace(0, np.nan)
    ret_car = pd.to_numeric(df.get("returning_carries"), errors="coerce").replace(0, np.nan)
    share_tgt = (prior_tgt / ret_tgt).clip(0, 0.7)
    share_car = (prior_car / ret_car).clip(0, 0.7)
    years = pd.to_numeric(df.get("years_exp"), errors="coerce").fillna(0)
    # Rookies have no prior share — use depth chart only.
    alloc_tgt = np.where(years <= 0, df["depth_alloc"], 0.55 * share_tgt.fillna(df["depth_alloc"]) + 0.45 * df["depth_alloc"])
    alloc_car = np.where(years <= 0, df["depth_alloc"], 0.55 * share_car.fillna(df["depth_alloc"]) + 0.45 * df["depth_alloc"])
    vac_tgt = pd.to_numeric(df.get("vacated_target_share"), errors="coerce").fillna(0)
    vac_car = pd.to_numeric(df.get("vacated_carry_share"), errors="coerce").fillna(0)
    team_tgt = pd.to_numeric(df.get("team_targets"), errors="coerce").fillna(0)
    team_car = pd.to_numeric(df.get("team_carries"), errors="coerce").fillna(0)
    df["player_vacated_targets"] = alloc_tgt * vac_tgt * team_tgt
    df["player_vacated_carries"] = alloc_car * vac_car * team_car
    df["player_vacated_boost"] = np.where(
        df["position"].isin(["WR", "TE"]),
        df["player_vacated_targets"],
        np.where(df["position"].eq("RB"), df["player_vacated_carries"], 0.0),
    )

    rec = pd.to_numeric(df.get("receptions_lag"), errors="coerce")
    tgt = pd.to_numeric(df.get("targets_lag"), errors="coerce")
    rec_yd = pd.to_numeric(df.get("receiving_yards_lag"), errors="coerce")
    rec_td = pd.to_numeric(df.get("receiving_tds_lag"), errors="coerce")
    car = pd.to_numeric(df.get("carries_lag"), errors="coerce")
    rush_yd = pd.to_numeric(df.get("rushing_yards_lag"), errors="coerce")
    rush_td = pd.to_numeric(df.get("rushing_tds_lag"), errors="coerce")
    ez = pd.to_numeric(df.get("ez_targets"), errors="coerce").fillna(0)
    in10 = pd.to_numeric(df.get("inside10_targets"), errors="coerce").fillna(0)
    in5 = pd.to_numeric(df.get("inside5_carries"), errors="coerce").fillna(0)
    df["catch_rate_lag"] = np.where(tgt > 0, rec / tgt, np.nan)
    df["ypr_lag"] = np.where(rec > 0, rec_yd / rec, np.nan)
    df["ypc_lag"] = np.where(car > 0, rush_yd / car, np.nan)
    df["rec_td_luck"] = rec_td.fillna(0) - (TD_RATES["inside10_targets"] * in10 + TD_RATES["ez_targets"] * ez)
    df["rush_td_luck"] = rush_td.fillna(0) - TD_RATES["inside5_carries"] * in5
    df["td_luck"] = df["rec_td_luck"] + df["rush_td_luck"]
    df["carry_load"] = car
    age = pd.to_numeric(df.get("age"), errors="coerce")
    df["workload_cliff"] = (
        ((car.fillna(0) >= 240) & (age.fillna(0) >= 26.5))
        | ((car.fillna(0) >= 280) & (age.fillna(0) >= 26))
    ).astype(float)
    pos = df["position"]
    df["breakout_window"] = (
        ((pos.eq("RB") & years.eq(1)) | (pos.eq("WR") & years.isin([1, 2])) | (pos.eq("TE") & years.isin([2, 3])))
    ).astype(float)
    df["sophomore_leap"] = (years.eq(1) & pd.to_numeric(df.get("ppr_lag"), errors="coerce").between(70, 200)).astype(float)
    games = pd.to_numeric(df.get("games_lag"), errors="coerce")
    df["games_pct_lag"] = games / games_prev.replace(0, np.nan)
    inj = pd.to_numeric(df.get("injury_weeks"), errors="coerce").fillna(0)
    df["availability"] = (1 - (inj / games_prev.clip(lower=1))).clip(0, 1)
    recent = df.get("recent_team_lag")
    df["team_change"] = ((recent.notna()) & (recent != df["team"])).astype(float)
    df["new_hc_pass_catcher"] = pd.to_numeric(df.get("new_hc"), errors="coerce").fillna(0) * pos.isin(["WR", "TE", "QB"]).astype(float)
    df["pos_competition"] = grp["player_id"].transform("size")
    df["starter"] = (depth.fillna(9) <= 1).astype(float)
    team_car = pd.to_numeric(df.get("team_carries"), errors="coerce").replace(0, np.nan)
    df["carry_share_lag"] = (car / team_car).clip(0, 0.85)
    df["pass_catch_rb"] = np.where(pos.eq("RB"), pd.to_numeric(df.get("target_share_lag"), errors="coerce").fillna(0), 0.0)
    df["qb_rush_share"] = np.where(
        pos.eq("QB"),
        pd.to_numeric(df.get("rushing_yards_lag"), errors="coerce").fillna(0) / 800.0,
        0.0,
    )
    snap = pd.to_numeric(df.get("off_snap_pct"), errors="coerce")
    df["role_expand"] = ((df["starter"] > 0) & snap.fillna(1).lt(0.55) & years.ge(1)).astype(float)
    df["injury_bounce"] = (
        (df["games_pct_lag"].fillna(1) < 0.70)
        & age.between(23, 31)
        & pd.to_numeric(df.get("ppr_ppg_lag"), errors="coerce").fillna(0).ge(8)
    ).astype(float)
    chronic = pd.to_numeric(df.get("injury_weeks_3yr"), errors="coerce").fillna(0)
    df["chronic_injury"] = (chronic >= 8).astype(float)
    df["draft_capital"] = 1.0 - np.log(pd.to_numeric(df.get("draft_pick_num"), errors="coerce").fillna(250).clip(1)) / np.log(250)
    df["young_capital"] = np.where(years.le(2), df["draft_capital"], 0.0)
    df["new_starter_vacated"] = df["starter"] * pd.to_numeric(df.get("player_vacated_boost"), errors="coerce").fillna(0)
    snap_z = _z_within(df, "off_snap_pct", ["season", "position"])
    share_z = _z_within(df, "target_share_lag", ["season", "position"])
    ppg_z = _z_within(df, "ppr_ppg_lag", ["season", "position"])
    inv = df.assign(_inv=1 / depth.clip(lower=1))
    depth_z = _z_within(inv, "_inv", ["season", "position"]) if "depth_rank" in df.columns else pd.Series(0.0, index=df.index)
    df["usage_index"] = snap_z.fillna(0) + share_z.fillna(0) + ppg_z.fillna(0) + depth_z.fillna(0)
    df["overproduction"] = ppg_z.fillna(0) - 0.5 * (snap_z.fillna(0) + share_z.fillna(0))
    df["eff_ypc_z"] = _z_within(df, "ypc_lag", ["season", "position"])
    df["eff_ypr_z"] = _z_within(df, "ypr_lag", ["season", "position"])
    df["eff_catch_z"] = _z_within(df, "catch_rate_lag", ["season", "position"])
    df["eff_index"] = df["eff_ypc_z"].fillna(0) + df["eff_ypr_z"].fillna(0) + df["eff_catch_z"].fillna(0)
    hv = pd.to_numeric(df.get("hv_rz"), errors="coerce")
    df["rz_index"] = _z_within(df.assign(_hv=hv), "_hv", ["season", "position"]) if "hv_rz" in df.columns else np.nan
    chunk = pd.to_numeric(df.get("chunk_rating"), errors="coerce")
    df["chunk_index"] = _z_within(df.assign(_ch=chunk), "_ch", ["season", "position"]) if "chunk_rating" in df.columns else np.nan

    # Tiny team-level context: keep, but pre-shrink so GBDT cannot dominate with O-line/PROE.
    for col, scale in {
        "oline_index": 0.35,
        "coaching_C": 0.35,
        "proe": 0.35,
        "pace_neutral": 0.35,
        "sack_rate": 0.35,
        "pass_rate": 0.35,
        "sos_def_pass_epa": 0.4,
        "sos_def_rush_epa": 0.4,
        "sos_def_rec_epa": 0.4,
        "pct_indoor": 0.4,
        "avg_total": 0.4,
        "avg_spread_for": 0.4,
    }.items():
        if col in df.columns:
            df[f"{col}_shrunk"] = pd.to_numeric(df[col], errors="coerce") * scale

    df["situation_proxy"] = df.apply(situation_proxy, axis=1)
    df["blend_proj"] = df.apply(blend_projection, axis=1)
    return df
