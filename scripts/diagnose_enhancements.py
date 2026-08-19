#!/usr/bin/env python3
"""Walk-forward diagnostics for additive/redactive model changes.

Does not write 2026 rankings. Prints JSON to stdout and a sidecar file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffmodel.config import BACKTEST_SEASONS, PROCESSED_DIR, RAW_DIR  # noqa: E402
from ffmodel.ingest.nflverse import download_release, load_win_totals, read_parquet  # noqa: E402
from ffmodel.model import (  # noqa: E402
    FEATURE_COLS,
    eligible,
    fit_models,
    predict_models,
)
from ffmodel.names import normalize_name  # noqa: E402

OUT = PROCESSED_DIR / "enhance_diagnostics.json"
ADP_BAND = (18, 132)

DROP_GROUPS = {
    "drop_team_context": [
        "oline_index_shrunk",
        "coaching_C_shrunk",
        "proe_shrunk",
        "pace_neutral_shrunk",
        "sos_def_pass_epa_shrunk",
        "sos_def_rush_epa_shrunk",
        "sos_def_rec_epa_shrunk",
        "pct_indoor_shrunk",
        "avg_total_shrunk",
        "avg_spread_for_shrunk",
        "new_hc",
        "new_hc_pass_catcher",
    ],
    "drop_physical": [
        "forty",
        "speed_score",
        "burst",
        "bmi",
        "draft_pick_num",
        "log_draft_pick",
        "undrafted",
        "physical_proxy",
        "draft_capital",
        "young_capital",
    ],
    "drop_luck": [
        "td_luck",
        "rec_td_luck",
        "rush_td_luck",
        "overproduction",
        "eff_index",
        "eff_ypc_z",
        "eff_ypr_z",
        "eff_catch_z",
    ],
    "drop_injury_flags": [
        "availability",
        "chronic_injury",
        "injury_bounce",
        "games_pct_lag",
    ],
    "drop_vacated_role": [
        "player_vacated_targets",
        "player_vacated_carries",
        "player_vacated_boost",
        "new_starter_vacated",
        "role_expand",
        "breakout_window",
        "sophomore_leap",
        "starter",
        "workload_cliff",
    ],
    "drop_raw_volume_lags": [
        "attempts_lag",
        "passing_yards_lag",
        "passing_tds_lag",
        "carries_lag",
        "rushing_yards_lag",
        "rushing_tds_lag",
        "targets_lag",
        "receptions_lag",
        "receiving_yards_lag",
        "receiving_tds_lag",
        "ppr_lag",
        "ppr_ppg_lag",
    ],
}


def _rho(x, y) -> float:
    mask = pd.notna(x) & pd.notna(y)
    if int(mask.sum()) < 25 or pd.Series(x)[mask].nunique() < 2:
        return float("nan")
    val = spearmanr(pd.Series(x)[mask], pd.Series(y)[mask], nan_policy="omit").statistic
    return float(val) if val == val else float("nan")


def adp_pool(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["adp"].notna() & df["ppr_actual"].notna() & df["market_fp"].notna()].copy()


def residual_table(panel: pd.DataFrame, preds: pd.DataFrame) -> list[dict]:
    keys = ["season", "player_id"]
    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    extra = ["injury_weeks", "injury_weeks_3yr", "major_injury", "oline_index", "coaching_C", "proe"]
    cols = list(dict.fromkeys(keys + feat_cols + extra))
    src = panel[cols].drop_duplicates(keys)
    d = preds.merge(src, on=keys, how="left")
    d = adp_pool(d)
    d["resid"] = d["ppr_actual"] - d["market_fp"]
    d["rank_lift"] = d["adp_rank"] - d["actual_rank"]
    rows = []
    for col in feat_cols + extra:
        if col not in d.columns:
            continue
        x = pd.to_numeric(d[col], errors="coerce")
        n = int(x.notna().sum())
        if n < 40:
            continue
        rec = {
            "feature": col,
            "n": n,
            "rho_resid": _rho(x, d["resid"]),
            "rho_rank_lift": _rho(x, d["rank_lift"]),
            "rho_ppr": _rho(x, d["ppr_actual"]),
            "rho_market": _rho(x, d["market_fp"]),
        }
        rec["unique"] = rec["rho_resid"]  # alias
        rows.append(rec)
    rows.sort(key=lambda r: abs(r["rho_resid"]) if r["rho_resid"] == r["rho_resid"] else 0, reverse=True)
    return rows


def residual_by_pos(panel: pd.DataFrame, preds: pd.DataFrame) -> list[dict]:
    keys = ["season", "player_id"]
    watch = [
        "depth_rank",
        "usage_index",
        "avg_spread_for_shrunk",
        "avg_total_shrunk",
        "injury_bounce",
        "overproduction",
        "player_vacated_boost",
        "oline_index_shrunk",
        "td_luck",
        "age_alpha",
        "pass_catch_rb",
        "workload_cliff",
        "forty",
        "team_change",
        "explosive_rate",
        "hv_rz",
        "breakout_window",
    ]
    src = panel[[c for c in ["season", "player_id"] + watch if c in panel.columns]].drop_duplicates(keys)
    d = adp_pool(preds.merge(src, on=keys, how="left"))
    d["resid"] = d["ppr_actual"] - d["market_fp"]
    out = []
    for pos, g in d.groupby("position"):
        for col in watch:
            if col not in g.columns:
                continue
            x = pd.to_numeric(g[col], errors="coerce")
            out.append({"position": pos, "feature": col, "n": int(x.notna().sum()), "rho_resid": _rho(x, g["resid"])})
    return out


def flag_audit(preds: pd.DataFrame, panel: pd.DataFrame) -> dict:
    keys = ["season", "player_id"]
    why_cols = [
        "breakout_window",
        "injury_bounce",
        "role_expand",
        "sophomore_leap",
        "player_vacated_boost",
        "usage_index",
        "pass_catch_rb",
        "td_luck",
        "new_starter_vacated",
        "workload_cliff",
        "chronic_injury",
        "overproduction",
        "age_alpha",
        "eff_index",
    ]
    src = panel[[c for c in keys + why_cols if c in panel.columns]].drop_duplicates(keys)
    d = adp_pool(preds.merge(src, on=keys, how="left"))
    d["beat"] = (d["adp_rank"] - d["actual_rank"]) >= 4
    d["miss"] = (d["actual_rank"] - d["adp_rank"]) >= 4
    base_beat = float(d["beat"].mean())
    base_miss = float(d["miss"].mean())
    cheap = d.loc[d["adp"].between(*ADP_BAND) & ((d["adp_rank"] - d["model_rank"]) >= 5)]
    expensive = d.loc[d["adp"].between(*ADP_BAND) & ((d["model_rank"] - d["adp_rank"]) >= 5)]
    rows = []
    tests = [
        ("breakout_window", d["breakout_window"].fillna(0).gt(0), "steal_why"),
        ("injury_bounce", d["injury_bounce"].fillna(0).gt(0), "steal_why"),
        ("role_expand", d["role_expand"].fillna(0).gt(0), "steal_why"),
        ("sophomore_leap", d["sophomore_leap"].fillna(0).gt(0), "steal_why"),
        ("player_vacated_boost>=8", pd.to_numeric(d["player_vacated_boost"], errors="coerce").ge(8), "steal_why"),
        ("usage_index>=1.25", pd.to_numeric(d["usage_index"], errors="coerce").ge(1.25), "steal_why"),
        ("pass_catch_rb>=0.08", pd.to_numeric(d["pass_catch_rb"], errors="coerce").ge(0.08), "steal_why"),
        ("td_luck<=-1", pd.to_numeric(d["td_luck"], errors="coerce").le(-1), "steal_why"),
        ("new_starter_vacated>=6", pd.to_numeric(d["new_starter_vacated"], errors="coerce").ge(6), "steal_why"),
        ("workload_cliff", d["workload_cliff"].fillna(0).gt(0), "fade_why"),
        ("chronic_injury", d["chronic_injury"].fillna(0).gt(0), "fade_why"),
        ("td_luck>=1.5", pd.to_numeric(d["td_luck"], errors="coerce").ge(1.5), "fade_why"),
        ("overproduction>=0.55", pd.to_numeric(d["overproduction"], errors="coerce").ge(0.55), "fade_why"),
        ("age_alpha<=0.85", pd.to_numeric(d["age_alpha"], errors="coerce").le(0.85), "fade_why"),
        ("eff_index>=1.4", pd.to_numeric(d["eff_index"], errors="coerce").ge(1.4), "fade_why"),
    ]
    for name, mask, kind in tests:
        sub = d.loc[mask]
        rec = {"feature": name, "kind": kind, "n": int(len(sub)), "base_beat": base_beat, "base_miss": base_miss}
        if kind == "steal_why":
            rec["hit"] = float(sub["beat"].mean()) if len(sub) else float("nan")
            rec["vs_base"] = rec["hit"] - base_beat if rec["hit"] == rec["hit"] else float("nan")
        else:
            rec["hit"] = float(sub["miss"].mean()) if len(sub) else float("nan")
            rec["vs_base"] = rec["hit"] - base_miss if rec["hit"] == rec["hit"] else float("nan")
        rows.append(rec)
    steals = d.loc[d["steal_label"] == "steal"]
    fades = d.loc[d["steal_label"] == "fade"]
    cheap_hit = float(cheap["beat"].mean()) if len(cheap) else float("nan")
    exp_hit = float(expensive["miss"].mean()) if len(expensive) else float("nan")
    return {
        "base_beat_rate": base_beat,
        "base_miss_rate": base_miss,
        "model_cheap_ge5_n": int(len(cheap)),
        "model_cheap_ge5_hit": cheap_hit,
        "model_expensive_ge5_n": int(len(expensive)),
        "model_expensive_ge5_hit": exp_hit,
        "painted_steal_n": int(len(steals)),
        "painted_steal_hit": float(steals["beat"].mean()) if len(steals) else float("nan"),
        "painted_fade_n": int(len(fades)),
        "painted_fade_hit": float(fades["miss"].mean()) if len(fades) else float("nan"),
        "confirming": rows,
    }


def walk_metrics(panel: pd.DataFrame, pred_col: str | None = None) -> dict:
    """If pred_col is set, use that column instead of fitting."""
    lifts = []
    maes = []
    cheap_hits = []
    n_cheap = 0
    for season in BACKTEST_SEASONS:
        test = eligible(panel.loc[panel["season"] == season])
        scored = test.loc[test["adp"].notna() & test["ppr_actual"].notna()].copy()
        if pred_col:
            scored["model_fp"] = pd.to_numeric(scored[pred_col], errors="coerce")
        else:
            train = eligible(panel.loc[panel["season"] < season])
            models = fit_models(train)
            test = test.copy()
            test["model_fp"] = predict_models(models, test)
            scored = test.loc[test["adp"].notna() & test["ppr_actual"].notna()].copy()
        scored["model_rank"] = scored.groupby("position")["model_fp"].rank(ascending=False, method="min")
        scored["adp_rank"] = scored.groupby("position")["adp"].rank(method="min")
        scored["actual_rank"] = scored.groupby("position")["ppr_actual"].rank(ascending=False, method="min")
        m = _rho(scored["model_fp"], scored["ppr_actual"])
        a = _rho(scored["market_fp"], scored["ppr_actual"])
        lifts.append(m - a if m == m and a == a else float("nan"))
        maes.append(float(mean_absolute_error(scored["ppr_actual"], scored["model_fp"])))
        cheap = scored.loc[scored["adp"].between(*ADP_BAND) & ((scored["adp_rank"] - scored["model_rank"]) >= 5)]
        n_cheap += len(cheap)
        if len(cheap):
            cheap_hits.append(float(((cheap["adp_rank"] - cheap["actual_rank"]) >= 4).mean()))
        print(f"    {season}: ρ {m:.3f} vs ADP {a:.3f}  Δ {m-a:+.3f}")
    return {
        "mean_spearman_lift": float(np.nanmean(lifts)),
        "mean_mae": float(np.nanmean(maes)),
        "cheap_ge5_hit": float(np.nanmean(cheap_hits)) if cheap_hits else float("nan"),
        "cheap_ge5_n": int(n_cheap),
        "yearly_lift": [None if x != x else round(float(x), 4) for x in lifts],
    }


def run_ablations(panel: pd.DataFrame) -> dict:
    import ffmodel.model as M

    out = {}
    print("  ablation baseline")
    out["baseline"] = walk_metrics(panel)
    print("  ablation market_only")
    out["market_only"] = walk_metrics(panel, pred_col="market_fp")
    orig = list(M.FEATURE_COLS)
    for name, drop in DROP_GROUPS.items():
        print(f"  ablation {name}")
        M.FEATURE_COLS = [c for c in orig if c not in drop]
        try:
            out[name] = walk_metrics(panel)
        finally:
            M.FEATURE_COLS = orig
    return out


def _safe_download(tag: str, filename: str):
    try:
        return download_release(tag, filename)
    except Exception as exc:
        print(f"  skip {tag}/{filename}: {exc}")
        return None


def probe_ngs(preds: pd.DataFrame) -> list[dict]:
    rec_path = _safe_download("nextgen_stats", "ngs_receiving.parquet")
    rush_path = _safe_download("nextgen_stats", "ngs_rushing.parquet")
    pass_path = _safe_download("nextgen_stats", "ngs_passing.parquet")
    frames = []
    if rec_path:
        rec = read_parquet(rec_path)
        rec = rec.loc[rec["week"].fillna(-1).eq(0) | rec["week"].fillna(-1).eq(0)]
        if "week" in rec.columns:
            rec = rec.loc[pd.to_numeric(rec["week"], errors="coerce").fillna(-1).eq(0)]
        rec = rec.rename(columns={"player_gsis_id": "player_id"})
        keep = [c for c in ["season", "player_id", "avg_separation", "avg_cushion", "avg_yac_above_expectation", "percent_share_of_intended_air_yards", "catch_percentage"] if c in rec.columns]
        frames.append(("WR/TE NGS", rec[keep]))
    if rush_path:
        rush = read_parquet(rush_path)
        if "week" in rush.columns:
            rush = rush.loc[pd.to_numeric(rush["week"], errors="coerce").fillna(-1).eq(0)]
        rush = rush.rename(columns={"player_gsis_id": "player_id"})
        keep = [c for c in ["season", "player_id", "efficiency", "percent_attempts_gte_eight_defenders", "rush_yards_over_expected_per_att"] if c in rush.columns]
        frames.append(("RB NGS", rush[keep]))
    if pass_path:
        pas = read_parquet(pass_path)
        if "week" in pas.columns:
            pas = pas.loc[pd.to_numeric(pas["week"], errors="coerce").fillna(-1).eq(0)]
        pas = pas.rename(columns={"player_gsis_id": "player_id"})
        keep = [c for c in ["season", "player_id", "completion_percentage_above_expectation", "aggressiveness", "avg_time_to_throw", "avg_intended_air_yards"] if c in pas.columns]
        frames.append(("QB NGS", pas[keep]))
    scored = adp_pool(preds)
    scored["resid"] = scored["ppr_actual"] - scored["market_fp"]
    rows = []
    for label, feat in frames:
        feat = feat.copy()
        feat["season"] = pd.to_numeric(feat["season"], errors="coerce") + 1  # lag
        feat["player_id"] = feat["player_id"].astype(str)
        m = scored.merge(feat, on=["season", "player_id"], how="inner")
        for col in feat.columns:
            if col in {"season", "player_id"}:
                continue
            x = pd.to_numeric(m[col], errors="coerce")
            rows.append(
                {
                    "source": label,
                    "feature": col,
                    "n": int(x.notna().sum()),
                    "rho_resid": _rho(x, m["resid"]),
                    "rho_ppr": _rho(x, m["ppr_actual"]),
                }
            )
    return rows


def probe_weekly_h2(preds: pd.DataFrame) -> list[dict]:
    parts = []
    for season in range(2020, 2026):
        path = _safe_download("stats_player", f"stats_player_week_{season}.parquet")
        if path is None:
            continue
        w = read_parquet(path)
        if "season_type" in w.columns:
            w = w.loc[w["season_type"].astype(str).str.upper().eq("REG")]
        w["week"] = pd.to_numeric(w["week"], errors="coerce")
        pid = w.get("player_id", w.get("gsis_id"))
        w = w.assign(player_id=pid.astype(str), half=np.where(w["week"] >= 10, "h2", "h1"))
        num_cols = [c for c in ["targets", "target_share", "carries", "fantasy_points_ppr", "receptions"] if c in w.columns]
        g = w.groupby(["season", "player_id", "half"], as_index=False)[num_cols].mean(numeric_only=True)
        wide = g.pivot(index=["season", "player_id"], columns="half")
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.reset_index()
        for col in ["targets", "target_share", "carries", "fantasy_points_ppr"]:
            h1, h2 = f"{col}_h1", f"{col}_h2"
            if h1 in wide.columns and h2 in wide.columns:
                wide[f"{col}_h2_delta"] = pd.to_numeric(wide[h2], errors="coerce") - pd.to_numeric(wide[h1], errors="coerce")
        wide["season"] = pd.to_numeric(wide["season"], errors="coerce") + 1
        parts.append(wide)
    if not parts:
        return []
    feat = pd.concat(parts, ignore_index=True)
    scored = adp_pool(preds)
    scored["resid"] = scored["ppr_actual"] - scored["market_fp"]
    m = scored.merge(feat, on=["season", "player_id"], how="inner")
    rows = []
    for col in [c for c in feat.columns if c.endswith("_h2_delta") or c.endswith("_h2")]:
        x = pd.to_numeric(m[col], errors="coerce")
        rows.append({"feature": col, "n": int(x.notna().sum()), "rho_resid": _rho(x, m["resid"]), "rho_ppr": _rho(x, m["ppr_actual"])})
    rows.sort(key=lambda r: abs(r["rho_resid"]) if r["rho_resid"] == r["rho_resid"] else 0, reverse=True)
    return rows


def probe_win_totals(preds: pd.DataFrame, panel: pd.DataFrame) -> dict:
    try:
        wt = load_win_totals()
    except Exception as exc:
        return {"error": str(exc)}
    wt = wt.rename(columns={c: c.lower() for c in wt.columns})
    team_col = "team" if "team" in wt.columns else ("abbr" if "abbr" in wt.columns else None)
    season_col = "season" if "season" in wt.columns else None
    line_col = next((c for c in ["win_total", "wins", "line", "total"] if c in wt.columns), None)
    if not team_col or not season_col or not line_col:
        return {"columns": list(wt.columns)[:20], "n": int(len(wt))}
    slim = wt[[season_col, team_col, line_col]].rename(columns={season_col: "season", team_col: "team", line_col: "win_total"})
    teams = panel[["season", "player_id", "team"]].drop_duplicates(["season", "player_id"])
    d = preds.drop(columns=["team"], errors="ignore").merge(teams, on=["season", "player_id"], how="left")
    d = adp_pool(d).merge(slim, on=["season", "team"], how="left")
    d["resid"] = d["ppr_actual"] - d["market_fp"]
    x = pd.to_numeric(d["win_total"], errors="coerce")
    return {
        "n": int(x.notna().sum()),
        "rho_resid": _rho(x, d["resid"]),
        "rho_ppr": _rho(x, d["ppr_actual"]),
        "rho_market": _rho(x, d["market_fp"]),
    }


def probe_qb_change(preds: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Primary passer last year vs this year's roster team (from already-built panel team + lag pbp)."""
    from ffmodel.ingest.nflverse import load_pbp

    rows = []
    for season in range(2020, 2026):
        try:
            pbp = load_pbp(season, columns=["posteam", "passer_player_id", "pass_attempt", "season_type"])
        except Exception:
            continue
        if "season_type" in pbp.columns:
            pbp = pbp.loc[pbp["season_type"] == "REG"]
        pbp = pbp.loc[pbp["pass_attempt"].fillna(0).eq(1) & pbp["passer_player_id"].notna()]
        top = (
            pbp.groupby(["posteam", "passer_player_id"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .groupby("posteam", as_index=False)
            .head(1)
            .rename(columns={"posteam": "team", "passer_player_id": "qb_id"})
        )
        top["season"] = season
        rows.append(top[["season", "team", "qb_id"]])
    if not rows:
        return {}
    qb = pd.concat(rows, ignore_index=True)
    now = qb.rename(columns={"qb_id": "qb_now"})
    prev = qb.rename(columns={"season": "lag_season", "qb_id": "qb_prev"})
    prev["season"] = prev["lag_season"] + 1
    chg = now.merge(prev[["season", "team", "qb_prev"]], on=["season", "team"], how="left")
    chg["qb_change"] = (chg["qb_now"].notna() & chg["qb_prev"].notna() & (chg["qb_now"] != chg["qb_prev"])).astype(float)
    teams = panel[["season", "player_id", "team", "position"]].drop_duplicates(["season", "player_id"])
    d = adp_pool(preds.drop(columns=["team", "position"], errors="ignore").merge(teams, on=["season", "player_id"], how="left"))
    d = d.merge(chg[["season", "team", "qb_change"]], on=["season", "team"], how="left")
    d["resid"] = d["ppr_actual"] - d["market_fp"]
    out = {"overall": {"n": int(d["qb_change"].notna().sum()), "rho_resid": _rho(d["qb_change"], d["resid"])}}
    for pos, g in d.groupby("position"):
        out[pos] = {"n": int(g["qb_change"].notna().sum()), "rho_resid": _rho(g["qb_change"], g["resid"]), "rate": float(g["qb_change"].mean())}
    return out


def main() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "player_panel.parquet")
    preds = pd.read_csv(PROCESSED_DIR / "backtest_predictions.csv")
    print("Residual correlations (ADP-drafted)...")
    resid = residual_table(panel, preds)
    print("Position splits...")
    by_pos = residual_by_pos(panel, preds)
    print("Flag audit...")
    flags = flag_audit(preds, panel)
    print("NGS probe...")
    ngs = probe_ngs(preds)
    print("Weekly H2 probe...")
    h2 = probe_weekly_h2(preds)
    print("Win totals probe...")
    wt = probe_win_totals(preds, panel)
    print("QB-change probe...")
    qb = probe_qb_change(preds, panel)
    print("Walk-forward ablations (refit, no 2026 write)...")
    ablations = run_ablations(panel)
    payload = {
        "residual_adp_pool": resid[:40],
        "residual_by_pos": by_pos,
        "flags": flags,
        "ngs": ngs,
        "weekly_h2": h2[:20],
        "win_totals": wt,
        "qb_change": qb,
        "ablations": ablations,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
