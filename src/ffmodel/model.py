from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import (
    BACKTEST_SEASONS,
    MIN_TRAIN_GAMES,
    MIN_TRAIN_PPR,
    PREDICT_SEASON,
    PROCESSED_DIR,
    ROUND_SIZE,
    STEAL_ADP_MAX,
    STEAL_ADP_MIN,
    STEAL_POINT_EDGE,
    STEAL_RANK_LIFT,
)

FEATURE_COLS = [
    "years_exp",
    "age",
    "age_alpha",
    "games_lag",
    "games_pct_lag",
    "availability",
    "ppr_lag",
    "ppr_ppg_lag",
    "attempts_lag",
    "passing_yards_lag",
    "passing_tds_lag",
    "passing_epa_lag",
    "passing_cpoe_lag",
    "pacr_lag",
    "carries_lag",
    "rushing_yards_lag",
    "rushing_tds_lag",
    "rushing_epa_lag",
    "rushing_20_lag",
    "targets_lag",
    "receptions_lag",
    "receiving_yards_lag",
    "receiving_tds_lag",
    "receiving_epa_lag",
    "target_share_lag",
    "air_yards_share_lag",
    "wopr_lag",
    "receiving_20_lag",
    "receiving_40_lag",
    "explosive_rate",
    "chunk_rating",
    "chunk_index",
    "hv_rz",
    "hv_rz_per_touch",
    "rz_index",
    "inside5_carries",
    "inside10_targets",
    "ez_targets",
    "player_vacated_targets",
    "player_vacated_carries",
    "player_vacated_boost",
    "td_luck",
    "rec_td_luck",
    "rush_td_luck",
    "catch_rate_lag",
    "ypc_lag",
    "ypr_lag",
    "carry_load",
    "carry_share_lag",
    "pass_catch_rb",
    "qb_rush_share",
    "workload_cliff",
    "breakout_window",
    "sophomore_leap",
    "team_change",
    "new_hc_pass_catcher",
    "pos_competition",
    "starter",
    "role_expand",
    "injury_bounce",
    "chronic_injury",
    "overproduction",
    "eff_index",
    "eff_ypc_z",
    "eff_ypr_z",
    "eff_catch_z",
    "young_capital",
    "new_starter_vacated",
    "usage_index",
    "draft_capital",
    "off_snap_pct",
    "forty",
    "speed_score",
    "burst",
    "bmi",
    "draft_pick_num",
    "log_draft_pick",
    "undrafted",
    "depth_rank",
    "production_proxy",
    "situation_proxy",
    "physical_proxy",
    "blend_proj",
    # Pre-shrunk team context so they cannot dominate usage.
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
]

POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
TENURE_CODE = {"rookie": 0, "sophomore": 1, "developing": 2, "prime": 3, "veteran": 4}
LAM_GRID = (0.25, 0.40, 0.55, 0.75, 1.00)
K_GRID = (0.0, 1.5)


def available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def _encode_frame(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=feats).copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["position_code"] = df["position"].map(POS_CODE)
    x["tenure_code"] = df["tenure_bucket"].map(TENURE_CODE)
    return x


def eligible(df: pd.DataFrame, for_train: bool = True) -> pd.DataFrame:
    out = df.copy()
    if for_train:
        out = out.loc[out["ppr_actual"].notna()]
        enough = (
            out["games_lag"].fillna(0).ge(MIN_TRAIN_GAMES)
            | out["ppr_lag"].fillna(0).ge(MIN_TRAIN_PPR)
            | out["adp"].notna()
            | out["years_exp"].fillna(99).le(1)
        )
        out = out.loc[enough]
    else:
        out = out.loc[out["position"].isin(["QB", "RB", "WR", "TE"])]
        out = out.loc[out["adp"].notna() | out["ppr_lag"].fillna(0).ge(20) | out["years_exp"].fillna(99).le(1)]
    return out


def market_base(df: pd.DataFrame) -> pd.Series:
    base = df["market_fp"] if "market_fp" in df.columns else pd.Series(np.nan, index=df.index)
    if "vfp" in df.columns:
        markets = df["vfp_markets"] if "vfp_markets" in df.columns else 0
        use_vfp = df["vfp"].notna() & pd.to_numeric(markets, errors="coerce").fillna(0).ge(1)
        base = pd.Series(np.where(use_vfp, df["vfp"], base), index=df.index)
    blend = df["blend_proj"] if "blend_proj" in df.columns else pd.Series(np.nan, index=df.index)
    lag = df["ppr_lag"] if "ppr_lag" in df.columns else pd.Series(np.nan, index=df.index)
    return pd.Series(base, index=df.index).fillna(blend).fillna(lag).fillna(80.0)


def _gbdt(**kwargs) -> HistGradientBoostingRegressor:
    params = dict(
        max_depth=3,
        learning_rate=0.05,
        max_iter=280,
        l2_regularization=0.55,
        min_samples_leaf=22,
        random_state=7,
    )
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def _sample_weights(df: pd.DataFrame) -> np.ndarray:
    adp = pd.to_numeric(df.get("adp"), errors="coerce")
    w = np.ones(len(df))
    w = np.where(adp.notna() & adp.le(150), 4.0, w)
    w = np.where(adp.notna() & adp.le(72), 5.5, w)
    w = np.where(adp.notna() & adp.between(STEAL_ADP_MIN, STEAL_ADP_MAX), w * 1.2, w)
    return w


def _fit_estimators(train: pd.DataFrame) -> dict:
    feats = available_features(train)
    x = _encode_frame(train, feats)
    w = _sample_weights(train)
    points = _gbdt()
    points.fit(x, train["ppr_actual"].astype(float), sample_weight=w)
    points._feature_names = list(x.columns)  # type: ignore[attr-defined]

    t = train.loc[train["market_fp"].notna() & train["ppr_actual"].notna()].copy()
    resid = _gbdt(max_iter=220)
    xr = _encode_frame(t, feats)
    yr = (t["ppr_actual"] - t["market_fp"]).astype(float)
    resid.fit(xr, yr, sample_weight=_sample_weights(t))
    resid._feature_names = list(xr.columns)  # type: ignore[attr-defined]

    pos_resid: dict = {}
    for pos, grp in t.groupby("position"):
        if len(grp) < 120:
            continue
        m = _gbdt(max_iter=180, max_depth=3, min_samples_leaf=18)
        xp = _encode_frame(grp, feats)
        yp = (grp["ppr_actual"] - grp["market_fp"]).astype(float)
        m.fit(xp, yp, sample_weight=_sample_weights(grp))
        m._feature_names = list(xp.columns)  # type: ignore[attr-defined]
        pos_resid[pos] = m

    rank_m = _gbdt(max_iter=200, max_depth=2)
    r = t.loc[t["adp"].notna()].copy()
    if len(r) >= 80:
        r["adp_rank"] = r.groupby(["season", "position"])["adp"].rank(method="min")
        r["act_rank"] = r.groupby(["season", "position"])["ppr_actual"].rank(ascending=False, method="min")
        y_lift = (r["adp_rank"] - r["act_rank"]).astype(float)
        xk = _encode_frame(r, feats)
        rank_m.fit(xk, y_lift, sample_weight=_sample_weights(r))
        rank_m._feature_names = list(xk.columns)  # type: ignore[attr-defined]
    else:
        rank_m = None
    return {"points": points, "resid": resid, "pos_resid": pos_resid, "rank": rank_m, "feats": feats}


def fit_models(train: pd.DataFrame) -> dict:
    models = _fit_estimators(train)
    lam, k = _tune_blend(train)
    models["lam"] = lam
    models["k"] = k
    return models


def _predict_resid(models: dict, df: pd.DataFrame, x: pd.DataFrame) -> np.ndarray:
    xr = x.reindex(columns=list(models["resid"]._feature_names))
    global_resid = np.clip(models["resid"].predict(xr), -60, 60)
    out = pd.Series(global_resid, index=df.index)
    pos_resid = models.get("pos_resid") or {}
    for pos, m in pos_resid.items():
        mask = df["position"].eq(pos)
        if not mask.any():
            continue
        xp = x.loc[mask].reindex(columns=list(m._feature_names))
        local = np.clip(m.predict(xp), -60, 60)
        out.loc[mask] = 0.55 * local + 0.45 * out.loc[mask].to_numpy()
    return out.to_numpy()


def _predict_components(models: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = list(models["points"]._feature_names)
    x = _encode_frame(df, models["feats"]).reindex(columns=cols)
    points = models["points"].predict(x)
    resid = _predict_resid(models, df, x)
    if models["rank"] is not None:
        xr = x.reindex(columns=list(models["rank"]._feature_names))
        lift = np.clip(models["rank"].predict(xr), -18, 18)
    else:
        lift = np.zeros(len(df))
    base = market_base(df).to_numpy()
    return base, points, resid, lift


def combine_prediction(base, points, resid, lift, lam, k, df: pd.DataFrame) -> np.ndarray:
    has_market = df["market_fp"].notna().to_numpy() if "market_fp" in df.columns else np.zeros(len(df), dtype=bool)
    years = pd.to_numeric(df["years_exp"], errors="coerce").fillna(2).to_numpy()
    blend = df["blend_proj"].fillna(pd.Series(points, index=df.index)).to_numpy()
    tenure_lam = np.where(years <= 0, lam * 1.15, np.where(years <= 2, lam * 1.05, np.where(years >= 7, lam * 0.85, lam)))
    mixed = np.where(
        has_market,
        base + tenure_lam * resid + np.clip(k * lift, -22.0, 22.0),
        0.6 * points + 0.4 * blend,
    )
    rookie_w = np.where(years <= 0, 0.40, np.where(years <= 1, 0.18, 0.05))
    # Historical backtests have no player props. When a 2026 rookie *does* have VFP,
    # do not drag 40% off the books toward a physical prior.
    if "vfp" in df.columns:
        vfp_ok = df["vfp"].notna().to_numpy()
        if "vfp_markets" in df.columns:
            vfp_ok = vfp_ok & pd.to_numeric(df["vfp_markets"], errors="coerce").fillna(0).ge(1).to_numpy()
        rookie_w = np.where(vfp_ok, np.minimum(rookie_w, 0.08), rookie_w)
    return (1 - rookie_w) * mixed + rookie_w * blend


def _spearman_adp(df: pd.DataFrame, pred: np.ndarray) -> float:
    scored = df.loc[df["adp"].notna()].copy()
    if len(scored) < 25:
        return np.nan
    yhat = pd.Series(pred, index=df.index).loc[scored.index]
    rho = spearmanr(yhat, scored["ppr_actual"], nan_policy="omit").statistic
    return float(rho) if rho == rho else np.nan


def _tune_blend(train: pd.DataFrame) -> tuple[float, float]:
    val_year = int(train["season"].max())
    val = train.loc[train["season"] == val_year]
    inner = train.loc[train["season"] < val_year]
    if val["adp"].notna().sum() < 40 or inner.empty:
        return 0.45, 0.0
    inner_models = _fit_estimators(inner)
    base, pts, res, lift = _predict_components(inner_models, val)
    best = (-1.0, 0.40, 0.0)
    for lam in LAM_GRID:
        for k in K_GRID:
            pred = combine_prediction(base, pts, res, lift, lam, k, val)
            rho = _spearman_adp(val, pred)
            if pd.notna(rho) and rho > best[0]:
                best = (rho, lam, k)
    return best[1], best[2]


def predict_models(models: dict, df: pd.DataFrame) -> np.ndarray:
    base, points, resid, lift = _predict_components(models, df)
    return combine_prediction(base, points, resid, lift, models["lam"], models["k"], df)


@dataclass
class BacktestResult:
    by_season: pd.DataFrame
    predictions: pd.DataFrame
    extra_corrs: pd.DataFrame
    steal_eval: dict


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(0)


def _flag_steals(df: pd.DataFrame) -> pd.Series:
    adp = pd.to_numeric(df["adp"], errors="coerce")
    band = adp.between(STEAL_ADP_MIN, STEAL_ADP_MAX)
    lift = df["adp_rank"] - df["model_rank"]
    edge = df["model_fp"] - df["market_fp"]
    steal_why = (
        _col(df, "breakout_window").gt(0)
        | _col(df, "injury_bounce").gt(0)
        | _col(df, "role_expand").gt(0)
        | _col(df, "sophomore_leap").gt(0)
        | _col(df, "player_vacated_boost").ge(8)
        | _col(df, "usage_index").ge(1.25)
        | _col(df, "pass_catch_rb").ge(0.08)
        | _col(df, "td_luck").le(-1.0)
        | _col(df, "new_starter_vacated").ge(6)
    )
    fade_why = (
        _col(df, "workload_cliff").gt(0)
        | _col(df, "chronic_injury").gt(0)
        | _col(df, "td_luck").ge(1.5)
        | _col(df, "overproduction").ge(0.55)
        | _col(df, "age_alpha").le(0.85)
        | _col(df, "eff_index").ge(1.4)
    )
    steal = band & (lift >= STEAL_RANK_LIFT) & (edge >= STEAL_POINT_EDGE) & steal_why
    fade = band & (lift <= -STEAL_RANK_LIFT) & (edge <= -STEAL_POINT_EDGE) & fade_why
    # Round-1/2 fades must actually be below the ADP points curve, except aging
    # workhorses where the rank drop is the story even if points are close.
    cliff = _col(df, "workload_cliff").gt(0)
    early = (
        adp.lt(STEAL_ADP_MIN)
        & adp.ge(1)
        & (lift <= -4)
        & fade_why
        & ((edge <= -10) | (cliff & edge.le(5)))
    )
    fade = fade | early
    return np.where(steal, "steal", np.where(fade, "fade", "fair"))


def run_backtest(panel: pd.DataFrame) -> BacktestResult:
    preds = []
    rows = []
    for season in BACKTEST_SEASONS:
        train = eligible(panel.loc[panel["season"] < season])
        test = eligible(panel.loc[panel["season"] == season])
        if train.empty or test.empty:
            continue
        models = fit_models(train)
        test = test.copy()
        test["model_fp"] = predict_models(models, test)
        test["lam"] = models["lam"]
        test["k"] = models["k"]
        test["error"] = test["model_fp"] - test["ppr_actual"]
        test["market_error"] = test["market_fp"] - test["ppr_actual"]
        test["residual_vs_market"] = test["ppr_actual"] - test["market_fp"]
        test["model_rank"] = test.groupby("position")["model_fp"].rank(ascending=False, method="min")
        test["adp_rank"] = test.groupby("position")["adp"].rank(method="min")
        test["actual_rank"] = test.groupby("position")["ppr_actual"].rank(ascending=False, method="min")
        test["steal_label"] = _flag_steals(test)
        test["steal_flag"] = test["steal_label"].eq("steal")
        scored = test.loc[test["adp"].notna()]
        mae = mean_absolute_error(scored["ppr_actual"], scored["model_fp"]) if len(scored) else np.nan
        rmse = mean_squared_error(scored["ppr_actual"], scored["model_fp"]) ** 0.5 if len(scored) else np.nan
        spear = _spearman_adp(test, test["model_fp"].to_numpy())
        market_spear = (
            spearmanr(scored["market_fp"], scored["ppr_actual"], nan_policy="omit").statistic if len(scored) > 10 else np.nan
        )
        lift_actual = scored["adp_rank"] - scored["actual_rank"]
        steal_rho = spearmanr(scored["adp_rank"] - scored["model_rank"], lift_actual, nan_policy="omit").statistic if len(scored) > 10 else np.nan
        rows.append(
            {
                "season": season,
                "n_adp": int(len(scored)),
                "lam": models["lam"],
                "k": models["k"],
                "mae": mae,
                "rmse": rmse,
                "spearman": spear,
                "market_spearman": market_spear,
                "spearman_lift": spear - market_spear if pd.notna(spear) and pd.notna(market_spear) else np.nan,
                "steal_dir_spearman": steal_rho,
                "mae_vs_market": mean_absolute_error(scored["ppr_actual"], scored["market_fp"]) if len(scored) else np.nan,
            }
        )
        preds.append(test)
        print(f"  backtest {season}: model {spear:.3f} vs ADP {market_spear:.3f} (λ={models['lam']}, k={models['k']})")
    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    extra = extra_stat_correlations(predictions if not predictions.empty else panel)
    steal_eval = evaluate_steals(predictions)
    by_season = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    by_season.to_csv(PROCESSED_DIR / "backtest_by_season.csv", index=False)
    extra.to_csv(PROCESSED_DIR / "extra_correlations.csv", index=False)
    (PROCESSED_DIR / "steal_eval.json").write_text(json.dumps(steal_eval, indent=2, default=float))
    if not predictions.empty:
        slim = [c for c in [
            "season", "player_id", "player_name", "position", "team", "adp",
            "market_fp", "model_fp", "ppr_actual", "model_rank", "adp_rank",
            "actual_rank", "steal_label", "error", "market_error",
        ] if c in predictions.columns]
        predictions[slim].to_csv(PROCESSED_DIR / "backtest_predictions.csv", index=False)
    return BacktestResult(by_season, predictions, extra, steal_eval)


def extra_stat_correlations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ppr_actual" not in df.columns:
        return pd.DataFrame()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    skip = {
        "ppr_actual",
        "games_actual",
        "ppg_actual",
        "model_fp",
        "error",
        "market_error",
        "residual_vs_market",
        "model_rank",
        "actual_rank",
        "season",
        "lam",
        "k",
    }
    rows = []
    target = df["ppr_actual"]
    beat_market = df["ppr_actual"] - df["market_fp"] if "market_fp" in df.columns else None
    for col in numeric:
        if col in skip:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() < 80 or x.nunique(dropna=True) < 2:
            continue
        sp = spearmanr(x, target, nan_policy="omit").statistic
        if pd.isna(sp):
            continue
        rec = {"feature": col, "corr_ppr": sp, "n": int(x.notna().sum())}
        if beat_market is not None:
            rec["corr_beat_market"] = spearmanr(x, beat_market, nan_policy="omit").statistic
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("corr_ppr", key=lambda s: s.abs(), ascending=False)


def evaluate_steals(preds: pd.DataFrame) -> dict:
    if preds.empty:
        return {}
    drafted = preds.loc[preds["adp"].notna()]
    flagged = preds.loc[preds.get("steal_label", "") == "steal"]
    fades = preds.loc[preds.get("steal_label", "") == "fade"]
    out: dict = {"n_steals": int(len(flagged)), "n_fades": int(len(fades))}
    if len(drafted) > 20:
        lift = drafted["adp_rank"] - drafted["actual_rank"]
        out["steal_dir_spearman"] = float(
            spearmanr(drafted["adp_rank"] - drafted["model_rank"], lift, nan_policy="omit").statistic or 0
        )
    if not flagged.empty:
        hit = (flagged["adp_rank"] - flagged["actual_rank"]) >= 4
        out["steal_hit_rate"] = float(hit.mean())
        out["steal_mean_rank_gain"] = float((flagged["adp_rank"] - flagged["actual_rank"]).mean())
        out["steal_mean_adp"] = float(flagged["adp"].mean())
    if not fades.empty:
        fade_hit = (fades["actual_rank"] - fades["adp_rank"]) >= 4
        out["fade_hit_rate"] = float(fade_hit.mean())
        out["fade_mean_rank_loss"] = float((fades["actual_rank"] - fades["adp_rank"]).mean())
    return out


def round_value(adp_or_proj_pick: float) -> str:
    """12-team slot: early/mid/late within the round (e.g. 'mid 3rd')."""
    if pd.isna(adp_or_proj_pick):
        return "undrafted"
    pick = float(adp_or_proj_pick)
    if pick < 1:
        return "undrafted"
    if pick > ROUND_SIZE * 16:
        return "undrafted"
    idx = pick - 1.0
    rnd = int(idx // ROUND_SIZE) + 1
    pos = idx % ROUND_SIZE
    third = "early" if pos < 4 else ("mid" if pos < 8 else "late")
    return f"{third} {_ordinal(rnd)}"


def _ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def apply_board_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute position ranks, 12-team round slots, and steal/fade flags."""
    out = df.copy()
    out["model_rank_pos"] = out.groupby("position")["model_fp"].rank(ascending=False, method="min")
    out["adp_rank_pos"] = out.groupby("position")["adp"].rank(method="min")
    out["model_rank_ov"] = out["model_fp"].rank(ascending=False, method="min")
    out["implied_pick"] = np.nan
    for _, grp in out.groupby("position"):
        adp_sorted = grp["adp"].dropna().sort_values().to_numpy()
        if len(adp_sorted) == 0:
            continue
        picks = [
            adp_sorted[min(max(int(rank) - 1, 0), len(adp_sorted) - 1)]
            for rank in grp["model_rank_pos"].fillna(len(adp_sorted))
        ]
        out.loc[grp.index, "implied_pick"] = picks
    out["implied_pick"] = out["implied_pick"].fillna(out["adp"]).fillna(out["model_rank_ov"])
    out["round_value"] = out["implied_pick"].map(round_value)
    out["adp_round_value"] = out["adp"].map(round_value)
    out["steal_score"] = (out["adp_rank_pos"] - out["model_rank_pos"]).fillna(0)
    if "market_fp" in out.columns:
        out["value_over_adp"] = out["model_fp"] - out["market_fp"]
    tmp = out.rename(columns={"model_rank_pos": "model_rank", "adp_rank_pos": "adp_rank"})
    out["steal_label"] = _flag_steals(tmp)
    return out


def predict_season(panel: pd.DataFrame, season: int = PREDICT_SEASON) -> pd.DataFrame:
    train = eligible(panel.loc[panel["season"] < season])
    test = eligible(panel.loc[panel["season"] == season], for_train=False)
    models = fit_models(train)
    print(f"  2026 blend λ={models['lam']} k={models['k']}")
    out = test.copy()
    out["model_fp"] = predict_models(models, out)
    rec = pd.to_numeric(out["v_rec"], errors="coerce") if "v_rec" in out.columns else pd.Series(np.nan, index=out.index)
    if "receptions_lag" in out.columns:
        rec = rec.fillna(pd.to_numeric(out["receptions_lag"], errors="coerce"))
    out["rec_proj"] = rec.fillna(0).clip(lower=0)
    out = apply_board_ranks(out)
    for col, name in [
        ("usage_index", "z_usage"),
        ("chunk_index", "z_explosive"),
        ("rz_index", "z_redzone"),
        ("player_vacated_boost", "z_vacated"),
        ("td_luck", "z_td_luck"),
        ("age_alpha", "z_age"),
    ]:
        if col in out.columns:
            mu, sd = out[col].mean(), out[col].std(ddof=0)
            out[name] = (out[col] - mu) / sd if sd and sd > 0 else 0.0
    keep = [
        "season",
        "player_id",
        "player_name",
        "position",
        "team",
        "years_exp",
        "tenure_bucket",
        "age",
        "adp",
        "market_fp",
        "vfp",
        "vfp_markets",
        "model_fp",
        "rec_proj",
        "blend_proj",
        "model_rank_pos",
        "adp_rank_pos",
        "model_rank_ov",
        "implied_pick",
        "round_value",
        "adp_round_value",
        "steal_score",
        "value_over_adp",
        "steal_label",
        "usage_index",
        "player_vacated_boost",
        "td_luck",
        "overproduction",
        "injury_bounce",
        "role_expand",
        "pass_catch_rb",
        "workload_cliff",
        "breakout_window",
        "explosive_rate",
        "chunk_rating",
        "hv_rz",
        "depth_rank",
        "ppr_lag",
        "age_alpha",
        "z_usage",
        "z_explosive",
        "z_redzone",
        "z_vacated",
        "z_td_luck",
        "z_age",
    ]
    keep = [c for c in keep if c in out.columns]
    ranked = out[keep].sort_values(["position", "model_rank_pos"])
    ranked.to_csv(PROCESSED_DIR / f"rankings_{season}.csv", index=False)
    for pos in ["QB", "RB", "WR", "TE"]:
        ranked.loc[ranked["position"] == pos].to_csv(PROCESSED_DIR / f"rankings_{season}_{pos}.csv", index=False)
    return ranked
