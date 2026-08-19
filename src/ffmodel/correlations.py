from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import BACKTEST_SEASONS, PREDICT_SEASON, PROCESSED_DIR

# The extra edges requested for the model, mapped onto panel columns.
EDGE_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Vegas season totals",
        [
            ("vfp", "VFP (props → PPR)"),
            ("v_pass_yd", "Pass yards line"),
            ("v_pass_td", "Pass TD line"),
            ("v_rush_yd", "Rush yards line"),
            ("v_rush_td", "Rush TD line"),
            ("v_rec_yd", "Rec yards line"),
            ("v_rec", "Receptions line"),
            ("v_rec_td", "Rec TD line"),
            ("v_rush_att", "Rush attempts line"),
            ("v_pass_att", "Pass attempts line"),
            ("market_fp", "ADP-implied PPR"),
            ("adp", "ADP (lower = better)"),
        ],
    ),
    (
        "Offensive line",
        [
            ("oline_index", "O-line index (higher = better)"),
            ("sack_rate", "Sack rate allowed"),
        ],
    ),
    (
        "Age curve",
        [
            ("age", "Age on Sept 1"),
            ("age_alpha", "Age-curve multiplier"),
            ("years_exp", "Years of experience"),
        ],
    ),
    (
        "Physical / combine",
        [
            ("forty", "40-yard dash"),
            ("speed_score", "Speed score"),
            ("burst", "Burst (vert + broad)"),
            ("bmi", "BMI"),
            ("draft_pick_num", "Draft pick (lower = better)"),
            ("log_draft_pick", "Log draft pick"),
        ],
    ),
    (
        "Injury history",
        [
            ("injury_weeks", "Injury weeks (prior year)"),
            ("injury_weeks_3yr", "Injury weeks (3-year)"),
            ("major_injury", "Major injury flag (prior year)"),
            ("major_injury_3yr", "Major injury flag (3-year)"),
        ],
    ),
    (
        "OC / coaching tendency",
        [
            ("coaching_C", "Coaching multiplier C"),
            ("proe", "Pass rate over expected"),
            ("pace_neutral", "Neutral-script pace"),
            ("pass_rate", "Team pass rate"),
            ("new_hc", "New head coach"),
        ],
    ),
    (
        "Chunk / big plays",
        [
            ("explosive_rate", "Explosive play rate"),
            ("chunk_rating", "Chunk rating (rate × EPA)"),
            ("receiving_20_lag", "20+ yard receptions"),
            ("rushing_20_lag", "20+ yard rushes"),
            ("receiving_40_lag", "40+ yard receptions"),
        ],
    ),
    (
        "Red zone",
        [
            ("hv_rz", "High-value RZ index"),
            ("hv_rz_per_touch", "HV-RZ per touch"),
            ("inside5_carries", "Carries inside the 5"),
            ("inside10_targets", "Targets inside the 10"),
            ("ez_targets", "End-zone targets"),
        ],
    ),
    (
        "Strength of schedule",
        [
            ("sos_def_pass_epa", "Opp. pass EPA allowed"),
            ("sos_def_rush_epa", "Opp. rush EPA allowed"),
            ("sos_def_rec_epa", "Opp. rec EPA allowed"),
            ("pct_indoor", "Share of indoor games"),
            ("avg_total", "Avg opponent game total"),
            ("avg_spread_for", "Avg spread (positive = favored)"),
        ],
    ),
    (
        "Usage / production share",
        [
            ("target_share_lag", "Target share"),
            ("wopr_lag", "WOPR"),
            ("off_snap_pct", "Offensive snap %"),
            ("vacated_target_share", "Vacated target share"),
            ("vacated_carry_share", "Vacated carry share"),
            ("player_vacated_boost", "Player-assigned vacated usage"),
            ("depth_rank", "Depth-chart rank (1 = starter)"),
            ("ppr_lag", "Prior-year PPR"),
            ("usage_index", "Usage index"),
            ("pass_catch_rb", "RB target share"),
            ("role_expand", "New/expanded starter"),
        ],
    ),
    (
        "Luck / regression / aging workload",
        [
            ("td_luck", "TD luck vs RZ opportunity"),
            ("overproduction", "PPG vs usage residual"),
            ("eff_index", "Efficiency vs position (high = regression)"),
            ("workload_cliff", "Aging RB workload cliff"),
            ("injury_bounce", "Injury bounce-back window"),
            ("breakout_window", "Positional breakout window"),
            ("chronic_injury", "Chronic injury flag"),
        ],
    ),
]


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 20 or x.loc[mask].nunique() < 2 or y.loc[mask].nunique() < 2:
        return (np.nan, n)
    rho = spearmanr(x.loc[mask], y.loc[mask], nan_policy="omit").statistic
    return (float(rho) if rho == rho else np.nan, n)


def edge_correlation_table(panel: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Each extra-edge feature vs holdout actuals and vs 2026 model projection."""
    test = panel.loc[panel["season"].isin(BACKTEST_SEASONS) & panel["ppr_actual"].notna()].copy()
    proj = panel.loc[panel["season"] == PREDICT_SEASON].merge(
        rankings[["player_id", "model_fp"]],
        on="player_id",
        how="inner",
    )
    rows = []
    for group, feats in EDGE_GROUPS:
        for col, label in feats:
            rec = {"group": group, "feature": col, "label": label}
            if col in test.columns:
                rho, n = _spearman(pd.to_numeric(test[col], errors="coerce"), test["ppr_actual"])
                rec["corr_test_actual"] = rho
                rec["n_test"] = n
            else:
                rec["corr_test_actual"] = np.nan
                rec["n_test"] = 0
            if col in proj.columns:
                rho, n = _spearman(pd.to_numeric(proj[col], errors="coerce"), proj["model_fp"])
                rec["corr_projection"] = rho
                rec["n_proj"] = n
            else:
                rec["corr_projection"] = np.nan
                rec["n_proj"] = 0
            rows.append(rec)
    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_DIR / "edge_correlations.csv", index=False)
    return out


def edge_cross_correlation(panel: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman among representative extras + projection/test targets."""
    reps = [
        ("vfp", "VFP"),
        ("oline_index", "O-line"),
        ("age_alpha", "Age curve"),
        ("speed_score", "Speed score"),
        ("injury_weeks_3yr", "Injury 3yr"),
        ("coaching_C", "Coaching C"),
        ("explosive_rate", "Chunk rate"),
        ("hv_rz_per_touch", "Red zone"),
        ("pct_indoor", "Indoor SOS"),
        ("avg_total", "Game totals"),
        ("target_share_lag", "Target share"),
        ("depth_rank", "Depth rank"),
        ("off_snap_pct", "Snap %"),
    ]
    test = panel.loc[panel["season"].isin(BACKTEST_SEASONS) & panel["ppr_actual"].notna()].copy()
    proj = panel.loc[panel["season"] == PREDICT_SEASON].merge(
        rankings[["player_id", "model_fp"]],
        on="player_id",
        how="inner",
    )
    # Cross among extras on the historical test panel (has actuals).
    cols = [c for c, _ in reps if c in test.columns]
    labels = {c: lab for c, lab in reps if c in test.columns}
    mat_rows = []
    names = cols + ["ppr_actual"]
    label_names = [labels.get(c, c) for c in cols] + ["Test actual PPR"]
    frame = test[cols].copy()
    frame["ppr_actual"] = test["ppr_actual"]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            rho, n = _spearman(pd.to_numeric(frame[a], errors="coerce"), pd.to_numeric(frame[b], errors="coerce"))
            mat_rows.append(
                {
                    "a": label_names[i],
                    "b": label_names[j],
                    "feature_a": a,
                    "feature_b": b,
                    "spearman": rho,
                    "n": n,
                    "scope": "test_panel",
                }
            )
    # Projection column against the same extras on 2026.
    for col, lab in reps:
        if col not in proj.columns:
            continue
        rho, n = _spearman(pd.to_numeric(proj[col], errors="coerce"), proj["model_fp"])
        mat_rows.append(
            {
                "a": lab,
                "b": "2026 projection",
                "feature_a": col,
                "feature_b": "model_fp",
                "spearman": rho,
                "n": n,
                "scope": "proj_2026",
            }
        )
    out = pd.DataFrame(mat_rows)
    out.to_csv(PROCESSED_DIR / "edge_cross_correlations.csv", index=False)
    return out
