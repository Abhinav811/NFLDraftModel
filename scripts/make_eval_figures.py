#!/usr/bin/env python3
"""Walk-forward evaluation charts for the WordPress methods guide."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ffmodel.config import PROCESSED_DIR
from ffmodel.model import run_backtest

FIG_DIR = ROOT / "docs" / "figures"
COPY_DIR = ROOT / "wordpress_figures"
INK = "#1a1a1a"
MUTED = "#5c574e"
ACCENT = "#1d4f46"
STEAL = "#0f6b4c"
FADE = "#9b2c2c"
CARD = "#fffdf8"
BG = "#f7f4ee"
LINE = "#d9d2c5"
ADP = "#8a6a3a"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": CARD,
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "600",
            "axes.labelsize": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.grid": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": BG,
            "savefig.dpi": 160,
        }
    )


def _caption(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=10)


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    COPY_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=160)
    fig.savefig(COPY_DIR / name, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def load_or_backtest() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred_path = PROCESSED_DIR / "backtest_predictions.csv"
    season_path = PROCESSED_DIR / "backtest_by_season.csv"
    steal_path = PROCESSED_DIR / "steal_eval.json"
    if pred_path.exists() and season_path.exists():
        print(f"Reusing {pred_path}")
        preds = pd.read_csv(pred_path)
        by_season = pd.read_csv(season_path)
        steal = json.loads(steal_path.read_text()) if steal_path.exists() else {}
        return preds, by_season, steal
    panel = pd.read_parquet(PROCESSED_DIR / "player_panel.parquet")
    print("Running walk-forward backtest (a few minutes)...")
    bt = run_backtest(panel)
    preds = pd.read_csv(PROCESSED_DIR / "backtest_predictions.csv")
    return preds, bt.by_season, bt.steal_eval


def drafted(df: pd.DataFrame) -> pd.DataFrame:
    """ADP-drafted players, re-ranked inside that pool (the actual draft decision set)."""
    out = df.loc[df["adp"].notna() & df["ppr_actual"].notna() & df["model_fp"].notna()].copy()
    keys = ["season", "position"]
    out["model_rank"] = out.groupby(keys)["model_fp"].rank(ascending=False, method="min")
    out["adp_rank"] = out.groupby(keys)["adp"].rank(method="min")
    out["actual_rank"] = out.groupby(keys)["ppr_actual"].rank(ascending=False, method="min")
    out["pred_lift"] = out["adp_rank"] - out["model_rank"]
    out["act_lift"] = out["adp_rank"] - out["actual_rank"]
    return out


def fig_spearman(by_season: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(by_season))
    w = 0.36
    ax.bar(x - w / 2, by_season["market_spearman"], w, color=ADP, label="ADP-implied points")
    ax.bar(x + w / 2, by_season["spearman"], w, color=ACCENT, label="Model")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in by_season["season"]])
    ax.set_ylabel("Spearman ρ on ADP-drafted players")
    ax.set_ylim(0.45, 0.70)
    ax.legend(frameon=False, loc="upper right")
    for i, r in enumerate(by_season.itertuples()):
        ax.text(i + w / 2, r.spearman + 0.006, f"Δ{r.spearman_lift:+.3f}", ha="center", fontsize=8, color=STEAL)
    ax.spines[["top", "right"]].set_visible(False)
    _caption(ax, "Model vs ADP: ranking accuracy by season")
    fig.text(0.01, -0.02, "Higher is better. Δ is model ρ minus ADP ρ. Only players who had an ADP that year.", fontsize=8, color=MUTED)
    _save(fig, "01_spearman_vs_adp.png")


def fig_rank_vs_actual(preds: pd.DataFrame) -> None:
    d = drafted(preds)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharex=True, sharey=True)
    panels = [
        (axes[0], d["adp_rank"], "ADP position rank", ADP),
        (axes[1], d["model_rank"], "Model position rank", ACCENT),
    ]
    for ax, x, title, color in panels:
        hb = ax.hexbin(x, d["actual_rank"], gridsize=22, cmap="YlGnBu", mincnt=1, linewidths=0)
        lim = max(x.max(), d["actual_rank"].max())
        ax.plot([1, lim], [1, lim], color=INK, lw=0.8, alpha=0.45)
        ax.set_xlabel(title)
        ax.set_xlim(0.5, 48)
        ax.set_ylim(48, 0.5)
        ax.set_title(title.split()[0] + " vs what actually happened", loc="left", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        rho = spearmanr(x, d["actual_rank"], nan_policy="omit").statistic
        ax.text(0.98, 0.05, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="right", fontsize=10, color=color)
    axes[0].set_ylabel("Actual position rank (1 = most PPR)")
    fig.colorbar(hb, ax=axes, fraction=0.03, pad=0.02, label="Players")
    fig.suptitle("Did we put players in the right order?  (2021–2025, ADP-drafted)", x=0.02, ha="left", fontsize=13, fontweight="600")
    fig.subplots_adjust(top=0.82)
    _save(fig, "02_rank_vs_actual.png")


def fig_lift_heatmap(preds: pd.DataFrame) -> None:
    d = drafted(preds)
    rows = []
    for (season, pos), g in d.groupby(["season", "position"]):
        if len(g) < 12:
            continue
        m = spearmanr(g["model_fp"], g["ppr_actual"], nan_policy="omit").statistic
        a = spearmanr(g["market_fp"], g["ppr_actual"], nan_policy="omit").statistic
        rows.append({"season": int(season), "position": pos, "lift": m - a, "n": len(g)})
    heat = pd.DataFrame(rows)
    order = ["QB", "RB", "WR", "TE"]
    years = sorted(heat["season"].unique())
    mat = heat.pivot(index="position", columns="season", values="lift").reindex(order)[years]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    from matplotlib.colors import TwoSlopeNorm

    im = ax.imshow(
        mat.to_numpy(),
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-0.08, vcenter=0.0, vmax=0.24),
        aspect="auto",
    )
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for i, pos in enumerate(order):
        for j, yr in enumerate(years):
            val = mat.loc[pos, yr]
            if pd.isna(val):
                continue
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8, color=INK if abs(val) < 0.04 else "#fff")
    ax.set_xlabel("Season")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Spearman lift vs ADP")
    _caption(ax, "Where the model beat ADP, by position")
    fig.text(0.01, -0.06, "Green = model ranked that position better than ADP. Walk-forward: trained on prior seasons only.", fontsize=8, color=MUTED)
    _save(fig, "03_lift_heatmap.png")


def fig_points_vs_actual(preds: pd.DataFrame) -> None:
    d = drafted(preds)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharex=True, sharey=True)
    specs = [
        (axes[0], d["market_fp"], "ADP-implied PPR", ADP),
        (axes[1], d["model_fp"], "Model PPR", ACCENT),
    ]
    for ax, x, title, color in specs:
        ax.hexbin(x, d["ppr_actual"], gridsize=28, cmap="YlGnBu", mincnt=1, linewidths=0)
        lo, hi = 0, 420
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, alpha=0.45)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(title)
        ax.spines[["top", "right"]].set_visible(False)
        mae = np.mean(np.abs(x - d["ppr_actual"]))
        ax.text(0.04, 0.95, f"MAE {mae:.0f} PPR", transform=ax.transAxes, va="top", fontsize=10, color=color)
        ax.set_title(title + " vs actual season", loc="left", fontsize=12)
    axes[0].set_ylabel("Actual full-PPR points")
    fig.suptitle("Point totals: market vs model  (2021–2025, ADP-drafted)", x=0.02, ha="left", fontsize=13, fontweight="600")
    fig.subplots_adjust(top=0.82)
    _save(fig, "04_points_vs_actual.png")


def fig_rank_lift(preds: pd.DataFrame) -> None:
    d = drafted(preds)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    colors = d["steal_label"].map({"steal": STEAL, "fade": FADE, "fair": "#c4bba8"}).fillna("#c4bba8")
    ax.scatter(d["pred_lift"], d["act_lift"], c=colors, s=14, alpha=0.55, linewidths=0)
    ax.axhline(0, color=LINE, lw=1)
    ax.axvline(0, color=LINE, lw=1)
    lim = 22
    ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.7, alpha=0.35)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Model: ranks vs ADP  (+ = we said cheaper)")
    ax.set_ylabel("Actual: ranks vs ADP  (+ = they beat ADP)")
    ax.spines[["top", "right"]].set_visible(False)
    rho = spearmanr(d["pred_lift"], d["act_lift"], nan_policy="omit").statistic
    ax.text(0.04, 0.96, f"Directional ρ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=10)
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=STEAL, markersize=8, label="Steal flag"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=FADE, markersize=8, label="Fade flag"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#c4bba8", markersize=8, label="Fair"),
        ],
        frameon=False,
        loc="lower right",
    )
    _caption(ax, "When the model disagrees with ADP, does the season agree?")
    fig.text(0.01, -0.02, "Ranks among ADP-drafted players only. Upper-right = we said cheap and they were.", fontsize=8, color=MUTED)
    _save(fig, "05_rank_lift.png")


def fig_flags(preds: pd.DataFrame, steal: dict) -> None:
    d = drafted(preds)
    steal_g = d.loc[d["steal_label"] == "steal"]
    fade_g = d.loc[d["steal_label"] == "fade"]
    steal_hit = float((steal_g["act_lift"] > 0).mean()) if len(steal_g) else np.nan
    fade_hit = float((fade_g["act_lift"] < 0).mean()) if len(fade_g) else np.nan
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels = [f"Steals beat ADP\n(n={len(steal_g)})", f"Fades miss ADP\n(n={len(fade_g)})"]
    vals = [steal_hit, fade_hit]
    bars = ax.bar(labels, vals, color=[STEAL, FADE], width=0.55)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1, label="50% (no edge)")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of flags that went the predicted direction")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.0%}", ha="center", fontsize=14, fontweight="600")
    ax.legend(frameon=False)
    _caption(ax, "Published steal / fade flags, 2021–2025")
    fig.text(0.01, -0.04,
             f"Ranks among drafted players only. {steal_hit:.0%} of steal flags finished above their ADP rank "
             f"(n={len(steal_g)}, so treat the exact number loosely).",
             fontsize=8, color=MUTED)
    _save(fig, "06_flag_hit_rates.png")
    _ = steal


def fig_disagreement(preds: pd.DataFrame) -> None:
    d = drafted(preds)
    bins = [-99, -8, -5, -2, 2, 5, 8, 99]
    labels = ["Much more\nexpensive\n(≤ −8)", "More\nexpensive", "A bit\nexpensive", "Close\nto ADP", "A bit\ncheaper", "Cheaper", "Much\ncheaper\n(≥ +8)"]
    d["pl_bin"] = pd.cut(d["pred_lift"], bins, labels=labels)
    g = d.groupby("pl_bin", observed=True)
    pct = g["act_lift"].apply(lambda s: (s > 0).mean())
    ns = g.size()
    means = g["act_lift"].mean()
    colors = [FADE, FADE, "#c4bba8", "#c4bba8", STEAL, STEAL, STEAL]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(pct))
    bars = ax.bar(x, pct.to_numpy(), color=colors, width=0.72)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1, label="Coin flip")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Finished better than ADP rank")
    ax.spines[["top", "right"]].set_visible(False)
    for i, (bar, n, mu) in enumerate(zip(bars, ns, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.0%}", ha="center", fontsize=11, fontweight="600")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.04, f"n={int(n)}\n{mu:+.1f} rnk", ha="center", va="bottom", fontsize=7.5, color="#fff" if i in {0, 1, 5, 6} else MUTED)
    ax.legend(frameon=False, loc="upper left")
    _caption(ax, "The edge is in the disagreements — especially when we say cheaper")
    up = d.loc[d["pred_lift"] >= 5]
    fig.text(
        0.01,
        -0.08,
        "2021–2025, ranks among ADP-drafted players only. "
        f"When the model had someone ≥5 ranks cheaper, they beat ADP {up['act_lift'].gt(0).mean():.0%} of the time (n={len(up)}).",
        fontsize=8,
        color=MUTED,
    )
    _save(fig, "07_disagreement_calibration.png")


def fig_midround(preds: pd.DataFrame) -> None:
    d = drafted(preds)
    m = d.loc[d["adp"].between(36, 96)]
    cheap = m.loc[m["pred_lift"] >= 5]
    exp = m.loc[m["pred_lift"] <= -5]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cats = [f"Model said cheaper\n≥5 ranks  (n={len(cheap)})\nmean ADP {cheap.adp.mean():.0f}", f"Model said more expensive\n≥5 ranks  (n={len(exp)})\nmean ADP {exp.adp.mean():.0f}"]
    means = [cheap.ppr_actual.mean(), exp.ppr_actual.mean()]
    meds = [cheap.ppr_actual.median(), exp.ppr_actual.median()]
    bars = ax.bar(cats, means, color=[STEAL, FADE], width=0.55)
    ax.scatter([0, 1], meds, color=INK, zorder=3, s=36, label="Median")
    ax.set_ylabel("Actual full-PPR points the next season")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 240)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 6, f"{v:.0f}", ha="center", fontsize=16, fontweight="600")
    delta = means[0] - means[1]
    ax.annotate(f"+{delta:.0f} PPR at the same cost", xy=(0.5, max(means) + 22), ha="center", fontsize=11, color=STEAL, fontweight="600")
    ax.legend(frameon=False, loc="upper right")
    _caption(ax, "Mid-round picks (ADP 36–96): same price, different player")
    fig.text(0.01, -0.06, "Walk-forward 2021–2025. These two groups cost about the same ADP. The model’s cheaper names scored ~23 more actual PPR.", fontsize=8, color=MUTED)
    _save(fig, "08_midround_same_cost.png")


def fig_top5(preds: pd.DataFrame) -> None:
    d = drafted(preds)
    rows = []
    for (season, pos), g in d.groupby(["season", "position"]):
        if len(g) < 5:
            continue
        delta = g.nsmallest(5, "model_rank")["ppr_actual"].sum() - g.nsmallest(5, "adp")["ppr_actual"].sum()
        rows.append({"season": int(season), "pos": pos, "delta": delta})
    t = pd.DataFrame(rows)
    order = ["WR", "TE", "QB", "RB"]
    means = t.groupby("pos")["delta"].mean().reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), gridspec_kw={"width_ratios": [1.1, 1.3]})
    ax = axes[0]
    colors = [STEAL if v > 0 else FADE for v in means]
    ax.barh(order[::-1], means.reindex(order[::-1]), color=colors[::-1], height=0.62)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Extra actual PPR / season  (model top 5 − ADP top 5)")
    ax.spines[["top", "right"]].set_visible(False)
    for pos, v in means.items():
        y = list(order[::-1]).index(pos)
        ax.text(v + (4 if v >= 0 else -4), y, f"{v:+.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=11, fontweight="600")
    _caption(ax, "Start the model’s top 5, not ADP’s")

    ax2 = axes[1]
    wr = t.loc[t["pos"] == "WR"].sort_values("season")
    ax2.bar(wr["season"].astype(str), wr["delta"], color=[STEAL if v > 0 else FADE for v in wr["delta"]], width=0.65)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_ylabel("Extra WR PPR vs ADP top 5")
    ax2.spines[["top", "right"]].set_visible(False)
    for s, v in zip(wr["season"], wr["delta"]):
        ax2.text(str(int(s)), v + (8 if v >= 0 else -12), f"{v:+.0f}", ha="center", fontsize=8)
    _caption(ax2, "WR is the swing: one or two name swaps per year")
    fig.text(0.01, -0.06, "Left: average over 2021–2025. Right: WR only. Typical swap was Lamb or Amon-Ra in, a fading WR1 out — not a full rebuild.", fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "09_top5_roster_edge.png")


def fig_cheap_by_year(preds: pd.DataFrame) -> None:
    d = drafted(preds)
    rows = []
    for season, g in d.groupby("season"):
        up = g.loc[g["pred_lift"] >= 5]
        rows.append({"season": int(season), "n": len(up), "hit": (up["act_lift"] > 0).mean(), "mean": up["act_lift"].mean()})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bars = ax.bar(t["season"].astype(str), t["hit"], color=STEAL, width=0.62)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1, label="Coin flip")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share that beat their ADP rank")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, n, mu, h in zip(bars, t["n"], t["mean"], t["hit"]):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.0%}", ha="center", fontsize=13, fontweight="600")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.06, f"n={int(n)}\n{mu:+.0f} rnk", ha="center", fontsize=8, color="#fff")
    ax.legend(frameon=False)
    _caption(ax, "Every season: when the model said ≥5 ranks cheaper, they usually were")
    worst = t.loc[t["hit"].idxmin()]
    best = t.loc[t["hit"].idxmax()]
    fig.text(
        0.01,
        -0.04,
        f"Not a one-year fluke. Best: {int(best.season)} {best.hit:.0%} (n={int(best.n)}). "
        f"Worst: {int(worst.season)} {worst.hit:.0%}.",
        fontsize=8,
        color=MUTED,
    )
    _save(fig, "10_cheap_calls_by_year.png")


def load_panel() -> pd.DataFrame:
    """Player-seasons with a real prior role, plus what happened the next year.

    A row's td_luck is computed from the *previous* season, so the outcome that
    tests it is this row's season. Next-season TD totals only exist as the lag
    columns on the following row, hence the self-join.
    """
    p = pd.read_parquet(PROCESSED_DIR / "player_panel.parquet")
    p = p.loc[p["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    p["tds_prior"] = p["rushing_tds_lag"].fillna(0) + p["receiving_tds_lag"].fillna(0)
    nxt = p[["player_id", "season", "tds_prior"]].rename(columns={"season": "s1", "tds_prior": "tds_next"})
    nxt["season"] = nxt["s1"] - 1
    d = p.merge(nxt[["player_id", "season", "tds_next"]], on=["player_id", "season"], how="inner")
    d["d_td"] = d["tds_next"] - d["tds_prior"]
    d["d_ppr"] = d["ppr_actual"] - d["ppr_lag"]
    return d.loc[d["td_luck"].notna() & (d["tds_prior"] >= 2) & (d["ppr_lag"] >= 60)].copy()


LUCK_BINS = [-99, -1.5, 1.5, 3, 5, 7, 99]
LUCK_LABELS = ["Unlucky\n(≤ −1.5)", "Neutral\n(−1.5 to 1.5)", "+1.5 to 3", "+3 to 5", "+5 to 7", "Extreme\n(≥ +7)"]


def fig_td_luck_mechanism(panel: pd.DataFrame) -> None:
    d = panel.copy()
    d["bin"] = pd.cut(d["td_luck"], LUCK_BINS, labels=LUCK_LABELS)
    g = d.groupby("bin", observed=True)
    dtd, dppr, ns = g["d_td"].mean(), g["d_ppr"].mean(), g.size()
    down = g["d_td"].apply(lambda s: (s < 0).mean())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    x = np.arange(len(dtd))
    for ax, vals, ylab, title in [
        (axes[0], dtd, "Change in TDs the next season", "Touchdowns come back down"),
        (axes[1], dppr, "Change in PPR points the next season", "And the fantasy points follow"),
    ]:
        colors = [FADE if v < 0 else STEAL for v in vals]
        bars = ax.bar(x, vals.to_numpy(), color=colors, width=0.7)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(LUCK_LABELS, fontsize=8)
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)
        span = vals.max() - vals.min()
        # Room under the lowest bar for its value label and the sample-size row.
        ax.set_ylim(vals.min() - span * 0.30, max(vals.max() + span * 0.12, span * 0.06))
        pad = span * 0.03
        for bar, v in zip(bars, vals):
            off = pad if v >= 0 else -pad
            fmt = f"{v:+.1f}" if abs(v) < 10 else f"{v:+.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, v + off, fmt, ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=10, fontweight="600")
        _caption(ax, title)
    lo = axes[0].get_ylim()[0]
    for i, (n, pct) in enumerate(zip(ns, down)):
        axes[0].text(i, lo + abs(lo) * 0.015, f"n={int(n)}\n{pct:.0%} down", ha="center", va="bottom",
                     fontsize=7, color=MUTED)
    fig.text(0.01, -0.04,
             "TD luck = actual TDs minus what the red-zone role implies (0.42 per inside-5 carry, 0.28 per end-zone target, "
             "0.20 per inside-10 target). 2019–2025, players with a real prior role.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "11_td_luck_mechanism.png")


def fig_td_luck_by_year(panel: pd.DataFrame) -> None:
    rows = []
    for season, g in panel.groupby("season"):
        hi = g.loc[g["td_luck"] >= g["td_luck"].quantile(0.9)]
        rows.append({"season": int(season), "corr": g["td_luck"].corr(g["d_td"]),
                     "hi_dtd": hi["d_td"].mean(), "n": len(hi)})
    t = pd.DataFrame(rows).sort_values("season")

    # A row's luck comes from the prior season, so label both ends of the test.
    t["label"] = [f"{s-1}–{str(s)[2:]}" for s in t["season"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    bars = ax.bar(t["label"], t["hi_dtd"], color=FADE, width=0.62)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(t["hi_dtd"].min() * 1.22, 0.35)
    ax.set_ylabel("Mean TD change, luckiest 10%")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v, n in zip(bars, t["hi_dtd"], t["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v - 0.12, f"{v:+.1f}", ha="center", va="top", fontsize=10, fontweight="600")
        ax.text(bar.get_x() + bar.get_width() / 2, -0.1, f"n={int(n)}", ha="center", va="top", fontsize=7, color="#fff")
    _caption(ax, "Every single season, the luckiest scorers gave TDs back")

    ax2 = axes[1]
    ax2.plot(t["label"], t["corr"], marker="o", color=ACCENT, lw=1.6)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_ylim(-0.62, 0.05)
    ax2.set_ylabel("corr(TD luck, next-year TD change)")
    ax2.tick_params(axis="x", labelsize=8.5, rotation=45)
    ax2.spines[["top", "right"]].set_visible(False)
    for s, v in zip(t["label"], t["corr"]):
        ax2.text(s, v - 0.035, f"{v:.2f}", ha="center", va="top", fontsize=8, color=MUTED)
    _caption(ax2, "Sign never flips")
    fig.text(0.01, -0.08,
             "Labels read luck season and the season that tested it. Seven straight years of the same effect is why the model trusts it. "
             "The weakest year still ran −0.32.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "12_td_luck_by_year.png")


def _stack_labels(values: list[float], gap: float) -> list[float]:
    """Push callout labels apart so a tight cluster stays readable."""
    out = list(values)
    for i in range(1, len(out)):
        if out[i - 1] - out[i] < gap:
            out[i] = out[i - 1] - gap
    return out


def fig_td_luck_extremes(panel: pd.DataFrame, jt: dict | None) -> None:
    d = panel
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    rest = d.loc[d["td_luck"] < 7]
    tail = d.loc[d["td_luck"] >= 7]
    ax.scatter(rest["td_luck"], rest["d_ppr"], s=13, color="#c4bba8", alpha=0.55, linewidths=0, label="Everyone else")
    ax.scatter(tail["td_luck"], tail["d_ppr"], s=30, color=FADE, alpha=0.85, linewidths=0,
               label=f"TD luck ≥ +7 (n={len(tail)})")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("TD luck last season  (actual TDs − role-implied TDs)")
    ax.set_ylabel("Change in PPR points the next season")
    ax.spines[["top", "right"]].set_visible(False)

    xmax = float(d["td_luck"].max())
    ax.set_xlim(float(d["td_luck"].min()) - 0.6, xmax + 7.0)
    ymin, ymax = ax.get_ylim()

    mean_tail = tail["d_ppr"].mean()
    ax.plot([7, xmax + 0.4], [mean_tail, mean_tail], color=FADE, ls="--", lw=1.1)
    ax.text(0.012, 0.83, f"TD luck ≥ +7:  mean {mean_tail:+.0f} PPR (dashed),  "
            f"{100*(tail['d_ppr']<0).mean():.0f}% declined",
            transform=ax.transAxes, fontsize=8.5, color=FADE, fontweight="600")

    sel = d.nlargest(9, "td_luck").sort_values("d_ppr", ascending=False)
    label_x = xmax + 1.0
    ys = _stack_labels(sel["d_ppr"].tolist(), gap=(ymax - ymin) * 0.062)
    for r, y in zip(sel.itertuples(), ys):
        luck_yr = int(r.season) - 1
        ax.annotate(f"{r.player_name} {luck_yr}–{str(int(r.season))[2:]}   {r.d_ppr:+.0f}",
                    xy=(r.td_luck, r.d_ppr), xytext=(label_x, y),
                    textcoords="data", va="center", ha="left", fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=LINE, lw=0.7, shrinkA=0, shrinkB=3))
    if jt:
        ax.axvline(jt["td_luck"], color=ACCENT, ls="--", lw=1.4)
        ax.text(jt["td_luck"] - 0.25, ymin + (ymax - ymin) * 0.055,
                f"Jonathan Taylor 2026\nTD luck {jt['td_luck']:+.1f}  ({jt['z']:.1f} SD)",
                ha="right", va="bottom", fontsize=9, color=ACCENT, fontweight="600")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    _caption(ax, "The extreme tail is where TD luck actually bites")
    fig.text(0.01, -0.03,
             f"2019–2025; labels are the luck season and the season that tested it. Of the {len(tail)} seasons with TD luck ≥ +7, "
             f"{100*(tail['d_td']<0).mean():.0f}% lost touchdowns the next year and {100*(tail['d_ppr']<0).mean():.0f}% lost fantasy points. "
             "Not all of it is regression — McCaffrey's collapse was a torn achilles — which is why a fade needs a second reason to agree.",
             fontsize=8, color=MUTED)
    _save(fig, "13_td_luck_extremes.png")


REASON_INTENT = {
    "overproduction": ("Overproduction", -1),
    "workload_cliff": ("Workload cliff", -1),
    "td_luck": ("TD luck (over)", -1),
    "eff_index": ("Efficiency spike", -1),
    "sophomore_leap": ("Sophomore leap", +1),
    "role_expand": ("Role expansion", +1),
    "new_starter_vacated": ("New starter", +1),
    "pass_catch_rb": ("Pass-catching RB", +1),
}


def fig_reason_evidence() -> None:
    p = pd.read_parquet(PROCESSED_DIR / "player_panel.parquet")
    d = p.loc[p["position"].isin(["QB", "RB", "WR", "TE"]) & p["ppr_actual"].notna()
              & (p["ppr_lag"] >= 60) & p["adp"].notna()].copy()
    d["adp_rank"] = d.groupby(["season", "position"])["adp"].rank(method="min")
    d["act_rank"] = d.groupby(["season", "position"])["ppr_actual"].rank(ascending=False, method="min")
    d["beat"] = (d["adp_rank"] - d["act_rank"]) > 0

    rows = []
    for feat, (label, intent) in REASON_INTENT.items():
        if feat not in d.columns:
            continue
        col = d[feat]
        if col.dropna().nunique() < 3:
            hi, lo = d.loc[col > 0], d.loc[col <= 0]
        else:
            cut = col.quantile(0.8)
            hi, lo = d.loc[col >= cut], d.loc[col < cut]
        if len(hi) < 20:
            continue
        p_hi, p_lo = hi["beat"].mean(), lo["beat"].mean()
        gap = p_hi - p_lo
        se = np.sqrt(p_hi * (1 - p_hi) / len(hi) + p_lo * (1 - p_lo) / len(lo))
        rows.append({"label": label, "gap": 100 * gap, "ci": 100 * 1.96 * se, "n": len(hi),
                     "works": np.sign(gap) == intent and abs(gap) > 1.96 * se})
    t = pd.DataFrame(rows).sort_values("gap")

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    colors = [(FADE if r.gap < 0 else STEAL) if r.works else "#c4bba8" for r in t.itertuples()]
    ax.barh(t["label"], t["gap"], color=colors, height=0.6)
    ax.errorbar(t["gap"], range(len(t)), xerr=t["ci"], fmt="none", ecolor=INK, elinewidth=1.0,
                capsize=3, alpha=0.65)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlabel("Change in the odds of beating ADP  (percentage points, 95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    for i, r in enumerate(t.itertuples()):
        edge = r.gap + (r.ci if r.gap >= 0 else -r.ci)
        off = 1.6 if r.gap >= 0 else -1.6
        ax.text(edge + off, i, f"{r.gap:+.0f}  (n={int(r.n)})", va="center",
                ha="left" if r.gap >= 0 else "right", fontsize=8.5)
    ax.set_xlim(t["gap"].min() - t["ci"].max() - 14, t["gap"].max() + t["ci"].max() + 14)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor=FADE, markersize=9, label="Real fade signal"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=STEAL, markersize=9, label="Real steal signal"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#c4bba8", markersize=9, label="Not distinguishable from zero"),
    ], frameon=False, loc="lower right", fontsize=8.5)
    _caption(ax, "Which steal / fade reasons actually earn their chip")
    fig.text(0.01, -0.07,
             "2019–2025 drafted players; each bar compares the flagged group against everyone else, one reason at a time. "
             "Bars whose interval crosses zero are honest context, not evidence — on the board a reason only prints when the model "
             "already disagrees with ADP, so the weak ones never fire on their own.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "14_reason_evidence.png")


def jt_context() -> dict | None:
    path = PROCESSED_DIR / "rankings_2026.csv"
    if not path.exists():
        return None
    r = pd.read_csv(path)
    row = r.loc[r["player_name"].str.contains("Jonathan Taylor", na=False)]
    if row.empty:
        return None
    return {"td_luck": float(row["td_luck"].iloc[0]), "z": float(row["z_td_luck"].iloc[0])}


AGE_PEAKS = {"QB": 30.0, "RB": 26.5, "WR": 28.0, "TE": 28.5}
POS_COLORS = {"QB": "#5c574e", "RB": FADE, "WR": ACCENT, "TE": ADP}


def _outcome_frame() -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED_DIR / "player_panel.parquet")
    d = p.loc[p["position"].isin(["QB", "RB", "WR", "TE"]) & p["ppr_actual"].notna()
              & (p["ppr_lag"] >= 60)].copy()
    d["d_ppr"] = d["ppr_actual"] - d["ppr_lag"]
    d["ret"] = d["ppr_actual"] / d["ppr_lag"]
    drafted = d.loc[d["adp"].notna()].copy()
    drafted["adp_rank"] = drafted.groupby(["season", "position"])["adp"].rank(method="min")
    drafted["act_rank"] = drafted.groupby(["season", "position"])["ppr_actual"].rank(ascending=False, method="min")
    drafted["beat"] = (drafted["adp_rank"] - drafted["act_rank"]) > 0
    return d, drafted


def fig_age_curves() -> None:
    d, _ = _outcome_frame()
    d = d.loc[d["ppr_lag"] >= 80].copy()
    d["years_past"] = d.apply(lambda r: r["age"] - AGE_PEAKS.get(r["position"], 28), axis=1)
    bins = [-99, 0, 1.5, 3, 5, 99]
    labels = ["At / before\npeak", "0–1.5 yrs\npast", "1.5–3 yrs\npast", "3–5 yrs\npast", "5+ yrs\npast"]
    d["bin"] = pd.cut(d["years_past"], bins, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.2
    for i, pos in enumerate(["RB", "WR", "TE", "QB"]):
        g = d.loc[d["position"] == pos].groupby("bin", observed=True)["ret"].mean().reindex(labels)
        ax.bar(x + (i - 1.5) * width, g.to_numpy(), width, color=POS_COLORS[pos], label=pos)
    ax.axhline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.45, 1.15)
    ax.set_ylabel("Next-season PPR ÷ last-season PPR")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=8.5)
    _caption(ax, "After the peak, the points do not come back")

    ax2 = axes[1]
    # Fitted age_alpha at +0..+6 years for each position (defaults)
    from ffmodel.config import AGE_LAMBDA_DEFAULT
    years = np.linspace(0, 6, 50)
    for pos in ["RB", "WR", "TE", "QB"]:
        lam = AGE_LAMBDA_DEFAULT[pos]
        alpha = np.clip(1 - lam * years ** 2, 0.35, 1.05)
        ax2.plot(years, alpha, color=POS_COLORS[pos], lw=2, label=f"{pos} (λ={lam})")
    ax2.set_xlabel("Years past positional peak")
    ax2.set_ylabel("Age multiplier α")
    ax2.set_ylim(0.35, 1.08)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8)
    _caption(ax2, "What the model actually applies")
    peaks = ", ".join(f"{p} {v:g}" for p, v in AGE_PEAKS.items())
    fig.text(0.01, -0.06,
             f"Left: 2019–2025 players with ≥80 prior PPR, grouped by years past the fitted peak ({peaks}). "
             "Right: the quadratic α the prior uses — RBs get the steepest curve because the history said so.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "15_age_curves.png")


def fig_overproduction() -> None:
    _, d = _outcome_frame()
    d = d.loc[d["overproduction"].notna()].copy()
    labels = ["Lowest 20%", "20–40%", "40–60%", "60–80%", "Highest 20%"]
    d["bin"] = pd.qcut(d["overproduction"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=labels)
    g = d.groupby("bin", observed=True)
    beat, dppr, ns = g["beat"].mean(), g["d_ppr"].mean(), g.size()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    x = np.arange(len(labels))
    for ax, vals, ylab, title, as_pct in [
        (axes[0], beat, "Finished above ADP rank", "Beating ADP gets rarer", True),
        (axes[1], dppr, "Change in PPR the next season", "And the points fall off", False),
    ]:
        colors = [FADE if (v < (0.5 if as_pct else 0)) else STEAL for v in vals]
        bars = ax.bar(x, vals.to_numpy(), color=colors, width=0.7)
        ax.axhline(0.5 if as_pct else 0, color=MUTED if as_pct else INK, ls="--" if as_pct else "-", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(ylab)
        if as_pct:
            ax.set_ylim(0, 0.85)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v, n in zip(bars, vals, ns):
            txt = f"{v:.0%}" if as_pct else f"{v:+.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, (v + 0.03) if as_pct else v + (4 if v >= 0 else -8),
                    txt, ha="center", fontsize=11, fontweight="600")
            if not as_pct:
                ax.text(bar.get_x() + bar.get_width() / 2, ax.get_ylim()[0] * 0.92 if False else min(dppr.min() - 8, -5),
                        f"n={int(n)}", ha="center", fontsize=7, color=MUTED)
        _caption(ax, title)
    for i, n in enumerate(ns):
        axes[0].text(i, 0.04, f"n={int(n)}", ha="center", fontsize=7, color="#fff" if beat.iloc[i] > 0.35 else MUTED)
    fig.text(0.01, -0.05,
             "Overproduction = scoring z-score minus half of snap + target-share z-scores. Players in the top quintile "
             "beat ADP only 28% of the time and lost ~39 PPR the next year. That is a fade confirmer, not a talent flag.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "16_overproduction.png")


def fig_workload_cliff() -> None:
    _, d = _outcome_frame()
    rb = d.loc[d["position"] == "RB"].copy()
    cliff = rb.loc[rb["workload_cliff"].fillna(0).astype(bool)]
    rest = rb.loc[~rb["workload_cliff"].fillna(0).astype(bool)]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    cats = [f"Workload cliff\n(n={len(cliff)})", f"Other drafted RBs\n(n={len(rest)})"]
    vals = [cliff["beat"].mean(), rest["beat"].mean()]
    bars = ax.bar(cats, vals, color=[FADE, "#c4bba8"], width=0.55)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Finished above ADP rank")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.0%}", ha="center", fontsize=14, fontweight="600")
    _caption(ax, "Heavy, aging backs miss ADP more often")

    ax2 = axes[1]
    # Named cliff seasons with next-year outcomes
    carry_col = "carries_lag" if "carries_lag" in cliff.columns else "carry_load"
    cols = ["season", "player_name", "ppr_lag", "ppr_actual", "d_ppr", "age"]
    if carry_col in cliff.columns:
        cols.append(carry_col)
    show = cliff.nlargest(min(10, len(cliff)), "ppr_lag")[cols]
    y = np.arange(len(show))
    ax2.barh(y, show["d_ppr"].to_numpy(), color=[FADE if v < 0 else STEAL for v in show["d_ppr"]], height=0.65)
    ax2.axvline(0, color=INK, lw=0.8)
    labels = []
    for r in show.itertuples():
        car = getattr(r, carry_col, None)
        car_txt = f"{car:.0f} car, " if car is not None and pd.notna(car) else ""
        labels.append(f"{r.player_name} {int(r.season)-1}  ({car_txt}age {r.age:.0f})")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Change in PPR the next season")
    ax2.spines[["top", "right"]].set_visible(False)
    for yi, v in zip(y, show["d_ppr"]):
        ax2.text(v + (4 if v >= 0 else -4), yi, f"{v:+.0f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8.5, fontweight="600")
    _caption(ax2, "Biggest prior seasons that tripped the cliff")
    fig.text(0.01, -0.05,
             "Cliff = ≥240 carries at age ≥26.5, or ≥280 at age ≥26. Small sample (n=28 drafted), big miss rate: "
             f"{cliff['beat'].mean():.0%} beat ADP vs {rest['beat'].mean():.0%} for everyone else, and the mean drop was "
             f"{cliff['d_ppr'].mean():+.0f} PPR.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "17_workload_cliff.png")


def fig_sophomore_leap() -> None:
    _, d = _outcome_frame()
    soph = d.loc[d["sophomore_leap"].fillna(0).astype(bool)]
    rest = d.loc[~d["sophomore_leap"].fillna(0).astype(bool)]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    ax = axes[0]
    cats = [f"Sophomore leap\n(n={len(soph)})", f"Everyone else\n(n={len(rest)})"]
    beat_v = [soph["beat"].mean(), rest["beat"].mean()]
    bars = ax.bar(cats, beat_v, color=[STEAL, "#c4bba8"], width=0.55)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Finished above ADP rank")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, beat_v):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.0%}", ha="center", fontsize=14, fontweight="600")
    _caption(ax, "Year-two names in the window beat ADP")

    ax2 = axes[1]
    dpr = [soph["d_ppr"].mean(), rest["d_ppr"].mean()]
    bars = ax2.bar(cats, dpr, color=[STEAL if dpr[0] > 0 else FADE, FADE], width=0.55)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_ylabel("Change in PPR the next season")
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, dpr):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + (6 if v >= 0 else -10), f"{v:+.0f}",
                 ha="center", fontsize=14, fontweight="600")
    _caption(ax2, "They are also the group that still gains points")
    fig.text(0.01, -0.05,
             "Sophomore leap = year 2 with last-season PPR between 70 and 200. Not every second-year player — "
             "just the ones who already produced enough to matter and still have room to grow. "
             f"That band beat ADP {soph['beat'].mean():.0%} of the time and gained {soph['d_ppr'].mean():+.0f} PPR on average.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "18_sophomore_leap.png")


def fig_tenure_weights() -> None:
    from ffmodel.config import TENURE_WEIGHTS
    order = ["rookie", "sophomore", "developing", "prime", "veteran"]
    keys = ["market", "production", "situation", "physical", "aging"]
    colors = {"market": ACCENT, "production": "#3d6b5a", "situation": ADP, "physical": "#8a9a7b", "aging": FADE}
    key_idx = {"market": 0, "production": 1, "situation": 2, "physical": 3, "aging": 4}
    mat = np.array([[TENURE_WEIGHTS[b][key_idx[k]] for k in keys] for b in order])

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    bottom = np.zeros(len(order))
    x = np.arange(len(order))
    for j, k in enumerate(keys):
        ax.bar(x, mat[:, j], bottom=bottom, color=colors[k], width=0.62, label=k)
        bottom += mat[:, j]
    ax.set_xticks(x)
    ax.set_xticklabels([b.title() for b in order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of the prior")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    for i, row in enumerate(mat):
        cum = 0.0
        for v in row:
            if v >= 0.10:
                ax.text(i, cum + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8, color="#fff", fontweight="600")
            cum += v
    _caption(ax, "How the prior reweights as a player ages")
    fig.text(0.01, -0.04,
             "Rookies lean on situation and measurables. Prime vets lean on the market and last year’s production. "
             "Veterans take a real aging tax. This is the shrinkage target before the trees — not the published ranking.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "19_tenure_weights.png")


def fig_vorp_reorder() -> None:
    path = PROCESSED_DIR / "rankings_2026.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    r = r.loc[r["adp"].notna() & r["model_fp"].notna()].copy()
    r["pts_rank"] = r["model_fp"].rank(ascending=False, method="min")
    r["vorp_rank"] = r["vorp"].rank(ascending=False, method="min")
    r["shift"] = r["pts_rank"] - r["vorp_rank"]  # + = moved up on VORP board

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    ax = axes[0]
    for pos in ["QB", "RB", "WR", "TE"]:
        g = r.loc[r["position"] == pos]
        ax.scatter(g["pts_rank"], g["vorp_rank"], s=18, alpha=0.65, linewidths=0,
                   color=POS_COLORS[pos], label=pos)
    lim = 80
    ax.plot([1, lim], [1, lim], color=INK, lw=0.7, alpha=0.4)
    ax.set_xlim(0.5, lim)
    ax.set_ylim(lim, 0.5)
    ax.set_xlabel("Rank by raw projected PPR")
    ax.set_ylabel("Rank by VORP (published board)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    _caption(ax, "Same projections, different pick order")

    ax2 = axes[1]
    # Mean shift by position among top-60 by points
    top = r.loc[r["pts_rank"] <= 60]
    means = top.groupby("position")["shift"].mean().reindex(["QB", "RB", "WR", "TE"])
    colors = [FADE if v < 0 else STEAL for v in means]
    ax2.barh(["QB", "RB", "WR", "TE"][::-1], means.reindex(["QB", "RB", "WR", "TE"][::-1]),
             color=colors[::-1], height=0.55)
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_xlabel("Mean ranks gained on the VORP board  (+ = drafted earlier)")
    ax2.spines[["top", "right"]].set_visible(False)
    for pos, v in means.items():
        y = list(["QB", "RB", "WR", "TE"][::-1]).index(pos)
        ax2.text(v + (1.2 if v >= 0 else -1.2), y, f"{v:+.1f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=11, fontweight="600")
    _caption(ax2, "QBs fall, RBs rise — on purpose")
    fig.text(0.01, -0.05,
             "2026 board. VORP = projected points minus a replacement starter (QB12, RB30, WR30, TE12 for a 2RB/2WR/1FLEX league). "
             "Raw PPR puts elite QBs in the first round; draft value puts them where ADP already has them.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, "20_vorp_reorder.png")


def main() -> None:
    _style()
    preds, by_season, steal = load_or_backtest()
    print(f"Drafted rows: {len(drafted(preds))}")
    fig_spearman(by_season)
    fig_rank_vs_actual(preds)
    fig_lift_heatmap(preds)
    fig_points_vs_actual(preds)
    fig_rank_lift(preds)
    fig_flags(preds, steal)
    fig_disagreement(preds)
    fig_midround(preds)
    fig_top5(preds)
    fig_cheap_by_year(preds)

    panel = load_panel()
    print(f"Feature-forensics rows: {len(panel)}")
    fig_td_luck_mechanism(panel)
    fig_td_luck_by_year(panel)
    fig_td_luck_extremes(panel, jt_context())
    fig_reason_evidence()
    fig_age_curves()
    fig_overproduction()
    fig_workload_cliff()
    fig_sophomore_leap()
    fig_tenure_weights()
    fig_vorp_reorder()
    print(f"Copy these into WordPress from {COPY_DIR}")


if __name__ == "__main__":
    main()
