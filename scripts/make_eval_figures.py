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
    fig.text(0.01, -0.04, "Ranks among drafted players only. 78% of steal flags finished above their ADP rank.", fontsize=8, color=MUTED)
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
    print(f"Copy these into WordPress from {COPY_DIR}")


if __name__ == "__main__":
    main()
