from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PREDICT_SEASON, PROCESSED_DIR, ROOT
from .model import apply_board_ranks
from .names import normalize_name


POS_LIMITS = {"QB": 16, "RB": 36, "WR": 48, "TE": 20}


def _fmt(n, digits=1) -> str:
    if pd.isna(n):
        return "—"
    return f"{float(n):.{digits}f}"


def expected_receptions(df: pd.DataFrame) -> pd.Series:
    rec = pd.Series(np.nan, index=df.index, dtype=float)
    if "rec_proj" in df.columns:
        rec = pd.to_numeric(df["rec_proj"], errors="coerce")
    if "v_rec" in df.columns:
        rec = rec.fillna(pd.to_numeric(df["v_rec"], errors="coerce"))
    if "receptions_lag" in df.columns:
        rec = rec.fillna(pd.to_numeric(df["receptions_lag"], errors="coerce"))
    if "targets_lag" in df.columns:
        tgt = pd.to_numeric(df["targets_lag"], errors="coerce")
        catch = np.where(df["position"].isin(["WR", "TE"]), 0.62, np.where(df["position"].eq("RB"), 0.75, 0.0))
        rec = rec.fillna(tgt * catch)
    return rec.fillna(0).clip(lower=0)


def enrich_rankings(rankings: pd.DataFrame, panel: pd.DataFrame | None) -> pd.DataFrame:
    out = rankings.copy()
    if panel is None or panel.empty:
        if "rec_proj" not in out.columns:
            out["rec_proj"] = expected_receptions(out)
        return apply_board_ranks(out)
    season = int(out["season"].iloc[0]) if "season" in out.columns and out["season"].notna().any() else PREDICT_SEASON
    src = panel.loc[panel["season"] == season].copy()
    cols = [
        c
        for c in [
            "player_id",
            "v_rec",
            "receptions_lag",
            "targets_lag",
            "chronic_injury",
            "eff_index",
            "age_alpha",
            "new_starter_vacated",
            "name_norm",
        ]
        if c in src.columns
    ]
    if "player_id" in cols:
        extra = src[cols].drop_duplicates("player_id")
        overlap = [c for c in extra.columns if c != "player_id" and c in out.columns]
        extra = extra.drop(columns=overlap)
        out = out.merge(extra, on="player_id", how="left")
    out["rec_proj"] = expected_receptions(out)
    return apply_board_ranks(out)


def to_half_ppr(rankings: pd.DataFrame, half_adp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rescore full-PPR projections to half-PPR by removing 0.5 × projected receptions."""
    out = rankings.copy()
    rec = expected_receptions(out)
    out["rec_proj"] = rec
    out["model_fp"] = pd.to_numeric(out["model_fp"], errors="coerce") - 0.5 * rec
    if "vfp" in out.columns:
        out["vfp"] = pd.to_numeric(out["vfp"], errors="coerce") - 0.5 * rec
    if "market_fp" in out.columns:
        out["market_fp"] = pd.to_numeric(out["market_fp"], errors="coerce") - 0.5 * rec
    if half_adp is not None and not half_adp.empty:
        names = out["player_name"].map(normalize_name)
        lookup = half_adp.drop_duplicates("adp_name_norm").set_index("adp_name_norm")["adp"]
        mapped = names.map(lookup)
        out["adp"] = mapped.fillna(out["adp"])
    return apply_board_ranks(out)


def _listed(df: pd.DataFrame, n: int | None = None, by: str = "model_rank_pos", ascending: bool = True) -> pd.DataFrame:
    out = df.loc[df["adp"].notna()].copy()
    if by and by in out.columns:
        out = out.sort_values(by, ascending=ascending)
    out["display_rank"] = range(1, len(out) + 1)
    if n is not None:
        out = out.head(n)
    return out


def _player_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in df.itertuples(index=False):
        steal = 0 if pd.isna(rec.steal_score) else int(rec.steal_score)
        pid = getattr(rec, "player_id", None)
        ident = "" if pid is None or pd.isna(pid) else str(pid)
        if not ident:
            ident = f"{rec.player_name}|{rec.position}"
        pos_n = getattr(rec, "listed_pos", getattr(rec, "model_rank_pos", rec.display_rank))
        ov_n = getattr(rec, "listed_ov", getattr(rec, "display_rank", 0))
        rows.append(
            {
                "id": ident,
                "rank": int(rec.display_rank),
                "ov": int(ov_n) if pd.notna(ov_n) else int(rec.display_rank),
                "name": rec.player_name,
                "team": rec.team if pd.notna(rec.team) else "FA",
                "pos": rec.position,
                "posRank": f"{rec.position}{int(pos_n)}",
                "adp": None if pd.isna(rec.adp) else round(float(rec.adp), 1),
                "fp": None if pd.isna(rec.model_fp) else round(float(rec.model_fp), 1),
                "vs": steal,
                "flag": rec.steal_label if rec.steal_label in {"steal", "fade"} else "",
            }
        )
    return rows


def _top_table(df: pd.DataFrame, n: int = 24) -> str:
    df = _listed(df, n)
    lines = [
        "| Rank | Player | Team | ADP | Model pts | vs ADP |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for rec in df.itertuples(index=False):
        steal = 0 if pd.isna(rec.steal_score) else rec.steal_score
        vs = f"+{int(steal)}" if steal >= 0 else str(int(steal))
        flag = f" ({rec.steal_label})" if rec.steal_label in {"steal", "fade"} else ""
        lines.append(
            f"| {int(rec.display_rank)} | {rec.player_name}{flag} | {rec.team} | "
            f"{_fmt(rec.adp)} | {_fmt(rec.model_fp)} | {vs} |"
        )
    return "\n".join(lines)


def _steal_bullets(df: pd.DataFrame) -> str:
    steals = df.loc[df["steal_label"] == "steal"].sort_values("steal_score", ascending=False)
    if steals.empty:
        return "_No steal flags at the current threshold._"
    lines = []
    for r in steals.head(12).itertuples(index=False):
        edges = ", ".join(
            x
            for x in [
                "chunk plays" if pd.notna(getattr(r, "z_explosive", None)) and r.z_explosive > 0.8 else "",
                "red zone" if pd.notna(getattr(r, "z_redzone", None)) and r.z_redzone > 0.8 else "",
                "vacated usage" if pd.notna(getattr(r, "z_vacated", None)) and r.z_vacated > 0.7 else "",
                "TD luck fade" if pd.notna(getattr(r, "z_td_luck", None)) and r.z_td_luck < -0.7 else "",
                "injury bounce" if getattr(r, "injury_bounce", 0) else "",
                "role expansion" if getattr(r, "role_expand", 0) else "",
                "workload cliff" if getattr(r, "workload_cliff", 0) else "",
                "breakout window" if getattr(r, "breakout_window", 0) else "",
            ]
            if x
        ) or "situation/age vs market"
        lines.append(
            f"- **{r.player_name} ({r.position}, {r.team})** — ADP {_fmt(r.adp)}, "
            f"model {r.position}{int(r.model_rank_pos)}. Edges: {edges}"
        )
    return "\n".join(lines)


def _load_half_adp(season: int) -> pd.DataFrame:
    try:
        from .ingest.markets import load_adp

        df = load_adp(season, scoring="half-ppr")
        if df is not None and not df.empty:
            print(f"  Half-PPR ADP: {len(df)} players")
            return df
    except Exception as exc:
        print(f"  Half-PPR ADP unavailable ({exc}); using PPR ADP ranks")
    return pd.DataFrame()


def _with_listed_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df["adp"].notna()].copy()
    out["listed_pos"] = out.groupby("position")["model_fp"].rank(ascending=False, method="min")
    out["listed_ov"] = out["model_fp"].rank(ascending=False, method="min")
    return out


def _board_payload(df: pd.DataFrame) -> dict:
    listed = _with_listed_ranks(df)
    steals = _listed(listed.loc[listed["steal_label"] == "steal"], by="steal_score", ascending=False)
    fades = _listed(listed.loc[listed["steal_label"] == "fade"], by="steal_score", ascending=True)
    full = listed.sort_values(["implied_pick", "listed_ov", "player_name"], na_position="last").copy()
    full["display_rank"] = range(1, len(full) + 1)
    full["listed_ov"] = full["display_rank"]
    full = full.head(192)
    return {
        "steals": _player_rows(steals),
        "fades": _player_rows(fades),
        "full": _player_rows(full),
        "QB": _player_rows(_listed(listed.loc[listed["position"] == "QB"], POS_LIMITS["QB"], by="listed_pos")),
        "RB": _player_rows(_listed(listed.loc[listed["position"] == "RB"], POS_LIMITS["RB"], by="listed_pos")),
        "WR": _player_rows(_listed(listed.loc[listed["position"] == "WR"], POS_LIMITS["WR"], by="listed_pos")),
        "TE": _player_rows(_listed(listed.loc[listed["position"] == "TE"], POS_LIMITS["TE"], by="listed_pos")),
    }


def write_article(
    rankings: pd.DataFrame,
    backtest: pd.DataFrame,
    extra: pd.DataFrame,
    steal_eval: dict,
    season: int = PREDICT_SEASON,
    panel: pd.DataFrame | None = None,
) -> str:
    ppr = enrich_rankings(rankings, panel)
    half_adp = _load_half_adp(season)
    half = to_half_ppr(ppr, half_adp)
    ppr = ppr.sort_values(["position", "model_rank_pos"])
    half = half.sort_values(["position", "model_rank_pos"])
    ppr.to_csv(PROCESSED_DIR / f"rankings_{season}.csv", index=False)
    half.to_csv(PROCESSED_DIR / f"rankings_{season}_half.csv", index=False)
    for pos in ["QB", "RB", "WR", "TE"]:
        ppr.loc[ppr["position"] == pos].to_csv(PROCESSED_DIR / f"rankings_{season}_{pos}.csv", index=False)

    extra_beat = extra.dropna(subset=["corr_beat_market"]).copy() if not extra.empty and "corr_beat_market" in extra.columns else extra
    extra_beat = extra_beat.reindex(extra_beat["corr_beat_market"].abs().sort_values(ascending=False).index).head(12) if not extra_beat.empty else extra_beat

    bt_lines = ""
    if not backtest.empty:
        n_col = "n_adp" if "n_adp" in backtest.columns else "n"
        bt_lines = "\n".join(
            f"- **{int(r.season)}:** model Spearman {r.spearman:.3f} vs ADP {r.market_spearman:.3f}"
            + (
                f" (Δ {getattr(r, 'spearman_lift', float('nan')):+.3f})"
                if hasattr(r, "spearman_lift") and pd.notna(getattr(r, "spearman_lift", None))
                else ""
            )
            + f"; MAE {r.mae:.1f} PPR (n={int(getattr(r, n_col))})"
            for r in backtest.itertuples(index=False)
        )

    extra_lines = ""
    if extra_beat is not None and not extra_beat.empty:
        extra_lines = "\n".join(
            f"- `{r.feature}`: corr with next-year PPR {r.corr_ppr:.3f}"
            + (
                f"; corr with beating ADP {r.corr_beat_market:.3f}"
                if hasattr(r, "corr_beat_market") and pd.notna(r.corr_beat_market)
                else ""
            )
            for r in extra_beat.head(10).itertuples(index=False)
        )

    steal_lines = _steal_bullets(ppr)
    html_path = _write_html(
        season=season,
        ppr=ppr,
        half=half,
        backtest_md=bt_lines,
        extra_md=extra_lines,
        steal_eval=steal_eval,
        embed=False,
    )
    _write_html(
        season=season,
        ppr=ppr,
        half=half,
        backtest_md=bt_lines,
        extra_md=extra_lines,
        steal_eval=steal_eval,
        embed=True,
    )
    md = _write_markdown(
        season=season,
        rankings=ppr,
        bt_lines=bt_lines,
        extra_lines=extra_lines,
        steal_lines=steal_lines,
        steal_eval=steal_eval,
        html_path=html_path,
    )
    path = PROCESSED_DIR / f"article_{season}.md"
    path.write_text(md)
    return html_path


def _write_markdown(
    season: int,
    rankings: pd.DataFrame,
    bt_lines: str,
    extra_lines: str,
    steal_lines: str,
    steal_eval: dict,
    html_path: str,
) -> str:
    standouts = rankings.assign(standout=lambda d: d.filter(regex=r"^z_").abs().max(axis=1)).sort_values(
        "standout", ascending=False
    )
    md = f"""# {season} Fantasy Rankings: Finding the Steals the Market Still Misprices

*{date.today().isoformat()} · 12-team · model trained on {season-8}–{season-1}*

**Interactive boards (PPR / Half PPR toggle):** open [`article_{season}.html`]({Path(html_path).name}).

Vegas season totals are the cleanest public summary of how sharp books think a player’s counting stats will land. This model starts there — converting no-vig player props into a fantasy-point baseline (VFP) — then looks for the things a posted yardage line is structurally bad at seeing: aging, offensive line quality, coordinator pace/pass rate, chunk-play creation, red-zone roles, indoor/outdoor schedule, vacated touches, and injury residue. Where a player has no posted prop (most of the league, historically), the market proxy is Fantasy Football Calculator ADP translated into expected points.

The point is not to out-project every volume total. Books are better at that than a public model. The point is to **rank players relative to the market** so the article can tell you who is a round too cheap.

Use the HTML toggle to switch **Full PPR** (1 point per reception) vs **Half PPR** (0.5). The writeup does not change. Rankings subtract half a point per projected reception; Half PPR ADP is used for vs-market ranks when it is available.

## How the projection is built

1. **Vegas Fantasy Points (VFP).** Season-long over/unders for passing/rushing/receiving yards, TDs, and receptions are vig-stripped and scored. Juice like −120/−100 is not treated as a 50/50 median.
2. **Tenure-weighted blend.** Rookies lean on draft capital, measurables, vacated opportunity, and scheme. Prime veterans lean on the market plus last year’s production. Aging veterans get a quadratic positional decay (QB after 30, RB after 26.5, WR after 28, TE after 28.5) with lambdas fit from history, not guessed.
3. **Gradient boosting residual.** A position-aware HistGradientBoosting model is trained leave-one-season-out on the engineered panel. The residual is *what the ADP/VFP market misses*: TD luck, workload cliffs, injury bounce-backs, vacated usage assigned to a specific player, pass-catching RB work, and efficiency regression. Blend strength (λ) and a small rank-lift term are tuned to maximize Spearman vs actual PPR **on the same ADP-drafted players** used to score ADP.
4. **Steal / fade flags.** ADP 18–132 (skip 1.01s and dart throws), plus at least **5 position ranks** of model lift **and** **12+ PPR** of point edge vs the ADP-implied curve. Both have to agree on sign.

## Backtest

The model is always trained on *prior* seasons only. Spearman below is computed on **ADP-drafted players only**, so it is a fair comparison against ADP.

{bt_lines or "_Backtest table will populate after the first full run._"}

Steal flags (ADP 18–132, ≥5 rank lift and ≥12 PPR edge) historically hit **{steal_eval.get('steal_hit_rate', float('nan')):.0%}** of the time (≥4 actual ranks gained vs ADP, {steal_eval.get('n_steals', 0)} flags). Fades hit **{steal_eval.get('fade_hit_rate', float('nan')):.0%}** ({steal_eval.get('n_fades', 0)} flags). Directional Spearman of predicted vs actual rank lift: **{steal_eval.get('steal_dir_spearman', float('nan')):.3f}**.

## Extra stats that still beat the market

These lagged stats show the strongest Spearman correlation with next-year PPR and with *beating ADP-implied points*. They are the candidates to overweight in the next iteration.

{extra_lines or "_Run the pipeline to fill this._"}

## {season} steals

{steal_lines}

## Position rankings

Open the HTML article for the **full board vs positional** toggle. Overall rank on the full board is the 12-team pick (pick 25 is the start of the 3rd). Markdown tables below are positional.

### Quarterback

{_top_table(rankings.loc[rankings['position']=='QB'], POS_LIMITS['QB'])}

### Running back

{_top_table(rankings.loc[rankings['position']=='RB'], POS_LIMITS['RB'])}

### Wide receiver

{_top_table(rankings.loc[rankings['position']=='WR'], POS_LIMITS['WR'])}

### Tight end

{_top_table(rankings.loc[rankings['position']=='TE'], POS_LIMITS['TE'])}

## Statistical standouts

Players with the most extreme z-scores on chunk-play rate, red-zone value, vacated usage, or aging (regardless of rank):

| Player | Pos | Team | Extreme edge | Model pts | ADP |
| --- | --- | --- | --- | ---: | ---: |
"""
    zcols = [c for c in ["z_explosive", "z_redzone", "z_vacated", "z_td_luck", "z_usage", "z_age"] if c in standouts.columns]
    for rec in standouts.head(15).itertuples(index=False):
        edge = max(zcols, key=lambda c: abs(getattr(rec, c, 0) or 0)) if zcols else ""
        md += f"| {rec.player_name} | {rec.position} | {rec.team} | {edge}={_fmt(getattr(rec, edge, 0), 2)} | {_fmt(rec.model_fp)} | {_fmt(rec.adp)} |\n"

    md += """

## Notes for the next iteration

- Drop a fuller prop board into `data/external/season_props.csv` (player, team, market, line, over_odds, under_odds). Markets: `pass_yd`, `pass_td`, `rush_yd`, `rush_td`, `rec`, `rec_yd`, `rec_td`, `int`. Running this on a home network can also pick up DraftKings futures that datacenter IPs block.
- Coordinator tracking is currently head-coach + team pass rate over expected. Mapping OC job changes explicitly is the highest-leverage data upgrade.
- Historical player props are the missing piece for a true VFP backtest; ADP is the stand-in until an archive exists.

*Sources: nflverse play-by-play/stats/rosters/injuries/combine (CC BY 4.0), Fantasy Football Calculator ADP, publicly posted season-long totals. Not betting advice.*
"""
    return md


def _write_html(
    season: int,
    ppr: pd.DataFrame,
    half: pd.DataFrame,
    backtest_md: str,
    extra_md: str,
    steal_eval: dict,
    embed: bool = False,
) -> str:
    payload = {"ppr": _board_payload(ppr), "half": _board_payload(half)}
    extra_html = _md_list_to_html(extra_md)
    backtest_html = _md_list_to_html(backtest_md)
    hit = steal_eval.get("steal_hit_rate", float("nan"))
    fade = steal_eval.get("fade_hit_rate", float("nan"))
    dir_s = steal_eval.get("steal_dir_spearman", float("nan"))
    n_s = steal_eval.get("n_steals", 0)
    n_f = steal_eval.get("n_fades", 0)
    data_json = json.dumps(payload, ensure_ascii=False)
    body_class = "embed" if embed else ""
    intro = (
        ""
        if embed
        else f"""
    <h1>Finding the steals the market still misprices</h1>
    <p class="dek">{date.today().isoformat()} · Model trained on {season-8}–{season-1}. Toggle scoring to rerank; switch Full board vs Positional for overall pick order vs position lists. Start Draft to cross off names as they come off the board.</p>

    <div class="prose">
      <p>Vegas season totals are the cleanest public summary of how sharp books think a player’s counting stats will land. This model starts there — converting no-vig player props into a fantasy-point baseline (VFP) — then looks for the things a posted yardage line is structurally bad at seeing: aging, offensive line quality, coordinator pace/pass rate, chunk-play creation, red-zone roles, indoor/outdoor schedule, vacated touches, and injury residue. Where a player has no posted prop, the market proxy is Fantasy Football Calculator ADP translated into expected points.</p>
      <p>The point is not to out-project every volume total. Books are better at that than a public model. The point is to <strong>rank players relative to the market</strong> so you can see who is a round too cheap.</p>
      <p class="note">Half PPR subtracts 0.5 points per projected reception from the Full PPR projection and, when available, uses Half PPR ADP for the vs-market column. Pass-catching backs and slot receivers move down relative to rushing volume and touchdown-driven work. The full board is overall model rank among drafted-pool players — pick 1 is 1.01, pick 25 is the start of the 3rd in a 12-team league — and still shows positional rank (QB5, RB12).</p>

      <h2>How the projection is built</h2>
      <ol>
        <li><strong>Vegas Fantasy Points (VFP).</strong> Season-long over/unders for passing/rushing/receiving yards, TDs, and receptions are vig-stripped and scored. Juice like −120/−100 is not treated as a 50/50 median.</li>
        <li><strong>Tenure-weighted blend.</strong> Rookies lean on draft capital, measurables, vacated opportunity, and scheme. Prime veterans lean on the market plus last year’s production. Aging veterans get a quadratic positional decay (QB after 30, RB after 26.5, WR after 28, TE after 28.5) with lambdas fit from history, not guessed.</li>
        <li><strong>Gradient boosting residual.</strong> A position-aware model is trained leave-one-season-out. The residual is what ADP/VFP miss: TD luck, workload cliffs, injury bounce-backs, player-assigned vacated usage, pass-catching RB work, and efficiency regression.</li>
        <li><strong>Steal / fade flags.</strong> ADP 18–132, plus at least 5 position ranks of model lift <em>and</em> 12+ points of edge vs the ADP-implied curve. Both have to agree on sign.</li>
      </ol>

      <h2>Backtest</h2>
      <p>Trained on prior seasons only. Spearman is on ADP-drafted players only, so it is a fair comparison against ADP.</p>
      {backtest_html}
      <p>Steal flags historically hit <strong>{hit:.0%}</strong> ({n_s} flags). Fades hit <strong>{fade:.0%}</strong> ({n_f} flags). Directional Spearman of predicted vs actual rank lift: <strong>{dir_s:.3f}</strong>.</p>

      <h2>Extra stats that still beat the market</h2>
      {extra_html}
    </div>
"""
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{season} Fantasy Rankings — PPR vs Half PPR</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --ink: #1a1a1a;
      --muted: #5c574e;
      --line: #d9d2c5;
      --card: #fffdf8;
      --accent: #1d4f46;
      --steal: #0f6b4c;
      --fade: #9b2c2c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      line-height: 1.55;
    }}
    .bar {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      padding: 12px 24px;
      background: var(--card);
      border-bottom: 1px solid var(--line);
    }}
    .bar-title {{ font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
    .bar-controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .toggle {{
      display: inline-flex;
      border: 1px solid var(--ink);
      border-radius: 999px;
      overflow: hidden;
    }}
    .toggle button, .draft-btn {{
      appearance: none;
      border: 0;
      background: transparent;
      padding: 8px 16px;
      font: 600 13px/1.2 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
      color: var(--ink);
    }}
    .toggle button.on {{ background: var(--accent); color: #fff; }}
    .draft-btn {{
      border: 1px solid var(--ink);
      border-radius: 999px;
      background: var(--card);
    }}
    .draft-btn.on {{ background: var(--ink); color: #fff; }}
    .draft-btn.ghost {{ border-style: dashed; color: var(--muted); }}
    .pick-count {{ font: 600 12px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--muted); }}
    main {{ max-width: 980px; margin: 0 auto; padding: 32px 24px 80px; }}
    h1 {{ font-size: 34px; line-height: 1.15; margin: 0 0 8px; }}
    .dek {{ color: var(--muted); font-size: 18px; margin: 0 0 28px; }}
    h2 {{ font-size: 22px; margin: 36px 0 12px; }}
    h3 {{ font-size: 16px; letter-spacing: 0.06em; text-transform: uppercase; margin: 28px 0 10px; }}
    p, li {{ font-size: 17px; }}
    .prose ol {{ padding-left: 22px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
      background: var(--card);
    }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.check, th.check {{ width: 28px; text-align: center; }}
    body:not(.draft-on) .check {{ display: none; }}
    tr.taken td {{ color: var(--muted); }}
    tr.taken td.name {{ text-decoration: line-through; }}
    tr.steal td.name {{ color: var(--steal); font-weight: 650; }}
    tr.fade td.name {{ color: var(--fade); font-weight: 650; }}
    tr.taken.steal td.name, tr.taken.fade td.name {{ color: var(--muted); }}
    tr.round-break td {{
      background: var(--bg);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
      padding: 10px 8px 6px;
    }}
    .pill {{
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-left: 6px;
    }}
    .pill.steal {{ color: var(--steal); }}
    .pill.fade {{ color: var(--fade); }}
    .pos-tag {{
      font: 600 11px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--muted);
      margin-left: 6px;
    }}
    .vs-pos {{ color: var(--steal); }}
    .vs-neg {{ color: var(--fade); }}
    .note {{ color: var(--muted); font-size: 14px; }}
    footer {{ color: var(--muted); font-size: 13px; margin-top: 48px; }}
    body.embed {{ background: var(--card); }}
    body.embed main {{ max-width: none; padding: 8px 12px 28px; }}
    body.embed h2 {{ margin-top: 20px; font-size: 18px; }}
    @media (max-width: 720px) {{
      h1 {{ font-size: 26px; }}
      .bar {{ flex-direction: column; align-items: flex-start; }}
    }}
  </style>
</head>
<body class="{body_class}">
  <div class="bar">
    <div class="bar-title">{season} rankings · <span id="scoring-label">Full PPR</span> · 12-team</div>
    <div class="bar-controls">
      <div class="toggle" role="tablist" aria-label="Scoring format">
        <button type="button" class="on" data-score="ppr" onclick="setScoring('ppr')">Full PPR</button>
        <button type="button" data-score="half" onclick="setScoring('half')">Half PPR</button>
      </div>
      <div class="toggle" role="tablist" aria-label="Board layout">
        <button type="button" class="on" data-board="full" onclick="setBoard('full')">Full board</button>
        <button type="button" data-board="pos" onclick="setBoard('pos')">Positional</button>
      </div>
      <button type="button" class="draft-btn" id="draft-toggle" onclick="toggleDraft()">Start Draft</button>
      <button type="button" class="draft-btn ghost" id="draft-clear" hidden onclick="clearDraft()">Clear picks</button>
      <span class="pick-count" id="pick-count" hidden></span>
    </div>
  </div>
  <main>
    {intro}
    <h2>Steals</h2>
    <p class="note" id="steal-empty" hidden>No steal flags at the current threshold.</p>
    <table id="steals-table"><thead></thead><tbody></tbody></table>

    <h2>Fades</h2>
    <p class="note" id="fade-empty" hidden>No fade flags at the current threshold.</p>
    <table id="fades-table"><thead></thead><tbody></tbody></table>

    <h2 id="rankings-title">Full board</h2>
    <p class="note" id="board-note">Overall rank is the 12-team pick. Round breaks mark every 12 picks. Positional rank stays on the name (QB5).</p>
    <div id="full-wrap">
      <table id="table-full"><thead></thead><tbody></tbody></table>
    </div>
    <div id="pos-wrap" hidden>
      <h3>Quarterback</h3>
      <table id="table-QB"><thead></thead><tbody></tbody></table>
      <h3>Running back</h3>
      <table id="table-RB"><thead></thead><tbody></tbody></table>
      <h3>Wide receiver</h3>
      <table id="table-WR"><thead></thead><tbody></tbody></table>
      <h3>Tight end</h3>
      <table id="table-TE"><thead></thead><tbody></tbody></table>
    </div>

    <footer>Sources: nflverse play-by-play/stats/rosters/injuries/combine (CC BY 4.0), Fantasy Football Calculator ADP, publicly posted season-long totals. Not betting advice.</footer>
  </main>
  <script>
    const DATA = {data_json};
    const STORAGE_KEY = "ffmodel-drafted-{season}";
    const ROUND_SIZE = 12;
    let scoring = "ppr";
    let board = "full";
    let draftOn = false;
    const drafted = new Set(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"));

    function setScoring(next) {{
      scoring = next;
      document.querySelectorAll("[data-score]").forEach(b => b.classList.toggle("on", b.dataset.score === next));
      document.getElementById("scoring-label").textContent = next === "ppr" ? "Full PPR" : "Half PPR";
      render();
    }}

    function setBoard(next) {{
      board = next;
      document.querySelectorAll("[data-board]").forEach(b => b.classList.toggle("on", b.dataset.board === next));
      document.getElementById("full-wrap").hidden = next !== "full";
      document.getElementById("pos-wrap").hidden = next !== "pos";
      document.getElementById("rankings-title").textContent = next === "full" ? "Full board" : "Positional rankings";
      document.getElementById("board-note").textContent = next === "full"
        ? "Overall rank is the 12-team pick. Round breaks mark every 12 picks. Positional rank stays on the name (QB5)."
        : "Each list is ranked within position. Switch to Full board to see overall pick / round.";
    }}

    function toggleDraft() {{
      draftOn = !draftOn;
      document.body.classList.toggle("draft-on", draftOn);
      const btn = document.getElementById("draft-toggle");
      btn.classList.toggle("on", draftOn);
      btn.textContent = draftOn ? "Exit Draft" : "Start Draft";
      document.getElementById("draft-clear").hidden = !draftOn;
      updatePickCount();
      render();
    }}

    function clearDraft() {{
      drafted.clear();
      localStorage.setItem(STORAGE_KEY, "[]");
      updatePickCount();
      render();
    }}

    function markTaken(id) {{
      if (drafted.has(id)) drafted.delete(id);
      else drafted.add(id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify([...drafted]));
      updatePickCount();
      render();
    }}

    function updatePickCount() {{
      const el = document.getElementById("pick-count");
      el.hidden = !draftOn;
      el.textContent = drafted.size ? drafted.size + " off the board" : "no picks yet";
    }}

    function ordinal(n) {{
      const j = n % 10, k = n % 100;
      if (j === 1 && k !== 11) return n + "st";
      if (j === 2 && k !== 12) return n + "nd";
      if (j === 3 && k !== 13) return n + "rd";
      return n + "th";
    }}

    function vsCell(n) {{
      const t = n > 0 ? "+" + n : String(n);
      const cls = n > 0 ? "vs-pos" : (n < 0 ? "vs-neg" : "");
      return `<td class="num ${{cls}}">${{t}}</td>`;
    }}

    function nameCell(row, showPosRank) {{
      const pill = row.flag ? `<span class="pill ${{row.flag}}">${{row.flag}}</span>` : "";
      const tag = showPosRank ? `<span class="pos-tag">${{escapeHtml(row.posRank)}}</span>` : "";
      return `<td class="name">${{escapeHtml(row.name)}}${{tag}}${{pill}}</td>`;
    }}

    function checkCell(row) {{
      const on = drafted.has(row.id);
      return `<td class="check"><input type="checkbox" ${{on ? "checked" : ""}} onclick="markTaken('${{String(row.id).replace(/'/g, "\\\\'")}}')" aria-label="Drafted ${{escapeHtml(row.name)}}"></td>`;
    }}

    function escapeHtml(s) {{
      return String(s).replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
    }}

    function numHead(h) {{
      return h === "Pick" || h === "Rank" || h === "ADP" || h === "Model pts" || h === "vs ADP";
    }}

    function writeHead(table, heads) {{
      table.querySelector("thead").innerHTML = "<tr>" + heads.map(h =>
        `<th class="${{h === "" ? "check" : (numHead(h) ? "num" : "")}}">${{h}}</th>`
      ).join("") + "</tr>";
    }}

    function playerRow(row, rank, extraCells, showPosRank) {{
      const taken = drafted.has(row.id) ? " taken" : "";
      return `<tr class="${{row.flag}}${{taken}}">${{checkCell(row)}}<td class="num">${{rank}}</td>${{nameCell(row, showPosRank)}}${{extraCells}}<td>${{escapeHtml(row.team)}}</td><td class="num">${{row.adp ?? "—"}}</td><td class="num">${{row.fp ?? "—"}}</td>${{vsCell(row.vs)}}</tr>`;
    }}

    function fillPosTable(id, rows) {{
      const table = document.getElementById(id);
      writeHead(table, ["", "Rank", "Player", "Team", "ADP", "Model pts", "vs ADP"]);
      table.querySelector("tbody").innerHTML = rows.map(r => playerRow(r, r.rank, "", false)).join("");
    }}

    function fillListTable(id, rows) {{
      const table = document.getElementById(id);
      writeHead(table, ["", "Rank", "Player", "Pos", "Team", "ADP", "Model pts", "vs ADP"]);
      table.querySelector("tbody").innerHTML = rows.map(r =>
        playerRow(r, r.rank, `<td>${{escapeHtml(r.posRank)}}</td>`, false)
      ).join("");
    }}

    function fillFullTable(rows) {{
      const table = document.getElementById("table-full");
      writeHead(table, ["", "Pick", "Player", "Team", "ADP", "Model pts", "vs ADP"]);
      const parts = [];
      rows.forEach((r, i) => {{
        const pick = r.ov || r.rank;
        if (i === 0 || (pick - 1) % ROUND_SIZE === 0) {{
          const rnd = Math.floor((pick - 1) / ROUND_SIZE) + 1;
          const start = (rnd - 1) * ROUND_SIZE + 1;
          const end = rnd * ROUND_SIZE;
          parts.push(`<tr class="round-break"><td colspan="7">Round ${{rnd}} · ${{ordinal(rnd)}} · picks ${{start}}–${{end}}</td></tr>`);
        }}
        parts.push(playerRow(r, pick, "", true));
      }});
      table.querySelector("tbody").innerHTML = parts.join("");
    }}

    function renderList(kind, rows) {{
      const empty = document.getElementById(kind + "-empty");
      const table = document.getElementById(kind + "s-table");
      if (!rows.length) {{
        empty.hidden = false;
        table.hidden = true;
        return;
      }}
      empty.hidden = true;
      table.hidden = false;
      fillListTable(kind + "s-table", rows);
    }}

    function render() {{
      const d = DATA[scoring];
      renderList("steal", d.steals);
      renderList("fade", d.fades);
      fillFullTable(d.full);
      ["QB","RB","WR","TE"].forEach(pos => fillPosTable("table-" + pos, d[pos]));
    }}
    setBoard("full");
    updatePickCount();
    render();
  </script>
</body>
</html>
"""
    if embed:
        dest = ROOT / "docs"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ".nojekyll").write_text("")
        path = dest / "index.html"
    else:
        path = PROCESSED_DIR / f"article_{season}.html"
    path.write_text(page)
    return str(path)


def _md_list_to_html(md: str) -> str:
    if not md:
        return "<p class='note'>_Run the pipeline to fill this._</p>"
    items = []
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(f"<li>{_bold_md(line[2:])}</li>")
    if not items:
        return f"<p>{html.escape(md)}</p>"
    return "<ul>" + "".join(items) + "</ul>"


def _bold_md(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        if text.startswith("**", i):
            end = text.find("**", i + 2)
            if end != -1:
                out.append("<strong>" + html.escape(text[i + 2 : end]) + "</strong>")
                i = end + 2
                continue
        if text.startswith("`", i):
            end = text.find("`", i + 1)
            if end != -1:
                out.append("<code>" + html.escape(text[i + 1 : end]) + "</code>")
                i = end + 1
                continue
        out.append(html.escape(text[i]))
        i += 1
    return "".join(out)
