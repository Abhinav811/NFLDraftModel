"""Lagged edges from weekly stats, NGS, primary QB, ECR, and OC changes."""
from __future__ import annotations

import json
import re
from urllib.parse import quote

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, PREDICT_SEASON, RAW_DIR
from ..http import fetch_bytes
from ..ingest.nflverse import load_ngs, load_pbp, load_player_stats_week
from ..names import canon_team, normalize_name

SKILL_POS = {"QB", "RB", "WR", "TE", "FB", "HB", "TB"}
SKILL_ABB = {
    "QB", "RB", "WR", "TE", "FB", "HB", "TB",
    "SLWR", "FL", "SE", "WRX", "WRY", "WRZ", "SWR", "LWR", "RWR",
}


def _empty(*cols: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(cols))


def depth_snapshot(raw: pd.DataFrame, season: int, current: bool) -> pd.DataFrame:
    """Week-1 / last pre-kickoff depth. Avoids end-of-season leakage on holdouts."""
    if raw is None or raw.empty:
        return _empty("player_id", "depth_rank")
    if "depth_team" in raw.columns and "gsis_id" in raw.columns:
        d = raw.loc[raw["position"].isin(SKILL_POS)].copy()
        if "game_type" in d.columns:
            d = d.loc[d["game_type"].fillna("REG").eq("REG")].copy()
        if "week" in d.columns and d["week"].notna().any():
            week = d["week"].max() if current else (1 if (d["week"] == 1).any() else d["week"].min())
            d = d.loc[d["week"] == week].copy()
        d["depth_rank"] = pd.to_numeric(d["depth_team"], errors="coerce")
    elif "pos_rank" in raw.columns and "gsis_id" in raw.columns:
        d = raw.copy()
        abb = d["pos_abb"].astype(str).str.upper() if "pos_abb" in d.columns else pd.Series("", index=d.index)
        grp = d["pos_grp"].astype(str) if "pos_grp" in d.columns else pd.Series("", index=d.index)
        skill = abb.isin(SKILL_ABB) | grp.str.contains(r"WR|RB|TE|QB", case=False, na=False)
        d = d.loc[skill].copy()
        d["dt"] = pd.to_datetime(d.get("dt"), utc=True, errors="coerce")
        if d["dt"].notna().any():
            if current:
                d = d.loc[d["dt"] == d["dt"].max()].copy()
            else:
                cutoff = pd.Timestamp(f"{season}-09-08", tz="UTC")
                prior = d.loc[d["dt"] <= cutoff]
                use = prior if not prior.empty else d
                d = use.loc[use["dt"] == use["dt"].max()].copy()
        d["depth_rank"] = pd.to_numeric(d["pos_rank"], errors="coerce")
    else:
        return _empty("player_id", "depth_rank")
    d = d.dropna(subset=["gsis_id", "depth_rank"])
    d["player_id"] = d["gsis_id"].astype(str).str.strip()
    d = d.sort_values("depth_rank").drop_duplicates("player_id", keep="first")
    return d[["player_id", "depth_rank"]]


def build_h2_features(seasons: list[int]) -> pd.DataFrame:
    weekly = load_player_stats_week(seasons)
    if weekly.empty:
        return _empty("player_id", "season", "carries_h2_delta", "ppr_h2_delta")
    w = weekly.copy()
    if "season_type" in w.columns:
        w = w.loc[w["season_type"].astype(str).str.upper().eq("REG")]
    w["week"] = pd.to_numeric(w["week"], errors="coerce")
    w["player_id"] = w["player_id"].astype(str)
    w["half"] = np.where(w["week"] >= 10, "h2", "h1")
    g = w.groupby(["season", "player_id", "half"], as_index=False).agg(
        carries=("carries", "mean"),
        ppr=("fantasy_points_ppr", "mean"),
    )
    wide = g.pivot(index=["season", "player_id"], columns="half")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["carries_h2_delta"] = pd.to_numeric(wide.get("carries_h2"), errors="coerce") - pd.to_numeric(
        wide.get("carries_h1"), errors="coerce"
    )
    wide["ppr_h2_delta"] = pd.to_numeric(wide.get("ppr_h2"), errors="coerce") - pd.to_numeric(
        wide.get("ppr_h1"), errors="coerce"
    )
    wide["season"] = pd.to_numeric(wide["season"], errors="coerce") + 1
    return wide[["player_id", "season", "carries_h2_delta", "ppr_h2_delta"]]


def _take_named(df: pd.DataFrame, dest: str, aliases: list[str]) -> pd.DataFrame:
    out = df.copy()
    src = next((c for c in aliases if c in out.columns), None)
    if src:
        out[dest] = pd.to_numeric(out[src], errors="coerce")
    return out


def build_ngs_lag() -> pd.DataFrame:
    frames = []
    try:
        rec = _take_named(
            _ngs_season(load_ngs("receiving")),
            "ngs_air_share",
            ["percent_share_of_intended_air_yards", "share_of_intended_air_yards"],
        )
        keep = [c for c in ["player_id", "season", "ngs_air_share"] if c in rec.columns]
        frames.append(rec[keep])
    except Exception as exc:
        print(f"  NGS receiving skipped ({exc})")
    try:
        rush = _take_named(
            _ngs_season(load_ngs("rushing")),
            "ngs_ryoe_att",
            ["rush_yards_over_expected_per_att", "ryoe_per_att", "expected_rush_yards_per_att"],
        )
        keep = [c for c in ["player_id", "season", "ngs_ryoe_att"] if c in rush.columns]
        frames.append(rush[keep])
    except Exception as exc:
        print(f"  NGS rushing skipped ({exc})")
    try:
        pas = _take_named(
            _ngs_season(load_ngs("passing")),
            "ngs_intended_air_yards",
            ["avg_intended_air_yards", "intended_air_yards", "avg_air_yards_to_sticks"],
        )
        keep = [c for c in ["player_id", "season", "ngs_intended_air_yards"] if c in pas.columns]
        frames.append(pas[keep])
    except Exception as exc:
        print(f"  NGS passing skipped ({exc})")
    if not frames:
        return _empty("player_id", "season", "ngs_air_share", "ngs_ryoe_att", "ngs_intended_air_yards")
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on=["player_id", "season"], how="outer")
    out["season"] = pd.to_numeric(out["season"], errors="coerce") + 1
    return out


def _ngs_season(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "week" in out.columns:
        out = out.loc[pd.to_numeric(out["week"], errors="coerce").fillna(-1).eq(0)]
    if "season_type" in out.columns:
        out = out.loc[out["season_type"].astype(str).str.upper().isin(["REG", "nan", ""]) | out["season_type"].isna()]
    pid = out.get("player_gsis_id", out.get("player_id"))
    out["player_id"] = pid.astype(str)
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    return out.drop_duplicates(["player_id", "season"], keep="last")


def qb_from_depth(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Starting QB from a depth snapshot when PBP is not available yet."""
    if raw is None or raw.empty:
        return _empty("season", "team", "qb_id")
    d = raw.copy()
    if "depth_team" in d.columns:
        d = d.loc[d["position"].eq("QB")].copy()
        if "week" in d.columns and d["week"].notna().any():
            d = d.loc[d["week"] == d["week"].max()]
        d["rank"] = pd.to_numeric(d["depth_team"], errors="coerce")
        team_col = "club_code" if "club_code" in d.columns else "team"
    else:
        abb = d["pos_abb"].astype(str).str.upper() if "pos_abb" in d.columns else pd.Series("", index=d.index)
        d = d.loc[abb.eq("QB")].copy()
        d["dt"] = pd.to_datetime(d.get("dt"), utc=True, errors="coerce")
        if d["dt"].notna().any():
            d = d.loc[d["dt"] == d["dt"].max()]
        d["rank"] = pd.to_numeric(d["pos_rank"], errors="coerce")
        team_col = "team" if "team" in d.columns else "club_code"
    d = d.dropna(subset=["gsis_id", "rank"])
    d = d.sort_values("rank").drop_duplicates(team_col, keep="first")
    return pd.DataFrame(
        {
            "season": season,
            "team": d[team_col].map(canon_team),
            "qb_id": d["gsis_id"].astype(str),
        }
    )


def build_team_qb(seasons: list[int]) -> pd.DataFrame:
    """Primary passer by team-season from regular-season PBP, else empty."""
    rows = []
    for season in seasons:
        try:
            pbp = load_pbp(season, columns=["posteam", "passer_player_id", "pass_attempt", "season_type"])
        except Exception:
            continue
        if pbp.empty:
            continue
        plays = pbp.loc[pbp["pass_attempt"].fillna(0).eq(1) & pbp["passer_player_id"].notna()]
        if plays.empty:
            continue
        top = (
            plays.groupby(["posteam", "passer_player_id"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .groupby("posteam", as_index=False)
            .head(1)
        )
        top["season"] = season
        top["team"] = top["posteam"].map(canon_team)
        top["qb_id"] = top["passer_player_id"].astype(str)
        rows.append(top[["season", "team", "qb_id"]])
    if not rows:
        return _empty("season", "team", "qb_id")
    return pd.concat(rows, ignore_index=True).drop_duplicates(["season", "team"])


def attach_qb_change(panel: pd.DataFrame, qb: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if qb.empty:
        out["qb_change"] = 0.0
        out["wr_new_qb"] = 0.0
        out["new_starter_qb"] = 0.0
        return out
    now = qb.rename(columns={"qb_id": "qb_now"})
    prev = qb.rename(columns={"season": "_lag", "qb_id": "qb_prev"})
    prev["season"] = prev["_lag"] + 1
    chg = now.merge(prev[["season", "team", "qb_prev"]], on=["season", "team"], how="left")
    chg["qb_change"] = (
        chg["qb_now"].notna() & chg["qb_prev"].notna() & (chg["qb_now"] != chg["qb_prev"])
    ).astype(float)
    out["team"] = out["team"].map(canon_team)
    out = out.merge(chg[["season", "team", "qb_change", "qb_now"]], on=["season", "team"], how="left")
    out["qb_change"] = out["qb_change"].fillna(0.0)
    pos = out["position"]
    out["wr_new_qb"] = out["qb_change"] * pos.isin(["WR", "TE"]).astype(float)
    out["new_starter_qb"] = (
        out["qb_change"] * pos.eq("QB").astype(float) * (out["player_id"].astype(str) == out["qb_now"].astype(str))
    ).astype(float)
    return out.drop(columns=["qb_now"], errors="ignore")


def load_ecr() -> pd.DataFrame:
    """FantasyPros ECR archive (DynastyProcess). Preseason redraft PPR when tagged."""
    dest = RAW_DIR / "db_fpecr.parquet"
    url = "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr.parquet"
    try:
        if not dest.exists() or dest.stat().st_size < 1000:
            fetch_bytes(url, dest=dest, timeout=180)
        df = pd.read_parquet(dest)
    except Exception:
        gz = RAW_DIR / "db_fpecr.csv.gz"
        try:
            fetch_bytes(
                "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr.csv.gz",
                dest=gz,
                timeout=180,
            )
            df = pd.read_csv(gz)
        except Exception as exc:
            print(f"  ECR archive skipped ({exc})")
            return _empty("season", "name_norm", "position", "ecr")
    if df.empty:
        return _empty("season", "name_norm", "position", "ecr")
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    name_col = next((c for c in ["player", "player_name", "name"] if c in out.columns), None)
    pos_col = next((c for c in ["pos", "position"] if c in out.columns), None)
    ecr_col = "ecr" if "ecr" in out.columns else None
    if not name_col or not pos_col or not ecr_col:
        print(f"  ECR columns unexpected: {list(out.columns)[:20]}")
        return _empty("season", "name_norm", "position", "ecr")
    if "ecr_type" in out.columns:
        out = out.loc[out["ecr_type"].astype(str).str.lower().isin(["rp", "redraft", "ppr", "nan", ""])]
    if "page_type" in out.columns:
        page = out["page_type"].astype(str).str.lower()
        overall = page.str.contains("overall") | page.eq("consensus-overall") | page.str.endswith("overall")
        if overall.any():
            out = out.loc[overall]
    out["name_norm"] = out[name_col].map(normalize_name)
    out["position"] = out[pos_col].astype(str).str.upper().replace({"HB": "RB", "FB": "RB"})
    out["ecr"] = pd.to_numeric(out[ecr_col], errors="coerce")
    date_col = next((c for c in ["scrape_date", "date", "as_of"] if c in out.columns), None)
    if date_col:
        out["_dt"] = pd.to_datetime(out[date_col], errors="coerce")
        out["season"] = out["_dt"].dt.year
        # August–September scrapes are the draft board; later weeks are in-season.
        month = out["_dt"].dt.month
        out = out.loc[month.between(7, 9) | month.isna()]
    elif "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce")
    else:
        out["season"] = PREDICT_SEASON
    out = out.dropna(subset=["name_norm", "ecr"])
    out = out.sort_values("ecr").drop_duplicates(["season", "name_norm", "position"], keep="first")
    print(f"  ECR rows={len(out):,} seasons={sorted(out['season'].dropna().unique())[-6:]}")
    return out[["season", "name_norm", "position", "ecr"]]


def load_oc_table() -> pd.DataFrame:
    """Team-season OC names. Prefers a local CSV; else Wikipedia staff boxes."""
    path = EXTERNAL_DIR / "offensive_coordinators.csv"
    if path.exists() and path.stat().st_size > 50:
        df = pd.read_csv(path)
        df["team"] = df["team"].map(canon_team)
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df["oc"] = df["oc"].astype(str).str.strip()
        df = df.dropna(subset=["season", "team", "oc"])
        if PREDICT_SEASON not in set(pd.to_numeric(df["season"], errors="coerce")):
            try:
                cur = _wiki_current_ocs()
                if not cur.empty:
                    df = pd.concat([df, cur], ignore_index=True).drop_duplicates(["season", "team"], keep="last")
                    df.to_csv(path, index=False)
                    print(f"  OC {PREDICT_SEASON} current list: {len(cur)} teams")
            except Exception as exc:
                print(f"  Current OC list skipped ({exc})")
        return df
    frames = []
    for season in range(2018, PREDICT_SEASON + 1):
        try:
            chunk = _wiki_ocs(season)
        except Exception as exc:
            print(f"  Wiki OC {season} skipped ({exc})")
            chunk = _empty("season", "team", "oc")
        if chunk.empty:
            try:
                chunk = _pfr_ocs(season)
            except Exception as exc:
                print(f"  PFR OC {season} skipped ({exc})")
                chunk = _empty("season", "team", "oc")
        if not chunk.empty:
            frames.append(chunk)
            print(f"  OC {season}: {len(chunk)} teams")
    try:
        current = _wiki_current_ocs()
        if not current.empty:
            frames.append(current)
            print(f"  OC {PREDICT_SEASON} current list: {len(current)} teams")
    except Exception as exc:
        print(f"  Current OC list skipped ({exc})")
    if not frames:
        return _empty("season", "team", "oc")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["season", "team", "oc"])
    out = out.drop_duplicates(["season", "team"], keep="last")
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"  Wrote {path} ({len(out)} rows)")
    return out


_OC_LINE = re.compile(
    r"\*\s*Offensive coordinator\s*[–—-]\s*(?:\[\[([^\]|]+)(?:\|[^\]]*)?\]\]|([^\n]+))",
    re.I,
)

WIKI_FRANCHISE = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def _wiki_team_name(team: str, season: int) -> str:
    if team == "WAS":
        if season <= 2019:
            return "Washington Redskins"
        if season <= 2021:
            return "Washington Football Team"
        return "Washington Commanders"
    if team == "LV":
        return "Oakland Raiders" if season <= 2019 else "Las Vegas Raiders"
    return WIKI_FRANCHISE[team]


def _wiki_ocs(season: int) -> pd.DataFrame:
    import time

    rows = []
    cache = RAW_DIR / "wiki_oc"
    cache.mkdir(parents=True, exist_ok=True)
    for team in WIKI_FRANCHISE:
        title = f"{season} {_wiki_team_name(team, season)} season"
        dest = cache / f"{season}_{team}.json"
        url = (
            "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json"
            f"&page={quote(title)}"
        )
        text = ""
        for attempt in range(3):
            try:
                if dest.exists():
                    raw = dest.read_bytes()
                    if raw[:1] != b"{" or b"too many requests" in raw.lower() or b"<html" in raw.lower()[:80]:
                        dest.unlink(missing_ok=True)
                if not dest.exists():
                    time.sleep(1.1)
                    fetch_bytes(url, dest=dest, timeout=60)
                data = json.loads(dest.read_text())
            except Exception as exc:
                dest.unlink(missing_ok=True)
                if attempt == 2:
                    print(f"  wiki OC {season} {team} skipped ({exc})")
                continue
            text = data.get("parse", {}).get("wikitext", {}).get("*", "") or ""
            if text:
                break
            dest.unlink(missing_ok=True)
        match = _OC_LINE.search(text)
        if not match:
            continue
        oc = (match.group(1) or match.group(2) or "").split("<")[0].strip(" '\"")
        oc = re.sub(r"\s*\(.*$", "", oc).strip()
        if oc:
            rows.append({"season": int(season), "team": team, "oc": oc})
    return pd.DataFrame(rows) if rows else _empty("season", "team", "oc")


def _wiki_current_ocs() -> pd.DataFrame:
    dest = RAW_DIR / "wiki_oc" / "current_ocs.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json"
        f"&page={quote('List of current NFL offensive coordinators')}"
    )
    if not dest.exists() or dest.stat().st_size < 500 or dest.read_bytes()[:1] != b"{":
        fetch_bytes(url, dest=dest, timeout=60)
    data = json.loads(dest.read_text())
    text = data.get("parse", {}).get("wikitext", {}).get("*", "") or ""
    name_to_team = {v: k for k, v in WIKI_FRANCHISE.items()}
    name_to_team.update(
        {
            "Washington Redskins": "WAS",
            "Washington Football Team": "WAS",
            "Oakland Raiders": "LV",
            "Los Angeles Rams": "LA",
            "St. Louis Rams": "LA",
        }
    )
    sortname = re.compile(r"\{\{sortname\|([^}|]+)\|([^}|]+)", re.I)
    rows = []
    for line in text.splitlines():
        if "||" not in line or "[[" not in line:
            continue
        parts = [p.strip() for p in line.split("||")]
        if len(parts) < 2:
            continue
        tm = re.search(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", parts[0])
        if not tm:
            continue
        team = name_to_team.get(tm.group(1).strip())
        if not team:
            continue
        sm = sortname.search(parts[1])
        oc = f"{sm.group(1).strip()} {sm.group(2).strip()}" if sm else ""
        oc = re.sub(r"<.*$", "", oc).strip()
        if oc:
            rows.append({"season": PREDICT_SEASON, "team": team, "oc": oc})
    return pd.DataFrame(rows).drop_duplicates("team") if rows else _empty("season", "team", "oc")


def _pfr_ocs(season: int) -> pd.DataFrame:
    url = f"https://www.pro-football-reference.com/years/{season}/coaches.htm"
    dest = RAW_DIR / f"pfr_coaches_{season}.html"
    if dest.exists() and dest.stat().st_size > 200:
        head = dest.read_text(errors="replace")[:400].lower()
        if "just a moment" in head or "cloudflare" in head or "captcha" in head:
            dest.unlink(missing_ok=True)
    if not dest.exists() or dest.stat().st_size < 500:
        fetch_bytes(url, dest=dest, timeout=60)
    head = dest.read_text(errors="replace")[:400].lower() if dest.exists() else ""
    if "just a moment" in head or "cloudflare" in head:
        return _empty("season", "team", "oc")
    tables = pd.read_html(dest, flavor="lxml")
    rows = []
    for tbl in tables:
        cols = [str(c).lower() for c in tbl.columns]
        flat = [" ".join(map(str, c)).lower() if isinstance(c, tuple) else str(c).lower() for c in tbl.columns]
        blob = " ".join(flat)
        if "off" not in blob and "coordinator" not in blob and "oc" not in blob:
            # Some PFR pages put assistants in a Role column.
            if "role" not in blob and "coordinator" not in blob:
                continue
        team_col = next((c for c in tbl.columns if str(c).lower() in {"tm", "team", "t"}), None)
        name_col = next((c for c in tbl.columns if str(c).lower() in {"coach", "name"}), tbl.columns[0])
        role_col = next((c for c in tbl.columns if "role" in str(c).lower() or "coord" in str(c).lower()), None)
        oc_col = next((c for c in tbl.columns if "off" in str(c).lower() and "coord" in str(c).lower()), None)
        if oc_col is not None and team_col is not None:
            for rec in tbl[[team_col, oc_col]].itertuples(index=False):
                team, oc = rec[0], rec[1]
                if pd.notna(oc) and str(oc).strip() not in {"", "nan"}:
                    rows.append({"season": season, "team": canon_team(team), "oc": str(oc).strip()})
            continue
        if role_col is not None and team_col is not None:
            sub = tbl.loc[tbl[role_col].astype(str).str.contains("offens", case=False, na=False)]
            for rec in sub[[team_col, name_col]].itertuples(index=False):
                rows.append({"season": season, "team": canon_team(rec[0]), "oc": str(rec[1]).strip()})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.loc[out["team"].ne("")].drop_duplicates(["season", "team"])
    return out


def attach_new_oc(panel: pd.DataFrame, oc: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if oc.empty:
        out["new_oc"] = 0.0
        return out
    now = oc.rename(columns={"oc": "oc_now"})
    prev = oc.rename(columns={"season": "_lag", "oc": "oc_prev"})
    prev["season"] = prev["_lag"] + 1
    chg = now.merge(prev[["season", "team", "oc_prev"]], on=["season", "team"], how="left")
    now_s = chg["oc_now"].astype(str).str.strip().str.lower()
    prev_s = chg["oc_prev"].astype(str).str.strip().str.lower()
    chg["new_oc"] = (
        chg["oc_now"].notna()
        & chg["oc_prev"].notna()
        & now_s.ne("nan")
        & prev_s.ne("nan")
        & now_s.ne(prev_s)
    ).astype(float)
    out["team"] = out["team"].map(canon_team)
    out = out.merge(chg[["season", "team", "new_oc"]], on=["season", "team"], how="left")
    out["new_oc"] = out["new_oc"].fillna(0.0)
    return out
