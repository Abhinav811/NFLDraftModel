from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from ..config import EXTERNAL_DIR, FTA_PROPS_URL, PREDICT_SEASON, ROOT
from ..http import fetch_json, fetch_text
from ..names import normalize_name

MARKET_MAP = {
    "passing yards": "pass_yd",
    "pass yards": "pass_yd",
    "passing tds": "pass_td",
    "passing touchdowns": "pass_td",
    "rushing yards": "rush_yd",
    "rush yards": "rush_yd",
    "rushing tds": "rush_td",
    "rushing touchdowns": "rush_td",
    "receiving yards": "rec_yd",
    "receptions": "rec",
    "receiving tds": "rec_td",
    "receiving touchdowns": "rec_td",
    "interceptions": "int",
}

# Wide season-long board (user-supplied). Ignore Projections / 7-Day Delta.
WIDE_PROP_COLUMNS = {
    "Pass Yards": "pass_yd",
    "Pass TDs": "pass_td",
    "Ints": "int",
    "Receptions": "rec",
    "Rec Yards": "rec_yd",
    "Rec TDs": "rec_td",
    "Rush Yards": "rush_yd",
    "Rush TDs": "rush_td",
    "Fumbles": "fum",
    "Attempts": "pass_att",
    "Comps": "pass_comp",
    "Rush Attempts": "rush_att",
}

WIDE_PROP_SEARCH_PATHS = [
    EXTERNAL_DIR / "season_long_props_wide.csv",
    ROOT / "data" / "external" / "season_long_props_wide.csv",
    Path.home() / "Downloads" / "season_long_proj_table (1).csv",
    Path.home() / "Downloads" / "season_long_proj_table.csv",
]

MARKET_MAP = {
    "passing yards": "pass_yd",
    "pass yards": "pass_yd",
    "passing tds": "pass_td",
    "passing touchdowns": "pass_td",
    "rushing yards": "rush_yd",
    "rush yards": "rush_yd",
    "rushing tds": "rush_td",
    "rushing touchdowns": "rush_td",
    "receiving yards": "rec_yd",
    "receptions": "rec",
    "receiving tds": "rec_td",
    "receiving touchdowns": "rec_td",
    "interceptions": "int",
}


def load_adp(year: int, teams: int = 12, scoring: str = "ppr") -> pd.DataFrame:
    scoring = {"half": "half-ppr", "half_ppr": "half-ppr", "0.5": "half-ppr"}.get(scoring, scoring)
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{scoring}?teams={teams}&year={year}"
    payload = fetch_json(url)
    players = payload.get("players") or []
    df = pd.DataFrame(players)
    if df.empty:
        return df
    df["season"] = year
    df["adp_name_norm"] = df["name"].map(normalize_name)
    df["position"] = df["position"].replace({"PK": "K", "DEF": "DST"})
    return df


def load_adp_history(years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        try:
            frames.append(load_adp(year))
        except Exception as exc:
            print(f"  ADP {year} failed: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class _FTATableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.section = ""
        self.in_td = False
        self.row: list[str] = []
        self.rows: list[tuple[str, list[str]]] = []
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h2", "h3"}:
            self._buf = []
        if tag == "td":
            self.in_td = True
            self._buf = []
        if tag == "tr":
            self.row = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h2", "h3"}:
            text = " ".join(self._buf).strip().lower()
            if "o/u" in text or "yards" in text or "tds" in text or "reception" in text:
                self.section = text
        if tag == "td" and self.in_td:
            self.row.append(" ".join(self._buf).strip())
            self.in_td = False
        if tag == "tr" and self.row:
            self.rows.append((self.section, self.row))
            self.row = []

    def handle_data(self, data: str) -> None:
        if self.in_td or True:
            self._buf.append(data.strip())


def scrape_fta_props() -> pd.DataFrame:
    try:
        html = fetch_text(FTA_PROPS_URL)
    except Exception as exc:
        print(f"  FTA scrape failed: {exc}")
        return pd.DataFrame()
    parser = _FTATableParser()
    parser.feed(html)
    records = []
    for section, row in parser.rows:
        market = None
        for key, code in MARKET_MAP.items():
            if key in section:
                market = code
                break
        if market is None or len(row) < 4:
            continue
        name = row[0]
        if not re.search(r"[A-Za-z]", name) or name.lower() in {"player", "name"}:
            continue
        line = None
        over_odds = under_odds = None
        book = row[2] if len(row) > 2 else ""
        nums = []
        for cell in row:
            cell = cell.replace(",", "").strip()
            if re.fullmatch(r"-?\d+\.5", cell) or re.fullmatch(r"-?\d+", cell):
                nums.append(cell)
        if not nums:
            continue
        # Typical layout: player, team, book, line, over, under
        if len(nums) >= 3:
            line, over_odds, under_odds = nums[0], nums[1], nums[2]
        elif len(nums) >= 1:
            line = nums[0]
        records.append(
            {
                "player_name": name,
                "team": row[1] if len(row) > 1 else "",
                "market": market,
                "line": float(line),
                "over_odds": pd.to_numeric(over_odds, errors="coerce"),
                "under_odds": pd.to_numeric(under_odds, errors="coerce"),
                "book": book,
                "source": "fta",
            }
        )
    return pd.DataFrame(records)


def scrape_draftkings_props() -> pd.DataFrame:
    """Best-effort DK futures pull. Datacenter IPs are often blocked; local runs may work."""
    try:
        from ..http import fetch_json
        from ..config import DK_NFL_EVENTGROUP

        payload = fetch_json(DK_NFL_EVENTGROUP)
    except Exception as exc:
        print(f"  DraftKings blocked or unavailable: {exc}")
        return pd.DataFrame()
    cats = (payload.get("eventGroup") or {}).get("offerCategories") or []
    interesting = []
    for cat in cats:
        name = (cat.get("name") or "").lower()
        if any(tok in name for tok in ("player", "future", "award", "passing", "rushing", "receiving")):
            interesting.append((cat.get("offerCategoryId"), cat.get("name")))
    if not interesting:
        print("  DraftKings returned categories but no player-future group.")
    return pd.DataFrame()


def load_wide_prop_board() -> pd.DataFrame:
    """Load the user season-long prop table. Drops Rank / Projections / 7-Day Delta."""
    path = next((p for p in WIDE_PROP_SEARCH_PATHS if p.exists()), None)
    if path is None:
        print("  No wide season-long prop board found.")
        return pd.DataFrame()
    print(f"  Wide props: {path}")
    raw = pd.read_csv(path)
    raw = raw.loc[raw["Pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    drop = [c for c in ["Rank", "Projections", "7-Day Delta"] if c in raw.columns]
    raw = raw.drop(columns=drop)
    value_cols = [c for c in WIDE_PROP_COLUMNS if c in raw.columns]
    long = raw.melt(
        id_vars=["Name", "Pos"],
        value_vars=value_cols,
        var_name="src",
        value_name="line",
    ).dropna(subset=["line"])
    long["player_name"] = long["Name"].astype(str)
    long["position"] = long["Pos"].astype(str)
    long["team"] = ""
    long["market"] = long["src"].map(WIDE_PROP_COLUMNS)
    long["over_odds"] = -110
    long["under_odds"] = -110
    long["book"] = "season_long_board"
    long["source"] = "wide_csv"
    return long[
        ["player_name", "position", "team", "market", "line", "over_odds", "under_odds", "book", "source"]
    ]


def load_season_props() -> pd.DataFrame:
    parts = []
    seeded_path = EXTERNAL_DIR / "season_props.csv"
    if seeded_path.exists():
        parts.append(pd.read_csv(seeded_path))
    scraped = scrape_fta_props()
    if scraped is not None and not scraped.empty:
        parts.append(scraped)
    dk = scrape_draftkings_props()
    if dk is not None and not dk.empty:
        parts.append(dk)
    wide = load_wide_prop_board()
    if wide is not None and not wide.empty:
        parts.append(wide)
    if not parts:
        return pd.DataFrame()
    props = pd.concat(parts, ignore_index=True)
    props["player_name"] = props["player_name"].astype(str).str.strip()
    props["name_norm"] = props["player_name"].map(normalize_name)
    props["season"] = PREDICT_SEASON
    props = props.drop_duplicates(["name_norm", "market"], keep="last")
    print(f"  Prop rows={len(props):,} players={props['name_norm'].nunique()}")
    return props
