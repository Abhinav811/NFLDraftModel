from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..config import PREDICT_SEASON
from .nflverse import (
    load_injuries,
    load_release_timestamp,
    load_rosters,
    load_weekly_rosters,
)
from .sleeper import load_sleeper_players

# Official weekly report_status, plus practice-only labels before game designations.
_REPORT_BADGE = {
    "out": "O",
    "doubtful": "D",
    "questionable": "Q",
    "injured reserve": "IR",
    "ir": "IR",
    "pup": "PUP",
    "physically unable to perform": "PUP",
}

_PRACTICE_BADGE = {
    "did not participate in practice": "DNP",
    "limited participation in practice": "LP",
}

_SLEEPER_INJ = {
    "questionable": "Q",
    "doubtful": "D",
    "out": "O",
    "ir": "IR",
    "pup": "PUP",
    "sus": "SUS",
    "suspended": "SUS",
}

_SLEEPER_STATUS = {
    "injured reserve": "IR",
    "physically unable to perform": "PUP",
    "pup": "PUP",
    "suspended": "SUS",
}

# status_description_abbr → short board label. Skip A01 (active) and R09
# (current-season rookies often sit here before the 90-man list is clean).
_ABBR_BADGE = {
    "R01": "IR",
    "R04": "PUP",
    "R05": "NFI",
    "R27": "NFI",
    "R34": "IR",
    "R36": "IR",
    "R37": "NFI",
    "R47": "NFI",
    "R48": "IR",
    "R02": "RET",
    "R30": "SUS",
    "R33": "SUS",
    "R40": "SUS",
    "R41": "NFI",
}

_STATUS_BADGE = {
    "PUP": "PUP",
    "SUS": "SUS",
    "RET": "RET",
    "IR": "IR",
    "NFI": "NFI",
}

_SKIP_ABBR = {"A01", "E14", "R09"}
_SKIP_STATUS = {"ACT", "E14", "DEV", "INA"}
_HARD = {"IR", "PUP", "NFI", "SUS", "RET"}


def _clean(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    text = str(val).strip()
    if not text or text.lower() in {"nan", "none", "nat", "na", "null"}:
        return ""
    return text


def _latest_roster(season: int) -> pd.DataFrame:
    weekly = load_weekly_rosters([season])
    if not weekly.empty and "week" in weekly.columns:
        latest = weekly.loc[weekly["week"] == weekly["week"].max()].copy()
        if not latest.empty:
            return latest
    return load_rosters([season])


def _roster_badge(row) -> tuple[str, str]:
    abbr = _clean(row.get("status_description_abbr")).upper()
    status = _clean(row.get("status")).upper()
    if abbr in _SKIP_ABBR or status in _SKIP_STATUS:
        return "", ""
    badge = _ABBR_BADGE.get(abbr) or _STATUS_BADGE.get(status)
    if not badge:
        return "", ""
    tip = {
        "IR": "injured reserve",
        "PUP": "physically unable to perform",
        "NFI": "non-football injury/illness",
        "SUS": "suspended",
        "RET": "retired",
    }.get(badge, badge.lower())
    return badge, tip


def _injury_badge(row) -> tuple[str, str]:
    report = _clean(row.get("report_status")).lower()
    practice = _clean(row.get("practice_status")).lower()
    injury = _clean(row.get("report_primary_injury")) or _clean(row.get("practice_primary_injury"))
    badge = _REPORT_BADGE.get(report)
    if not badge:
        badge = _PRACTICE_BADGE.get(practice)
    if not badge:
        return "", ""
    tip = " · ".join(p for p in [injury.lower() if injury else "", badge] if p)
    return badge, tip


def _sleeper_badge(row) -> tuple[str, str]:
    inj = _clean(row.get("injury_status")).lower()
    status = _clean(row.get("status")).lower()
    badge = _SLEEPER_INJ.get(inj) or _SLEEPER_STATUS.get(status)
    if not badge:
        return "", ""
    body = _clean(row.get("injury_body_part"))
    notes = _clean(row.get("injury_notes"))
    tip = " · ".join(p for p in [body, notes] if p)
    if not tip:
        tip = {
            "Q": "questionable",
            "D": "doubtful",
            "O": "out",
            "IR": "injured reserve",
            "PUP": "physically unable to perform",
            "SUS": "suspended",
        }.get(badge, badge.lower())
    return badge, tip


def _nflverse_reports_ready(injuries: pd.DataFrame) -> bool:
    if injuries.empty or "report_status" not in injuries.columns:
        return False
    status = injuries["report_status"].fillna("").astype(str).str.strip().str.lower()
    return int(status.isin(_REPORT_BADGE).sum()) >= 20


def _sleeper_reports(roster: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    players = load_sleeper_players()
    if players.empty:
        return pd.DataFrame(columns=["player_id", "report_inj", "report_tip"]), ""
    cross = roster.dropna(subset=["gsis_id"]).copy()
    cross["player_id"] = cross["gsis_id"].astype(str).str.strip()
    if "sleeper_id" in cross.columns:
        cross["sleeper_id"] = cross["sleeper_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    else:
        cross["sleeper_id"] = ""

    by_gsis = players.loc[players["gsis_id"].ne("")].drop_duplicates("gsis_id")
    mapped = by_gsis.rename(columns={"gsis_id": "player_id"})[["player_id", "injury_status", "status", "injury_body_part", "injury_notes"]]

    if "sleeper_id" in cross.columns:
        by_sl = players.drop_duplicates("sleeper_id")
        via = cross.loc[cross["sleeper_id"].ne("") & cross["sleeper_id"].ne("nan"), ["player_id", "sleeper_id"]]
        via = via.merge(by_sl, on="sleeper_id", how="inner")
        extra = via[["player_id", "injury_status", "status", "injury_body_part", "injury_notes"]]
        mapped = pd.concat([mapped, extra], ignore_index=True).drop_duplicates("player_id")

    rows = []
    for rec in mapped.to_dict("records"):
        badge, tip = _sleeper_badge(rec)
        if badge:
            rows.append({"player_id": rec["player_id"], "report_inj": badge, "report_tip": tip})
    cached = players["_cached_at"].iloc[0] if "_cached_at" in players.columns and len(players) else None
    stamp = ""
    if cached:
        stamp = datetime.fromtimestamp(float(cached), tz=timezone.utc).strftime("%Y-%m-%d")
        stamp = f"Sleeper · {stamp}"
    else:
        stamp = "Sleeper"
    return pd.DataFrame(rows), stamp


def load_live_status(season: int = PREDICT_SEASON) -> tuple[pd.DataFrame, str]:
    """Live designations keyed by gsis_id. Sleeper until nflverse weekly reports exist."""
    roster = _latest_roster(season)
    if roster.empty or "gsis_id" not in roster.columns:
        return pd.DataFrame(columns=["player_id", "inj", "inj_tip"]), ""

    rost = roster.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id", keep="last").copy()
    rost_rows = []
    for rec in rost.to_dict("records"):
        badge, tip = _roster_badge(rec)
        rost_rows.append({"player_id": str(rec["gsis_id"]).strip(), "roster_inj": badge, "roster_tip": tip})
    status = pd.DataFrame(rost_rows)

    injuries = load_injuries([season])
    source = "nflverse roster"
    if _nflverse_reports_ready(injuries):
        inj = injuries.dropna(subset=["gsis_id"]).copy()
        if "week" in inj.columns:
            inj = inj.loc[inj["week"] == inj["week"].max()]
        inj = inj.drop_duplicates("gsis_id", keep="last")
        inj_rows = []
        for rec in inj.to_dict("records"):
            badge, tip = _injury_badge(rec)
            if badge:
                inj_rows.append({"player_id": str(rec["gsis_id"]).strip(), "report_inj": badge, "report_tip": tip})
        reports = pd.DataFrame(inj_rows)
        nv_ts = load_release_timestamp("injuries")
        source = f"nflverse injury report{(' · ' + nv_ts) if nv_ts else ''}"
    else:
        reports, source = _sleeper_reports(rost)

    if reports.empty:
        status["report_inj"] = ""
        status["report_tip"] = ""
    else:
        status = status.merge(reports, on="player_id", how="outer")

    def pick(row) -> pd.Series:
        roster_badge = _clean(row.get("roster_inj"))
        report_badge = _clean(row.get("report_inj"))
        if roster_badge in _HARD:
            tip = _clean(row.get("report_tip")) or _clean(row.get("roster_tip"))
            return pd.Series({"inj": roster_badge, "inj_tip": tip})
        if report_badge:
            return pd.Series({"inj": report_badge, "inj_tip": _clean(row.get("report_tip"))})
        return pd.Series({"inj": roster_badge, "inj_tip": _clean(row.get("roster_tip"))})

    picked = status.apply(pick, axis=1)
    out = pd.concat([status[["player_id"]], picked], axis=1)
    out = out.loc[out["inj"].fillna("").ne("")].copy()
    return out, source


def attach_live_status(df: pd.DataFrame, status: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=["inj", "inj_tip"], errors="ignore")
    if status is None or status.empty or "player_id" not in out.columns:
        out["inj"] = ""
        out["inj_tip"] = ""
        return out
    extra = status[["player_id", "inj", "inj_tip"]].drop_duplicates("player_id")
    extra["player_id"] = extra["player_id"].astype(str).str.strip()
    out["player_id"] = out["player_id"].astype(str).str.strip()
    out = out.merge(extra, on="player_id", how="left")
    out["inj"] = out["inj"].fillna("")
    out["inj_tip"] = out["inj_tip"].fillna("")
    return out
