from __future__ import annotations

import json
import time

import pandas as pd

from ..config import RAW_DIR, SLEEPER_MAX_AGE_HOURS, SLEEPER_PLAYERS_URL
from ..http import fetch_bytes
from ..names import normalize_name


def load_sleeper_players() -> pd.DataFrame:
    """Sleeper player dump. Cached locally; call at most ~once per day."""
    dest = RAW_DIR / "sleeper_players_nfl.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fresh = dest.exists() and dest.stat().st_size > 1000
    if fresh:
        age_h = (time.time() - dest.stat().st_mtime) / 3600
        if age_h > SLEEPER_MAX_AGE_HOURS:
            fresh = False
    if not fresh:
        fetch_bytes(SLEEPER_PLAYERS_URL, dest=dest, timeout=180)
    payload = json.loads(dest.read_text())
    if not isinstance(payload, dict):
        return pd.DataFrame()
    rows = []
    for pid, rec in payload.items():
        if not isinstance(rec, dict):
            continue
        gsis = rec.get("gsis_id")
        gsis = str(gsis).strip() if gsis else ""
        if gsis.lower() in {"", "none", "nan", "null"}:
            gsis = ""
        name = rec.get("full_name") or " ".join(
            p for p in [rec.get("first_name"), rec.get("last_name")] if p
        )
        rows.append(
            {
                "sleeper_id": str(rec.get("player_id") or pid),
                "gsis_id": gsis,
                "player_name": name,
                "name_norm": normalize_name(name) if name else "",
                "position": rec.get("position"),
                "team": rec.get("team") or rec.get("team_abbr"),
                "status": rec.get("status"),
                "injury_status": rec.get("injury_status"),
                "injury_body_part": rec.get("injury_body_part"),
                "injury_notes": rec.get("injury_notes"),
                "practice_participation": rec.get("practice_participation"),
            }
        )
    out = pd.DataFrame(rows)
    out["_cached_at"] = dest.stat().st_mtime
    print(f"  Sleeper players: {len(out):,} ({dest.name})")
    return out
