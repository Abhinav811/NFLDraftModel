from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..config import NFLDATA_RAW, NFLVERSE_RELEASE, PREDICT_SEASON, RAW_DIR
from ..http import fetch_bytes

# Tiny GitHub 404 bodies look like b"Not Found". Real parquet/json is larger.
_MIN_CACHE_BYTES = 20
_LIVE_MAX_AGE_HOURS = 6.0


def _cache_path(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / name


def _looks_missing(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < _MIN_CACHE_BYTES:
        return True
    head = path.read_bytes()[:32].strip().lower()
    return head in {b"not found", b"404"} or head.startswith(b"<!doctype") or head.startswith(b"<html")


def download_release(tag: str, filename: str, max_age_hours: float | None = None) -> Path:
    dest = _cache_path(filename)
    fresh = dest.exists() and not _looks_missing(dest)
    if fresh and max_age_hours is not None:
        age_h = (time.time() - dest.stat().st_mtime) / 3600
        if age_h > max_age_hours:
            fresh = False
    if fresh:
        return dest
    url = f"{NFLVERSE_RELEASE}/{tag}/{filename}"
    fetch_bytes(url, dest=dest)
    if _looks_missing(dest):
        dest.unlink(missing_ok=True)
        raise FileNotFoundError(url)
    return dest


def load_release_timestamp(tag: str) -> str:
    try:
        path = download_release(tag, "timestamp.json", max_age_hours=1)
        data = json.loads(path.read_text())
        return str(data.get("last_updated") or "")
    except Exception:
        return ""


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    schema_names = set(pq.read_schema(path).names)
    cols = [c for c in (columns or []) if c in schema_names] if columns else None
    return pd.read_parquet(path, columns=cols or None)


def load_player_stats(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = download_release("stats_player", f"stats_player_reg_{season}.parquet")
        df = read_parquet(path)
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_pbp(season: int, columns: list[str] | None = None) -> pd.DataFrame:
    path = download_release("pbp", f"play_by_play_{season}.parquet")
    df = read_parquet(path, columns=columns)
    if "season_type" in df.columns:
        df = df.loc[df["season_type"] == "REG"].copy()
    return df


def load_rosters(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        max_age = _LIVE_MAX_AGE_HOURS if season >= PREDICT_SEASON else None
        path = download_release("rosters", f"roster_{season}.parquet", max_age_hours=max_age)
        df = read_parquet(path)
        df["season"] = season
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "week" in out.columns:
        out = (
            out.sort_values(["season", "gsis_id", "week"])
            .groupby(["season", "gsis_id"], as_index=False)
            .tail(1)
        )
    return out


def load_players() -> pd.DataFrame:
    path = download_release("players", "players.parquet")
    return read_parquet(path)


def load_combine() -> pd.DataFrame:
    path = download_release("combine", "combine.parquet")
    return read_parquet(path)


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        max_age = _LIVE_MAX_AGE_HOURS if season >= PREDICT_SEASON else None
        try:
            path = download_release("injuries", f"injuries_{season}.parquet", max_age_hours=max_age)
        except FileNotFoundError:
            continue
        frames.append(read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_weekly_rosters(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        max_age = _LIVE_MAX_AGE_HOURS if season >= PREDICT_SEASON else None
        try:
            path = download_release(
                "weekly_rosters",
                f"roster_weekly_{season}.parquet",
                max_age_hours=max_age,
            )
        except FileNotFoundError:
            continue
        df = read_parquet(path)
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_snap_counts(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = download_release("snap_counts", f"snap_counts_{season}.parquet")
        frames.append(read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_draft_picks() -> pd.DataFrame:
    path = download_release("draft_picks", "draft_picks.parquet")
    return read_parquet(path)


def load_schedules() -> pd.DataFrame:
    path = download_release("schedules", "games.parquet")
    return read_parquet(path)


def load_depth_charts(season: int) -> pd.DataFrame:
    path = download_release("depth_charts", f"depth_charts_{season}.parquet")
    return read_parquet(path)


def load_win_totals() -> pd.DataFrame:
    dest = _cache_path("win_totals.csv")
    if not dest.exists():
        fetch_bytes(f"{NFLDATA_RAW}/win_totals.csv", dest=dest)
    return pd.read_csv(dest)
