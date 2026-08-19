from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..config import NFLDATA_RAW, NFLVERSE_RELEASE, RAW_DIR
from ..http import fetch_bytes


def _cache_path(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / name


def download_release(tag: str, filename: str) -> Path:
    dest = _cache_path(filename)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{NFLVERSE_RELEASE}/{tag}/{filename}"
    fetch_bytes(url, dest=dest)
    return dest


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
        path = download_release("rosters", f"roster_{season}.parquet")
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
        path = download_release("injuries", f"injuries_{season}.parquet")
        frames.append(read_parquet(path))
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
