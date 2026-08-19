from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

NFLVERSE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
NFLDATA_RAW = "https://raw.githubusercontent.com/nflverse/nfldata/master/data"
FFC_ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
FTA_PROPS_URL = "https://fantasyteamadvice.com/nfl/season-long-props"
DK_NFL_EVENTGROUP = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/88808?format=json"

# Feature years include a lag year before the first training target.
FIRST_SEASON = 2018
LAST_COMPLETED_SEASON = 2025
PREDICT_SEASON = 2026

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
GAMES_PER_SEASON = {2018: 16, 2019: 16, 2020: 16, 2021: 17, 2022: 17, 2023: 17, 2024: 17, 2025: 17, 2026: 17}

# Full-PPR scoring used for VFP and realized points.
PPR = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "int": -2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec": 1.0,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fum": -2.0,
}

# Age peaks from the feature spec; lambdas are estimated from history.
AGE_PEAKS = {"QB": 30.0, "RB": 26.5, "WR": 28.0, "TE": 28.5}
AGE_LAMBDA_DEFAULT = {"QB": 0.012, "RB": 0.035, "WR": 0.018, "TE": 0.016}

# High-value red-zone weights.
RZ_WEIGHTS = {"inside5_carries": 1.4, "inside10_targets": 1.0, "ez_targets": 1.6}

# 12-team, full-PPR draft windows.
ROUND_SIZE = 12
EARLY_ROUNDS = (1, 4)
MID_ROUNDS = (5, 8)
LATE_ROUNDS = (9, 14)

# Explicit tenure blends used on top of the learned model.
# Weights are (market, production, situation, physical, aging).
TENURE_WEIGHTS = {
    "rookie": (0.22, 0.08, 0.40, 0.30, 0.00),
    "sophomore": (0.22, 0.32, 0.32, 0.12, 0.02),
    "developing": (0.28, 0.38, 0.24, 0.06, 0.04),
    "prime": (0.40, 0.34, 0.16, 0.04, 0.06),
    "veteran": (0.38, 0.24, 0.10, 0.02, 0.26),
}

# Steal/fade band: skip the 1.01 and dart-throw names where ADP is noisy.
STEAL_ADP_MIN = 18
STEAL_ADP_MAX = 132
STEAL_RANK_LIFT = 5
STEAL_POINT_EDGE = 12.0

MIN_TRAIN_GAMES = 3
MIN_TRAIN_PPR = 25.0
BACKTEST_SEASONS = (2021, 2022, 2023, 2024, 2025)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

PBP_COLUMNS = [
    "season",
    "week",
    "game_id",
    "posteam",
    "defteam",
    "play_type",
    "season_type",
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "epa",
    "wp",
    "xpass",
    "pass_oe",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "pass_attempt",
    "rush_attempt",
    "complete_pass",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "sack",
    "qb_hit",
    "qb_dropback",
    "qb_scramble",
    "goal_to_go",
    "success",
    "roof",
    "surface",
    "home_team",
    "away_team",
    "spread_line",
    "total_line",
    "penalty",
    "two_point_attempt",
    "special",
    "pass",
    "rush",
    "qb_kneel",
    "qb_spike",
]
