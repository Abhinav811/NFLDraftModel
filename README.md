# NFL fantasy model

Position rankings built from free data: **nflverse** (play-by-play, stats, rosters, injuries, combine, schedule), **Fantasy Football Calculator ADP**, and **season-long Vegas props** (CSV overlay + optional DraftKings/FTA scrape).

The model is designed to catch players the market underweights, not to beat Vegas on raw yardage totals.

## Pipeline

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_pipeline.py
```

First run downloads ~200MB of nflverse parquet into `data/raw/` and takes several minutes (play-by-play aggregation). Later runs reuse the cache.

Fast debug without PBP:

```bash
python3 scripts/run_pipeline.py --skip-pbp
```

Reuse a built panel:

```bash
python3 scripts/run_pipeline.py --reuse-panel
```

Refresh injury designations on the hosted board (no retraining):

```bash
python3 scripts/refresh_board.py
```

## Outputs

Written to `data/processed/` (gitignored) and `docs/` (GitHub Pages):

- `docs/index.html` — board-only page for a WordPress iframe (toggles, draft mode, tables)
- `article_2026.md` / `article_2026.html` — full writeup
- `rankings_2026_{QB,RB,WR,TE}.csv`, `backtest_by_season.csv`, `extra_correlations.csv`

## WordPress embed

Prose stays in the Saintistics post. The iframe is only the hosted board:

```html
<iframe src="https://abhinav811.github.io/NFLDraftModel/" style="width:100%;min-height:1600px;border:0;" title="2026 fantasy board"></iframe>
```

Requires a paid WordPress.com plan with hosting/plugins activated. On the free plan, link to the Pages URL instead.

## Vegas props

Books are the volume prior. Drop a fuller board into `data/external/season_props.csv`:

`player_name,team,market,line,over_odds,under_odds,book,source`

Markets: `pass_yd`, `pass_td`, `rush_yd`, `rush_td`, `rec`, `rec_yd`, `rec_td`, `int`.

DraftKings’ public JSON is geo-blocked from many servers; running the pipeline on a home machine is more likely to ingest live futures. Historical player props are not available for free, so **backtests use ADP as the market**.

## Features

| Edge | Source |
| --- | --- |
| VFP (no-vig props → PPR) | season_props.csv / FTA / DK |
| Age curve | roster birthdays; λ fit on YoY PPR |
| O-line | PBP sack/stuff/rush EPA index |
| OC/scheme | team pass OE, neutral pace, new HC |
| Chunk plays | 15+ rushes, 20+ receptions × explosive EPA |
| Red zone | inside-5 carries, inside-10 targets, end-zone targets |
| SOS / indoor | 2026 schedule × prior-year defensive EPA, roof; **remaining-season SOS overlay** after games are played |
| Usage / vacated touches | target/carry share + 2026 roster returning volume |
| Combine / draft | nflverse combine + draft capital |
| Injury | weekly injury reports, major-keyword flag, 3-year rollup; live IR/PUP/Q/O badges from Sleeper until nflverse weekly reports exist |
| Tenure weights | rookies vs vets blend market / production / situation / athletic / aging |

## Scoring

Full PPR: 0.04 pass yard, 4 pass TD, −2 INT, 0.1 rush/rec yard, 6 rush/rec TD, 1 reception. Round value assumes a 12-team draft (early 1–4, mid 5–8, late 9–14).
