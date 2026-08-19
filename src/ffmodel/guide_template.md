# {{SEASON}} Fantasy Ranking Model — Methods Guide

*{{DATE}} · 12-team · Full PPR primary · trained on {{TRAIN_START}}–{{TRAIN_END}}*

**[Open the {{SEASON}} board](https://abhinav811.github.io/NFLDraftModel/)** (new tab). Toggle Full PPR / Half PPR, switch Full board vs Positional, Start Draft to cross names off.

This guide is how the board is built. Rankings live on that page — including steal and fade flags — not here.

**If you just want to draft:** use the board. Green = steal, red = fade, everyone else is close enough to ADP that I would not force the issue. **If you want the engine:** each section starts in plain English, then the formula.

The model is not trying to beat Vegas on yardage. Books are better at volume. It is trying to **rank players relative to the market** by estimating the things a posted season total is bad at seeing: aging, vacated touches assigned to a *specific* player, high-value red-zone role, chunk plays, injury residue, and workload cliffs.

---

## How to use the board

1. Open the [hosted table](https://abhinav811.github.io/NFLDraftModel/) in a new tab.
2. Full PPR vs Half PPR re-ranks the same engine. Half PPR is full PPR minus ½ × projected receptions — not a second model.
3. Full board is 12-team pick order with round headers (early / mid / late within the round). Positional is QB/RB/WR/TE lists.
4. Start Draft / Exit Draft stores crossed-off names in this browser. Clear picks wipes them.
5. A name can sit a couple of ranks off ADP without being flagged. Flags require a real gap *and* a confirming reason (section 7).

Scoring (full PPR):

```
pass yd  0.04     rush/rec yd  0.10     reception  1.00
pass TD  4.00     rush/rec TD  6.00     INT / fum  −2.00
```

Round labels assume a 12-team snake. Picks 1–4 of a round are early, 5–8 mid, 9–12 late. “Mid 3rd” is picks 29–32.

---

## Evaluation: model vs ADP vs what actually happened

Walk-forward, 2021–2025: trained on prior seasons only. Ranks below are **among players who actually had an ADP** — the people you are choosing between on draft day. Overall Spearman vs ADP is a small lift (~+0.018). That is the wrong headline. The edge shows up when the model *disagrees* with ADP.

**In English:** when we say someone is five or more ranks cheaper than ADP, they usually were. That is the reason to open the board instead of a consensus cheat sheet.

![Disagreement calibration](https://abhinav811.github.io/NFLDraftModel/figures/07_disagreement_calibration.png)

*Figure 1. When the model had a player ≥8 ranks cheaper than ADP, they beat ADP rank 81% of the time (mean +12 ranks). Close-to-ADP names are a coin flip — as they should be.*

![Cheap calls every season](https://abhinav811.github.io/NFLDraftModel/figures/10_cheap_calls_by_year.png)

*Figure 2. Not one lucky year. If the model said ≥5 ranks cheaper, they beat ADP in 68–93% of those calls every season.*

![Same ADP cost, different player](https://abhinav811.github.io/NFLDraftModel/figures/08_midround_same_cost.png)

*Figure 3. Mid-rounds (ADP 36–96): two groups that cost about the same. The names the model liked scored 177 actual PPR; the names it faded scored 154. That is a starter vs a flex, at the same pick.*

![Model top 5 vs ADP top 5](https://abhinav811.github.io/NFLDraftModel/figures/09_top5_roster_edge.png)

*Figure 4. Extra actual PPR if you started the model’s top 5 at a position instead of ADP’s. WR is the swing — usually one swap (Lamb or Amon-Ra in). 2022 and 2024 lost; the other three years more than paid for it.*

![Steal and fade flags](https://abhinav811.github.io/NFLDraftModel/figures/06_flag_hit_rates.png)

*Figure 5. The published green/red flags on the board (stricter than “any 5-rank disagreement”). {{STEAL_HIT}} of steal flags finished above ADP ({{N_STEALS}} flags); {{FADE_HIT}} of fades finished below ({{N_FADES}} flags). Names stay on the board.*

![Model vs ADP Spearman by season](https://abhinav811.github.io/NFLDraftModel/figures/01_spearman_vs_adp.png)

*Figure 6. Full-sample ranking correlation. Mean lift vs ADP is about +0.018. Most of the board is close to ADP; use figures 1–3 for the draftable edge.*

---

## 1. What the model is optimizing

For player *i* in season *t*, *y* is season-long fantasy points. *m* is the market: Vegas Fantasy Points when a prop exists, otherwise an ADP-implied points curve. *ŷ* is the model projection.

**In English:** I care about *order* among players who actually get drafted, not about nailing every undrafted dart throw’s point total.

Primary metric: Spearman ρ(*ŷ*, *y*) **only on players with an ADP**. That is the fair comparison against ADP. MAE on the whole universe is reported but is not the tuning target.

---

## 2. Data

| Layer | Source | Use |
| --- | --- | --- |
| Play-by-play, stats, rosters, injuries, snaps, combine, draft, schedule | nflverse (CC BY 4.0) | features + realized *y* |
| ADP | Fantasy Football Calculator, 12-team PPR (half-PPR ADP when available) | market proxy, sample weights, steal band |
| Season-long player props | public over/unders | VFP baseline for {{SEASON}} |

Train on {{TRAIN_START}}–{{TRAIN_END}} ({{TRAIN_START}} is a lag year). Walk-forward tests: 2021–2025. {{SEASON}} uses models fit on all prior seasons.

**Limitation that matters:** I do not have a historical archive of player-level season props. Backtests therefore use ADP as *m*. {{SEASON}} uses VFP when at least one market is posted, else ADP.

---

## 3. Market layer

### 3.1 Vegas Fantasy Points (VFP)

**In English:** Take the posted over/under, strip the juice so −120/−100 is not treated as 50/50, nudge the line a little toward the fair mean, then score those counting stats as PPR. That number is the book’s implied fantasy projection.

American odds → implied probability:

```
p = |o| / (|o| + 100)     if o < 0
p = 100 / (o + 100)       if o > 0
```

The posted number *L* is a median. Fair over probability and a damped location shift:

```
p_fair = p_over / (p_over + p_under)
z      = clip( Φ⁻¹(p_fair), −1.5, 1.5 )
E[X]   = L + 0.35 · σ_m · z
```

σ_m is a market-specific scale (pass yd 450, rush yd 180, rec yd 160, rec 12, pass TD 4.5, rush/rec TD 2.4–2.8, INT 2.2). The 0.35 keeps a noisy juice line from yanking a total several σ.

```
VFP = Σ_k  E[X_k] · s_k
```

*s* is the PPR vector above. Used as *m* only if `vfp_markets ≥ 1`.

### 3.2 ADP-implied points

**In English:** Historically, earlier ADP means more points, but not in a straight line. I fit that curve by position, then read off “what ADP usually scores.”

```
m_ADP = β_{0,p} + β_{1,p} · log(ADP)
```

OLS on historical (ADP, *y*) pairs. This is a *points* market: two WRs at ADP 24 and 36 can be 40 PPR apart even if they sit next to each other on a cheat sheet.

---

## 4. Features a yardage prop cannot see

### 4.1 Age

**In English:** Players hold value until a position-specific peak, then decline faster and faster — not a gentle linear fade. RBs fall off harder than WRs because the history said so, not because I assumed it.

Age is days from birth to Sept 1 of season *t* / 365.25. Peaks *a\*_p*: QB 30, RB 26.5, WR 28, TE 28.5.

```
α_age(a, p) = clip( 1 − λ_p · [a − a*_p]₊² ,  0.35,  1.05 )
```

λ_p is fit on players already past the peak with last-year PPR > 50:

```
λ_p = clip(  (xᵀ y) / (xᵀ x) ,  0.004,  0.08  )
```

*x = (a − a\*)²*, *y = 1 − PPR_t / PPR_{t−1}*. Defaults if the subsample is thin: QB 0.012, RB 0.035, WR 0.018, TE 0.016.

### 4.2 Tenure

These buckets only change the **prior** (section 5). The trees still see years of experience.

```
years_exp ≤ 0  rookie
          = 1  sophomore
        2–3    developing
        4–6    prime
        ≥ 7    veteran
```

### 4.3 Player-level vacated usage

**In English:** If a WR leaves, you should not give the leftover targets to every teammate. I split vacated volume by depth chart and who was already getting the work.

Team leftover:

```
v_tgt = 1 − returning_targets / team_targets
v_car = 1 − returning_carries / team_carries
```

Assigned to a person:

```
depth weight w  =  {1: 0.52,  2: 0.28,  3: 0.13,  else 0.04}
depth_alloc     =  w / Σ w     within (season, team, position)
prior_share     =  last year’s touches / returning touches, clipped to 0.70

alloc           =  depth_alloc                              if rookie
                =  0.55 · prior_share + 0.45 · depth_alloc  otherwise

player_vacated_targets = alloc · v_tgt · team_targets
player_vacated_carries = alloc · v_car · team_carries
```

`player_vacated_boost` is targets for WR/TE, carries for RB, 0 for QB.

### 4.4 High-value red zone and TD luck

**In English:** Some TDs come from role (carries inside the 5, end-zone targets). The leftover is luck. Under-scored relative to role can be a steal confirm; last year’s TD binge can be a fade confirm.

From PBP (kneels, spikes, penalties, 2-pt dropped):

```
inside-5 carry     yardline_100 ≤ 5
inside-10 target   yardline_100 ≤ 10
end-zone target    air_yards ≥ yardline_100 − 0.5  and  yardline_100 ≤ 20

HV-RZ = 1.4 · carries_in5  +  1.0 · targets_in10  +  1.6 · ez_targets

rec_TD_luck  = rec_TD  − (0.20 · targets_in10 + 0.28 · ez_targets)
rush_TD_luck = rush_TD −  0.42 · carries_in5
TD_luck      = rec_TD_luck + rush_TD_luck
```

Those coefficients convert role → expected TDs. They are not a second fantasy model.

### 4.5 Chunk plays

Explosive rush ≥15 yards; explosive reception ≥20. Rate times EPA, so a cheap long play is not treated like a tackle-breaking one.

```
explosive_rate = (exp_rushes + exp_receptions) / touches
chunk_rating   = explosive_rate · (EPA on those explosive plays)
```

### 4.6 Usage vs overproduction

**In English:** `usage_index` is how involved they were. `overproduction` is scoring that usage does not justify — a regression flag, not a “he’s good” flag.

Within (season, position), z-score snap rate, target share, PPR/game, and inverse depth:

```
usage_index    = z_snap + z_tgt_share + z_ppg + z_{1/depth}
overproduction = z_ppg − 0.5 · (z_snap + z_tgt_share)
eff_index      = z_YPC + z_YPR + z_catch_rate
```

### 4.7 O-line, scheme, schedule (shrunk)

**In English:** A good offense helps, but I shrink these so the model cannot ignore the player and just pick “plays on a good team.”

```
OL = clip( 0.4 · (−z_sack_rate) + 0.3 · (−z_stuff_rate) + 0.3 · z_rush_EPA , −3, 3 )
C  = ((PROE + 1) / 2) · (neutral_pace / league_neutral_pace)
```

Neutral pace: plays/game with win probability in [0.20, 0.80] and |score differential| ≤ 8. PROE is nflfastR pass rate over expected.

Pre-shrink before the trees: ×0.35 for OL/scheme/pace, ×0.40 for SOS / indoor / implied totals. SOS is last year’s opponent defensive EPA averaged over *this* year’s schedule.

### 4.8 Physical prior (young players only)

**In English:** Combine and draft capital matter for rookies. They are not allowed to keep re-drafting a 29-year-old.

```
speed_score = wt · 200 / forty⁴
burst       = vertical + broad_jump / 12
draft_cap   = max( 0.15,  1.15 − log(pick) / log(250) )
```

Undrafted pick = 250. RBs vs speed_score 90; WR/TE vs burst 36. Years_exp < 3: `pos_base · draft_cap · athletic` with pos_base {QB 220, RB 160, WR 140, TE 110}. Veterans fall back to last year’s PPR.

### 4.9 Situation flags

These are on/off switches. A steal or fade still needs the point/rank gap in section 7.

```
workload_cliff   = (carries ≥ 240 and age ≥ 26.5)  or  (carries ≥ 280 and age ≥ 26)
breakout_window  = RB year 1  |  WR years 1–2  |  TE years 2–3
sophomore_leap   = years_exp = 1 and last PPR ∈ [70, 200]
injury_bounce    = games_pct < 0.70, age 23–31, last PPR/G ≥ 8
chronic_injury   = ≥ 8 missed-week flags in a 3-year rollup
role_expand      = listed starter, snap rate < 55%, years_exp ≥ 1
pass_catch_rb    = RB target share (0 otherwise)
```

---

## 5. Tenure-weighted prior

**In English:** Before the machine-learning step, mix five views of the player. Rookies lean on situation and athleticism. Prime vets lean on the market and last year’s production. Aging vets get a heavier age penalty.

| Bucket | Market | Production | Situation | Physical | Aging |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rookie | 0.22 | 0.08 | 0.40 | 0.30 | 0.00 |
| Sophomore | 0.22 | 0.32 | 0.32 | 0.12 | 0.02 |
| Developing | 0.28 | 0.38 | 0.24 | 0.06 | 0.04 |
| Prime | 0.40 | 0.34 | 0.16 | 0.04 | 0.06 |
| Veteran | 0.38 | 0.24 | 0.10 | 0.02 | 0.26 |

```
prod = PPR_eff · α_age · (0.85 + 0.15 · availability)

PPR_eff = (PPR/G)_{t−1} · G_t     if missed weeks ≥ 3
        = PPR_{t−1}               otherwise

sit  = m · (1 + 0.03·OL) · (0.92 + 0.08·C) · (1 + 0.08·SOS)
         · depth_mult · (1 + 0.0025 · player_vacated_boost)
         · cliff / breakout / team-change / role-expand multipliers

aging = α_age · prod
blend = w_m·m + w_p·prod + w_s·sit + w_ph·phys + w_a·aging
```

No market → zero *w_m* and renormalize the rest. Situation also applies 0.90 on a cliff, 1.06 breakout, 0.97 team change, 1.04 role expand, 0.96 chronic injury, plus a small indoor bump for WR/QB. Injury-bounce is still computed as a tree feature; it is not a situation multiplier or a steal confirmer.

This prior is **not** the published ranking. It is a shrinkage target when the market is missing, and a feature (`blend_proj`) the trees can use.

---

## 6. Residual model

**In English:** Four small gradient-boosted trees. One predicts points. Two predict *how the market was wrong* (everyone, then by position). One predicts rank movement vs ADP. Shallow trees on purpose: I want corrections from TD luck, vacated work, and cliffs — not a memorized 2022 WR class.

| Model | Target | Caps |
| --- | --- | --- |
| Points | *y* | — |
| Global residual | *y − m* | clip ±60 |
| Position residual | *y − m* within position if *n* ≥ 120 | clip ±60, then 0.55 local + 0.45 global |
| Rank lift | ADP rank − actual rank | clip ±18 |

HistGradientBoosting, sklearn, `random_state=7`. Shared: `max_depth=3`, `learning_rate=0.05`, `l2_regularization=0.55`, `min_samples_leaf=22`. Iterations 280 / 220 / 180 / 200.

Sample weights — a 12-team draft cares more about pick 8 than pick 180:

```
w = 5.5   if ADP ≤ 72
  = 4.0   if ADP ≤ 150
  = 1.0   otherwise
w ← 1.2 w  if ADP ∈ [18, 132]   (steal band)
```

### 6.1 Published projection

**In English:** Start from the market. Add a fraction of “what the market missed.” For rookies without a book, pull toward the physical/situation prior. For rookies *with* a book, barely do that — do not drag them 40% off Vegas toward a combine number.

*b* = VFP if present, else ADP-implied *m*, else blend, else last-year PPR, else 80. *r̂* = mixed residual. *ℓ̂* = rank-lift prediction. *λ*, *k* = tuned scalars.

```
λ_t = 1.15 λ    years_exp ≤ 0
    = 1.05 λ    years_exp ≤ 2
    = 0.85 λ    years_exp ≥ 7
    = λ         otherwise

w_r = 0.40 / 0.18 / 0.05     for years_exp = 0 / 1 / ≥2
w_r ← min(w_r, 0.08)         if VFP exists

ŷ = (1 − w_r) · mixed + w_r · blend

mixed = b + λ_t · r̂ + clip(k · ℓ̂, −22, 22)     if market exists
      = 0.6 · ŷ_points + 0.4 · blend            otherwise
```

### 6.2 How λ and k are chosen

Grid: λ ∈ {0.25, 0.40, 0.55, 0.75, 1.00}, k ∈ {0, 1.5}. Fit on seasons before the last training year; score Spearman on ADP-drafted players in that last year. **Maximize ρ, not steal count.** Turning λ up until more names go green is how you lose to ADP in a year like 2025 (Δρ was only +0.004).

---

## 7. Steal / fade rule

**In English:** Disagreement with ADP is not enough. The board only paints a name green or red if the gap is large in both ranks *and* points, *and* there is a football reason. The names themselves are on the board, not listed again here.

Position rank is dense min-rank of *ŷ* within QB/RB/WR/TE. Implied 12-team pick maps that rank onto where the market actually drafts that slot (QB3 lands where people draft the third QB, not at overall 3).

Steal if all of:

1. ADP ∈ [18, 132] (skip 1.01s and dart throws)
2. model position rank ≤ ADP position rank − 5
3. *ŷ − m* ≥ 12 PPR
4. at least one of: role expand, sophomore leap, `pass_catch_rb ≥ 0.08`, `TD_luck ≤ −1`, `new_starter_vacated ≥ 6`

Fade if (2)–(3) flip sign and at least one of: workload cliff, chronic injury, `TD_luck ≥ 2.0`, `overproduction ≥ 0.55`, `eff_index ≥ 1.4`.

Early-round fades (ADP < 18) also need either a ≥10 PPR hole vs *m*, or a workload cliff with edge ≤ +5. You do not fade a 1.01 because a tree twitched.

Walk-forward, on ranks among ADP-drafted players only, **{{STEAL_HIT}}** of steal flags finished above their ADP rank ({{N_STEALS}} flags) and **{{FADE_HIT}}** of fades finished below ({{N_FADES}} flags). The bigger edge is in *any* ≥5-rank cheap call (figure 1), not only the painted flags. Extreme “much more expensive” calls are not a strong fade by themselves — use the red flags, which require a confirming feature.

---

## 8. Walk-forward backtest

Trained on seasons `< t` only. Spearman on ADP-drafted players only.

{{BT_LINES}}

Mean Spearman lift vs ADP is about **+0.018**. Point MAE is ~55–60 PPR — normal season-long fantasy error, not a bug. The claim is ranking lift, not a tighter crystal ball.

Features most correlated with *beating* ADP-implied points (not just with raw PPR):

{{EXTRA_LINES}}

`depth_rank` (negative) is the loudest beat-the-market correlate: listed depth still has information ADP does not fully price. `usage_index` and `touches` predict *y*; the beat-market column is the residual story.

---

## What this is not

- Not a betting model. VFP is an input, not a +EV claim on the props.
- Not a causal O-line or coaching paper. Those are shrunk covariates.
- Scheme includes head-coach change, a binary OC-change flag when mapped, and team pass rate over expected. Primary-QB change is a separate tree feature (WR/TE vs new starter).
- Not a claim that every ADP disagreement is a lock. Close-to-ADP names are a coin flip. The usable edge is when the model says someone is clearly cheaper.
- Half PPR is a reception tax on the full-PPR fit, not a re-estimated residual.

Sources: nflverse play-by-play, stats, rosters, injuries, snaps, combine, draft, and schedules (CC BY 4.0); Fantasy Football Calculator ADP; publicly posted season-long totals. Not betting advice.
