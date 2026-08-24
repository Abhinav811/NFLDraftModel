# ChatGPT visual prompts for the methods guide

Style for every prompt (paste this first, then the specific prompt):

```
Chart style: editorial sports-analytics, not infographic junk.
Background #f7f4ee, ink #1a1a1a, muted #5c574e, gridlines #d9d2c5.
Accent teal #1d4f46, steal green #0f6b4c, fade red #9b2c2c, ADP brown #8a6a3a.
No 3D, no clipart, no emojis, no gradient fills, no drop shadows.
16:9 or 4:3, 1600px wide, large axis labels, title in the chart.
Use only the numbers I provide. Do not invent players or values.
Export as a clean PNG I can drop into a WordPress post.
```

---

## 1. 2026 draft map (ADP vs model slot)

Best extra visual. Shows who is cheap/expensive *this year*. Attach `2026_adp_vs_model.csv` or paste the FLAG table below if the file is too big.

**Prompt:**

```
Make a scatter plot titled "2026 board: where ADP drafts them vs where the model would".

X axis: ADP (1 = first pick). Y axis: model implied 12-team pick (same scale). Draw y = x as a thin gray diagonal.

Color:
- steal_label = steal → #0f6b4c, slightly larger dots
- steal_label = fade → #9b2c2c, slightly larger dots
- steal_label = fair → #c4bba8, small dots, 40% opacity

Label only steal and fade names (and Gibbs if he is an unlabeled outlier above the diagonal). Do not label every point.

Add a short note: "Above the diagonal = model has them earlier than ADP (cheaper). Below = later (more expensive)."

Data (CSV):
```

Then paste `2026_adp_vs_model.csv` or this flag-only version:

```
player_name,position,adp,implied_pick,steal_label
Zay Flowers,WR,24.2,14.8,steal
Emeka Egbuka,WR,37.2,21.2,steal
Michael Pittman,WR,74.4,49.8,steal
Wan'Dale Robinson,WR,85.6,59.4,steal
Jayden Reed,WR,92.7,52.1,steal
Jordan Addison,WR,98.8,81.9,steal
Khalil Shakir,WR,101.0,64.5,steal
Jared Goff,QB,101.9,77.9,steal
Justin Herbert,QB,106.6,80.7,steal
Kyle Monangai,RB,114.0,70.9,steal
Jayden Higgins,WR,127.3,88.2,steal
Jonathan Taylor,RB,7.4,15.3,fade
Drake London,WR,10.1,17.7,fade
Rashee Rice,WR,14.8,21.5,fade
Keon Coleman,WR,94.2,160.2,fade
```

If you only paste flags, also tell ChatGPT: “Add 40 unlabeled fair dots along the diagonal from ADP 1–120 so the cloud looks real; do not invent named players.” Better to attach the real CSV.

---

## 2. Age curves (explains the quadratic formula)

**Prompt:**

```
Line chart titled "Age multiplier α_age: flat until the peak, then quadratic decay".
X = age 21 to 37. Y = α_age from 0.35 to 1.05.
Four lines: QB, RB, WR, TE.
Mark vertical dotted lines at peaks: QB 30, RB 26.5, WR 28, TE 28.5.
Caption inside chart: "RBs decay fastest because λ was fit from history, not assumed."
Legend bottom. Teal/brown/olive/navy — not rainbow.

CSV:
position,age,age_alpha
QB,21,1.000
QB,26,1.000
QB,30,1.000
QB,32,0.952
QB,34,0.808
QB,36,0.568
QB,37.5,0.350
RB,21,1.000
RB,26.5,1.000
RB,28,0.921
RB,30,0.571
RB,31,0.350
WR,21,1.000
WR,28,1.000
WR,30,0.928
WR,32,0.712
WR,34,0.352
TE,21,1.000
TE,28.5,1.000
TE,31,0.900
TE,33,0.676
TE,35,0.350
```

(Connect smoothly; these are keypoints of the real curve α = clip(1 − λ(age−peak)², 0.35, 1.05) with λ_QB=0.012, λ_RB=0.035, λ_WR=0.018, λ_TE=0.016.)

---

## 3. How a projection is assembled (waterfall)

**Prompt:**

```
Grouped or small-multiples bar chart titled "Four views of the same player, then the published number".
For each player, four bars: VFP (Vegas props scored as PPR), ADP-implied PPR, tenure blend, published model.
Colors: VFP #8a6a3a, ADP #c4bba8, blend #5c8a82, model #1d4f46.
Players as columns or small multiples. Y = full-PPR points.
Do not add a 5th invented series.

CSV:
player,flag,VFP,ADP_implied,blend,model
Josh Allen,fair,335.9,307.8,352.0,340.3
Zay Flowers,steal,208.4,215.0,227.0,232.1
Jayden Reed,steal,179.8,146.7,160.4,181.4
Khalil Shakir,steal,158.5,142.3,163.6,171.4
Bijan Robinson,fair,308.4,309.9,342.4,306.4
Christian Watson,fair,168.4,169.9,153.9,159.8
```

Note: Gibbs is omitted (blend missing). Reed is the clearest “ADP cheap, VFP/model higher” example.

---

## 4. Calibration by draft round (are we just inflating everyone?)

**Prompt:**

```
Grouped bar chart titled "By draft round: actual PPR vs ADP-implied vs model (2021–2025)".
X = round bins. Three bars per bin: mean actual, mean ADP-implied, mean model.
Colors: actual #1a1a1a, ADP #8a6a3a, model #1d4f46.
Y = mean full-PPR points.
Annotate n under each bin.
This should show the model tracks the actual curve, not a constant boost.

CSV:
bin,n,mean_actual,mean_ADP_implied,mean_model
R1,60,266.4,271.0,261.3
R2,62,234.2,224.1,219.1
R3,61,212.2,209.9,210.8
R4,60,205.6,192.0,190.7
R5–6,118,175.2,176.9,175.7
R7–8,116,161.0,158.3,156.8
R9–11,181,149.9,147.2,146.8
R12+,203,119.3,131.2,128.5
```

---

## 5. What actually beats the market (feature split)

**Prompt:**

```
Horizontal dot plot or two-column lollipop titled "Lagged stats: predicting PPR vs beating ADP".
Each feature is a row. Two dots/bars:
- corr with next-year PPR (teal #1d4f46)
- corr with (actual PPR − ADP-implied PPR) (brown #8a6a3a)
Vertical line at 0. Sort by |beat-ADP correlation|.
Callout: "depth_rank is mediocre for raw PPR but the strongest beat-the-market signal."

CSV:
feature,corr_ppr,corr_beat_adp
depth_rank,-0.545,-0.264
usage_index,0.624,0.114
touches,0.604,0.113
injury_bounce,0.062,-0.111
off_snaps,0.628,0.107
forty,-0.084,0.097
overproduction,0.165,-0.086
hv_rz,0.547,0.055
age_alpha,-0.032,0.033
chunk_rating,0.468,-0.032
draft_capital,0.536,0.025
td_luck,0.374,0.020
```

---

## 6. Tenure prior (stacked bars)

**Prompt:**

```
100% stacked horizontal bars titled "Tenure prior: what the model trusts before the residual trees".
Y = bucket (Rookie → Veteran). X = 0 to 1. Five segments: Market, Production, Situation, Physical, Aging.
Colors: Market #8a6a3a, Production #1d4f46, Situation #5c8a82, Physical #c4bba8, Aging #9b2c2c.
Label percentages on segments >10%.
Caption: "Rookies: situation + measurables. Prime: books + last year. Veterans: aging gets a real weight."

CSV:
bucket,market,production,situation,physical,aging
Rookie,0.22,0.08,0.40,0.30,0.00
Sophomore,0.22,0.32,0.32,0.12,0.02
Developing,0.28,0.38,0.24,0.06,0.04
Prime,0.40,0.34,0.16,0.04,0.06
Veteran,0.38,0.24,0.10,0.02,0.26
```

---

## 7. Pipeline diagram (no data)

**Prompt:**

```
A left-to-right architecture diagram, technical but readable by a non-analyst.

Boxes in order:
1. Data: nflverse PBP/stats · FFC ADP · season props
2. Market m: VFP if a prop exists, else log-ADP points curve
3. Features a prop cannot see: age α, vacated allocation, HV-RZ / TD luck, chunk×EPA, usage vs overproduction, shrunk OL/scheme
4. Tenure blend prior
5. Four shallow GBDTs: points, residual, position residual, rank lift
6. Published ŷ = market + λ·residual  (rookies pulled slightly toward the prior)
7. Steal/fade rule: ADP 18–132 AND ≥5 ranks AND ≥12 PPR AND a confirming flag

Title: "What the 2026 board is actually doing".
Flat boxes, #f7f4ee fill, #1d4f46 borders, no cheesy 3D arrows.
```

---

## 8. Position lift summary (simple companion to the heatmap)

Already have the heatmap PNG. If you want a simpler one-number chart:

```
Horizontal bars titled "Pooled 2021–2025: Spearman lift vs ADP by position".
QB +0.112, TE +0.063, WR +0.011, RB −0.002.
Color positive #0f6b4c, RB near-zero #c4bba8.
n: QB 125, RB 293, WR 342, TE 101.
Caption: "The model’s ranking edge is not uniform. It is mostly QB/TE; RB is a wash."
```
