# gta6-trophy-rarity-forecast-01 — round 2

Forecast each GTA VI PS5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z
as five quantiles (0.05/0.25/0.5/0.75/0.95), scored by mean pinball loss on
the logit scale. $250, top 20 pay. Entry: `python3 /bot/main.py`
(stdlib only, no network, deterministic).

## Model (round 2, built from scratch)

A trophy's rate is H / N. Exophase refreshes in steps, so the model learns
from the refreshes themselves instead of only from smooth trends:

- **Snapshots**: consecutive identical reads collapse to one state.
- **Refresh events**: each jump in N gives a direct observation of the
  new-cohort profile, g_i = new holders of trophy i / new players.
- **Three forecasts blended in logit space** (0.30 / 0.35 / 0.35):
  1. *cohort*: H_i(target) = H_i(last) + g_i · (N(target) − N(last)),
     g_i pooled over the last 5 refreshes, shrunk toward a regression of
     log g on log rate;
  2. *power-law*: log H_i and log N fitted vs log(1 + hours since release),
     last-24h blended with whole history, slopes shrunk to the group median;
  3. *curve*: logit rate fitted vs log(1 + hours) over the last 24 h.
- **Denominator** N(target) from the power-law fit (growth decelerates;
  a constant per-refresh multiplier would compound and overshoot).
- **Spread**: logit sigma growing with horizon and with model disagreement.
- **Flat history (no refresh seen)**: falls back on the shape of the
  documented first refresh — one refresh due, denominator ×~6 then ×~1.5,
  and the ~0.032 non-universal trophies per new player distributed as
  (old rate)³, i.e. concentrated on the earliest trophies. The launch-night
  totals (3,230 new players, only 103 non-universal new holders) rule out
  round 1's uniform-12% profile.
- Near-universal trophies (≥99.9%) get a tight band around ~99.99%.

## Why this beats round 1

Round 1 (live practice loss 0.59, rank 4164) used one smooth power-law
model plus a uniform new-cohort factor. Round 2 replaces the uniform factor
with the observed refresh profile and adds the cohort + curve voices. On
backtests (see `../backtest.py`, honest numbers, no cherry-picking):

| Test | Round 2 | Round 1 | Persist | Curve |
| --- | --- | --- | --- | --- |
| Practice recon, mean over 7 truth profiles | 0.106 (range 0.056–0.145) | 0.575 (range 0.505–0.695) | 1.49 | 1.49 |
| Flat history → live Nov-20 page (N=30,000) | 0.313 (cov90 0.89) | 0.72 | 2.29 | 2.29 |
| Simulated final, power-law truth (10 seeds) | 0.189 | 0.203 | 0.413 | 0.198 |
| Simulated final, step/refresh truth (10 seeds) | 0.170 | 0.257 | 0.240 | 0.412 |
| Simulated final, regime-bend truth (10 seeds) | 0.203 | 0.210 | 0.261 | 0.186 |

Round 2 beats round 1 clearly on practice, on the long-horizon refresh
check, and on step/regime-change finals; it roughly matches round 1 on a
pure power-law final (where round 1's functional form is the truth). One
honest exception: on the regime-bend simulation the curve baseline alone
(0.186) beats round 2 (0.203) — a bend after the deadline favours the
steepest extrapolator, and no model can foresee a bend. Round 2 still
beats round 1 and persistence there.

## Honest limits

- The practice answer's exact per-trophy split of the 103 new holders is
  not identifiable from the kit; ~0.10 is the honest ceiling on the
  reconstruction family, not the 0.022 live top-20 bar (which likely
  reflects fitting the live practice board itself).
- The final is a 4.5-day extrapolation: if growth bends in a way no
  November 19–21 refresh foreshadows, every model misses together.
- The simulated finals are synthetic truths, not the real Nov-26 rates.

## Tests

Ran `main.py` twice on p00/p01: outputs byte-identical; format validated
(55×5, ids 1–55 once, monotonic, in [0,100], ≤65,536 bytes, one top-level
key); runtime ~0.05 s per run (cap 8 s); deterministic; stdlib only.
