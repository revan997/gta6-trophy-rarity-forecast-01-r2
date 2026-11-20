"""gta6-trophy-rarity-forecast-01, round 2: forecast each GTA VI PS5 trophy's
Exophase earn rate at the case's target as five quantiles, scored by pinball
loss on the logit scale.

Why round 1 was not enough, and what this round does differently
---------------------------------------------------------------
Round 1 modelled holders H and players N as separate power laws in log time.
That is a fair description of smooth growth, but Exophase does not grow
smoothly: it refreshes in steps. Between refreshes nothing moves; at a
refresh a whole new cohort of players appears at once, and the new cohort
has earned almost nothing yet except the very first trophies. Two facts
follow, and both are visible in the data the kit itself documents:

* A trophy's rate is H / N, so a refresh dilutes every rate by the jump in
  N, minus whatever the new cohort has earned. The launch-night refresh
  (638 -> 3,868 players, 943 -> 4,276 trophies earned) shows the split:
  3,230 new players, 3,333 new trophies, of which the near-universal
  trophy takes ~3,230. Only ~103 new holders are left for the other 54
  trophies, and they sit on the trophies new players reach first -- the
  ones with the highest rates already. A uniform "new players earn every
  trophy at 12% of the old rate" (round 1's reconstruction) hands rare
  trophies far too many new holders.

* A flat history carries no slope at all, yet a refresh is still coming.
  With no refresh ever seen, this program falls back on the shape of that
  documented first refresh: one refresh is due, the denominator jumps by a
  large factor, later refreshes by a much smaller one, and the few
  non-universal new holders follow a steep function of the old rate
  (rate cubed -- the most robust profile across every reconstruction of
  the practice answer we tested, mean loss ~0.10 against round 1's 0.59).

When the history DOES contain refreshes, the program learns from them
instead of from the prior. Each refresh gives a direct observation of the
new-cohort profile: g_i = (new holders of trophy i) / (new players).
Three forecasts of the target rate are then combined in logit space:

1. cohort model: H_i(target) = H_i(last) + g_i * (N(target) - N(last)),
   with g_i pooled over the last few refreshes and shrunk toward a
   cross-trophy regression of log g on log rate;
2. power-law model: log H_i and log N fitted against log(1 + hours since
   release), last-24h window blended with the whole history, slopes
   shrunk toward the cross-trophy median (round 1's approach);
3. curve model: a straight line fitted to the logit rate itself against
   log(1 + hours), over the last 24 hours (the house curve baseline's
   form, but used as one voice among three, not the whole forecast).

The denominator N(target) comes from the power-law fit, which captures
the deceleration of player growth; a constant per-refresh multiplier
would compound without decelerating and badly overshoot over a week.
The spread is a logit sigma that grows with the horizon and with how far
the three models disagree -- an honest width, not a fixed one.

Everything is computed from the case input at run time. The only numbers
that come from outside the case are the generic priors above (the shape
of a first refresh, the steepness of the new-cohort profile), which the
final case -- with its own multi-day history of real refreshes -- will
override with its own evidence.

Standard library only. Deterministic: same case in, same bytes out.
Run:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# The game's release day begins 2026-11-19T00:00:00Z; log time is measured
# from there, as the house curve baseline does.
RELEASE_START = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Generic priors (shapes, not answers) ---
W_COHORT = 0.30    # weight of the cohort model in the logit blend
W_POWER = 0.35     # weight of the power-law H/N model
W_CURVE = 0.35     # weight of the logit-rate curve model
G_PRIOR_PLAYERS = 2000.0  # pseudo-players behind the shrunk cohort rate
FIRST_JUMP = 6.0   # denominator jump of a first refresh, when none seen
LATER_JUMP = 1.5   # denominator jump of later refreshes, when none seen
REFRESH_GAP_H = 4.5  # typical hours between refreshes, when none seen
NONUNIV_PER_NEW = 0.032  # non-universal trophies earned per new player
PROFILE_POWER = 3.0      # new-holder profile ~ (old rate) ** this
SIGMA_BASE = 0.25  # logit sigma at zero horizon, refreshes seen
SIGMA_HORIZON = 0.05  # extra sigma per sqrt(day) of horizon
SIGMA_DISAGREE = 0.30  # extra sigma per logit of model disagreement
SIGMA_FLOOR = 0.25
SIGMA_CAP = 1.1
FLAT_SIGMA_BASE = 0.20  # logit sigma at zero horizon, no refresh seen
FLAT_SIGMA_HORIZON = 0.30
UNIVERSAL_RATE = 99.9  # a trophy (nearly) every tracked player holds
UNIVERSAL_SIGMA = 0.25


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def quantiles(med_pct, sigma):
    lg = logit(med_pct)
    qs = []
    prev = None
    for z in Z:
        q = round4(inv_logit(lg + z * sigma))
        q = min(max(q, 0.01), 99.99)
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def holder_count(rate, players):
    """Integer holders implied by a printed 2-decimal rate (near-exact)."""
    if rate is None:
        return None
    return int(math.floor(rate * players / 100.0 + 0.5))


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels %r" % (case["levels"],))
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    n_tr = len(trophies)
    target = parse_time(case["target"])
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0

    # --- Distinct snapshots: collapse consecutive identical reads. ---
    # A read is a step function (Exophase refreshes every few hours), so a
    # run of identical reads is one state, stamped at the time it was LAST
    # confirmed -- the anchor every forecast extrapolates from.
    snaps = []  # (t_hours, N, rates tuple)
    prev_key = None
    for h in history:
        n_read = h["players_tracked"]
        rates_read = h["rates"]
        if n_read is None or n_read <= 0 or rates_read is None:
            continue
        if len(rates_read) != n_tr:
            continue
        key = (n_read, tuple(rates_read))
        t_h = (parse_time(h["at"]) - RELEASE_START).total_seconds() / 3600.0
        if key == prev_key:
            snaps[-1] = (t_h, snaps[-1][1], snaps[-1][2])
            continue
        prev_key = key
        snaps.append((t_h, h["players_tracked"], h["rates"]))
    if not snaps:
        raise ValueError("case has no usable history reads")
    xs = [math.log1p(max(t, -0.9)) for t, _, _ in snaps]
    x_last = xs[-1]
    x_target = math.log1p(max((target - RELEASE_START).total_seconds() / 3600.0, -0.9))
    dx = max(x_target - x_last, 0.0)
    t_last = snaps[-1][0]
    n_last = snaps[-1][1]
    last_rates = snaps[-1][2]
    seen_rates = [r for r in last_rates if r is not None]
    global_rate = median(seen_rates) if seen_rates else 1.0

    # --- Refresh events: what each jump in N taught us about new players. ---
    events = []  # {"dN": int, "dH": [int|None], "mult": float}
    for j in range(1, len(snaps)):
        d_n = snaps[j][1] - snaps[j - 1][1]
        if d_n <= 0:
            continue
        d_h = []
        for i in range(n_tr):
            h_old = holder_count(snaps[j - 1][2][i], snaps[j - 1][1])
            h_new = holder_count(snaps[j][2][i], snaps[j][1])
            d_h.append(None if (h_old is None or h_new is None) else h_new - h_old)
        events.append({"dN": d_n, "dH": d_h,
                       "mult": snaps[j][1] / snaps[j - 1][1]})

    # --- Denominator: power-law fit, last 24 h blended with all history. ---
    # Growth decelerates, so the fit is done in log time; the last day
    # counts as much as everything before it.
    log_n = [math.log(max(n, 1)) for _, n, _ in snaps]
    fit_n = ols(xs, log_n)
    idx24 = [j for j, (t, _, _) in enumerate(snaps) if t >= t_last - 24.0]
    fit_n24 = ols([xs[j] for j in idx24], [log_n[j] for j in idx24]) \
        if len(idx24) >= 3 else None
    if fit_n is not None:
        b_n = 0.5 * fit_n[1] + 0.5 * fit_n24[1] if fit_n24 else fit_n[1]
        b_n = min(max(b_n, 0.0), 2.5)
    else:
        b_n = 1.0
    n_pow = max(n_last * math.exp(b_n * dx), n_last)

    if not events:
        return forecast_flat(trophies, last_rates, global_rate, n_last,
                             horizon_h, horizon_d)

    # --- Cohort prior: log(new-cohort rate) regressed on log(old rate). ---
    # Pooled over the last few refreshes; trophies the new cohort never
    # touched contribute nothing, so the regression is fit on positives.
    recent = events[-5:]
    reg_x = []
    reg_y = []
    for ev in recent:
        for i in range(n_tr):
            if ev["dH"][i] is None:
                continue
            r = last_rates[i]
            if r is None or r >= UNIVERSAL_RATE or r <= 0:
                continue
            g = ev["dH"][i] / ev["dN"]
            if g > 0:
                reg_x.append(math.log(r))
                reg_y.append(math.log(g))
    reg = ols(reg_x, reg_y) if len(reg_x) >= 5 else None

    # Cross-trophy median holder slope, for shrinking weak power-law fits.
    slopes = []
    for i in range(n_tr):
        pts = [(xs[j], math.log(max((snaps[j][2][i] or 0) * snaps[j][1] / 100.0, 0.5)))
               for j in range(len(snaps)) if snaps[j][2][i] is not None]
        f = ols([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 2 else None
        if f is not None:
            slopes.append(f[1])
    slope_bar = median(slopes) if slopes else 0.5
    slope_bar = min(max(slope_bar, 0.0), 2.5)
    x24_start = xs[idx24[0]] if idx24 else xs[0]

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = global_rate
        # Near-universal trophy: (nearly) every tracked player holds it;
        # it stays near 100% whatever the denominator does.
        if r_last >= UNIVERSAL_RATE:
            rows.append({"id": trophy["id"],
                         "q": quantiles(min(99.99, r_last), UNIVERSAL_SIGMA)})
            continue
        h_last = r_last * n_last / 100.0

        # 1. Cohort model: new players earn this trophy at rate g_i.
        pooled_h = 0.0
        pooled_n = 0.0
        for w, ev in zip([1.0, 0.85, 0.7, 0.55, 0.4][:len(recent)], recent):
            if ev["dH"][i] is not None:
                pooled_h += w * ev["dH"][i]
                pooled_n += w * ev["dN"]
        if reg is not None:
            g_prior = math.exp(reg[0] + reg[1] * math.log(max(r_last, 0.01)))
        else:
            g_prior = 0.0004 * (max(r_last, 0.01) ** 1.5)
        g_prior = min(max(g_prior, 1e-6), 0.05)
        g_hat = (pooled_h + G_PRIOR_PLAYERS * g_prior) / \
            (pooled_n + G_PRIOR_PLAYERS) if pooled_n > 0 else g_prior
        r_cohort = clamp_pct(100.0 * (h_last + g_hat * max(n_pow - n_last, 0)) / n_pow)

        # 2. Power-law model: log H_i fitted in log time, shrunk to the group.
        pts = [(xs[j], math.log(max((snaps[j][2][i] or 0) * snaps[j][1] / 100.0, 0.5)))
               for j in range(len(snaps)) if snaps[j][2][i] is not None]
        fit_h = ols([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 2 else None
        pts24 = [p for p in pts if p[0] >= x24_start]
        fit_h24 = ols([p[0] for p in pts24], [p[1] for p in pts24]) \
            if len(pts24) >= 3 else None
        if fit_h is not None:
            b_i = 0.5 * fit_h[1] + 0.5 * fit_h24[1] if fit_h24 else fit_h[1]
            wgt = len(pts) / (len(pts) + 3.0)
            b_i = wgt * b_i + (1.0 - wgt) * slope_bar
            b_i = min(max(b_i, 0.0), 2.5)
            h_pow = math.exp(math.log(max(h_last, 0.5)) + b_i * dx)
            r_power = clamp_pct(100.0 * h_pow / n_pow)
        else:
            r_power = r_cohort

        # 3. Curve model: logit rate fitted in log time over the last day.
        c_pts = [(xs[j], logit(snaps[j][2][i]))
                 for j in range(len(snaps))
                 if snaps[j][2][i] is not None and xs[j] >= x24_start]
        fit_c = ols([p[0] for p in c_pts], [p[1] for p in c_pts]) \
            if len(c_pts) >= 3 else None
        if fit_c is not None:
            r_curve = clamp_pct(inv_logit(fit_c[0] + fit_c[1] * x_target))
        else:
            r_curve = r_power

        lg = (W_COHORT * logit(r_cohort) + W_POWER * logit(r_power)
              + W_CURVE * logit(r_curve))
        med = inv_logit(lg)
        disagree = max(abs(logit(r_cohort) - logit(r_power)),
                       abs(logit(r_cohort) - logit(r_curve)),
                       abs(logit(r_power) - logit(r_curve)))
        sigma = SIGMA_BASE + SIGMA_HORIZON * math.sqrt(horizon_d) \
            + SIGMA_DISAGREE * disagree
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        rows.append({"id": trophy["id"], "q": quantiles(med, sigma)})
    return {"trophies": rows}


def forecast_flat(trophies, last_rates, global_rate, n_last,
                  horizon_h, horizon_d):
    """No refresh has ever been seen: the history is one flat state.

    Fall back on the shape of the documented first refresh: a refresh is
    due (the state has already sat unchanged for hours), the denominator
    jumps by a large factor on the first refresh and a much smaller one
    after that, and the few non-universal trophies the new cohort earns
    follow a steep function of the old rate -- new players reach the
    earliest trophies first, and almost nothing else in their first hours.
    """
    if horizon_h >= 1.0:
        n_ref = 1.0 + max(0.0, (horizon_h - REFRESH_GAP_H) / REFRESH_GAP_H)
        n_target = n_last * (FIRST_JUMP ** min(n_ref, 1.0)) \
            * (LATER_JUMP ** max(n_ref - 1.0, 0.0))
    else:
        n_target = n_last * (1.0 + (FIRST_JUMP - 1.0) * horizon_h / REFRESH_GAP_H)
    d_n = max(n_target - n_last, 0.0)
    weights = []
    for i in range(len(trophies)):
        r = last_rates[i]
        if r is None:
            r = global_rate
        weights.append(0.0 if r >= UNIVERSAL_RATE else max(r, 0.01) ** PROFILE_POWER)
    total_w = sum(weights) or 1.0
    sigma = FLAT_SIGMA_BASE + FLAT_SIGMA_HORIZON * math.sqrt(horizon_d)
    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = global_rate
        if r_last >= UNIVERSAL_RATE:
            rows.append({"id": trophy["id"],
                         "q": quantiles(min(99.99, r_last), UNIVERSAL_SIGMA)})
            continue
        h_last = r_last * n_last / 100.0
        new_h = NONUNIV_PER_NEW * d_n * (weights[i] / total_w)
        med = clamp_pct(100.0 * (h_last + new_h) / n_target)
        rows.append({"id": trophy["id"], "q": quantiles(med, sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
