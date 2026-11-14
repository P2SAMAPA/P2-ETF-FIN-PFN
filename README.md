# P2-ETF-FIN-PFN — Financial Prior-Data Fitted Network

Predict each ETF's next-day return **distribution conditional on the most recent feature-return pairs**, for all ETFs in a
universe jointly, then rank by expected risk-adjusted return (mean / predictive sd). The question it exists to answer is not
"does the model rank ETFs" but **"does in-context learning add anything beyond a ridge regression refit every day on the same
recent window?"**

## About the cited paper
The spec cites "Meta-learning for return prediction in shifting market regimes" (Journal of Financial Markets, June 2026). **I could
not find that paper** and the link carried a `utm_source=chatgpt.com` tag, so treat the citation as unverified. The method is
real and well established: prior-data fitted networks (Müller et al. 2022), TabPFN, and *Drift-Resilient TabPFN* (arXiv 2411.10634),
which applies the same in-context idea to temporal distribution shift. This repo is built from that framework, not from claims in the
unverified paper.

## How it works
* **Features (13)**, all known at the close of day t: momentum at 1/5/21/63 days and a volatility-regime ratio (vol-scaled), 60-day
  rolling betas to the universe, VIX, the 10y-2y curve and DXY (what each ETF is exposed to), and today's macro shocks (universe, VIX,
  curve, DXY relative to their trailing volatility). ETF-specific features are standardized across the universe each day. Macro:
  VIX, T10Y2Y, DXY (full history; the spread series start ~2022 and are zero-filled before, so they are not used).
* **Target**: next-day return relative to the universe mean, in units of the ETF's trailing volatility ("risk-adjusted").
* **In-context prediction**: for each day the network reads a *context* (random rows from the last 60 days: features, next-day return,
  age) and *queries* (today's ETFs). Queries attend only to the context, never to each other (tested). It outputs a Student-t
  predictive (mean, scale) per ETF. No weights change at prediction time; the context is rebuilt every day, so the mapping follows
  whatever relationship the recent window shows, with no regime label.
* **Stage 1 — prior pre-training** on synthetic tasks: regime-switching, drifting, fixed or absent (25% null) sparse linear
  relationships at realistic low signal-to-noise (median R² 0.4%), some nonlinear, fat-tailed noise. `prior_weights.npz` (151 KB,
  725 steps) ships with the repo; `python pretrain.py <seconds>` regenerates it (resumable).
* **Stage 2 — meta-training on real windows** before every refit date (each historical window is a separate task).
* **Evaluation**: walk-forward over the last 5 years; `pfn_ft` is scored against a rolling ridge on the same window, the network's mean
  alone (`stack_ridge`), a fixed ridge, LightGBM and momentum, on identical rows. Distribution quality: log score and interval coverage.

## What the network actually learns (read this)
Pre-training on a CPU in minutes cannot teach a transformer what GPU-scale PFNs learn, and the experiments showed it plainly:
1. A pure network showed **no in-context learning** after ~900 steps (IC ≈ 0 while a ridge on the same context gave +0.016).
2. Feeding each query the closed-form ridge predictions from the same context (a "Bayesian-style" input, four variants: two shrinkage
   levels, two recency decays) fixed that, but **letting the transformer correct the mean made the ranking worse** (validation IC fell
   from +0.016 to +0.001–0.005): the gradient signal for a correction is far below its noise at this scale. So `MU_CORRECTION = 0`.
3. Even the four combination weights were being destroyed by SGD noise (Adam random-walks them), so they are now fit in **closed form**
   (pooled least squares on the prior, then on real windows, shrunk toward the prior's weights).
4. The transformer therefore learns only the predictive scale, and **its scale beat the best constant scale by 0.0009 nats** (0.25%).

Result: FIN-PFN ≈ **a learned stack of recency-weighted in-context ridge regressions with an almost-constant Student-t scale**. The
dashboard's `pfn_ft minus stack_ridge` row is the transformer's own contribution; in validation it was ~0 in every world. Set
`MU_CORRECTION = 1.0` and pre-train for far longer (GPU) to test the real PFN.

## Validation (synthetic markets with a known relationship; 5 seeds each; same walk-forward and verdict code as real data)
| world | verdicts | FIN-PFN IC | rolling ridge | fixed ridge | LightGBM | PFN − rolling ridge | PFN − stack_ridge |
|---|---|---|---|---|---|---|---|
| null | Not supported ×4, Suggestive ×1 | +0.011 | +0.009 | – | – | +0.002 (t −0.2) | 0.000 |
| fixed linear | Not supported ×4, Suggestive ×1 | +0.079 | +0.079 | +0.101 | +0.082 | −0.0004 (t −0.1) | 0.000 |
| latent regimes | Not supported ×5 | +0.060 | +0.063 | +0.049 | +0.042 | −0.003 (t −0.6) | 0.000 |
| macro-gated nonlinear | Suggestive ×4, Supported ×1 | +0.065 | +0.048 | −0.004 | +0.062 | **+0.018 (t +1.8)** | 0.000 |

* It stays quiet under the null (no run reached "Supported") and its intervals are calibrated in every world (50% → 0.50, 90% → 0.90).
* It beats a **fixed** model whenever the relationship moves (+0.070 vs fixed ridge in the gated world) — but so does the rolling ridge.
* It beats the plain rolling ridge **only** in the gated world, and **all of that gain is the recency-weighted stacking, not the transformer**.
* LightGBM is as good as FIN-PFN where the relationship is fixed and nonlinear.
* With the network's ridge inputs computed from a 160-row subsample it was *worse* than the rolling ridge (which uses every row); that
  handicap was removed (inputs now use the whole window).

## Verdict (walk-forward period)
Supported = IC t ≥ 1.65 **and** beats the rolling ridge on the same window (paired t ≥ 1.65) **and** the daily top-3 pays after costs;
Suggestive = IC t ≥ 1 and beats it with t ≥ 1; otherwise Not supported. Costs 15 bps round trip × fraction of picks replaced.

## Limits and what is not verified
* **Not run on real data.** Real-scale timing was measured on synthetic prices: about 5–6 minutes per run on one core. The real-data
  walk-forward is the first run on your Actions.
* The prior bounds the network: it can only be as good as the prior resembles real markets, and real predictability is probably smaller
  and messier than any prior here. The macro-gated world is in the *spirit* of the prior but not drawn from it.
* `jax[cpu]` is a new dependency (installs in under a minute); it was tested here, not on Actions.
* Daily top-3 rebalancing costs ~7–10 bps/day; a signal with IC ≈ 0.02 does not pay that.

## Files
`config.py` · `prior.py` (synthetic prior) · `pfn.py` (network) · `pretrain.py` (stage 1) · `features.py` (real panel, episodes) ·
`evaluate.py` (in-context prediction, baselines, walk-forward, metrics, verdict) · `simworld.py` · `validation.py` · `trainer.py` ·
`streamlit_app.py` · `prior_weights.npz` · `validation_results.json` · `.github/workflows/daily_run.yml`.
Every module carries a `CODE_VERSION`; `trainer.py` refuses to run a mixed set. **Replace all files together.**

Run: `python trainer.py` (needs `HF_TOKEN`; results go to `P2SAMAPA/p2-etf-fin-pfn-results`). Not financial advice.
