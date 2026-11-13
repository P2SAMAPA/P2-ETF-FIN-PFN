"""
config.py  —  Configuration for P2 FIN-PFN (Financial Prior-Data Fitted Network)

A prior-data fitted network (PFN) is a transformer pre-trained on many synthetic prediction tasks drawn from a PRIOR,
so that at prediction time it performs inference IN CONTEXT: it is shown a context of labelled examples (here: recent
feature -> next-day-return rows) plus unlabelled queries (today's ETFs) and outputs a predictive distribution for each
query in one forward pass, with no weight update. Because the context is rebuilt every day, the mapping it applies
follows whatever feature-return relationship the recent past exhibits, without a hand-specified regime label.

This repo (1) pre-trains on a synthetic prior of regime-switching, low signal-to-noise relationships, (2) meta-trains
further on real historical windows (each window is a separate task), and (3) tests whether the result ranks ETFs
better than the obvious non-neural competitors given the SAME context: a ridge regression refit on the recent window
every day (rolling ridge), a fixed ridge and a fixed LightGBM trained on all history.
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-fin-pfn-results"
WEIGHTS_FILE = "prior_weights.npz"

# data_manager.py (shared with the other engines) builds its ticker list from UNIVERSES. Universes are scored
# separately (never combined in one scoring pass); one network serves all three because it is set-based.
UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
}
MIN_UNIVERSE = 6
TOP_N = 3
TRADING_COST_BPS = 15

# ---------------------------------------------------------------------------
# Features (13 real, padded to F_MAX slots). Every ETF-specific feature is vol-scaled and standardized across the
# universe each day; macro features are time z-scores (vs the trailing 60 days, history only). The target is the
# next-day return RELATIVE to the universe mean, in units of the ETF's trailing volatility ("risk-adjusted").
# ---------------------------------------------------------------------------
F_MAX = 16
N_REAL_FEATURES = 13
VOL_WINDOW = 63
BETA_WINDOW = 60
MACRO_Z_WINDOW = 60
MACRO_COLS = ["VIX", "T10Y2Y", "DXY"]          # full 2008+ history; the spread series only start ~2022 and are zero-filled before

# ---------------------------------------------------------------------------
# Network. Queries attend ONLY to the context (never to each other), so a query's prediction cannot depend on the
# other ETFs in the same query set. The head outputs a Student-t predictive (mean, scale); NU is fixed.
# (A binned "bar distribution" is the PFN signature, but with targets whose signal is ~1% of the variance the mean
# shift is far smaller than any bin; a parametric head estimates the mean directly.)
# ---------------------------------------------------------------------------
D_MODEL = 48
N_LAYERS = 2
N_HEADS = 4
D_FF = 96
M_CTX = 160            # context rows per episode (sampled from the recent window)
Q_MAX = 44             # max queries (ETFs) per episode
CTX_DAYS = 60          # the recent window the context is drawn from
NU = 5.0
ENSEMBLE_R = 8         # at inference average over R random context subsets of the same window

# Training
BATCH = 32
LR = 2e-3
LR_FINETUNE = 3e-4
WARMUP = 200
PRETRAIN_STEPS = 6000
FINETUNE_STEPS = 300
GRAD_CLIP = 1.0

# Synthetic prior: realistic LOW signal-to-noise (R^2 of the true relationship per row, log-uniform), 25% pure-null
# tasks, regime-switching / drifting / fixed coefficients, sparse, some nonlinear, fat-tailed noise.
PRIOR_R2_LOW = 3e-4
PRIOR_R2_HIGH = 5e-2
PRIOR_NULL_SHARE = 0.25
CURRICULUM_R2_START = 0.25     # training starts with easier (higher signal) tasks and anneals to PRIOR_R2_HIGH over the first half

# Walk-forward evaluation on real data
EVAL_YEARS = 5.0
REFIT_DAYS = 252        # the network is meta-trained further on real windows before each refit date
MIN_HISTORY_DAYS = 756
MIN_ETFS_PER_DAY = 6
RIDGE_ALPHA = 100.0
ROLL_RIDGE_ALPHA = 100.0
LGBM_PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=200,
                   subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, n_jobs=2)

# Verdict (on the walk-forward period): Supported = IC t >= 1.65 AND beats the rolling ridge on the same context
# (paired t >= 1.65) AND top-3 pays after costs; Suggestive = IC t >= 1 and beats it with t >= 1.
IC_T_SUPPORTED = 1.65
IC_T_SUGGESTIVE = 1.0
SERIES_STEP = 3

CODE_VERSION = "2026-11-13.1"
