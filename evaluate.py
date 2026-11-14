"""
evaluate.py  —  Does in-context learning add anything beyond the obvious competitors?

Methods, scored on IDENTICAL (day, ETF) rows:
  pfn_ft        the network after prior pre-training AND meta-training on real historical windows (refit every REFIT_DAYS)
  pfn_prior     the network with prior pre-training only (no real-data training at all)
  stack_ridge   the network's MEAN alone: a learned stack of four in-context ridge predictions (two shrinkage levels, two recency decays), weights
                refit on real windows — NO transformer involved. pfn_ft minus stack_ridge is the transformer's own contribution (the scale head)
  roll_ridge    ridge regression refit EVERY DAY on the same recent window the network reads (all rows, not a subsample) —
                the direct non-neural analogue of in-context learning: the credit is PFN minus this
  fixed_ridge   one ridge on all history up to the last refit date (a single fixed relationship)
  lgbm          LightGBM on all history up to the last refit date (a fixed nonlinear relationship)
  momentum      the 21-day momentum feature alone
Metrics: daily rank IC against the next-day relative return, top-3 vs universe on realized next-day returns (gross and net
of costs), and for the network's predictive DISTRIBUTION: mean log score vs a no-skill distribution and the coverage of its
50% / 90% intervals. The network is ranked by expected risk-adjusted return (mean / predictive sd).
"""

import numpy as np
from scipy.stats import rankdata, t as tdist
from typing import Dict, List, Optional

import config
import prior
import features as FE
import pfn

import jax.numpy as jnp


def tstat(x) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def daily_ic(score: np.ndarray, y: np.ndarray, min_etfs: int = None) -> np.ndarray:
    min_etfs = config.MIN_ETFS_PER_DAY if min_etfs is None else min_etfs
    out = np.full(score.shape[0], np.nan)
    ok = np.isfinite(score) & np.isfinite(y)
    for t in np.where(ok.sum(axis=1) >= min_etfs)[0]:
        m = ok[t]
        a, b = rankdata(score[t, m]), rankdata(y[t, m])
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        out[t] = (a * b).sum() / den if den > 0 else np.nan
    return out


def topn_series(score, ret_next, top_n=None, cost_bps=None):
    top_n = config.TOP_N if top_n is None else top_n
    cost_bps = config.TRADING_COST_BPS if cost_bps is None else cost_bps
    n = score.shape[0]
    gross, net = np.full(n, np.nan), np.full(n, np.nan)
    prev = None
    for t in range(n):
        m = np.isfinite(score[t]) & np.isfinite(ret_next[t])
        if m.sum() < config.MIN_ETFS_PER_DAY:
            prev = None
            continue
        idx = np.where(m)[0]
        order = idx[np.argsort(-score[t, idx])][:top_n]
        g = ret_next[t, order].mean() - ret_next[t, idx].mean()
        repl = (top_n - len(set(order.tolist()) & prev)) / top_n if prev is not None else 0.0
        gross[t], net[t] = g, g - repl * cost_bps / 1e4
        prev = set(order.tolist())
    return gross, net


# ---------------------------------------------------------------------------
# The network, in context
# ---------------------------------------------------------------------------
def pfn_predict(params, panel: Dict, days: List[int], R: int = None, seed: int = 0, batch: int = 128) -> Dict[str, np.ndarray]:
    """
    For each query day: R random context subsets of the recent window -> forward pass -> ensemble. Returns arrays (n,E):
    mu (expected relative return in vol units), sd (predictive sd including disagreement between subsets), scale (t scale), p_up.
    """
    R = config.ENSEMBLE_R if R is None else R
    rng = np.random.default_rng(seed)
    n, E = panel["valid"].shape
    mu = np.full((n, E), np.nan); sd = np.full((n, E), np.nan); sc = np.full((n, E), np.nan)
    eps, meta = [], []
    for d in days:
        for r in range(R):
            out = FE.make_episode(panel, d, rng, with_target=False)
            if out is None:
                break
            eps.append(out[0]); meta.append((d, out[1]))
    if not eps:
        return {"mu": mu, "sd": sd, "scale": sc, "p_up": np.full((n, E), np.nan)}
    outs_mu, outs_ls = [], []
    for i in range(0, len(eps), batch):
        b = tuple(np.stack(a) for a in zip(*eps[i:i + batch]))
        xc, yc, ac, mc, xq, rq, yq, mq = [jnp.asarray(a) for a in b]
        m_, l_ = pfn.forward_jit(params, xc, yc, ac, mc, xq, rq, mq)
        outs_mu.append(np.asarray(m_)); outs_ls.append(np.asarray(l_))
    MU, LS = np.concatenate(outs_mu), np.concatenate(outs_ls)
    by_day: Dict[int, list] = {}
    for k, (d, qi) in enumerate(meta):
        by_day.setdefault(d, []).append((MU[k][:len(qi)], np.exp(LS[k][:len(qi)]), qi))
    nu = config.NU
    for d, lst in by_day.items():
        qi = lst[0][2]
        mus = np.stack([a for a, _, _ in lst]); ss = np.stack([b for _, b, _ in lst])
        var_t = (ss ** 2 * nu / (nu - 2)).mean(axis=0) + mus.var(axis=0)
        mu[d, qi] = mus.mean(axis=0)
        sd[d, qi] = np.sqrt(var_t)
        sc[d, qi] = np.sqrt(var_t * (nu - 2) / nu)                     # effective t scale including subset disagreement
    p_up = tdist.cdf(mu / np.where(sc > 0, sc, np.nan), df=nu)
    return {"mu": mu, "sd": sd, "scale": sc, "p_up": p_up}


def fit_real_stacking(params, panels: List[Dict], day_hi: Dict[str, int], seed: int = 0):
    """Stage 2a: refit the 4 stacking weights by closed-form least squares on real historical windows, shrunk toward the prior's weights."""
    rng = np.random.default_rng(seed + 7)
    A = np.zeros((4, 4)); b = np.zeros(4)
    for _ in range(config.STACK_EPISODES // config.BATCH):
        bt = FE.sample_real_batch(panels, day_hi, rng)
        if bt is None:
            break
        xc, yc, ac, mc, xq, rq, yq, mq = bt
        m = mq > 0
        A += rq[m].T @ rq[m]; b += rq[m].T @ yq[m]
    if not np.any(A):
        return params
    a0 = np.asarray(params["a_r"], dtype=np.float64)
    kappa = config.STACK_PRIOR_STRENGTH * np.mean(np.diag(A))
    a = np.linalg.solve(A + kappa * np.eye(4), b + kappa * a0)
    q = dict(params); q["a_r"] = jnp.asarray(a, dtype=jnp.float32)
    return q


def finetune(params, panels: List[Dict], day_hi: Dict[str, int], steps: int, seed: int = 0, lr: float = None):
    """
    Stage 2: meta-train on real historical windows (each episode a separate task) whose query targets are known by day_hi:
    (a) refit the stacking weights in closed form, (b) train the transformer (the predictive scale) with them held fixed.
    """
    p = fit_real_stacking(params, panels, day_hi, seed)
    if steps <= 0:
        return p
    lr = config.LR_FINETUNE if lr is None else lr
    rng = np.random.default_rng(seed)
    opt = pfn._adam_init(p)
    for s in range(steps):
        b = FE.sample_real_batch(panels, day_hi, rng)
        if b is None:
            break
        p, opt, _ = pfn.train_step(p, opt, pfn.lr_at(s, lr, warm=20, total=steps), *[jnp.asarray(a) for a in b])
    return p


# ---------------------------------------------------------------------------
# Non-neural competitors
# ---------------------------------------------------------------------------
def _rows(panel, lo, hi):
    ok = panel["valid"][lo:hi] & np.isfinite(panel["y"][lo:hi])
    t, e = np.where(ok)
    return panel["X"][lo:hi][t, e][:, :config.N_REAL_FEATURES], panel["y"][lo:hi][t, e]


def rolling_ridge(panel, days, K=config.CTX_DAYS, lam=config.ROLL_RIDGE_ALPHA):
    n, E = panel["valid"].shape
    mu = np.full((n, E), np.nan)
    for d in days:
        X, y = _rows(panel, max(d - K, 0), d)
        if len(y) < 40:
            continue
        beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
        qi = np.where(panel["valid"][d])[0]
        mu[d, qi] = panel["X"][d, qi][:, :config.N_REAL_FEATURES] @ beta
    return mu


def fixed_models(panel, block_starts, days_by_block, fit_lgbm=True):
    n, E = panel["valid"].shape
    mu_r = np.full((n, E), np.nan); mu_g = np.full((n, E), np.nan)
    for b0, days in zip(block_starts, days_by_block):
        X, y = _rows(panel, 0, b0)                                   # rows whose targets are known before the block starts
        if len(y) < 500:
            continue
        beta = np.linalg.solve(X.T @ X + config.RIDGE_ALPHA * np.eye(X.shape[1]), X.T @ y)
        gm = None
        if fit_lgbm:
            import lightgbm as lgb
            gm = lgb.LGBMRegressor(**config.LGBM_PARAMS).fit(X, y)
        for d in days:
            qi = np.where(panel["valid"][d])[0]
            Xq = panel["X"][d, qi][:, :config.N_REAL_FEATURES]
            mu_r[d, qi] = Xq @ beta
            if gm is not None:
                mu_g[d, qi] = gm.predict(Xq)
    return mu_r, mu_g


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def eval_window(panels: List[Dict], eval_days: int = None):
    n = len(panels[0]["dates"])
    ed = int(config.EVAL_YEARS * 252) if eval_days is None else eval_days
    d1 = n - 1                                                      # exclusive: the last day has no realized next-day target
    d0 = max(config.MIN_HISTORY_DAYS, d1 - ed)
    return d0, d1


def run_walk_forward(panels: List[Dict], params_prior, finetune_steps: int = None, refit_days: int = None, eval_days: int = None,
                     fit_lgbm: bool = True, R: int = None, log=None) -> Dict:
    finetune_steps = config.FINETUNE_STEPS if finetune_steps is None else finetune_steps
    refit_days = config.REFIT_DAYS if refit_days is None else refit_days
    d0, d1 = eval_window(panels, eval_days)
    starts = list(range(d0, d1, refit_days))
    blocks = [list(range(b, min(b + refit_days, d1))) for b in starts]
    res = {p["name"]: {} for p in panels}
    n, _ = panels[0]["valid"].shape
    store = {p["name"]: {k: np.full(p["valid"].shape, np.nan) for k in ("ft_mu", "ft_sd", "ft_scale", "ft_pup", "pr_mu", "pr_sd", "pr_scale")} for p in panels}
    for b0, days in zip(starts, blocks):
        hi = {p["name"]: b0 - 1 for p in panels}                    # the last query day whose target (day -> day+1) is known by b0
        params_ft = finetune(params_prior, panels, hi, finetune_steps, seed=b0)
        for p in panels:
            o = pfn_predict(params_ft, p, days, R=R, seed=b0); s = store[p["name"]]
            for k, v in (("ft_mu", o["mu"]), ("ft_sd", o["sd"]), ("ft_scale", o["scale"]), ("ft_pup", o["p_up"])):
                s[k][days] = v[days]
            o2 = pfn_predict(params_prior, p, days, R=R, seed=b0)
            for k, v in (("pr_mu", o2["mu"]), ("pr_sd", o2["sd"]), ("pr_scale", o2["scale"])):
                s[k][days] = v[days]
        if log:
            log("  block %d-%d done" % (days[0], days[-1]))
    for p in panels:
        s = store[p["name"]]
        allday = [d for b in blocks for d in b]
        roll = rolling_ridge(p, allday)
        fr, fg = fixed_models(p, starts, blocks, fit_lgbm=fit_lgbm)
        mom = np.where(p["valid"], p["X"][:, :, 2], np.nan)
        res[p["name"]] = {"scores": {"pfn_ft": s["ft_mu"] / s["ft_sd"], "stack_ridge": s["ft_mu"], "pfn_prior": s["pr_mu"] / s["pr_sd"], "roll_ridge": roll,
                                     "fixed_ridge": fr, "lgbm": fg, "momentum": mom},
                          "dist": {"mu": s["ft_mu"], "scale": s["ft_scale"], "sd": s["ft_sd"], "p_up": s["ft_pup"]},
                          "days": np.array([d for b in blocks for d in b]), "window": (d0, d1)}
    return res


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def summarize(panel: Dict, r: Dict) -> Dict:
    days = r["days"]
    mask = np.zeros(panel["valid"].shape[0], bool); mask[days] = True
    out = {"methods": {}, "paired": {}, "n_days": 0}
    ser = {}
    for k, sc in r["scores"].items():
        if not np.isfinite(sc[days]).any():
            continue
        ic = daily_ic(sc, panel["y"])
        gr, nt = topn_series(sc, panel["ret_next"])
        ser[k] = (ic, gr, nt)
        out["methods"][k] = {"ic_mean": float(np.nanmean(ic[mask])), "ic_t": tstat(ic[mask]), "gross_bps": float(np.nanmean(gr[mask]) * 1e4),
                             "net_bps": float(np.nanmean(nt[mask]) * 1e4), "net_t": tstat(nt[mask]), "n_days": int(np.isfinite(ic[mask]).sum())}
    out["n_days"] = out["methods"].get("pfn_ft", {}).get("n_days", 0)
    for base in ("roll_ridge", "stack_ridge", "fixed_ridge", "lgbm", "momentum", "pfn_prior"):
        if "pfn_ft" in ser and base in ser:
            d = (ser["pfn_ft"][0] - ser[base][0])[mask]; dn = (ser["pfn_ft"][2] - ser[base][2])[mask]
            out["paired"][base] = {"d_ic": float(np.nanmean(d)), "t": tstat(d), "d_net_bps": float(np.nanmean(dn) * 1e4), "net_t": tstat(dn)}
    # the predictive distribution
    dist = r["dist"]; y = panel["y"]; m = mask[:, None] & np.isfinite(dist["mu"]) & np.isfinite(y) & np.isfinite(dist["scale"])
    if m.sum() > 100:
        nu = config.NU
        yy, mu, sc = y[m], dist["mu"][m], dist["scale"][m]
        ls = tdist.logpdf((yy - mu) / sc, df=nu) - np.log(sc)
        s0 = float(np.std(yy)); ls0 = tdist.logpdf(yy / (s0 * np.sqrt((nu - 2) / nu)), df=nu) - np.log(s0 * np.sqrt((nu - 2) / nu))
        cov = {}
        for lvl in (0.5, 0.9):
            q = tdist.ppf(0.5 + lvl / 2, df=nu)
            cov[str(lvl)] = float((np.abs(yy - mu) <= q * sc).mean())
        out["distribution"] = {"mean_logscore": float(ls.mean()), "mean_logscore_noskill": float(ls0.mean()), "gain": float(ls.mean() - ls0.mean()),
                               "coverage_50": cov["0.5"], "coverage_90": cov["0.9"], "n_rows": int(m.sum()),
                               "mean_sd_pred": float(np.mean(dist["sd"][m])), "realized_sd": float(np.std(yy))}
    return out


def verdict(summary: Dict) -> Dict:
    m, pr = summary["methods"].get("pfn_ft"), summary["paired"].get("roll_ridge")
    if not m or not pr:
        return {"verdict": "Not enough data", "reason": "No walk-forward predictions."}
    if m["ic_t"] >= config.IC_T_SUPPORTED and pr["t"] >= config.IC_T_SUPPORTED and m["net_bps"] > 0:
        v, why = "Supported", "Significant IC, beats the rolling ridge on the same context, and the top-3 pays after costs."
    elif m["ic_t"] >= config.IC_T_SUGGESTIVE and pr["t"] >= config.IC_T_SUGGESTIVE:
        v, why = "Suggestive", "Some ranking information beyond the rolling ridge" + ("" if m["net_bps"] > 0 else ", but the daily top-3 does not pay after costs (%.1f bps/day)." % m["net_bps"])
    else:
        v, why = "Not supported", "The network does not rank ETFs better than a ridge refit on the same recent window (IC t = %.2f, vs rolling ridge dIC t = %.2f)." % (m["ic_t"], pr["t"])
    return {"verdict": v, "reason": why, "ic_mean": m["ic_mean"], "ic_t": m["ic_t"], "net_bps": m["net_bps"], "vs_roll_ridge": pr}


CODE_VERSION = config.CODE_VERSION
