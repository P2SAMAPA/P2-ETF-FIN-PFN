"""
features.py  —  From prices (and macro) to the panel the network reads.

For ETF e on day t (everything known at the close of t):
  0-3   momentum in vol units: r_t/sigma_t, 5d, 21d and 63d returns scaled by sigma_t*sqrt(horizon)
  4     volatility regime: log(vol21 / vol126)
  5-8   60-day rolling betas of the ETF's returns to: the universe return, the VIX change, the 10y-2y curve change and
        the dollar (DXY) change — what the ETF is exposed to, so that macro moves can be linked to its return
  9-12  today's macro environment, common to every ETF: the universe return, VIX, curve and DXY shocks relative to their
        trailing 60-day volatility (history only)
ETF-specific features (0-8) are standardized across the universe each day; all features are clipped to +-5.
Target: next-day RELATIVE return (minus the universe mean), in units of the ETF's trailing volatility ("risk-adjusted").

A row (day, ETF) is usable only if every feature and its volatility are finite. Context rows for a query day d come from
days d-CTX_DAYS .. d-1 only — their targets (the return from t to t+1, t <= d-1) are known at d.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)

import config
import prior

F = config.F_MAX
NF = config.N_REAL_FEATURES


def _roll_beta(Z: np.ndarray, R: np.ndarray, window: int) -> np.ndarray:
    """Rolling OLS betas of each column of R (n,E) on the regressors Z (n,k) over a trailing window: (n,E,k); NaN until enough history."""
    n, k = Z.shape
    E = R.shape[1]
    zz = np.einsum("ti,tj->tij", Z, Z).reshape(n, k * k)
    c_zz = np.concatenate([np.zeros((1, k * k)), np.cumsum(zz, axis=0)], axis=0)
    zr = np.einsum("ti,te->tie", Z, np.nan_to_num(R)).reshape(n, k * E)
    c_zr = np.concatenate([np.zeros((1, k * E)), np.cumsum(zr, axis=0)], axis=0)
    out = np.full((n, E, k), np.nan)
    if n < window:
        return out
    S = (c_zz[window:] - c_zz[:-window]).reshape(-1, k, k) + 1e-6 * np.eye(k)[None]
    C = (c_zr[window:] - c_zr[:-window]).reshape(-1, k, E)
    out[window - 1:] = np.linalg.solve(S, C).transpose(0, 2, 1)
    return out


def build_panel(prices: pd.DataFrame, macro: Optional[pd.DataFrame], tickers: List[str], name: str = "") -> Optional[Dict]:
    tickers = [t for t in tickers if t in prices.columns]
    if len(tickers) < config.MIN_UNIVERSE:
        return None
    px = prices[tickers].astype(float)
    lr = np.log(px / px.shift(1))
    R = lr.values                                              # (n,E) log returns, row t = return from t-1 to t
    n, E = R.shape
    sig = pd.DataFrame(R).rolling(config.VOL_WINDOW, min_periods=40).std().values        # sigma_t uses returns up to t
    mkt = np.nanmean(R, axis=1)

    def z_macro(x):                                           # today's shock vs the PREVIOUS window's volatility (history only)
        s = pd.Series(x).rolling(config.MACRO_Z_WINDOW, min_periods=30).std().shift(1).values
        return np.clip(x / np.where(s > 1e-12, s, np.nan), -5, 5)

    have_macro = macro is not None and all(c in macro.columns for c in config.MACRO_COLS)
    if have_macro:
        m = macro.reindex(px.index).ffill()
        d_vix = np.log(m["VIX"].clip(lower=1e-6)).diff().values
        d_curve = m["T10Y2Y"].diff().values
        d_dxy = np.log(m["DXY"].clip(lower=1e-6)).diff().values
    else:
        d_vix = d_curve = d_dxy = np.zeros(n)
    regs = np.stack([mkt, d_vix, d_curve, d_dxy], axis=1)
    scale = pd.DataFrame(regs).rolling(252, min_periods=60).std().values
    Zs = np.nan_to_num(regs / np.where(scale > 1e-12, scale, np.nan))
    betas = _roll_beta(Zs, R, config.BETA_WINDOW)              # (n,E,4)

    cs = lambda a: pd.DataFrame(R).rolling(a, min_periods=a).sum().values
    feats = np.full((n, E, F), np.nan)
    sg = np.where(sig > 1e-9, sig, np.nan)
    feats[:, :, 0] = R / sg
    feats[:, :, 1] = cs(5) / (sg * np.sqrt(5))
    feats[:, :, 2] = cs(21) / (sg * np.sqrt(21))
    feats[:, :, 3] = cs(63) / (sg * np.sqrt(63))
    v21 = pd.DataFrame(R).rolling(21, min_periods=21).std().values
    v126 = pd.DataFrame(R).rolling(126, min_periods=100).std().values
    feats[:, :, 4] = np.log(np.where(v21 > 1e-9, v21, np.nan) / np.where(v126 > 1e-9, v126, np.nan))
    feats[:, :, 5:9] = betas
    if not have_macro:
        feats[:, :, 6:9] = 0.0                                 # no macro series -> no macro sensitivities
    base = np.isfinite(feats[:, :, :9]).all(axis=2) & np.isfinite(sg)
    for j in range(9):                                         # cross-sectional standardization of the ETF-specific features
        col = np.where(base, feats[:, :, j], np.nan)
        mu = np.nanmean(col, axis=1, keepdims=True); sd = np.nanstd(col, axis=1, keepdims=True)
        # a feature that is constant across the universe on a day carries no cross-sectional information: 0, not NaN
        feats[:, :, j] = np.where(base, (col - mu) / np.where(sd > 1e-9, sd, 1.0), np.nan)
    mac = np.stack([z_macro(mkt / np.where(np.nanmean(sg, axis=1) > 0, np.nanmean(sg, axis=1), np.nan)), z_macro(d_vix), z_macro(d_curve), z_macro(d_dxy)], axis=1)
    mac = np.nan_to_num(mac)                                   # missing macro shock -> 0 ("no information")
    if not have_macro:
        mac[:] = 0.0
    feats[:, :, 9:13] = mac[:, None, :]
    feats = np.clip(feats, -5, 5)
    valid = np.isfinite(feats[:, :, :NF]).all(axis=2)
    nvalid = valid.sum(axis=1)
    valid &= (nvalid >= config.MIN_ETFS_PER_DAY)[:, None]
    X = np.where(valid[:, :, None], feats, np.nan)
    X[:, :, NF:] = 0.0
    X = np.where(valid[:, :, None], np.nan_to_num(X), 0.0)

    ret_next = np.vstack([R[1:], np.full((1, E), np.nan)])                      # next-day log return (raw units)
    z = np.vstack([R[1:] / sg[:-1], np.full((1, E), np.nan)])                   # next-day return in units of TODAY's volatility
    z = np.clip(z, -8, 8)
    zv = np.where(valid & np.isfinite(z), z, np.nan)
    y = zv - np.nanmean(zv, axis=1, keepdims=True)                              # relative to the universe, per day
    y = np.where(valid & np.isfinite(y), y, np.nan)
    return {"name": name, "dates": px.index, "etfs": tickers, "X": X.astype(np.float32), "valid": valid, "y": y, "ret_next": ret_next, "have_macro": have_macro}


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
def context_rows(panel: Dict, d: int, K: int = config.CTX_DAYS):
    """(t_idx, e_idx) of all usable context rows for query day d: days d-K..d-1 whose target is known."""
    lo = max(d - K, 0)
    ok = panel["valid"][lo:d] & np.isfinite(panel["y"][lo:d])
    t, e = np.where(ok)
    return t + lo, e


def make_episode(panel: Dict, d: int, rng: np.random.Generator, with_target: bool = True):
    """
    One episode for query day d: (xc,yc,ac,mc,xq,rq,yq,mq), qi — or None if there is too little context. The context TOKENS are a random
    subset of M_CTX rows of the recent window, but the in-context ridge inputs rq are computed from ALL rows of the window (a version
    that fed the network ridge predictions from the 160-row subset handicapped it against the rolling ridge, which uses every row).
    """
    t, e = context_rows(panel, d)
    if len(t) < 40:
        return None
    X_all = panel["X"][t, e][None]; y_all = panel["y"][t, e][None].astype(np.float32)
    a_all = ((d - t) / config.CTX_DAYS)[None].astype(np.float32); m_all = np.ones((1, len(t)), np.float32)
    qi = np.where(panel["valid"][d])[0]
    if len(qi) < config.MIN_ETFS_PER_DAY:
        return None
    xq = np.zeros((config.Q_MAX, F), np.float32); yq = np.zeros(config.Q_MAX, np.float32); mq = np.zeros(config.Q_MAX, np.float32)
    xq[:len(qi)] = panel["X"][d, qi]
    rq = prior.ridge_features(X_all, y_all, a_all, m_all, xq[None])[0]
    m = config.M_CTX
    sel = rng.choice(len(t), size=min(m, len(t)), replace=False)
    ts, es = t[sel], e[sel]
    xc = np.zeros((m, F), np.float32); yc = np.zeros(m, np.float32); ac = np.zeros(m, np.float32); mc = np.zeros(m, np.float32)
    n = len(ts)
    xc[:n] = panel["X"][ts, es]; yc[:n] = panel["y"][ts, es]; ac[:n] = (d - ts) / config.CTX_DAYS; mc[:n] = 1.0
    if with_target:
        yq[:len(qi)] = np.nan_to_num(panel["y"][d, qi]); mq[:len(qi)] = np.isfinite(panel["y"][d, qi]).astype(np.float32)
    else:
        mq[:len(qi)] = 1.0
    return (xc, yc, ac, mc, xq, rq, yq, mq), qi


def sample_real_batch(panels: List[Dict], day_hi: Dict[str, int], rng: np.random.Generator, B: int = config.BATCH):
    """B episodes from random historical days of random universes; query days are restricted to d <= day_hi[universe] (targets known)."""
    eps = []
    tries = 0
    while len(eps) < B and tries < B * 20:
        tries += 1
        pn = panels[int(rng.integers(len(panels)))]
        hi = day_hi[pn["name"]]
        lo = config.MIN_HISTORY_DAYS // 4
        if hi <= lo:
            continue
        d = int(rng.integers(lo, hi + 1))
        r = make_episode(pn, d, rng)
        if r is None or r[0][7].sum() < config.MIN_ETFS_PER_DAY:
            continue
        eps.append(r[0])
    if not eps:
        return None
    return tuple(np.stack(a) for a in zip(*eps))


CODE_VERSION = config.CODE_VERSION
