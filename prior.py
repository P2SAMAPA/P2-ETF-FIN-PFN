"""
prior.py  —  The synthetic prior the network is pre-trained on.

Each task is a small market: N assets (12-43) over a context window of CTX_DAYS plus a query day. Features follow
AR(1) processes (some persistent like betas, some noisy like returns) and are standardized across assets each day,
exactly as the real pipeline does. The next-day relative return is a LOW signal-to-noise function of the features
whose coefficients are one of: regime-switching (2-4 latent regimes with Markov switching), smoothly drifting,
fixed, or absent (25% null tasks). Some tasks carry a nonlinear interaction. Noise is fat-tailed (Student-t).
The prior is where the network's inductive bias comes from: it can only be as good as this prior resembles reality.
"""

import numpy as np
import config

F, K, M, QMAX = config.F_MAX, config.CTX_DAYS, config.M_CTX, config.Q_MAX
RIDGE_VARIANTS = [(30.0, None), (300.0, None), (30.0, 10.0), (30.0, 30.0)]     # (lambda, recency half-life-like tau in days)
RQ_SCALE = 5.0


def ridge_features(xc, yc, ac, mc, xq):
    """
    Closed-form Bayesian-linear (ridge) predictions for the queries from the SAME context: four variants (two shrinkage
    levels, two recency decays). xc (B,M,F) yc/ac/mc (B,M) xq (B,Q,F) -> (B,Q,4). These are inputs; the network learns corrections.
    """
    age = ac * K
    outs = []
    for lam, tau in RIDGE_VARIANTS:
        w = mc * (1.0 if tau is None else np.exp(-age / tau))
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-9) * mc.sum(1, keepdims=True)
        XtX = np.einsum("bmf,bmg->bfg", xc * w[..., None], xc)
        Xty = np.einsum("bmf,bm->bf", xc * w[..., None], yc)
        beta = np.linalg.solve(XtX + lam * np.eye(xc.shape[2])[None], Xty[..., None])[..., 0]
        outs.append(np.einsum("bqf,bf->bq", xq, beta))
    return (np.stack(outs, -1) * RQ_SCALE).astype(np.float32)



def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def sample_task(rng: np.random.Generator, with_meta: bool = False, r2_high: float = None):
    """One task -> (X (K+1,N,F), y (K+1,N)); y[t] is the next-day relative return of the features at day t."""
    N = int(rng.integers(12, config.Q_MAX))
    T = K + 1
    n_act = int(rng.integers(4, 14))
    act = rng.choice(F, n_act, replace=False)
    rho = rng.uniform(0.0, 0.95, n_act)
    e = rng.normal(size=(T, N, n_act))
    x = np.zeros((T, N, n_act)); x[0] = e[0]
    for t in range(1, T):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
    x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-9)

    mode = rng.choice(["null", "fixed", "regime", "drift"], p=[config.PRIOR_NULL_SHARE, 0.12, 0.43, 0.20])
    r2 = 10 ** rng.uniform(np.log10(config.PRIOR_R2_LOW), np.log10(config.PRIOR_R2_HIGH if r2_high is None else r2_high))
    mag = np.sqrt(r2 / (1 - r2))
    lin = np.zeros((T, N))
    nl_w = 0.0
    if mode != "null":
        def beta():
            k = int(rng.integers(1, min(5, n_act) + 1))
            b = np.zeros(n_act); idx = rng.choice(n_act, k, replace=False); b[idx] = rng.normal(size=k)
            return _unit(b) * mag
        if mode == "fixed":
            B = np.repeat(beta()[None, :], T, axis=0)
        elif mode == "regime":
            R = int(rng.integers(2, 5)); betas = np.stack([beta() for _ in range(R)])
            p_sw = 10 ** rng.uniform(np.log10(0.004), np.log10(0.06))
            s = np.zeros(T, dtype=int); s[0] = rng.integers(R)
            for t in range(1, T):
                s[t] = s[t - 1] if rng.random() > p_sw else rng.integers(R)
            B = betas[s]
        else:
            bA, bB = beta(), beta(); om = rng.uniform(0.01, 0.1); ph = rng.uniform(0, 2 * np.pi)
            ang = ph + om * np.arange(T)
            B = np.cos(ang)[:, None] * bA[None, :] + np.sin(ang)[:, None] * bB[None, :]
        lin = np.einsum("tnf,tf->tn", x, B)
        if rng.random() < 0.2:                       # a nonlinear interaction on top
            i, j = rng.choice(n_act, 2, replace=False)
            g = x[:, :, i] * x[:, :, j]; g = (g - g.mean(axis=1, keepdims=True)) / np.maximum(g.std(axis=1, keepdims=True), 1e-9)
            lin = lin + g * mag * rng.choice([-1, 1]) * 0.7
    nu = rng.choice([3.0, 5.0, 8.0])
    eps = rng.standard_t(nu, size=(T, N)) / np.sqrt(nu / (nu - 2))
    y = np.clip(lin + eps, -8, 8)                    # clip FIRST, then demean (clipping after demeaning shifts each day's mean)
    y = y - y.mean(axis=1, keepdims=True)            # relative to the universe, as in the real target
    X = np.zeros((T, N, F)); X[:, :, act] = x
    meta = {"mode": str(mode), "r2": float(r2) if mode != "null" else 0.0, "N": N}
    return (X, y, meta) if with_meta else (X, y)


def episode_from_panel(X, y, rng, K_days=K, m=M, qmax=QMAX):
    """
    Build one episode from a panel segment. X (K+1,N,F): the last day (index K) is the query day; days 0..K-1 are context
    days whose targets y[t] (the NEXT-day returns) are known at the query day. Returns padded arrays.
    """
    T, N, Fd = X.shape
    ctx_rows = np.arange(K_days * N)
    sel = rng.choice(ctx_rows, size=min(m, len(ctx_rows)), replace=False)
    t_idx, e_idx = sel // N, sel % N
    xc = np.zeros((m, Fd), np.float32); yc = np.zeros(m, np.float32); ac = np.zeros(m, np.float32); mc = np.zeros(m, np.float32)
    n = len(sel)
    xc[:n] = X[t_idx, e_idx]; yc[:n] = y[t_idx, e_idx]; ac[:n] = (K_days - t_idx) / K_days; mc[:n] = 1.0
    xq = np.zeros((qmax, Fd), np.float32); yq = np.zeros(qmax, np.float32); mq = np.zeros(qmax, np.float32)
    xq[:N] = X[K_days]; yq[:N] = y[K_days]; mq[:N] = 1.0
    return xc, yc, ac, mc, xq, yq, mq


def add_ridge(batch):
    """(xc,yc,ac,mc,xq,yq,mq) -> (xc,yc,ac,mc,xq,rq,yq,mq)"""
    xc, yc, ac, mc, xq, yq, mq = batch
    return (xc, yc, ac, mc, xq, ridge_features(xc, yc, ac, mc, xq), yq, mq)


def sample_batch(rng, B=config.BATCH, r2_high=None):
    eps = [episode_from_panel(*sample_task(rng, r2_high=r2_high), rng) for _ in range(B)]
    return add_ridge(tuple(np.stack(a) for a in zip(*eps)))
