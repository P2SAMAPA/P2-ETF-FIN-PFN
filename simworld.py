"""
simworld.py  —  Synthetic ETF markets with a KNOWN feature-return relationship, for validating the whole pipeline.

Panels have the same layout as the real ones (features 0-8 ETF-specific, standardized across ETFs each day; 9-12 slow
macro-common shocks), so the SAME evaluation code runs on them. Worlds:
  null          next-day relative return unrelated to the features
  fixed         one constant linear relationship
  regime        the linear relationship switches between latent regimes every ~90 days (no label is ever shown)
  interaction   a fixed NONLINEAR relationship: the sign of the effect depends on the macro state today
The target is the relative return in vol units; raw next-day returns are 1% x (relative + a common market component).
"""

import numpy as np
import pandas as pd
from typing import Dict

import config

F, NF = config.F_MAX, config.N_REAL_FEATURES


def make_world(kind: str, n: int = 900, E: int = 20, r2: float = 0.01, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.0, 0.9, NF)
    raw = np.zeros((n, E, NF))
    e = rng.normal(size=(n, E, NF))
    mac_e = rng.normal(size=(n, 4))
    raw[0] = e[0]
    mac = np.zeros((n, 4)); mac[0] = mac_e[0]
    for t in range(1, n):
        raw[t] = rho * raw[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
        mac[t] = 0.97 * mac[t - 1] + np.sqrt(1 - 0.97 ** 2) * mac_e[t]          # slow macro state
    X = np.zeros((n, E, F))
    for j in range(9):
        c = raw[:, :, j]; X[:, :, j] = (c - c.mean(axis=1, keepdims=True)) / np.maximum(c.std(axis=1, keepdims=True), 1e-9)
    X[:, :, 9:13] = mac[:, None, :]
    X = np.clip(X, -5, 5)
    mag = np.sqrt(r2 / (1 - r2))
    sig = np.zeros((n, E))
    if kind == "fixed":
        b = np.zeros(9); b[[0, 2, 5]] = [1.0, -0.7, 0.5]; b = b / np.linalg.norm(b) * mag
        sig = X[:, :, :9] @ b
    elif kind == "regime":
        betas = []
        for _ in range(3):
            b = np.zeros(9); idx = rng.choice(9, 3, replace=False); b[idx] = rng.normal(size=3); betas.append(b / np.linalg.norm(b) * mag)
        s = np.zeros(n, dtype=int); s[0] = rng.integers(3)
        for t in range(1, n):
            s[t] = s[t - 1] if rng.random() > 1 / 90 else rng.integers(3)
        sig = np.einsum("tef,tf->te", X[:, :, :9], np.stack(betas)[s])
    elif kind == "interaction":
        b = np.zeros(9); b[[1, 3]] = [1.0, 0.8]; b = b / np.linalg.norm(b) * mag * 1.4
        gate = np.tanh(2.0 * mac[:, 1])[:, None]                                   # sign of the effect follows the macro state
        sig = gate * (X[:, :, :9] @ b)
    noise = rng.standard_t(5, size=(n, E)) / np.sqrt(5 / 3)
    z_next = np.clip(sig + noise, -8, 8)                                           # row t: return from t to t+1 in vol units
    common = rng.normal(0, 1.2, n)[:, None]
    ret_next = (z_next - z_next.mean(axis=1, keepdims=True) + z_next.mean(axis=1, keepdims=True) * 0 + common) * 0.01
    y = z_next - z_next.mean(axis=1, keepdims=True)
    y[-1] = np.nan; ret_next[-1] = np.nan
    dates = pd.bdate_range("2019-01-02", periods=n)
    return {"name": "SIM_" + kind, "dates": dates, "etfs": ["E%02d" % i for i in range(E)], "X": X.astype(np.float32),
            "valid": np.ones((n, E), bool), "y": y, "ret_next": ret_next, "have_macro": True}
