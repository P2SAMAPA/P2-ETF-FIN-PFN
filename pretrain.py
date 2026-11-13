"""
pretrain.py  —  Stage 1: pre-train the PFN on the synthetic prior (resumable; ships prior_weights.npz).

    python pretrain.py <seconds>      # train for about that long, resuming from ckpt.npz if present

Reports, on a FIXED validation set of prior tasks the network never trains on: the predictive NLL and the rank IC of the
network versus (a) no skill (mean 0) and (b) a ridge regression fit on the SAME context (the best of a small lambda grid,
chosen after seeing the answers, i.e. a generous baseline).
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import spearmanr

import config
import prior
import pfn

CK = "ckpt.npz"


def make_val(n=800, seed=12345):
    rng = np.random.default_rng(seed)
    eps, metas = [], []
    for _ in range(n):
        X, y, m = prior.sample_task(rng, True)
        eps.append(prior.episode_from_panel(X, y, rng)); metas.append(m)
    return prior.add_ridge(tuple(np.stack(a) for a in zip(*eps))), metas


def ridge_mu(xc, yc, mc, xq, lam):
    out = np.zeros((xc.shape[0], xq.shape[1]))
    for i in range(xc.shape[0]):
        m = mc[i] > 0; X, y = xc[i][m], yc[i][m]
        beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
        out[i] = xq[i] @ beta
    return out


def mean_ic(mu, yq, mq, sel=None):
    ics = []
    for i in range(len(mu)):
        if sel is not None and not sel[i]:
            continue
        m = mq[i] > 0
        if m.sum() >= 8 and np.std(mu[i][m]) > 1e-12:
            ics.append(spearmanr(mu[i][m], yq[i][m])[0])
    return (float(np.mean(ics)), len(ics)) if ics else (float("nan"), 0)


def evaluate(p, val, metas):
    xc, yc, ac, mc, xq, rq, yq, mq = val
    mu, ls = pfn.forward_jit(p, *[jnp.asarray(a) for a in (xc, yc, ac, mc, xq, rq, mq)])
    mu, ls = np.asarray(mu), np.asarray(ls)
    nll = np.asarray(pfn.student_t_nll(jnp.asarray(yq), jnp.asarray(mu), jnp.asarray(ls)))
    nll0 = np.asarray(pfn.student_t_nll(jnp.asarray(yq), jnp.zeros_like(jnp.asarray(mu)), jnp.zeros_like(jnp.asarray(ls))))
    w = mq.sum()
    res = {"nll": float((nll * mq).sum() / w), "nll_noskill": float((nll0 * mq).sum() / w)}
    strong = np.array([m["r2"] >= 0.01 for m in metas]); regime = np.array([m["mode"] == "regime" for m in metas]); null = np.array([m["mode"] == "null" for m in metas])
    res["ic_all"], _ = mean_ic(mu, yq, mq)
    res["ic_strong"], res["n_strong"] = mean_ic(mu, yq, mq, strong)
    res["ic_regime"], _ = mean_ic(mu, yq, mq, regime)
    res["ic_null"], _ = mean_ic(mu, yq, mq, null)
    best = None
    for lam in (30, 100, 300, 1000):
        r = ridge_mu(xc, yc, mc, xq, lam); ic, _ = mean_ic(r, yq, mq); icr, _ = mean_ic(r, yq, mq, regime); ics, _ = mean_ic(r, yq, mq, strong)
        if best is None or ic > best[1]:
            best = (lam, ic, icr, ics)
    res.update({"ridge_lam": best[0], "ic_ridge_all": best[1], "ic_ridge_regime": best[2], "ic_ridge_strong": best[3]})
    return res


def fmt(r):
    return ("NLL %.4f vs no-skill %.4f (gain %+.4f) | IC all %+.4f  strong-signal %+.4f  regime %+.4f  null %+.4f || best in-context ridge (lam %d): IC all %+.4f  strong %+.4f  regime %+.4f"
            % (r["nll"], r["nll_noskill"], r["nll_noskill"] - r["nll"], r["ic_all"], r["ic_strong"], r["ic_regime"], r["ic_null"], r["ridge_lam"], r["ic_ridge_all"], r["ic_ridge_strong"], r["ic_ridge_regime"]))


def main(seconds: float, total=config.PRETRAIN_STEPS, eval_every=600):
    if os.path.exists(CK):
        p, extra = pfn.load_params(CK); step = int(extra["step"])
    else:
        p, step = pfn.init_params(0), 0
    opt = pfn._adam_init(p)
    val, metas = make_val()
    rng = np.random.default_rng(1000 + step)
    print("start at step %d | parameters %d" % (step, pfn.n_params(p)), flush=True)
    if step == 0 or "--eval0" in sys.argv:
        print("  eval:", fmt(evaluate(p, val, metas)), flush=True)
    t0 = time.time(); losses = []
    while time.time() - t0 < seconds and step < total:
        frac = min(1.0, step / (0.5 * total))
        r2h = config.CURRICULUM_R2_START * (config.PRIOR_R2_HIGH / config.CURRICULUM_R2_START) ** frac
        b = [jnp.asarray(a) for a in prior.sample_batch(rng, r2_high=r2h)]
        p, opt, l = pfn.train_step(p, opt, pfn.lr_at(step, config.LR, total=total), *b)
        losses.append(float(l)); step += 1
        if step % 200 == 0:
            pfn.save_params(p, CK, {"step": step}); print("  step %d | train NLL (last 200) %.4f | %.0fs" % (step, np.mean(losses[-200:]), time.time() - t0), flush=True)
        if step % eval_every == 0:
            print("  eval @%d:" % step, fmt(evaluate(p, val, metas)), flush=True)
    pfn.save_params(p, CK, {"step": step})
    print("stopped at step %d | eval:" % step, fmt(evaluate(p, val, metas)), flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 240.0)
