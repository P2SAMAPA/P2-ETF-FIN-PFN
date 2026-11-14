"""
pfn.py  —  The prior-data fitted network (JAX).

Tokens: context rows carry (features, next-day return, age); query rows carry features only. A pre-LN transformer
processes [context; query] with an attention mask in which EVERY row may attend to context rows only: context rows
learn from the other context rows, and each query reads the context without seeing any other query. The query
outputs are the parameters (mean, scale) of a Student-t predictive for that ETF's next-day risk-adjusted relative return.
Training is episodic: each episode is a separate task (a synthetic market, or a real historical window).
"""

import io
import numpy as np
import jax
import jax.numpy as jnp
import config

D, H, L, FF, F = config.D_MODEL, config.N_HEADS, config.N_LAYERS, config.D_FF, config.F_MAX


def init_params(seed: int = 0):
    k = jax.random.PRNGKey(seed)
    ks = iter(jax.random.split(k, 6 + 8 * L))
    dense = lambda a, b, s=1.0: jax.random.normal(next(ks), (a, b)) * (s / np.sqrt(a))
    p = {"w_x": dense(F, D), "w_c": dense(3, D), "w_r": dense(4, D), "q_emb": jnp.zeros(D), "layers": [],
         "lnf_g": jnp.ones(D), "lnf_b": jnp.zeros(D), "head": jnp.zeros((D, 2)), "head_b": jnp.zeros(2),
         "a_r": jnp.array([0.1, 0.0, 0.0, 0.0])}      # residual: mean = learned correction + a . (in-context ridge predictions); starts at half-shrunk ridge
    for _ in range(L):
        p["layers"].append({"wq": dense(D, D), "wk": dense(D, D), "wv": dense(D, D), "wo": dense(D, D, 0.5),
                            "ln1_g": jnp.ones(D), "ln1_b": jnp.zeros(D), "w1": dense(D, FF), "b1": jnp.zeros(FF),
                            "w2": dense(FF, D, 0.5), "b2": jnp.zeros(D), "ln2_g": jnp.ones(D), "ln2_b": jnp.zeros(D)})
    return p


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * g + b


def forward(p, xc, yc, ac, mc, xq, rq, mq):
    """xc (B,M,F) yc/ac/mc (B,M) xq (B,Q,F) rq (B,Q,4) in-context ridge predictions, mq (B,Q) -> mu (B,Q), log_s (B,Q)."""
    B, M, _ = xc.shape
    Q = xq.shape[1]
    ctx = xc @ p["w_x"] + jnp.stack([yc, ac, yc * ac], -1) @ p["w_c"]
    qry = xq @ p["w_x"] + rq @ p["w_r"] + p["q_emb"]
    h = jnp.concatenate([ctx, qry], axis=1)                              # (B, M+Q, D)
    key_ok = jnp.concatenate([mc, jnp.zeros((B, Q))], axis=1)            # only valid context rows can be attended to
    bias = jnp.where(key_ok > 0, 0.0, -1e9)[:, None, None, :]            # (B,1,1,M+Q), same for every query row
    dh = D // H
    for lp in p["layers"]:
        x = _ln(h, lp["ln1_g"], lp["ln1_b"])
        q = (x @ lp["wq"]).reshape(B, -1, H, dh).transpose(0, 2, 1, 3)
        k = (x @ lp["wk"]).reshape(B, -1, H, dh).transpose(0, 2, 1, 3)
        v = (x @ lp["wv"]).reshape(B, -1, H, dh).transpose(0, 2, 1, 3)
        att = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh) + bias, axis=-1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, -1, D)
        h = h + o @ lp["wo"]
        x = _ln(h, lp["ln2_g"], lp["ln2_b"])
        h = h + jax.nn.gelu(x @ lp["w1"] + lp["b1"]) @ lp["w2"] + lp["b2"]
    out = _ln(h[:, M:], p["lnf_g"], p["lnf_b"]) @ p["head"] + p["head_b"]
    # the stacking weights a_r are fit in closed form (fit_stacking), never by SGD: with this little signal their gradients are
    # almost pure noise and Adam random-walks them (they were being destroyed within ~900 steps)
    return out[..., 0] * config.MU_CORRECTION + rq @ jax.lax.stop_gradient(p["a_r"]), jnp.clip(out[..., 1], -3.0, 3.0)


def student_t_nll(y, mu, log_s, nu=config.NU):
    z = (y - mu) / jnp.exp(log_s)
    return log_s + 0.5 * (nu + 1.0) * jnp.log1p(z ** 2 / nu)


def loss_fn(p, xc, yc, ac, mc, xq, rq, yq, mq):
    mu, log_s = forward(p, xc, yc, ac, mc, xq, rq, mq)
    nll = student_t_nll(yq, mu, log_s)
    return (nll * mq).sum() / jnp.maximum(mq.sum(), 1.0)


def fit_stacking(batches, ridge=1e-6):
    """Closed-form least squares of the target on the 4 in-context ridge predictions, pooled over many episodes: the learned
    shrinkage / recency weights. batches: iterable of (xc,yc,ac,mc,xq,rq,yq,mq)."""
    A = np.zeros((4, 4)); b = np.zeros(4)
    for (xc, yc, ac, mc, xq, rq, yq, mq) in batches:
        m = mq > 0
        r, y = rq[m], yq[m]
        A += r.T @ r; b += r.T @ y
    return np.linalg.solve(A + ridge * np.eye(4), b).astype(np.float32)


def _adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, p), "t": jnp.zeros(())}


@jax.jit
def train_step(p, opt, lr, xc, yc, ac, mc, xq, rq, yq, mq):
    loss, g = jax.value_and_grad(loss_fn)(p, xc, yc, ac, mc, xq, rq, yq, mq)
    gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, config.GRAD_CLIP / (gn + 1e-9)), g)
    t = opt["t"] + 1.0
    m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, opt["m"], g)
    v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ ** 2, opt["v"], g)
    upd = jax.tree_util.tree_map(lambda m_, v_: (m_ / (1 - 0.9 ** t)) / (jnp.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8), m, v)
    p = jax.tree_util.tree_map(lambda p_, u_: p_ - lr * u_, p, upd)
    return p, {"m": m, "v": v, "t": t}, loss


forward_jit = jax.jit(forward)


def lr_at(step, base, warm=config.WARMUP, total=config.PRETRAIN_STEPS):
    w = min(1.0, (step + 1) / warm)
    c = 0.5 * (1 + np.cos(np.pi * min(1.0, step / max(total, 1))))
    return base * w * (0.1 + 0.9 * c)


# ---- persistence -----------------------------------------------------------
def _flatten(p, prefix=""):
    out = {}
    for k, v in p.items():
        if isinstance(v, list):
            for i, d in enumerate(v):
                out.update(_flatten(d, "%s%s.%d." % (prefix, k, i)))
        else:
            out[prefix + k] = np.asarray(v)
    return out


def save_params(p, path, extra=None):
    d = _flatten(p)
    for k, v in (extra or {}).items():
        d["__" + k] = np.asarray(v)
    np.savez(path, **d)


def load_params(path):
    z = np.load(path)
    p = {"layers": [dict() for _ in range(L)]}
    extra = {}
    for k in z.files:
        if k.startswith("__"):
            extra[k[2:]] = z[k]
        elif k.startswith("layers."):
            _, i, name = k.split(".", 2)
            p["layers"][int(i)][name] = jnp.asarray(z[k])
        else:
            p[k] = jnp.asarray(z[k])
    return p, extra


def n_params(p):
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


CODE_VERSION = config.CODE_VERSION
