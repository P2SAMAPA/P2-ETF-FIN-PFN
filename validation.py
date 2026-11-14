"""
validation.py  —  Does the pipeline find what is there, stay quiet when nothing is, and does the network beat a plain ridge
refit on the same recent window?

Synthetic ETF markets with a KNOWN relationship (simworld.py), several seeds each, through the SAME walk-forward and verdict code
as real data. Worlds: null / fixed / regime (latent switching) / interaction (nonlinear, macro-gated).

    python validation.py <seeds> <world 0-3>     # one world per call; saves validation_world_<i>.json
    python validation.py <seeds> merge            # -> validation_results.json
"""

import sys
import json
import time
import numpy as np

import config
import pfn
import simworld as S
import evaluate as E

WORLDS = ["null", "fixed", "regime", "interaction"]
EVAL_DAYS, REFIT, FT_STEPS, ENS = 400, 200, 60, 4


def one_run(kind, seed, params):
    P = S.make_world(kind, n=1000, seed=seed)
    r = E.run_walk_forward([P], params, finetune_steps=FT_STEPS, refit_days=REFIT, eval_days=EVAL_DAYS, fit_lgbm=True, R=ENS)
    sm = E.summarize(P, r[P["name"]])
    return {"seed": seed, "methods": sm["methods"], "paired": sm["paired"], "distribution": sm.get("distribution"), "verdict": E.verdict(sm)["verdict"]}


def run_world(i, seeds):
    params, _ = pfn.load_params("prior_weights.npz")
    kind = WORLDS[i]
    runs = [one_run(kind, 100 + s, params) for s in range(seeds)]
    w = {"world": kind, "n_runs": seeds, "verdicts": {}, "methods": {}, "paired": {}}
    for r in runs:
        w["verdicts"][r["verdict"]] = w["verdicts"].get(r["verdict"], 0) + 1
    for k in runs[0]["methods"]:
        w["methods"][k] = {m: float(np.mean([r["methods"][k][m] for r in runs if k in r["methods"]])) for m in ("ic_mean", "ic_t", "net_bps")}
    for k in runs[0]["paired"]:
        ts = [r["paired"][k]["t"] for r in runs]
        w["paired"][k] = {"d_ic": float(np.mean([r["paired"][k]["d_ic"] for r in runs])), "mean_t": float(np.mean(ts)),
                          "share_t_ge_1": float(np.mean([t >= 1.0 for t in ts])), "share_t_ge_165": float(np.mean([t >= 1.65 for t in ts]))}
    d = [r["distribution"] for r in runs if r.get("distribution")]
    if d:
        w["distribution"] = {k: float(np.mean([x[k] for x in d])) for k in ("gain", "coverage_50", "coverage_90")}
    return w


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    if len(sys.argv) > 2 and sys.argv[2] == "merge":
        out = {"seeds_per_world": seeds, "eval_days": EVAL_DAYS, "refit_days": REFIT, "finetune_steps": FT_STEPS, "ensemble": ENS,
               "worlds": [json.load(open("validation_world_%d.json" % i)) for i in range(len(WORLDS))]}
        json.dump(out, open("validation_results.json", "w"), indent=1); print("merged -> validation_results.json")
    else:
        i = int(sys.argv[2]); t0 = time.time()
        w = run_world(i, seeds); w["seconds"] = round(time.time() - t0)
        json.dump(w, open("validation_world_%d.json" % i, "w"), indent=1)
        print(w["world"], w["verdicts"], "in %ds" % w["seconds"], flush=True)
        for k, v in w["methods"].items():
            print("   %-12s IC %+.4f (mean t %+.2f) | top-3 net %+.1f bps/day" % (k, v["ic_mean"], v["ic_t"], v["net_bps"]))
        for k, v in w["paired"].items():
            print("   pfn_ft vs %-11s dIC %+.4f | mean t %+.2f | t>=1: %.0f%% | t>=1.65: %.0f%%" % (k, v["d_ic"], v["mean_t"], 100 * v["share_t_ge_1"], 100 * v["share_t_ge_165"]))
        if "distribution" in w: print("   distribution: log-score gain vs no-skill %+.4f | 50%% interval coverage %.3f | 90%% interval coverage %.3f" % (w["distribution"]["gain"], w["distribution"]["coverage_50"], w["distribution"]["coverage_90"]))
