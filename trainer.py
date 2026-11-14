"""
trainer.py  —  P2 FIN-PFN

Each run:
  1. load prices and macro from the master dataset; build one panel per universe (13 features, relative vol-scaled target);
  2. load the PRIOR-pretrained network (prior_weights.npz, shipped with the repo);
  3. walk-forward over the last EVAL_YEARS: before every refit date, meta-train on real historical windows (closed-form stacking
     weights refit + transformer scale trained), then predict each day IN CONTEXT; score against a rolling ridge on the same
     window, the network's mean alone (stack_ridge), a fixed ridge, LightGBM and momentum on identical rows;
  4. fit once more on all data and rank today's ETFs by expected risk-adjusted return (mean / predictive sd).
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import features as FE
import evaluate as E
import pfn

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

CODE_VERSION = config.CODE_VERSION
HERE = os.path.dirname(os.path.abspath(__file__))


def check_versions():
    stale = [m.__name__ + ".py" for m in (config, FE, E, pfn) if getattr(m, "CODE_VERSION", None) != CODE_VERSION]
    if stale:
        raise RuntimeError("Files from different releases: " + ", ".join(stale) + " are not version " + CODE_VERSION + ". Replace ALL files with the latest zip.")


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d):
    if isinstance(d, dict):
        return {str(k): _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return _f(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d


def run_pipeline(prices_df: pd.DataFrame, macro_df, params_prior, fast: bool = False) -> Dict:
    panels = []
    for name, tickers in config.UNIVERSES.items():
        p = FE.build_panel(prices_df, macro_df, tickers, name)
        if p is not None:
            panels.append(p)
    if not panels:
        return {"status": "no_data", "message": "No universe has enough ETFs."}
    n = len(panels[0]["dates"])
    logger.info("Panels: " + ", ".join("%s (%d ETFs, macro=%s)" % (p["name"], len(p["etfs"]), p["have_macro"]) for p in panels))
    logger.info("🧠 Walk-forward (meta-train on real windows before each refit, predict in context)...")
    res = E.run_walk_forward(panels, params_prior, fit_lgbm=True, log=logger.info)
    last = n - 1
    logger.info("🔮 Final fit on all history and today's ranking...")
    params_final = E.finetune(params_prior, panels, {p["name"]: last - 1 for p in panels}, config.FINETUNE_STEPS, seed=last)
    out = {"status": "ok", "last_data_date": str(panels[0]["dates"][-1].date()), "universes": {},
           "weights": {"prior_stacking": [float(x) for x in np.asarray(params_prior["a_r"])], "final_stacking": [float(x) for x in np.asarray(params_final["a_r"])]}}
    for p in panels:
        r = res[p["name"]]
        sm = E.summarize(p, r)
        v = E.verdict(sm)
        o = E.pfn_predict(params_final, p, [last], R=max(config.ENSEMBLE_R, 8), seed=last)
        roll = E.rolling_ridge(p, [last])
        qi = np.where(p["valid"][last])[0]
        today = []
        for j in qi:
            today.append({"etf": p["etfs"][j], "mu": _f(o["mu"][last, j]), "sd": _f(o["sd"][last, j]), "score": _f(o["mu"][last, j] / o["sd"][last, j]),
                          "p_up": _f(o["p_up"][last, j]), "roll_ridge_mu": _f(roll[last, j])})
        today.sort(key=lambda x: -(x["score"] if x["score"] is not None else -9))
        # cumulative net top-3 (decimated) over the evaluation window
        days = r["days"]
        ser = {"dates": [str(p["dates"][d].date()) for d in days[::config.SERIES_STEP]]}
        for k in ("pfn_ft", "roll_ridge", "stack_ridge", "momentum"):
            sc = r["scores"].get(k)
            if sc is None:
                continue
            _, nt = E.topn_series(sc, p["ret_next"])
            ser[k] = [float(x) * 100 for x in np.cumsum(np.nan_to_num(nt[days]))[::config.SERIES_STEP]]
        out["universes"][p["name"]] = {"n_etfs": len(p["etfs"]), "have_macro": p["have_macro"], "window": [str(p["dates"][r["window"][0]].date()), str(p["dates"][r["window"][1] - 1].date())],
                                       "summary": sm, "verdict": v, "picks": today[:config.TOP_N], "today": today, "series": ser}
        logger.info("  %s: IC %.4f (t %.2f) | vs rolling ridge dIC %.4f (t %.2f) | verdict %s" % (p["name"], sm["methods"]["pfn_ft"]["ic_mean"], sm["methods"]["pfn_ft"]["ic_t"],
                    sm["paired"]["roll_ridge"]["d_ic"], sm["paired"]["roll_ridge"]["t"], v["verdict"]))
    return _clean(out)


def run_trainer(prices_loader=None, push: bool = True) -> Dict:
    check_versions()
    logger.info("🔄 Loading data...")
    try:
        if prices_loader is None:
            from data_manager import load_master_data, validate_data
            prices_df, macro_df = load_master_data()
            validate_data(prices_df, macro_df)
        else:
            prices_df, macro_df = prices_loader()
    except Exception as e:
        import traceback
        logger.error(f"Failed to load data: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return {}
    wpath = os.path.join(HERE, config.WEIGHTS_FILE)
    if not os.path.exists(wpath):
        logger.error(f"{config.WEIGHTS_FILE} not found next to trainer.py — it ships with the repo (run pretrain.py to regenerate it).")
        return {}
    params_prior, extra = pfn.load_params(wpath)
    results = run_pipeline(prices_df, macro_df, params_prior)
    results.update({"run_date": datetime.now().strftime("%Y-%m-%d"), "algorithm": "FIN-PFN", "pretrain_steps": int(extra.get("step", 0)),
                    "settings": {"eval_years": config.EVAL_YEARS, "refit_days": config.REFIT_DAYS, "ctx_days": config.CTX_DAYS, "ensemble": config.ENSEMBLE_R,
                                 "finetune_steps": config.FINETUNE_STEPS, "cost_bps": config.TRADING_COST_BPS, "mu_correction": config.MU_CORRECTION}})
    vpath = os.path.join(HERE, "validation_results.json")
    if os.path.exists(vpath):
        try:
            results["validation"] = json.load(open(vpath))
        except Exception:
            pass
    out = f"pfn_results_{results['run_date']}.json"
    with open(out, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"💾 Saved: {out}")
    if push:
        try:
            from push_results import upload_results
            upload_results(out, hf_token=config.HF_TOKEN)
        except Exception as e:
            logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    out = run_trainer()
    if not out:
        logger.error("Run produced no results.")
        sys.exit(1)
