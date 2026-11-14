import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import plotly.graph_objects as go

st.set_page_config(page_title="P2 FIN-PFN", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

PRIMARY = "#0f766e"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
VCOL = {"Supported": "#16a34a", "Suggestive": "#d97706", "Not supported": "#dc2626", "Not enough data": "#64748b"}

CSS = """
<style>
    .stApp { background-color: __BG__; }
    #MainMenu, footer {visibility: hidden;}
    .app-title { font-size: 1.9rem; font-weight: 800; color: __INK__; margin: 0; }
    .app-subtitle { color: __SUB__; font-size: 0.95rem; margin-bottom: 1.1rem; }
    .banner { background: #ffffff; border: 1px solid __BD__; border-left: 6px solid var(--accent); border-radius: 10px;
        padding: 0.8rem 1.1rem; font-size: 0.92rem; color: __INK__; margin: 0.4rem 0 0.8rem 0; }
    .pick-card { background: #ffffff; border: 1px solid __BD__; border-radius: 14px; padding: 1rem 1.2rem; text-align: center; }
    .pick-rank { font-size: 0.72rem; color: __SUB__; text-transform: uppercase; letter-spacing: 0.05em; }
    .pick-ticker { font-size: 1.8rem; font-weight: 800; color: __INK__; }
    .pick-sub { font-size: 0.78rem; color: __SUB__; }
    .note { font-size: 0.82rem; color: __SUB__; margin: 0.4rem 0 0.9rem 0; }
    div[data-testid="stDataFrame"] { border: 1px solid __BD__; border-radius: 10px; overflow: hidden; }
</style>
"""
st.markdown(CSS.replace("__BG__", PAGE_BG).replace("__INK__", INK).replace("__SUB__", SUBTLE).replace("__BD__", CARD_BORDER), unsafe_allow_html=True)

RESULTS_REPO = "P2SAMAPA/p2-etf-fin-pfn-results"


def _latest(repo_id):
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        names = sorted(f for f in files if f.startswith("pfn_results_") and f.endswith(".json"))
        return (names[-1], None) if names else (None, "No pfn_results_*.json files found in " + repo_id + ".")
    except Exception as e:
        return None, "Could not list files in " + repo_id + ": " + str(e)


@st.cache_data(show_spinner="Loading results...")
def load_data(_k=""):
    notes = []
    local = glob.glob("pfn_results_*.json")
    if local:
        f = sorted(local)[-1]
        try:
            return json.load(open(f)), ["Loaded local file: " + f]
        except Exception as e:
            notes.append("Could not parse " + f + ": " + str(e))
    name, err = _latest(RESULTS_REPO)
    if err:
        notes.append(err)
    else:
        try:
            h = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
            r = requests.get("https://huggingface.co/datasets/" + RESULTS_REPO + "/resolve/main/" + name, headers=h, timeout=30)
            if r.status_code == 200:
                return r.json(), ["Loaded from HF dataset: " + name]
            notes.append("Fetching " + name + " returned HTTP " + str(r.status_code) + ".")
        except Exception as e:
            notes.append("Fetching " + name + " failed: " + str(e))
    return None, notes


def fnum(x, fmt="%.2f", na="n/a"):
    return na if x is None else fmt % x


def render_universe(name, u):
    v = u["verdict"]
    st.markdown('<div class="banner" style="--accent: ' + VCOL.get(v["verdict"], "#64748b") + ';"><b>' + name.replace("_", " ").title() + ': ' + v["verdict"] + '.</b> '
                + str(v.get("reason", "")) + '</div>', unsafe_allow_html=True)
    picks = u.get("picks", [])
    if picks:
        cols = st.columns(len(picks))
        for i, (c, p) in enumerate(zip(cols, picks)):
            with c:
                st.markdown('<div class="pick-card"><div class="pick-rank">#' + str(i + 1) + '</div><div class="pick-ticker">' + p["etf"] + '</div><div class="pick-sub">risk-adj. score '
                            + fnum(p["score"], "%+.3f") + ' | P(up vs universe) ' + fnum(p["p_up"], "%.2f") + '</div></div>', unsafe_allow_html=True)
    today = pd.DataFrame(u.get("today", []))
    if not today.empty:
        today = today.rename(columns={"etf": "ETF", "mu": "Expected (vol units)", "sd": "Predictive sd", "score": "Risk-adjusted score", "p_up": "P(beats universe)", "roll_ridge_mu": "Rolling ridge mean"})
        with st.expander("All " + str(len(today)) + " ETFs ranked"):
            st.dataframe(today.style.format({"Expected (vol units)": "{:+.4f}", "Predictive sd": "{:.3f}", "Risk-adjusted score": "{:+.4f}", "P(beats universe)": "{:.3f}", "Rolling ridge mean": "{:+.4f}"}, na_rep="n/a"),
                         use_container_width=True, hide_index=True)


def render_evidence(name, u):
    sm = u["summary"]
    rows = [{"Method": k, "IC": x["ic_mean"], "t": x["ic_t"], "Top-3 gross bps/day": x["gross_bps"], "Top-3 net bps/day": x["net_bps"], "Days": x["n_days"]} for k, x in sm["methods"].items()]
    st.markdown("**" + name.replace("_", " ").title() + "** — walk-forward " + u["window"][0] + " to " + u["window"][1])
    st.dataframe(pd.DataFrame(rows).style.format({"IC": "{:+.4f}", "t": "{:+.2f}", "Top-3 gross bps/day": "{:+.1f}", "Top-3 net bps/day": "{:+.1f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)
    pr = [{"FIN-PFN minus": k, "IC difference": x["d_ic"], "paired t": x["t"], "net bps/day difference": x["d_net_bps"]} for k, x in sm["paired"].items()]
    st.dataframe(pd.DataFrame(pr).style.format({"IC difference": "{:+.4f}", "paired t": "{:+.2f}", "net bps/day difference": "{:+.2f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)
    d = sm.get("distribution")
    if d:
        st.caption("Predictive distribution: log-score gain vs a no-skill distribution " + fnum(d["gain"], "%+.4f") + " nats; the 50% interval covered " + fnum(100 * d["coverage_50"], "%.1f")
                   + "% and the 90% interval " + fnum(100 * d["coverage_90"], "%.1f") + "% of outcomes (well calibrated = 50% and 90%).")
    ser = u.get("series", {})
    if ser.get("dates"):
        fig = go.Figure()
        for k, col in (("pfn_ft", PRIMARY), ("roll_ridge", "#94a3b8"), ("stack_ridge", "#6366f1"), ("momentum", "#f59e0b")):
            if k in ser:
                fig.add_trace(go.Scatter(x=ser["dates"], y=ser[k], mode="lines", name=k, line=dict(color=col, width=2 if k == "pfn_ft" else 1.4)))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
                          yaxis=dict(title="Cumulative net top-3 vs universe (%)", gridcolor="#eef2f7"), xaxis=dict(gridcolor="#eef2f7"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True, key="cum_" + name)


def render_findings(data):
    st.markdown('<div class="note"><b>What the network does and does not learn.</b> At CPU-scale pre-training the transformer could not learn mean corrections that beat the in-context '
                'ridge it is fed, so the mean is a learned stack of four in-context ridge predictions (two shrinkage levels, two recency decays) with weights refit on real windows '
                '(stack_ridge); the transformer learns only the predictive scale. In validation its scale beat the best constant scale by 0.0009 nats, and FIN-PFN minus stack_ridge was '
                'zero in every world. The row "pfn_ft minus stack_ridge" in the Evidence tab is the transformer\'s own contribution.</div>', unsafe_allow_html=True)
    v = data.get("validation")
    if not v:
        st.info("No validation_results.json shipped with this run.")
        return
    rows = []
    for w in v["worlds"]:
        pr, st_ = w["paired"].get("roll_ridge", {}), w["paired"].get("stack_ridge", {})
        rows.append({"World": w["world"], "Verdicts": json.dumps(w["verdicts"]), "FIN-PFN IC": w["methods"]["pfn_ft"]["ic_mean"], "Rolling ridge IC": w["methods"]["roll_ridge"]["ic_mean"],
                     "Fixed ridge IC": w["methods"]["fixed_ridge"]["ic_mean"], "LightGBM IC": w["methods"]["lgbm"]["ic_mean"], "PFN - rolling ridge": pr.get("d_ic"), "t": pr.get("mean_t"),
                     "PFN - stack_ridge": st_.get("d_ic"), "50% cov.": w.get("distribution", {}).get("coverage_50"), "90% cov.": w.get("distribution", {}).get("coverage_90")})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:+.4f}" for c in ("FIN-PFN IC", "Rolling ridge IC", "Fixed ridge IC", "LightGBM IC", "PFN - rolling ridge", "PFN - stack_ridge")} | {"t": "{:+.2f}", "50% cov.": "{:.3f}", "90% cov.": "{:.3f}"}, na_rep="n/a"),
                 use_container_width=True, hide_index=True)
    st.caption(str(v.get("seeds_per_world")) + " seeds per world, " + str(v.get("eval_days")) + " evaluation days, through the same walk-forward and verdict code as real data.")


def main():
    if "kb" not in st.session_state:
        st.session_state.kb = 0
    a, b = st.columns([5, 1])
    with a:
        st.markdown('<div class="app-title">🧠 P2 FIN-PFN</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Predict each ETF\'s next-day return distribution conditional on the most recent feature-return pairs — and test whether that beats a ridge refit on the same window.</div>', unsafe_allow_html=True)
    with b:
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear(); st.session_state.kb += 1; st.rerun()
    data, notes = load_data(str(st.session_state.kb))
    if not data:
        st.error("No results found.")
        with st.expander("Why? (click for details)", expanded=True):
            for n in notes or ["No diagnostic information was returned."]:
                st.write("• " + n)
        return
    st.caption("🕒 Results generated: **" + str(data.get("run_date", "Unknown")) + "** | data through " + str(data.get("last_data_date")))
    if data.get("status") != "ok":
        st.warning(str(data.get("message", "The run did not produce a ranking."))); return
    t1, t2, t3 = st.tabs(["🎯 Today", "📊 Evidence", "🔬 What it learns & validation"])
    with t1:
        for name, u in data["universes"].items():
            render_universe(name, u)
    with t2:
        for name, u in data["universes"].items():
            render_evidence(name, u)
    with t3:
        render_findings(data)


if __name__ == "__main__":
    main()
