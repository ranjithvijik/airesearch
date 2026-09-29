"""Prediction-PDF enhancement: plots + tables derived from the Sept 2026 pipeline run.

Reads only committed pipeline outputs in tables//figures/ (no refit, no new
parameters) and writes prediction-specific exhibits:
  figures/pred_*.png + .pdf  (Okabe-Ito, DejaVu Sans, 300 DPI)
  tables/pred_*.csv/.tex/.xlsx (+ pred_tables_index.csv registry)

Run: python3 build_prediction_enhancement.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox as _Bbox, ScaledTranslation as _ST

ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(ROOT, "tables")
FIGS = os.path.join(ROOT, "figures")

# Single house style for every pred_* exhibit (pipeline figures already use
# DejaVu Sans + Okabe-Ito, so the whole report speaks one visual language):
# titles 10 bold, axis labels 9, ticks/legend 8, data annotations 8,
# source footnotes 6 grey. No per-plot font overrides below this block.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "normal",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.title_fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 300, "savefig.dpi": 300,
    "figure.titlesize": 10, "figure.titleweight": "bold",
})
FOOT_SIZE = 6
# Okabe-Ito palette
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#000000"]
ARCH = ["Hardware", "Cloud Providers", "Foundation Models", "LLM Wrappers"]
ARCH_C = {"Hardware": OI[0], "Cloud Providers": OI[1],
          "Foundation Models": OI[2], "LLM Wrappers": OI[3]}

FOOT = ("Source: Sept 2026 pipeline run (tables/table_7.1x.csv, enhanced_valuation_metrics.csv). "
        "Ranks are the finding.")

def savefig(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, stem + "." + ext),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/{stem}.png/.pdf")

def save_table(df, stem):
    df.to_csv(os.path.join(TABLES, stem + ".csv"), index=False)
    with open(os.path.join(TABLES, stem + ".tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))
    try:
        df.to_excel(os.path.join(TABLES, stem + ".xlsx"), index=False)
    except Exception as e:
        print(f"  xlsx skipped for {stem}: {e}")
    print(f"  wrote tables/{stem}.csv/.tex/.xlsx")

def _assert_no_overlap(fig, texts, stem):
    """Fail loudly if any two annotation bboxes intersect (rendered)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    bad = [(a, b) for a in range(len(boxes)) for b in range(a + 1, len(boxes))
           if boxes[a].overlaps(boxes[b])]
    assert not bad, f"{stem}: {len(bad)} overlapping text pairs: {bad}"

# Quarter labels for prediction-block horizons.
def q_label(nq, vintage_y=2026, vintage_q=4):
    """Quarters elapsed from a run vintage as Qn'yy (e.g. 10.0 -> Q2'29)."""
    q = (vintage_q - 1) + int(round(float(nq)))
    return f"Q{q % 4 + 1}'{(vintage_y + q // 4) % 100:02d}"


# ---- Price gauges: convexity + crash hazard (GitHub-surveyed, scipy only) ----
# Two cheap, transparent price lenses triangulated against the formation and
# burst clocks. (a) Convexity: OLS log-price ~ t + t^2 per trailing window;
# the t-stat of the quadratic term reads super-exponential (bubble-like)
# acceleration without LPPL's fragile log-periodic fit (evaluated and
# rejected on this data: degenerate optima, no tc convergence). (b) Hazard:
# logistic map of standardized drawdown depth/duration, distance from the
# 52w high, and 13w realized vol, times a trailing-vol-tercile regime
# multiplier -- the regime-aware-hazard pattern, with stated (not fitted)
# weights, so the crash-capture backtest below is descriptive, not overfit.
CONVEX_WINDOWS = [26, 52, 78]
CONVEX_FIRE_T = 2.0
HAZ_WEIGHTS = {"dd": 1.0, "dur": 0.6, "dist": 0.8, "vol": 0.9}
HAZ_REGIME_MULT = {"High": 1.2, "Mid": 1.0, "Low": 0.85}
# Standardization floors: near-degenerate histories (a straight-up year)
# must not manufacture hazard out of microscopic dips.
HAZ_SD_FLOORS = {"dd": 2.0, "dur": 4.0, "dist": 2.0, "vol": 2.0}


def convex_tstat(y):
    """One-sided t-stat of the quadratic term in log-price ~ t + t^2."""
    y = np.asarray(y, dtype=float)
    if y.size and float(np.var(y)) == 0.0:
        return 0.0  # constant history: no curvature to measure, not a fire
    L = len(y)
    t = np.arange(L, dtype=float)
    X = np.column_stack([np.ones(L), t, t ** 2])
    beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    dof = L - 3
    s2 = float(res[0]) / dof if len(res) and dof > 0 else 0.0
    if s2 <= 0:
        return 0.0
    se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2]))
    return float(beta[2] / se) if se > 0 else 0.0


def price_features(px):
    """Weekly factor frame: 52w drawdown depth/duration, distance from high, 13w vol."""
    px = np.asarray(px, dtype=float)
    n = len(px)
    s = pd.Series(px)
    hi52 = s.rolling(52, min_periods=52).max().to_numpy()
    dd = np.where(hi52 > 0, (hi52 - px) / hi52 * 100.0, 0.0)
    runmax = s.rolling(52, min_periods=1).max().to_numpy()
    is_high = (px >= runmax - 1e-12)
    dur = np.zeros(n)
    for i in range(1, n):
        dur[i] = 0 if is_high[i] else dur[i - 1] + 1
    lr = np.zeros(n)
    lr[1:] = np.diff(np.log(np.maximum(px, 1e-12)))
    vol = pd.Series(lr).rolling(13, min_periods=13).std().to_numpy() * np.sqrt(52) * 100.0
    return pd.DataFrame({"dd": dd, "dur": dur, "dist": dd, "vol": vol}).fillna(0.0)


def hazard_frame(feats):
    """Hazard 0-100, regime, and top factor per week (vectorized, causal)."""
    mu = feats.rolling(260, min_periods=52).mean().shift(1)
    sd = feats.rolling(260, min_periods=52).std().shift(1)
    floors = pd.Series(HAZ_SD_FLOORS)
    sd = pd.DataFrame(np.maximum(sd.to_numpy(), floors.to_numpy()),
                      index=sd.index, columns=sd.columns)
    z = (feats - mu) / sd
    lin = sum(HAZ_WEIGHTS[k] * z[k] for k in HAZ_WEIGHTS)
    base = 100.0 / (1.0 + np.exp(-lin))
    qlo = feats["vol"].rolling(260, min_periods=52).quantile(1 / 3).shift(1)
    qhi = feats["vol"].rolling(260, min_periods=52).quantile(2 / 3).shift(1)
    regime = pd.Series(np.where(feats["vol"] >= qhi, "High",
                                np.where(feats["vol"] <= qlo, "Low", "Mid")),
                       index=feats.index)
    mult = regime.map(HAZ_REGIME_MULT)
    score = (base * mult).clip(0.0, 100.0)
    contrib = pd.DataFrame({k: HAZ_WEIGHTS[k] * z[k] for k in HAZ_WEIGHTS})
    # Unscored early weeks have no contributions; -inf keeps idxmax total
    # (those rows are never read -- only scored positions are consumed).
    top = contrib.fillna(-np.inf).idxmax(axis=1)
    return score, regime, top, z


# ---- ML horse-race (RSI evaluation): L2 logistic crash model vs stated weights ----
ML_FEATS = ["dd", "dur", "vol"]  # dist dropped: exactly collinear with dd (r = 1.00)
ML_C = 1.0  # stated regularization strength, in the stated-not-fitted convention
ML_CUTOFFS = {"wf": "2007-01-01", "late": "2016-01-01"}


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def fit_crash_logit(Ztr, ytr, C=ML_C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(Ztr, ytr)
    return clf


def era_scores(y, scores, era, q=0.80):
    """Capture share + AUC of scores over era crash weeks (NaN-safe)."""
    from sklearn.metrics import roc_auc_score
    m = era & (y == 1) & np.isfinite(scores)
    ref = era & np.isfinite(scores)
    n = int(m.sum())
    if n == 0:
        return {"n_crash": 0, "capture": float("nan"),
                "lift": float("nan"), "auc": float("nan")}
    thr = float(np.quantile(scores[ref], q))
    cap = float((scores[m] >= thr).mean())
    return {"n_crash": n, "capture": cap, "lift": cap / (1 - q),
            "auc": float(roc_auc_score(y[ref], scores[ref]))}


def crash_capture(px, scores, thresh=0.20):
    """Share of >=20%-drawdown weeks scoring in the top quintile (lift vs 0.20)."""
    px = np.asarray(px, dtype=float)
    peak = np.maximum.accumulate(px)
    dd = np.where(peak > 0, (peak - px) / peak, 0.0)
    ok = np.isfinite(scores.to_numpy())
    crash = (dd >= thresh) & ok
    if not crash.any() or not ok.any():
        return {"capture": 0.0, "lift": 0.0, "threshold": float("nan"),
                "crash_weeks": 0, "weeks": int(ok.sum())}
    thr = float(np.quantile(scores.to_numpy()[ok], 0.80))
    cap = float((scores.to_numpy()[crash] >= thr).mean())
    return {"capture": cap, "lift": cap / 0.20, "threshold": thr,
            "crash_weeks": int(crash.sum()), "weeks": int(ok.sum())}

def main():
    form = pd.read_csv(os.path.join(TABLES, "table_7.10_bubble_formation.csv"))
    burst = pd.read_csv(os.path.join(TABLES, "table_7.11_burst_ranking.csv"))
    gdp = pd.read_csv(os.path.join(TABLES, "table_7.12_gdp_impact.csv"))
    deb = pd.read_csv(os.path.join(TABLES, "table_7.14_debtrank_contagion.csv"))
    cred = pd.read_csv(os.path.join(TABLES, "table_7.15_credit_monitor.csv"))
    geo = pd.read_csv(os.path.join(TABLES, "table_7.19_geography_overlay.csv"))
    val = pd.read_csv(os.path.join(TABLES, "enhanced_valuation_metrics.csv"))
    synergy = pd.read_csv(os.path.join(TABLES, "table_7.6_policy_synergy.csv"))
    os.makedirs(FIGS, exist_ok=True)

    order = ["Foundation Models", "Hardware", "Cloud Providers", "LLM Wrappers"]
    f2 = form.set_index("Archetype").loc[order].reset_index()

    # ---- F1: formation scores with robustness bands ----
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(order))
    vals = f2["Formation_Prob_Pct"].values
    lo = vals - f2["Prob_Low_Pct"].values
    hi = f2["Prob_High_Pct"].values - vals
    ax.bar(x, vals, yerr=[lo, hi], capsize=4, color=[ARCH_C[a] for a in order],
           edgecolor="black", linewidth=0.6, error_kw={"ecolor": "black", "lw": 1.2})
    for i, v in enumerate(vals):
        ax.text(i, v + hi[i] + 1.2, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, order, rotation=0, ha="center")
    ax.set_ylabel("Bubble-condition score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Bubble-condition scores with within-model bands — labs densest, apps lightest")
    fig.text(0.01, 0.0, FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_formation_bands")

    # ---- F2: burst losses mild vs severe ----
    b2 = burst.set_index("Archetype").loc[["Hardware", "Cloud Providers",
                                           "Foundation Models", "LLM Wrappers"]].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(b2))
    ax.bar(x - 0.2, b2["Total_Loss_Severe_$B"], 0.4, label="Severe (70% impair)",
           color=[ARCH_C[a] for a in b2["Archetype"]], edgecolor="black", linewidth=0.6)
    ax.bar(x + 0.2, b2["Total_Loss_Mild_$B"], 0.4, label="Mild (30% impair)",
           color=[ARCH_C[a] for a in b2["Archetype"]], edgecolor="black",
           linewidth=0.6, alpha=0.35, hatch="//")
    for i, v in enumerate(b2["Total_Loss_Severe_$B"]):
        ax.text(i - 0.2, v + 4, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
    for i, v in enumerate(b2["Total_Loss_Mild_$B"]):
        ax.text(i + 0.2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7,
                color="0.25")
    ax.set_xticks(x, b2["Archetype"], rotation=0, ha="center")
    ax.set_ylabel("Total loss ($B)")
    ax.set_title("(b) Burst incidence: reverse order of the hype — Hardware, Cloud, labs, apps")
    ax.legend(frameon=True, fontsize=8)
    fig.text(0.01, 0.0, FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_burst_waterfall")

    # ---- F3: GDP transmission stacked ----
    sev = gdp[gdp["Severity"] == "severe"]; mild = gdp[gdp["Severity"] == "mild"]
    def ch(df, s):
        c = float(df[(df["Channel"] == "AI capex cut") & (df["Severity"] == s)]["GDP_Loss_$B"].iloc[0])
        w = float(df[(df["Channel"] == "Equity wealth effect") & (df["Severity"] == s)]["GDP_Loss_$B"].iloc[0])
        return c, w
    mc, mw = ch(gdp, "mild"); sc, sw = ch(gdp, "severe")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(["Mild\n$48.2B (0.157%)", "Severe\n$139.1B (0.452%)"], [mc, sc], 0.55,
           label="Canceled capex", color=OI[0], edgecolor="black")
    ax.bar(["Mild\n$48.2B (0.157%)", "Severe\n$139.1B (0.452%)"], [mw, sw], 0.55,
           bottom=[mc, sc], label="Wealth effect (4c/$)", color=OI[5], edgecolor="black")
    for i, (lbl, c, w) in enumerate((("mild", mc, mw), ("severe", sc, sw))):
        ax.text(i, c / 2, f"${c:.0f}B", ha="center", fontsize=9,
                fontweight="bold", color="white")
        ax.text(i, c + w / 2, f"${w:.0f}B", ha="center", fontsize=8,
                fontweight="bold", color="white")
    ax.set_ylabel("GDP cost ($B)")
    ax.set_title("(c) GDP transmission: capex dominates, "
                 "wealth adds ~\\$4B per \\$100B destroyed")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.0, "US GDP $30,767.1B (BEA NIPA 2025); no multiplier. " + FOOT,
             fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_gdp_waterfall")

    # ---- F4: scenario ladder ----
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cats = ["Mild\n(frame 675)", "Severe base\n(frame 675)", "Activation\n(frame ~735)"]
    frames = [675.40, 675.40, 735.0]
    mild_direct = float(burst["Total_Loss_Mild_$B"].sum())
    sev_direct = 478.3
    losses = [mild_direct, sev_direct, sev_direct * 735.0 / 675.40]  # activation = severe x735/675, scaled
    x = np.arange(3)
    b1 = ax.bar(x, frames, 0.45, label="Booked frame ($B)", color="0.75", edgecolor="black")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.0, losses, 0.18, label="Direct burst loss ($B)", color=OI[1], edgecolor="black")
    ax.set_xticks(x, cats); ax.set_ylabel("Committed frame ($B)"); ax2.set_ylabel("Direct loss ($B)")
    ax.legend(handles=[b1[0], b2[0]],
              labels=["Booked frame ($B, left axis)",
                      "Direct burst loss ($B, right axis)"],
              fontsize=8, loc="upper left")
    ax.set_title("(d) Scenario ladder: activation (+~9%) scales every burst number up")
    ax.text(2, 735 + 12, "+$60B cond.", ha="center", fontsize=8, style="italic")
    fig.text(0.01, 0.0, FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_scenario_ladder")

    # ---- F5: valuation scatter P/E vs circular dependency ----
    v = val.copy()
    v["pe"] = pd.to_numeric(v["pe_ratio"], errors="coerce")
    v["dep"] = pd.to_numeric(v["circular_revenue_dependency"], errors="coerce")
    v = v[np.isfinite(v["pe"]) & np.isfinite(v["dep"]) & (v["pe"] > 0) & (v["pe"] < 1000)]
    flag_c = {"Positive": OI[2], "Negative": OI[1], "Neutral": "0.55"}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for fl, sub in v.groupby("screen_flag"):
        ax.scatter(sub["dep"], sub["pe"], s=np.sqrt(sub["market_cap"]) * 4,
                   c=flag_c.get(fl, "0.55"), label=fl, alpha=0.8, edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Circular-revenue dependency (score)")
    ax.set_ylabel("Trailing P/E (log scale)")
    ax.set_yscale("log")
    ax.axhline(35, color="red", ls="--", lw=1, alpha=0.7)
    # Shade the danger zone (stretched multiple AND dense round-tripping) so
    # the screen's verdict reads without relying on crowded labels.
    _x0, _x1 = ax.get_xlim()
    ax.axvspan(1.5, _x1, color="red", alpha=0.06, zorder=0)
    # Zone tag first, right-aligned inside the axes, so labels route around it.
    zone = ax.text(v["dep"].max(), 37, "stretched multiple zone", fontsize=7,
                   color="red", ha="right", va="bottom",
                   bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_title("(e) Valuation screen: uncovered multiples meet dense round-tripping at top-right risk")
    ax.legend(title="Screen", fontsize=8)
    # Greedy de-collision for the top-8 labels: fixed candidate offsets (pt),
    # data order fixed by market cap, renderer-measured bboxes, thin leaders.
    candidates = [(7, 7), (7, -15), (-12, 7), (-12, -15), (16, -2),
                  (-42, -2), (7, 14), (-12, -22), (22, 10), (-52, 10),
                  (-72, -30), (30, -28), (-72, 18), (34, 22), (-20, 28),
                  (-20, -36), (50, -14), (-72, -8), (48, 16), (-30, 30)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [zone.get_window_extent(renderer)]
    _pad = 4  # pt margin inside the axes for label boxes
    inset = _Bbox([[ax.bbox.x0 + _pad, ax.bbox.y0 + _pad],
                   [ax.bbox.x1 - _pad, ax.bbox.y1 - _pad]])

    def _fits(box):
        return (inset.contains(box.x0, box.y0)
                and inset.contains(box.x1, box.y1))

    anns, fallback = [], []
    # Label only the names that carry the screen's message (outlier multiples
    # or dense round-tripping); the cloud-major cluster is described in the
    # caption instead of labeled point by point.
    lab = v[(v["pe"] > 60) | (v["dep"] >= 2.0)].sort_values("market_cap", ascending=False)
    for _, r in lab.iterrows():
        done = False
        for dx, dy in candidates:
            t = ax.text(r["dep"], r["pe"], r["company_name"], fontsize=7,
                        transform=(ax.transData
                                   + _ST(dx / 72., dy / 72.,
                                         fig.dpi_scale_trans)),
                        ha="left", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  pad=1.0))
            fig.canvas.draw()
            box = t.get_window_extent(fig.canvas.get_renderer())
            if _fits(box) and not any(box.overlaps(o) for o in occupied):
                occupied.append(box)
                anns.append((t, r, dx, dy))
                done = True
                break
            t.remove()
        if not done:
            t = ax.text(r["dep"], r["pe"], r["company_name"], fontsize=7,
                        transform=(ax.transData
                                   + _ST(candidates[0][0] / 72.,
                                         candidates[0][1] / 72.,
                                         fig.dpi_scale_trans)),
                        ha="left", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  pad=1.0))
            anns.append((t, r, candidates[0][0], candidates[0][1]))
            fallback.append(r["company_name"])
    assert not fallback, f"pred_valuation_scatter: no clean slot for {fallback}"
    _assert_no_overlap(fig, [t for t, _, _, _ in anns],
                       "pred_valuation_scatter")
    for t, r, dx, dy in anns:
        ax.annotate("", xy=(r["dep"], r["pe"]), xytext=(r["dep"], r["pe"]),
                    textcoords=(ax.transData
                                + _ST(dx / 72., dy / 72.,
                                      fig.dpi_scale_trans)),
                    arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6,
                                    shrinkA=1.5, shrinkB=1.5))
    fig.text(0.01, 0.0, "Bubble size = market cap. " + FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_valuation_scatter")

    # ---- F6: robustness survival ----
    # The Noise blue bar is a DIFFERENT metric (Nash reproduction at base,
    # not rank agreement): hatched + footnoted so it cannot be read as a
    # fourth rank-hold observation.
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels = ["OAT extremes\n(16 runs)", "Joint draws\n(300)", "Reseed\n(12 games)",
              "Noise*\n(2.5/5/10%)"]
    burst_surv = [100, 100, 100, 64.7]
    form_surv = [62.5, 65.0, 64.7, 64.7]  # formation 10/16 OAT, 65% joint (disclosed weaker link)
    x = np.arange(len(labels))
    bars_b = ax.bar(x - 0.2, burst_surv, 0.4, label="Burst rank holds (%)", color=OI[0],
                    edgecolor="black")
    bars_b[3].set_hatch("///")
    ax.bar(x + 0.2, form_surv, 0.4, label="Formation rank holds (%)", color=OI[5], edgecolor="black")
    ax.set_xticks(x, labels); ax.set_ylabel("Survival / agreement (%)"); ax.set_ylim(0, 112)
    for i, (b_, f_) in enumerate(zip(burst_surv, form_surv)):
        ax.text(i - 0.2, b_ + 1.5, f"{b_:.0f}" if b_ == int(b_) else f"{b_:.1f}",
                ha="center", fontsize=8, fontweight="bold")
        ax.text(i + 0.2, f_ + 1.5, f"{f_:.1f}", ha="center", fontsize=8)
    ax.set_title("(f) Stress record: burst rank holds under OAT/joint/reseed; "
                 "noise reads Nash reproduction")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.0, "* Noise blue bar = Nash reproduction at base (64.7%), "
             "not rank agreement. " + FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_robustness_survival")

    # ---- F7: cascade direct vs second round ----
    d2 = pd.merge(burst[["Archetype", "Direct_Loss_Severe_$B"]],
                  deb[["Archetype", "Second_Round_Loss_$B"]], on="Archetype")
    d2 = d2.sort_values("Direct_Loss_Severe_$B", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(d2["Archetype"], d2["Direct_Loss_Severe_$B"], color=[ARCH_C[a] for a in d2["Archetype"]],
            edgecolor="black")
    ax.barh(d2["Archetype"], d2["Second_Round_Loss_$B"], left=d2["Direct_Loss_Severe_$B"],
            color="white", hatch="///", edgecolor="black")
    for _, r in d2.iterrows():
        tot = r["Direct_Loss_Severe_$B"] + r["Second_Round_Loss_$B"]
        ax.text(tot + 2, r["Archetype"], f"${tot:.1f}B", va="center",
                fontsize=8, fontweight="bold")
    from matplotlib.patches import Patch as _Patch4
    ax.legend(handles=[_Patch4(facecolor="0.75", edgecolor="black",
                               label="Direct"),
                       _Patch4(facecolor="white", edgecolor="black", hatch="///",
                               label="Second round ($15.2B)")],
              fontsize=8, loc="lower right")
    ax.set_xlabel("Severe loss ($B)")
    ax.set_title("(g) Cascade: \\$478.3B direct + \\$15.2B reverberation "
                 "(x1.032) — labs absorb round two")
    fig.text(0.01, 0.0, FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_cascade_stack")

    # ---- F8: geography portable rule ----
    g2 = geo.sort_values("GDP_Share_Pct", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.barh(g2["Economy"], g2["GDP_Share_Pct"],
            color=[OI[0] if x == "N" else "0.7" for x in g2["Illustrative"]], edgecolor="black")
    for i, (_, r) in enumerate(g2.reset_index(drop=True).iterrows()):
        ax.text(r["GDP_Share_Pct"] + 0.012, i, f'{r["GDP_Share_Pct"]:.3f}% (${r["Combined_Loss_$B"]:.1f}B)',
                va="center", fontsize=8)
        if r["Illustrative"] == "Y":
            ax.text(r["GDP_Share_Pct"] / 2, i, "illustrative", ha="center", va="center",
                    fontsize=7, style="italic", color="white")
    ax.set_xlabel("Combined GDP share (%)")
    ax.set_title("(h) Portable rule applied: US-calibrated channels, local GDP/capex inputs")
    from matplotlib.patches import Patch as _Patch3
    ax.legend(handles=[_Patch3(facecolor=OI[0], edgecolor="black",
                               label="US (measured)"),
                       _Patch3(facecolor="0.7", edgecolor="black",
                               label="Illustrative scenario")],
              frameon=True, fontsize=8, loc="center left",
              bbox_to_anchor=(1.02, 0.5))
    fig.text(0.01, 0.0, "US row reproduces Table 7.12 severe; others are stated illustrative scenarios. " + FOOT,
             fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_geography_bars")

    # ---- F9: booked-frame layer composition (Table A.4) ----
    lay = pd.read_csv(os.path.join(TABLES, "table_A.4.layer.exposure.csv"))
    committed = lay[~lay["Layer"].str.startswith("TOTAL") & (lay["Exposure_Treatment"].str.contains("MEMO") == False)].copy()
    memo = lay[lay["Exposure_Treatment"].str.contains("MEMO")].copy()
    committed["Value_$B"] = pd.to_numeric(committed["Value_$B"], errors="coerce")
    committed["Share"] = pd.to_numeric(committed["Share_of_Committed_Pct"], errors="coerce")
    memo["Value_$B"] = pd.to_numeric(memo["Value_$B"], errors="coerce")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                                  gridspec_kw={"width_ratios": [3, 1.4]})
    cc = [OI[0], OI[1], OI[5]]
    bars = ax.bar(committed["Layer"].str.replace("-procurement", "\nproc.")
                  .str.replace("-equity", "\n eq.").str.replace("-guarantee", "\n guar."),
                  committed["Value_$B"], color=cc, edgecolor="black")
    for b, v, s in zip(bars, committed["Value_$B"], committed["Share"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"${v:.0f}B\n({s:.1f}%)",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 620)
    ax.set_ylabel("Booked value ($B)")
    ax.set_title("(i) Booked frame by exposure layer — procurement dominates")
    ax2.bar(["Prospective\ncapacity\n(MEMO)", "Vertical\nintegration\n(MEMO)"], memo["Value_$B"],
            color="0.8", edgecolor="black", hatch="///")
    for i, v in enumerate(memo["Value_$B"]):
        ax2.text(i, v + 12, f"${v:.0f}B", ha="center", fontsize=8, style="italic")
    ax2.set_title("(ii) Parked outside\n(tracked, not booked)")
    ax2.set_ylabel("Tracked value ($B)")
    ax2.set_ylim(0, 600)
    fig.suptitle("What counts as committed — and what is deliberately excluded",
                 fontsize=10, fontweight="bold")
    fig.text(0.01, 0.0, "Table A.4. Memo rows enter no loss math. " + FOOT,
             fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_layer_composition")

    # ---- F10: bubble-mitigation BCRs (Table 7.13, burst-denominated) ----
    pol = pd.read_csv(os.path.join(TABLES, "table_7.13_bubble_policy.csv"))
    pol = pol[~pol["Intervention"].str.startswith("Combined")].copy()
    short = ["Disclosure &\nmargin rules", "GPU-lending\ncaps", "Compute reserve\nbackstop"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    base = pol["Net_Benefit_$B_Base"].values
    lo = base - pol["Net_Benefit_$B_Conservative"].values
    hi = pol["Net_Benefit_$B_Optimistic"].values - base
    bars = ax.bar(short, base, yerr=[lo, hi], capsize=5, color=[OI[2], OI[0], OI[5]],
                  edgecolor="black", error_kw={"ecolor": "black", "lw": 1.2})
    # Headroom first: two-line labels used to clip into the axes top/title.
    ax.set_ylim(0, 120)
    anns = []
    for b, v, r, h in zip(bars, base, pol["BCR_Base"].values, hi):
        anns.append(ax.text(b.get_x() + b.get_width() / 2, v + h + 4,
                            f"${v:.1f}B\nBCR {r:.1f}", ha="center", va="bottom",
                            fontsize=8, fontweight="bold",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      pad=1.5)))
    _assert_no_overlap(fig, anns, "pred_bubble_policy_bcr")
    fig.canvas.draw()
    ab = ax.bbox
    for t in anns:
        bb = t.get_window_extent(fig.canvas.get_renderer())
        assert ab.contains(bb.x0, bb.y0) and ab.contains(bb.x1, bb.y1), \
            "pred_bubble_policy_bcr: label escapes axes"
    ax.set_ylabel("Net benefit, base case ($B; whiskers = cons./opt.)")
    ax.set_title("(j) Remedy value: disclosure saves most ($69.8B); lending caps pay 57-to-1")
    fig.text(0.01, 0.0, "Table 7.13 (expected severe burst loss); cf. DWL-denominated Table 6. " + FOOT,
             fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_bubble_policy_bcr")

    # ---- F11: equity impact vs revenue impact (Table 7.11) ----
    # Foundation Models sits top-right under the title: its label goes below
    # the point (all others above-right), then the de-collision asserts run.
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    anns = []
    for _, r in burst.iterrows():
        ax.scatter(r["Revenue_Impact_Severe_Pct"], r["Equity_Impact_Severe_Pct"],
                   s=r["Total_Loss_Severe_$B"] * 2.2, c=ARCH_C[r["Archetype"]],
                   alpha=0.8, edgecolors="black", linewidths=0.7)
        below = r["Archetype"] == "Foundation Models"
        # The Hardware bubble is ~3x the area of the next: its label needs
        # a wider berth to clear the disc edge.
        off = (14, 14) if r["Total_Loss_Severe_$B"] > 150 else (6, 6)
        anns.append(ax.annotate(f'{r["Archetype"]}\n(${r["Total_Loss_Severe_$B"]:.0f}B)',
                    (r["Revenue_Impact_Severe_Pct"], r["Equity_Impact_Severe_Pct"]),
                    fontsize=7, xytext=(-8, -14) if below else off,
                    textcoords="offset points",
                    va="top" if below else "bottom",
                    ha="right" if below else "left"))
    _assert_no_overlap(fig, anns, "pred_equity_revenue_impact")
    fig.canvas.draw()
    ab = ax.bbox
    for t in anns:
        bb = t.get_window_extent(fig.canvas.get_renderer())
        assert ab.contains(bb.x0, bb.y0) and ab.contains(bb.x1, bb.y1), \
            "pred_equity_revenue_impact: label escapes axes"
    ax.set_xlabel("Revenue impact, severe (%)")
    ax.set_ylabel("Equity impact, severe (%)")
    ax.set_title("(k) Two ways to die: labs by revenue share (80%), Hardware by dollars ($239B)")
    fig.text(0.01, 0.0, "Bubble size = severe total loss. Table 7.11. " + FOOT,
             fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_equity_revenue_impact")

    # ---- F12: open-weights formation scenario (validated S-LOGIT-w_open channel) ----
    # Exact derivation from published Table 7.10 Z-scores + documented parameters
    # (BubbleBurstAnalyzer.formation_probability: p = sig(z + w_open*margin),
    # w_open = 0.6; band = 0.4). No refit. FM 89.3% cross-checks the README.
    margins = {"Hardware": 0.05, "Cloud Providers": 0.10,
               "Foundation Models": 0.90, "LLM Wrappers": 0.45}
    sig = lambda t: 1.0 / (1.0 + np.exp(-t))
    ow_rows = []
    for _, r in form.iterrows():
        a = r["Archetype"]
        z = float(r["Z_Score"]) + 0.6 * margins[a]
        ow_rows.append({"Archetype": a,
                        "Base": float(r["Formation_Prob_Pct"]),
                        "OW": round(sig(z) * 100, 1),
                        "OW_Lo": round(sig(z - 0.4) * 100, 1),
                        "OW_Hi": round(sig(z + 0.4) * 100, 1)})
    ow = pd.DataFrame(ow_rows)
    assert abs(float(ow.set_index("Archetype").loc["Foundation Models", "OW"]) - 89.3) < 0.05
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(ow))
    b1 = ax.bar(x - 0.2, ow["Base"], 0.4, label="Base", color="0.65", edgecolor="black")
    b2 = ax.bar(x + 0.2, ow["OW"], 0.4, label="Open-weights scenario (archetype colors)",
                color=[ARCH_C[a] for a in ow["Archetype"]], edgecolor="black")
    for i, (_, r) in enumerate(ow.iterrows()):
        d = r["OW"] - r["Base"]
        ax.text(i + 0.2, r["OW"] + 1.5, f'{r["OW"]:.1f} (+{d:.1f})',
                ha="center", fontsize=8, fontweight="bold")
        ax.text(i - 0.2, r["Base"] + 1.5, f'{r["Base"]:.1f}', ha="center", fontsize=8)
    ax.set_xticks(x, ow["Archetype"], rotation=0, ha="center")
    ax.set_ylabel("Bubble-condition score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(l) Open-weights stress: free frontier models steepen fragility, order preserved")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.0, "Derived from Table 7.10 Z + documented w_open=0.6 (S-LOGIT-w_open). " + FOOT,
             fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_open_weights")

    # ---- F13: equilibrium stability screen (Table 7.4) ----
    stab = pd.read_csv(os.path.join(TABLES, "table_7.4_stability_screen.csv"))
    stab = stab.sort_values("NE_Stability_Pct", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    pocket = stab["Screen"].str.contains("pocket")
    bars = ax.barh(stab["Game"], stab["NE_Stability_Pct"],
                   color=["#D55E00" if p else "#0072B2" for p in pocket],
                   edgecolor="black")
    for b, v, p in zip(bars, stab["NE_Stability_Pct"], pocket):
        if p:
            ax.text(v + 0.7, b.get_y() + b.get_height() / 2,
                    f"{v:.1f}%  stable loss pocket",
                    va="center", fontsize=8, fontweight="bold")
        else:
            ax.text(v - 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                    va="center", ha="right", fontsize=8, color="white",
                    fontweight="bold")
    from matplotlib.patches import Patch as _Patch2
    ax.legend(handles=[_Patch2(facecolor="#D55E00", edgecolor="black",
                               label="Stable loss pocket"),
                       _Patch2(facecolor="#0072B2", edgecolor="black",
                               label="Not a pocket")],
              frameon=True, fontsize=8, loc="lower right")
    ax.set_xlim(0, 112)
    ax.set_xlabel("Nash reproduction across Monte Carlo draws (%)")
    ax.set_title("(m) Which standoffs self-correct and which sit still: 4 of 6 are stable loss pockets")
    fig.text(0.01, 0.0, "Table 7.4. Core HW-Cloud (64.7%) is the modal outcome, not a certainty. " + FOOT,
             fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_stability_pockets")

    # ---- F14: five-pillar second screen (Table 7.7) ----
    pil = pd.read_csv(os.path.join(TABLES, "table_7.7_five_pillar_screen.csv"))
    pil = pil.sort_values("Composite", ascending=True).reset_index(drop=True)
    cls_c = {"Bubble risk": "#D55E00", "Boom": "#E69F00", "Buildout": "#009E73"}
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    bars = ax.barh(pil["Archetype"], pil["Composite"],
                   color=[cls_c.get(c, "0.6") for c in pil["Classification"]],
                   edgecolor="black")
    for b, v, c in zip(bars, pil["Composite"], pil["Classification"]):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f} — {c}",
                va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("Five-pillar composite (P5 excluded by construction)")
    ax.set_title("(n) Second opinion: only frontier labs screen as Bubble risk")
    fig.text(0.01, 0.0, "Table 7.7; pillars: narrative tilt, multiple exuberance/expansion, "
             "circular-hype proxy, capex context. " + FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_five_pillar")

    # ---- F15: trapped value vs efficiency (Table 7.3 contestability screen) ----
    con = pd.read_csv(os.path.join(TABLES, "table_7.3_contestability_screen.csv"))
    short = {"Hardware-LLM Wrappers": "HW–Wrap", "Hardware-Foundation Models": "HW–FM",
             "Cloud Providers-LLM Wrappers": "Cloud–Wrap",
             "Cloud Providers-Foundation Models": "Cloud–FM",
             "Hardware-Cloud Providers": "HW–Cloud (core)",
             "Foundation Models-LLM Wrappers": "FM–Wrap"}
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for _, r in con.iterrows():
        ax.scatter(r["Efficiency_Pct"], r["DWL_$B"], s=160,
                   c="#D55E00" if "Dominant" in str(r["Game_Type"]) else "#0072B2",
                   alpha=0.85, edgecolors="black")
        _core = ("Cloud Providers" in r["Game"] and "Foundation" not in r["Game"]
                 and "Wrappers" not in r["Game"])
        # Core point sits at the right edge: label it from the empty left side.
        ax.annotate(short.get(r["Game"], r["Game"]),
                    (r["Efficiency_Pct"], r["DWL_$B"]),
                    fontsize=7, xytext=(-6, 5) if _core else (6, 5),
                    ha="right" if _core else "left",
                    textcoords="offset points",
                    fontweight="bold" if _core else "normal")
    ax.set_xlabel("Game efficiency (%)")
    ax.set_ylabel("Deadweight loss ($B)")
    ax.set_title("(o) The paradox of the core: the stuck game wastes the least (94.9% efficient)")
    ax.text(0.98, 0.06, "Red = inefficient dominant-strategy NE\nBlue = coordination failure",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7"))
    fig.text(0.01, 0.0, "Table 7.3. Smallest waste, largest contract base — hence the bottleneck. " + FOOT,
             fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_trapped_vs_efficiency")

    # ---- F16: fragility composite (Table 7.5) ----
    frag = pd.read_csv(os.path.join(TABLES, "table_7.5_fragility_composite.csv")).iloc[0]
    comps = ["Circularity", "Critical Deals", "Single Point", "Concentration", "Inefficiency"]
    fvals = [float(frag["Circularity"]), float(frag["Critical Deals"]),
             float(frag["Single Point"]), float(frag["Concentration"]),
             float(frag["Inefficiency"])]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    bars = ax.bar([f"Circularity\n{fvals[0]:.3f}", "Critical\ndeals", "Single\npoint",
                   "Concentration", "Inefficiency"],
                  fvals, color=["#D55E00" if v > 0.5 else "#0072B2" for v in fvals],
                  edgecolor="black")
    from matplotlib.patches import Patch as _Patch
    ax.legend(handles=[_Patch(facecolor="#D55E00", edgecolor="black",
                              label="High (> 0.5)"),
                       _Patch(facecolor="#0072B2", edgecolor="black",
                              label="Low (≤ 0.5)")],
              frameon=True, fontsize=8, loc="upper right")
    for b, v in zip(bars, fvals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.axhline(float(frag["Composite"]), color="black", ls="--", lw=1.2)
    ax.text(4.4, float(frag["Composite"]) + 0.03,
            f'Composite {float(frag["Composite"]):.3f} — {frag["Band"]}',
            fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fragility component (0–1)")
    ax.set_title("(p) Why P1 holds at Likely: circularity near 1.0 drags the composite to elevated")
    fig.text(0.01, 0.0, "Table 7.5. " + FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_fragility")

    # ---- F17: gatekeeper shares (Table 7.2) ----
    gk = pd.read_csv(os.path.join(TABLES, "table_7.2_gatekeeper_index.csv"))
    gk = gk.sort_values("Gatekeeper_Share_Pct", ascending=True).reset_index(drop=True)
    ext = gk["Archetype"].str.contains("external")
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.barh(gk["Archetype"], gk["Gatekeeper_Share_Pct"],
                   color=["0.7" if e else "#0072B2" for e in ext], edgecolor="black")
    for b, v in zip(bars, gk["Gatekeeper_Share_Pct"]):
        if v > 0:  # zero bars read off the axis; a floating 0.0% adds noise
            ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                    va="center", fontsize=8)
    ax.set_xlabel("Share of inbound stack dependency (%)")
    ax.set_title("(q) Gatekeepers: Hardware controls 53% of what the stack buys from itself")
    fig.text(0.01, 0.0, "Table 7.2; grey = external suppliers. Control precedes loss. " + FOOT,
             fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_gatekeeper")

    # ---- F18 (r): concentration decomposition (Table 7.1) ----
    cd = pd.read_csv(os.path.join(TABLES, "table_7.1_concentration_decomposition.csv"))
    cd = cd.sort_values("HHI_Contribution_Pct", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    bars = ax.barh(cd["Archetype"], cd["HHI_Contribution_Pct"],
                   color=[ARCH_C.get(a, "0.6") for a in cd["Archetype"]],
                   edgecolor="black")
    for b, v, h in zip(bars, cd["HHI_Contribution_Pct"], cd["HHI_Points"]):
        ax.text(v + 0.7, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%  ({h:,.0f} pts)", va="center", fontsize=8)
    ax.set_xlim(0, cd["HHI_Contribution_Pct"].max() * 1.32)
    ax.set_xlabel("Share of cross-stack concentration index (%)")
    ax.set_title("(r) Two archetypes own 97% of concentration: the macro stakes of one breakup")
    fig.text(0.01, 0.0, "Table 7.1; index total 3,851. " + FOOT, fontsize=FOOT_SIZE, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_concentration")

    # ---- F19 (s): valuation screen flags by archetype ----
    v = pd.read_csv(os.path.join(TABLES, "enhanced_valuation_metrics.csv"))
    cats = ["Hardware", "Cloud Providers", "Foundation Models", "LLM Wrappers"]
    flags = ["Positive", "Neutral", "Negative"]
    flag_c = {"Positive": "#009E73", "Neutral": "0.65", "Negative": "#D55E00"}
    counts = {c: {f: int(((v["category"] == c) & (v["screen_flag"] == f)).sum())
                  for f in flags} for c in cats}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(cats))
    bottom = np.zeros(len(cats))
    for f in flags:
        vals = [counts[c][f] for c in cats]
        bars = ax.bar(x, vals, 0.55, bottom=bottom, label=f,
                      color=flag_c[f], edgecolor="black")
        for i, (vv, bb) in enumerate(zip(vals, bottom)):
            if vv > 0:
                ax.text(i, bb + vv / 2, str(vv), ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if f != "Neutral" else "black")
        bottom += np.array(vals)
    ax.set_xticks(x, cats, rotation=8, ha="right")
    ax.set_ylabel("Companies screened")
    ax.set_title("(s) The screen in one chart: 4 Positive, 2 Negative, 17 Neutral")
    ax.legend(title="Screen flag", fontsize=8)
    fig.text(0.01, 0.0, "enhanced_valuation_metrics.csv (23 companies). " + FOOT,
             fontsize=FOOT_SIZE, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_screen_flags")

    # ================= TABLES =================
    # T-A scenario ledger. DebtRank reverberation is modeled severe-only, so the
    # mild row carries no second-round entry; activation rows are proportional
    # scalings (x735/675.4) marked as such, not recomputed equilibria.
    mild_total = round(float(burst["Total_Loss_Mild_$B"].sum()), 1)
    act_scale = 735.0 / 675.40
    scen = pd.DataFrame([
        {"Scenario": "Mild", "Frame_$B": 675.40, "Impair_share": 0.30, "Capex_cut": 0.10,
         "Direct_loss_$B": mild_total, "Second_round_$B": "n/a (severe-only)",
         "GDP_loss_$B": 48.2, "GDP_share_pct": 0.157,
         "Reading": "Bad quarter, no recession print"},
        {"Scenario": "Severe (base prediction)", "Frame_$B": 675.40, "Impair_share": 0.70,
         "Capex_cut": 0.30, "Direct_loss_$B": 478.3, "Second_round_$B": "15.2",
         "GDP_loss_$B": 139.1, "GDP_share_pct": 0.452, "Reading": "Upstream-concentrated repricing"},
        {"Scenario": "Activation-then-severe (scaled x1.088)", "Frame_$B": 735.0, "Impair_share": 0.70,
         "Capex_cut": 0.30, "Direct_loss_$B": round(478.3 * act_scale, 1),
         "Second_round_$B": str(round(15.2 * act_scale, 1)) + " (scaled)",
         "GDP_loss_$B": round(139.1 * act_scale, 1),
         "GDP_share_pct": round(0.452 * act_scale, 3), "Reading": "Every burst figure scales up ~9%"},
    ])
    save_table(scen, "pred_A_scenario_ledger")

    # T-B trigger watchlist (parameters stated in tex Sections 9-10 + credit monitor)
    trig = pd.DataFrame([
        {"Rank": 1, "Trigger": "Frontier-lab down-round / pulled mega-round",
         "Observable": "Announced round size, tenor, debt conversion", "Current_marker": "Anthropic $965B; xAI 400x rev; Databricks 905x",
         "Tripwire": "Round shrinks, stretches, or converts to debt"},
        {"Rank": 2, "Trigger": "Conditional tranches lapse",
         "Observable": "Drawdown announcements vs milestone dates", "Current_marker": "~$60B disclosed, unbooked (GOOG $30B milestones, AMZN $20B, NVDA up-to-$10B)",
         "Tripwire": "Deadline passes with no drawdown"},
        {"Rank": 3, "Trigger": "Neocloud refinancing failure",
         "Observable": "GPU-collateral paper spreads; 'strategic' extensions", "Current_marker": "CoreWeave-model funding; supplier resale guarantees",
         "Tripwire": "Failed roll / collateral-class repricing"},
        {"Rank": 4, "Trigger": "Rating/spread contagion",
         "Observable": "Single-name CDS + rating actions in burst order", "Current_marker": "Oracle 5Y 215bp; BBB-; $570B pipeline (1.43x capex)",
         "Tripwire": "Next leveraged name gaps like Oracle"},
        {"Rank": 5, "Trigger": "Hyperscaler capex guidance cut",
         "Observable": "Earnings-call capex language (joint cuts)", "Current_marker": "~$733B 2026 guidance sum = demand floor",
         "Tripwire": "Two hyperscalers cut together"},
    ])
    save_table(trig, "pred_B_trigger_watchlist")

    # T-C compact valuation watchlist (top fall-first / fall-furthest names)
    vc = val[["company_name", "category", "market_cap", "pe_ratio", "ev_revenue",
              "circular_revenue_dependency", "sustainability_score", "screen_flag"]].copy()
    vc = vc.sort_values("market_cap", ascending=False)
    vc.columns = ["Company", "Archetype", "MktCap_$B", "PE", "EV_Rev",
                  "Circ_dep", "Sustain", "Screen"]
    save_table(vc, "pred_C_valuation_watchlist")

    # T-D robustness ledger
    rob = pd.DataFrame([
        {"Attack": "16 one-at-a-time extremes", "Burst_rank": "Holds 16/16, HW first always",
         "Formation_rank": "Holds 10/16 (honest weaker link)", "Source": "RobustnessBattery OAT"},
        {"Attack": "300 joint draws", "Burst_rank": "100% agreement",
         "Formation_rank": "65% agreement", "Source": "Joint-draw check (seed 7)"},
        {"Attack": "2 reseed reruns (12 games)", "Burst_rank": "All replicate in tolerance",
         "Formation_rank": "All replicate in tolerance", "Source": "Seed screen"},
        {"Attack": "Noise 2.5/5/10%", "Burst_rank": "64.7% Nash reproduction at base",
         "Formation_rank": "Bands widen, order broadly kept", "Source": "Measurement-noise screen"},
        {"Attack": "Saltelli decomposition", "Burst_rank": "Shared circularity top driver (0.99)",
         "Formation_rank": "Circular exposure dominates logit", "Source": "Sensitivity (Table 8.x)"},
    ])
    save_table(rob, "pred_D_robustness_ledger")

    # T-E formation-driver decomposition (transcribe Table 7.10 drivers)
    drv = form[["Archetype", "Circular_Exposure", "Capex_Intensity", "HHI_Share",
                "Exuberance", "Z_Score", "Formation_Prob_Pct", "Prob_Low_Pct", "Prob_High_Pct"]].copy()
    drv.columns = ["Archetype", "Circ_expo", "Capex_int", "HHI_share", "Exuberance",
                   "Z", "Score_pct", "Band_lo", "Band_hi"]
    save_table(drv, "pred_E_formation_drivers")

    # T-F early-warning dashboard: five validated screens, one row per archetype.
    # A.3 mapping verified against the live model (HW 0.783, Cloud 0.746,
    # FM 0.381, Wrappers 0.124); OW column derived above; five-pillar classes
    # from Table 7.7; burst ranks from Table 7.11; watch order from Table 7.15.
    pillar = {"Hardware": "Boom (0.408)", "Cloud Providers": "Buildout (0.317)",
              "Foundation Models": "Bubble risk (0.507)", "LLM Wrappers": "Boom (0.352)"}
    tier = {"Hardware": "High (0.783)", "Cloud Providers": "High (0.746)",
            "Foundation Models": "Low (0.381)", "LLM Wrappers": "Low (0.124)"}
    brank = {"Hardware": 1, "Cloud Providers": 2, "Foundation Models": 3, "LLM Wrappers": 4}
    ewd = pd.DataFrame([{"Archetype": a,
                         "Formation_base_pct": float(form.set_index("Archetype").loc[a, "Formation_Prob_Pct"]),
                         "Formation_OW_pct": float(ow.set_index("Archetype").loc[a, "OW"]),
                         "Five_pillar": pillar[a], "Stability": tier[a],
                         "Burst_rank": brank[a], "Credit_watch": brank[a]}
                        for a in ARCH])
    save_table(ewd, "pred_F_early_warning")

    # T-G burst-timing window: constant-hazard translation of the formation
    # scores into quarters-from-vintage. Hazard h = p / TAU with TAU = 12
    # quarters (stated formation-to-burst lag: 2005->2008 and 1999->2002
    # vintages both run ~3y from dense formation to burst). Median quarters
    # to burst = ln(2)/h. Earliest/latest vary p over the Table 7.10 band at
    # base TAU; fast/slow vary TAU over [8, 20] at base p (the honest
    # uncertainty is the lag, not the band). System row = first-tip hazard
    # (max across archetypes). Scenario arithmetic, not a forecast: move TAU
    # and every quarter moves with it, stated in the Reading column.
    TAU_Q, TAU_FAST, TAU_SLOW = 12.0, 8.0, 20.0
    _LN2 = float(np.log(2.0))

    def _med_q(p_frac, tau):
        return _LN2 / (p_frac / tau) if p_frac > 0 else float("inf")

    _frow = form.set_index("Archetype")
    _trows, _sys = [], {"h": 0.0, "p": 0.0, "p_hi": 0.0, "p_lo": 0.0, "who": ""}
    for a in ARCH:
        p = float(_frow.loc[a, "Formation_Prob_Pct"]) / 100.0
        p_lo = float(_frow.loc[a, "Prob_Low_Pct"]) / 100.0
        p_hi = float(_frow.loc[a, "Prob_High_Pct"]) / 100.0
        h = p / TAU_Q
        med, early, late = _med_q(p, TAU_Q), _med_q(p_hi, TAU_Q), _med_q(p_lo, TAU_Q)
        fast, slow = _med_q(p, TAU_FAST), _med_q(p, TAU_SLOW)
        _trows.append({"Archetype": a, "Formation_pct": round(p * 100, 1),
                       "Hazard_per_q": round(h, 4), "Median_q": round(med, 1),
                       "Earliest_q": round(early, 1), "Latest_q": round(late, 1),
                       "Fast_q": round(fast, 1), "Slow_q": round(slow, 1),
                       "Median_quarter": q_label(med),
                       "Window": f"{q_label(fast)}-{q_label(slow)}",
                       "Reading": f"TAU={TAU_Q:.0f}q stated lag; slow/fast TAU {TAU_SLOW:.0f}/{TAU_FAST:.0f}q"})
        if h > _sys["h"]:
            _sys.update(h=h, p=p, p_hi=p_hi, p_lo=p_lo, who=a)
    _smed = _med_q(_sys["p"], TAU_Q)
    _sfast, _sslow = _med_q(_sys["p"], TAU_FAST), _med_q(_sys["p"], TAU_SLOW)
    _trows.append({"Archetype": "System (first-tip)", "Formation_pct": "",
                   "Hazard_per_q": round(_sys["h"], 4),
                   "Median_q": round(_smed, 1),
                   "Earliest_q": round(_med_q(_sys["p_hi"], TAU_Q), 1),
                   "Latest_q": round(_med_q(_sys["p_lo"], TAU_Q), 1),
                   "Fast_q": round(_sfast, 1),
                   "Slow_q": round(_sslow, 1),
                   "Median_quarter": q_label(_smed),
                   "Window": f"{q_label(_sfast)}-{q_label(_sslow)}",
                   "Reading": f"{_sys['who']} tips first at base TAU"})
    tmg = pd.DataFrame(_trows)
    save_table(tmg, "pred_G_burst_timing")

    # T-H company-level burst losses: each archetype's Table 7.11 severe/mild
    # totals split across its valuation-screen companies on two stated bases.
    # (a) market-cap pro-rata (size lens); (b) deal-flow lens: each company's
    # attributed circular-deal flow from industry_dependencies_enhanced.csv
    # (dependent-side exact match plus provider-side alias match; a row counts
    # for both ends since both are exposed to it; aggregate rows such as
    # "All Frontier Models" are unattributable and excluded). Companies with
    # no measured flow read $0 on the deal basis -- an honest zero, not a
    # model claim of safety. Severe_mid = mean of the two bases; haircut =
    # Severe_mid / market cap, capped at 100 (wipeout). Zero-flow archetypes
    # fall back to equal split. T-K caps the expected haircut the same way.
    deals = pd.read_csv(os.path.join(TABLES, "industry_dependencies_enhanced.csv"))
    _dval = pd.to_numeric(deals["Dependency_Value_Billions"], errors="coerce").fillna(0.0).clip(lower=0.0)
    _dep = deals["Dependent_Player"].fillna("").str.strip()
    _on = deals["Dependency_On"].fillna("").str.lower()
    _DEP_MAP = {"anthropic": "Anthropic", "openai": "OpenAI", "coreweave": "CoreWeave",
                "meta": "Meta (Llama)", "microsoft azure": "Microsoft Azure", "aws": "AWS"}
    _ON_MAP = {"nvidia": "NVIDIA", "tpu": "Google (TPU)",
               "azure": "Microsoft Azure", "microsoft": "Microsoft Azure",
               "aws": "AWS", "coreweave": "CoreWeave", "lambda": "Lambda Labs",
               "openai": "OpenAI", "anthropic": "Anthropic", "meta": "Meta (Llama)"}
    _flow = {}
    for _dp, _oo, _vv in zip(_dep, _on, _dval):
        _hit = set()
        if _dp.strip().lower() in _DEP_MAP:
            _hit.add(_DEP_MAP[_dp.strip().lower()])
        for _alias, _co in _ON_MAP.items():
            if _alias in _oo:
                _hit.add(_co)
        for _co in _hit:
            _flow[_co] = _flow.get(_co, 0.0) + float(_vv)
    print(f"  deal-flow attribution: {len(_flow)} screen companies share "
          f"${sum(_flow.values()):.1f}B attributed flow")
    _brow = burst.set_index("Archetype")
    _hrows = []
    for a in ARCH:
        sub = val[val["category"] == a].copy()
        cap = pd.to_numeric(sub["market_cap"], errors="coerce").fillna(0.0).clip(lower=0.0)
        expo = sub["company_name"].map(_flow).fillna(0.0).clip(lower=0.0)
        if cap.sum() > 0:
            mw = cap / cap.sum()
        else:
            mw = pd.Series([1.0 / len(sub)] * len(sub), index=sub.index)
        if expo.sum() > 0:
            ew = expo / expo.sum()
        else:
            ew = pd.Series([1.0 / len(sub)] * len(sub), index=sub.index)
        sev = float(_brow.loc[a, "Total_Loss_Severe_$B"])
        mild = float(_brow.loc[a, "Total_Loss_Mild_$B"])
        for (nm, c, m, e) in zip(sub["company_name"], cap, mw, ew):
            sm, se = sev * m, sev * e
            mm, me = mild * m, mild * e
            mid = (sm + se) / 2.0
            _hrows.append({"Company": nm, "Archetype": a, "MktCap_$B": round(float(c), 1),
                           "Mcap_wt_pct": round(float(m) * 100, 2),
                           "Expo_wt_pct": round(float(e) * 100, 2),
                           "Severe_mcap_$B": round(sm, 2), "Severe_expo_$B": round(se, 2),
                           "Mild_mcap_$B": round(mm, 2), "Mild_expo_$B": round(me, 2),
                           "Severe_mid_$B": round(mid, 2),
                           "Haircut_severe_pct": min(round(mid / float(c) * 100, 1), 100.0) if c > 0 else 0.0})
    cmp_loss = pd.DataFrame(_hrows).sort_values("Severe_mid_$B", ascending=False).reset_index(drop=True)
    save_table(cmp_loss, "pred_H_company_losses")

    # ---- F(t): burst-timing window ----
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _tn = tmg.set_index("Archetype").loc[ARCH + ["System (first-tip)"]]
    y = np.arange(len(_tn))
    ax.barh(y, _tn["Slow_q"] - _tn["Fast_q"], left=_tn["Fast_q"], height=0.45,
            color=[ARCH_C.get(a, "0.55") for a in _tn.index], edgecolor="black", linewidth=0.6,
            label="TAU 8-20q sensitivity")
    ax.barh(y, _tn["Latest_q"] - _tn["Earliest_q"], left=_tn["Earliest_q"], height=0.22,
            color="white", edgecolor="black", linewidth=0.8, label="Formation-band window")
    ax.scatter(_tn["Median_q"], y, s=70, marker="D", color="black", zorder=5, label="Median (TAU 12q)")
    anns = []
    for i, (m, q) in enumerate(zip(_tn["Median_q"], _tn["Median_quarter"])):
        anns.append(ax.text(m + 0.9, i, f"{m:.1f}q ({q})", va="center", fontsize=8,
                            bbox=dict(facecolor="white", edgecolor="none", pad=1)))
    _assert_no_overlap(fig, anns, "pred_timing_window")
    ax.set_yticks(y, _tn.index)
    ax.set_xlabel("Quarters from Sept 2026 vintage")
    ax.set_title("(t) Burst timing: formation-implied window with median — labs tip first")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.0, FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_timing_window")

    # ---- F(u): company-level severe losses, top 8 ----
    top8 = cmp_loss.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(top8))
    ax.barh(y - 0.2, top8["Severe_mcap_$B"], 0.4, label="Severe, mcap basis",
            color=OI[0], edgecolor="black", linewidth=0.6)
    ax.barh(y + 0.2, top8["Severe_expo_$B"], 0.4, label="Severe, exposure basis",
            color=OI[1], edgecolor="black", linewidth=0.6)
    anns = []
    for i, (m, e) in enumerate(zip(top8["Severe_mcap_$B"], top8["Severe_expo_$B"])):
        anns.append(ax.text(m + max(top8["Severe_mcap_$B"].max(),
                                    top8["Severe_expo_$B"].max()) * 0.02, i - 0.2,
                            f"{m:.1f}", va="center", fontsize=7))
        anns.append(ax.text(e + max(top8["Severe_mcap_$B"].max(),
                                    top8["Severe_expo_$B"].max()) * 0.02, i + 0.2,
                            f"{e:.1f}", va="center", fontsize=7))
    _assert_no_overlap(fig, anns, "pred_company_losses")
    ax.set_yticks(y, top8["Company"])
    ax.set_xlabel("Severe burst loss ($B)")
    ax.set_title("(u) Who loses what: top-8 company severe losses on two allocation bases")
    ax.legend(fontsize=8)
    _ufoot = FOOT
    if ((top8["Severe_mcap_$B"] < 0.5) | (top8["Severe_expo_$B"] < 0.5)).any():
        _ufoot = ("Bars under $0.5B labeled at axis (private/unmatched inputs "
                  "carry ~zero weight; exact weights: Table pred_H). " + FOOT)
    fig.text(0.01, 0.0, _ufoot, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_company_losses")

    # T-I price gauges: convexity + hazard on vendored weekly closes
    # (tables/price_history_weekly.csv; Yahoo Finance via yfinance,
    # auto-adjusted for splits/dividends, fetched Sept 2026; QQQ/SPY are ETF
    # proxies for Nasdaq-100/S&P 500). Six contexts: three live names plus
    # three frozen backtests (SPY Oct'07 sensitivity, QQQ Mar'00 sensitivity,
    # SPY Feb'25 specificity -- no crash followed, so the honest gauge reads
    # quiet). T-J backtests the hazard itself: crash-capture over SPY
    # 2000-2026 with stated (not fitted) weights, so the lift is descriptive.
    pxw = pd.read_csv(os.path.join(TABLES, "price_history_weekly.csv"),
                      parse_dates=["Date"]).sort_values(["Ticker", "Date"])
    _CONTEXTS = [("NVDA", "2026-09-30", "current"),
                 ("QQQ", "2026-09-30", "current"),
                 ("SPY", "2026-09-30", "current"),
                 ("SPY", "2007-10-31", "sensitivity"),
                 ("QQQ", "2000-03-31", "sensitivity"),
                 ("SPY", "2025-02-28", "specificity")]
    _grows, _ctx = [], {}
    for ticker, asof, check in _CONTEXTS:
        sub = pxw[(pxw["Ticker"] == ticker) & (pxw["Date"] <= asof)]
        sub = sub.sort_values("Date").reset_index(drop=True)
        y = np.log(sub["AdjClose"].to_numpy())
        n = len(sub)
        ts = {L: round(convex_tstat(y[n - L:]), 2) for L in CONVEX_WINDOWS if L <= n}
        cmax = max(ts.values()) if ts else float("nan")
        feats = price_features(sub["AdjClose"].to_numpy())
        scores, regimes, tops, _ = hazard_frame(feats)
        hz = float(scores.iloc[n - 1]) if n - 1 in scores.index else float("nan")
        rg = str(regimes.iloc[n - 1])
        tp = str(tops.iloc[n - 1])
        verdict = ("FIRES" if cmax >= CONVEX_FIRE_T else "quiet")
        _ctx[(ticker, asof)] = {"n": n, "ts": ts, "hazard": hz}
        _grows.append({"Name": ticker, "Asof": str(sub["Date"].iloc[-1].date()),
                       "Convex_t26": ts.get(26, ""), "Convex_t52": ts.get(52, ""),
                       "Convex_t78": ts.get(78, ""), "Convex_max": round(cmax, 2),
                       "Convex_verdict": verdict, "Hazard": round(hz, 1),
                       "Regime": rg, "Top_factor": tp, "Check": check,
                       "Reading": ("1y-window acceleration with 6mo/18mo context; "
                                   "hazard 0-100, regime scales sensitivity")})
    gauges = pd.DataFrame(_grows)
    save_table(gauges, "pred_I_price_gauges")
    _spy = pxw[pxw["Ticker"] == "SPY"].sort_values("Date").reset_index(drop=True)
    _sfeats = price_features(_spy["AdjClose"].to_numpy())
    _sscores, _, _, _ = hazard_frame(_sfeats)
    _cc = crash_capture(_spy["AdjClose"].to_numpy(), _sscores)
    cap = pd.DataFrame([
        {"Metric": "Crash weeks (>=20% drawdown, SPY 2000-2026)", "Value": _cc["crash_weeks"]},
        {"Metric": "Scored weeks", "Value": _cc["weeks"]},
        {"Metric": "Top-quintile threshold", "Value": round(_cc["threshold"], 1)},
        {"Metric": "Crash-week capture share", "Value": round(_cc["capture"], 3)},
        {"Metric": "Baseline share", "Value": 0.20},
        {"Metric": "Lift over baseline", "Value": round(_cc["lift"], 2)},
    ])
    save_table(cap, "pred_J_gauge_backtest")
    print(f"  gauge backtest: capture {_cc['capture']:.3f}, lift {_cc['lift']:.2f}")

    # T-K expected burst losses (quant memo): severity blend per company.
    # E_loss = P_burst * Severe_mid + (1 - P_burst) * Mild_mid, where P_burst
    # is the archetype formation probability (pred_G) used AS the
    # severe-burst probability inside the window -- an upper-bound
    # convention, stated in Reading, not a measured burst frequency.
    # Mild_mid is the mean of the two mild bases. Driver names the basis
    # that dominates the severe loss: deal-flow (exposed through circular
    # deals) or size (market-cap weight).
    _pmap = {a: float(tmg.set_index("Archetype").loc[a, "Formation_pct"]) / 100.0
             for a in ARCH}
    _krows = []
    for _, _r in cmp_loss.iterrows():
        _p = _pmap[_r["Archetype"]]
        _mild = round((float(_r["Mild_mcap_$B"]) + float(_r["Mild_expo_$B"])) / 2.0, 2)
        _sev = float(_r["Severe_mid_$B"])
        _exp = round(_p * _sev + (1.0 - _p) * _mild, 2)
        _cap = float(_r["MktCap_$B"])
        _drv = ("deal-flow" if float(_r["Severe_expo_$B"]) >= float(_r["Severe_mcap_$B"])
                else "size")
        _krows.append({"Company": _r["Company"], "Archetype": _r["Archetype"],
                       "P_burst_pct": round(_p * 100, 1),
                       "Mild_mid_$B": _mild, "Severe_mid_$B": _sev,
                       "E_loss_$B": _exp,
                       "E_haircut_pct": (min(round(_exp / _cap * 100, 1), 100.0)
                                         if _cap > 0 else 0.0),
                       "Driver": _drv,
                       "Reading": (f"P_burst = {_r['Archetype']} formation "
                                   f"{_p * 100:.1f}% (upper-bound convention); "
                                   f"severe loss is {_drv}-driven")})
    exp_loss = pd.DataFrame(_krows).sort_values("E_loss_$B", ascending=False)
    exp_loss = exp_loss.reset_index(drop=True)
    save_table(exp_loss, "pred_K_expected_losses")
    print(f"  expected losses: top E_loss {exp_loss['Company'].iloc[0]} "
          f"${exp_loss['E_loss_$B'].iloc[0]:.1f}B")

    # T-L timing rationale (quant memo): the why-ledger behind the
    # conditional call. Hazard base (system first-tip median) plus three
    # stated adjustments: trigger calm-credit (+1q judgment -- no pred_B
    # tripwire reads tripped on Sept-2026 price evidence), gauge
    # triangulation (+0q: acceleration without stress, no pull-forward),
    # precedent topping lag (+0q: supports window width, not the median).
    _sys = tmg.set_index("Archetype").loc["System (first-tip)"]
    _sys_med = float(_sys["Median_q"])
    _fm_pct = float(tmg.set_index("Archetype").loc["Foundation Models", "Formation_pct"])
    _nv = gauges[(gauges["Name"] == "NVDA") & (gauges["Check"] == "current")].iloc[0]
    _q00 = gauges[(gauges["Name"] == "QQQ") & (gauges["Check"] == "sensitivity")].iloc[0]
    _s07 = gauges[(gauges["Name"] == "SPY") & (gauges["Check"] == "sensitivity")].iloc[0]
    _CALM_CREDIT = 1.0
    _cond = round(_sys_med + _CALM_CREDIT, 1)
    _lrows = [
        {"Reason": "Hazard base — system first-tip",
         "Detail": (f"{_sys['Reading']}: FM formation {_fm_pct:.1f}% x TAU=12q "
                    f"stated lag -> median {_sys_med:.1f}q ({_sys['Median_quarter']}); "
                    f"slow/fast {_sys['Fast_q']:.1f}-{_sys['Slow_q']:.1f}q"),
         "Adj_q": 0.0, "Weight": "High"},
        {"Reason": "Trigger conditioning — calm credit",
         "Detail": ("no pred_B tripwire reads tripped on Sept-2026 price evidence "
                    f"(hazard Low on NVDA/QQQ/SPY: {float(_nv['Hazard']):.0f}/"
                    f"{float(gauges[(gauges['Name'] == 'QQQ') & (gauges['Check'] == 'current')].iloc[0]['Hazard']):.0f}/"
                    f"{float(gauges[(gauges['Name'] == 'SPY') & (gauges['Check'] == 'current')].iloc[0]['Hazard']):.0f}); "
                    "stated +1q judgment, not a measurement"),
         "Adj_q": _CALM_CREDIT, "Weight": "Medium"},
        {"Reason": "Gauge triangulation — acceleration without stress",
         "Detail": (f"NVDA convex-max {float(_nv['Convex_max']):.1f} (fires at 2.0) "
                    f"with hazard {float(_nv['Hazard']):.0f} (Low) vs QQQ@Mar'00 "
                    f"{float(_q00['Convex_max']):.1f}/hazard {float(_q00['Hazard']):.0f} "
                    f"and SPY@Oct'07 {float(_s07['Convex_max']):.1f}/hazard "
                    f"{float(_s07['Hazard']):.0f}: no pull-forward"),
         "Adj_q": 0.0, "Weight": "Medium"},
        {"Reason": "Precedent topping lag",
         "Detail": ("2000/2007 tops preceded crash-week clusters by 2-8 quarters; "
                    "supports window width, not the median"),
         "Adj_q": 0.0, "Weight": "Low"},
        {"Reason": "CONDITIONAL CALL (level)",
         "Detail": (f"median {_cond:.1f}q ({q_label(_cond)}); window "
                    f"{q_label(float(_sys['Fast_q']) + _CALM_CREDIT)}-"
                    f"{q_label(float(_sys['Slow_q']) + _CALM_CREDIT)} (system TAU "
                    "8-20q band shifted by the calm credit)"),
         "Adj_q": _cond, "Weight": "High"},
    ]
    # Sensitivity: the call under no credit and double credit, so the one
    # judgment in the bridge carries its own error bar.
    for _credit, _label in ((0.0, "no calm credit"), (2.0, "double calm credit")):
        _lvl = round(_sys_med + _credit, 1)
        _lrows.append({"Reason": f"Sensitivity — {_label} (level)",
                       "Detail": f"median {_lvl:.1f}q ({q_label(_lvl)})",
                       "Adj_q": _lvl, "Weight": "Low"})
    rationale = pd.DataFrame(_lrows)
    save_table(rationale, "pred_L_timing_rationale")
    print(f"  conditional call: {_cond:.1f}q ({q_label(_cond)})")

    # ---- F(w): gauges across contexts ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    _labels = [f"{r['Name']}@{r['Asof'][:7]}" for _, r in gauges.iterrows()]
    x = np.arange(len(gauges))
    _ct = gauges["Convex_max"].to_numpy(dtype=float)
    ax1.bar(x, _ct, color=[OI[1] if v >= CONVEX_FIRE_T else "0.7" for v in _ct],
            edgecolor="black", linewidth=0.6)
    ax1.axhline(CONVEX_FIRE_T, color="red", ls="--", lw=1)
    ax1.text(len(gauges) - 1, CONVEX_FIRE_T + 0.15, "fires at 2.0", ha="right",
             fontsize=8, color="red")
    ax1.set_ylabel("Convexity max t-stat")
    ax1.set_title("(w) Price gauges: convexity fires at both historical tops; hazard concentrates in crash weeks")
    _hz = gauges["Hazard"].to_numpy(dtype=float)
    _regc = {"High": OI[1], "Mid": OI[5], "Low": OI[2]}
    ax2.bar(x, _hz, color=[_regc.get(r, "0.7") for r in gauges["Regime"]],
            edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Hazard score")
    ax2.set_xticks(x, _labels, rotation=10, ha="right")
    ax2.set_ylim(0, 105)
    anns = []
    for i, (c, h) in enumerate(zip(_ct, _hz)):
        if c >= 0:
            anns.append(ax1.text(i, c + 0.25, f"{c:.1f}", ha="center",
                                 va="bottom", fontsize=8))
        else:
            anns.append(ax1.text(i, c - 0.4, f"{c:.1f}", ha="center",
                                 va="top", fontsize=8))
        anns.append(ax2.text(i, h + 1.5, f"{h:.0f}", ha="center", fontsize=8))
    _assert_no_overlap(fig, anns, "pred_gauge_backtest")
    fig.text(0.01, 0.0, FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_gauge_backtest")

    # ---- F(x): expected losses, top 10 ----
    top10 = exp_loss.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(top10))
    _e = top10["E_loss_$B"].to_numpy(dtype=float)
    _lo = top10["Mild_mid_$B"].to_numpy(dtype=float)
    _hi = top10["Severe_mid_$B"].to_numpy(dtype=float)
    ax.barh(y, _e, height=0.55, color=OI[0], edgecolor="black", linewidth=0.6,
            xerr=[_e - _lo, _hi - _e], ecolor="black", capsize=3,
            label="Expected loss (whiskers: mild-mid to severe-mid)")
    anns = [ax.text(h + _hi.max() * 0.015, i, f"${v:.1f}B", va="center", fontsize=7)
            for i, v, h in zip(range(len(_e)), _e, _hi)]
    _assert_no_overlap(fig, anns, "pred_expected_losses")
    ax.set_yticks(y, top10["Company"])
    ax.set_xlabel("Expected burst loss ($B)")
    ax.set_title("(x) Expected loss per player: burst-probability-weighted, severity range shown")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.0, FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_expected_losses")

    # ---- F(y): timing rationale bridge ----
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _steps = ["Hazard base", "+ triggers", "+ gauges", "+ precedent", "Conditional"]
    _lvls = [_sys_med, _sys_med + _CALM_CREDIT, _sys_med + _CALM_CREDIT,
             _sys_med + _CALM_CREDIT, _cond]
    x = np.arange(len(_steps))
    ax.bar(x, _lvls, color=["0.7", "0.7", "0.7", "0.7", OI[1]],
           edgecolor="black", linewidth=0.6)
    anns = [ax.text(i, v + 0.25, f"{v:.1f}q\n({q_label(v)})", ha="center",
                    va="bottom", fontsize=8,
                    fontweight="bold" if i == len(_steps) - 1 else "normal")
            for i, v in enumerate(_lvls)]
    _assert_no_overlap(fig, anns, "pred_timing_call")
    ax.set_xticks(x, _steps)
    ax.set_ylabel("Quarters from Sept 2026 vintage")
    ax.set_ylim(0, max(_lvls) + 3.5)
    ax.set_title("(y) Why this timing: hazard base plus stated adjustments = conditional call")
    fig.text(0.01, 0.0, FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_timing_call")

    # T-M ML horse-race (RSI evaluation, negative result): 3-feature L2
    # logistic crash model (dd/dur/vol; dist exactly collinear with dd, r =
    # 1.00, so it carries no independent signal) vs the stated-weight hazard
    # on three fits x era evaluations. Verdict: no stable winner -- weights
    # fit to the dot-com bust misfire on the GFC and vice versa -- so the
    # stated gauge stands and no ML upgrade ships. Scaler and fit see train
    # eras only; every test era starts after its fit's train era ends.
    _spy = pxw[pxw["Ticker"] == "SPY"].sort_values("Date").reset_index(drop=True)
    _spx = _spy["AdjClose"].to_numpy()
    _sfeats = price_features(_spx)
    _sX = _sfeats[ML_FEATS].to_numpy(dtype=float)
    _sscores, _, _, _ = hazard_frame(_sfeats)
    _sbase = _sscores.to_numpy(dtype=float)
    _speak = np.maximum.accumulate(_spx)
    _sy = np.where(_speak > 0, (_speak - _spx) / _speak, 0.0)
    _sy = (_sy >= 0.20).astype(int)
    _sdates = _spy["Date"].to_numpy(dtype="datetime64[D]")
    _WF = _sdates < np.datetime64(ML_CUTOFFS["wf"])
    _MID = ((_sdates >= np.datetime64(ML_CUTOFFS["wf"]))
            & (_sdates < np.datetime64(ML_CUTOFFS["late"])))
    _LATE = _sdates >= np.datetime64(ML_CUTOFFS["late"])
    _mrows = []
    for _fname, _ftr in (("FULL train 2000-15", _WF | _MID),
                         ("WF train 2000-06", _WF),
                         ("REV train 2007+", _MID | _LATE)):
        _mu, _sd = standardize_fit(_sX[_ftr])
        _Z = (_sX - _mu) / _sd
        _clf = fit_crash_logit(_Z[_ftr], _sy[_ftr])
        _p = _clf.predict_proba(_Z)[:, 1]
        _eras = {"train": _ftr}
        if _fname.startswith("FULL"):
            _eras["late-test"] = _LATE
        elif _fname.startswith("WF"):
            _eras.update({"wf-test": _MID, "late-test": _LATE})
        else:
            _eras["rev-test"] = _WF
        for _ename, _era in _eras.items():
            _gm = era_scores(_sy, _p, _era)
            _gb = era_scores(_sy, _sbase, _era)
            _mrows.append({"Fit": _fname, "Context": _ename,
                           "N_crash": _gm["n_crash"],
                           "Model_cap": round(_gm["capture"], 3),
                           "Base_cap": round(_gb["capture"], 3),
                           "Model_auc": round(_gm["auc"], 3),
                           "Base_auc": round(_gb["auc"], 3),
                           "Winner": ("model" if _gm["capture"] > _gb["capture"]
                                      else "base" if _gb["capture"] > _gm["capture"]
                                      else "tie"),
                           "Coef_dd": round(float(_clf.coef_[0][0]), 3),
                           "Coef_dur": round(float(_clf.coef_[0][1]), 3),
                           "Coef_vol": round(float(_clf.coef_[0][2]), 3)})
    mlrace = pd.DataFrame(_mrows)
    save_table(mlrace, "pred_M_ml_horserace")
    _oos = mlrace[mlrace["Context"].str.contains("test")]
    print(f"  ML horse-race: OOS winners {list(_oos['Winner'])}")

    # ---- F(z): ML horse-race, out-of-sample capture ----
    _zlab = (_oos["Fit"].str.split().str[0] + ": " + _oos["Context"]).tolist()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(_oos))
    ax.bar(x - 0.2, _oos["Model_cap"], 0.4, label="L2 logit (fit pre-era)",
           color=OI[0], edgecolor="black", linewidth=0.6)
    ax.bar(x + 0.2, _oos["Base_cap"], 0.4, label="Stated-weight hazard",
           color=OI[2], edgecolor="black", linewidth=0.6)
    anns = []
    for i, (m, b) in enumerate(zip(_oos["Model_cap"], _oos["Base_cap"])):
        anns.append(ax.text(i - 0.2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8))
        anns.append(ax.text(i + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8))
    _assert_no_overlap(fig, anns, "pred_ml_horserace")
    ax.set_xticks(x, _zlab, rotation=8, ha="right")
    ax.set_ylabel("Crash-week capture share")
    ax.set_ylim(0, max(_oos["Model_cap"].max(), _oos["Base_cap"].max()) + 0.25)
    ax.set_title("(z) ML horse-race out-of-sample: fitted weights win one bust, lose the other")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.0, FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "pred_ml_horserace")

    idx = pd.DataFrame([
        {"stem": "pred_A_scenario_ledger", "figures": "pred_scenario_ladder, pred_gdp_waterfall",
         "use": "Section 10 scenarios: mild/severe/activation ledger"},
        {"stem": "pred_B_trigger_watchlist", "figures": "pred_valuation_scatter",
         "use": "Section 9 triggers: observable + tripwire per trigger"},
        {"stem": "pred_C_valuation_watchlist", "figures": "pred_valuation_scatter",
         "use": "Section 11 watchlist: 23-company screen, compact"},
        {"stem": "pred_D_robustness_ledger", "figures": "pred_robustness_survival",
         "use": "Section 8 stress record: burst vs formation survival split"},
        {"stem": "pred_E_formation_drivers", "figures": "pred_formation_bands",
         "use": "Section 6 bubble scores: driver decomposition + bands"},
        {"stem": "pred_F_early_warning", "figures": "pred_open_weights",
         "use": "Sections 6-7: five-screen early-warning dashboard per archetype"},
        {"stem": "pred_G_burst_timing", "figures": "pred_timing_window",
         "use": "Burst-timing addendum: hazard-implied window + median per archetype (Sept 2026 vintage)"},
        {"stem": "pred_H_company_losses", "figures": "pred_company_losses",
         "use": "Burst-loss addendum: 23-company severe/mild split on mcap and exposure bases"},
        {"stem": "pred_I_price_gauges", "figures": "pred_gauge_backtest",
         "use": "Price gauges: convexity t-stats + hazard per context (live + frozen backtests)"},
        {"stem": "pred_J_gauge_backtest", "figures": "pred_gauge_backtest",
         "use": "Hazard backtest: SPY crash-week capture vs baseline, stated weights"},
        {"stem": "pred_K_expected_losses", "figures": "pred_expected_losses",
         "use": "Quant memo: per-player expected burst loss (severity blend) with driver"},
        {"stem": "pred_L_timing_rationale", "figures": "pred_timing_call",
         "use": "Quant memo: why-ledger behind the conditional burst-timing call"},
        {"stem": "pred_M_ml_horserace", "figures": "pred_ml_horserace",
         "use": "RSI evaluation: L2-logit vs stated hazard, walk-forward scoreboard (negative: no stable winner)"},
    ])
    idx.to_csv(os.path.join(TABLES, "pred_tables_index.csv"), index=False)
    print("  wrote tables/pred_tables_index.csv")
    print(f"  synergy check: combined DWL reduction ${float(synergy['Combined_DWL_Reduction_$B'].iloc[0]):.1f}B, "
          f"ratio {float(synergy['Synergy_Ratio'].iloc[0]):.3f}")

if __name__ == "__main__":
    main()
