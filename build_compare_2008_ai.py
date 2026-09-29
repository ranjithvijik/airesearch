"""AI vs 2008 head-to-head comparison exhibits.

Reads the AI vintage (tables/table_7.1x.csv, enhanced portfolio A.2) and the
2008 backtest (tables/backtest/*.csv). No refit, no new parameters.
Outputs figures/compare_*.png/.pdf + tables/compare_summary.csv/.tex/.xlsx.
Run: python3 build_compare_2008_ai.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(ROOT, "tables")
BT = os.path.join(TABLES, "backtest")
FIGS = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 300, "savefig.dpi": 300,
})
AI_C, Y08_C = "#0072B2", "#D55E00"
FOOT = "AI: Sept 2026 pipeline vintage. 2008: sourced-2006 backtest frame."


def savefig(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, stem + "." + ext),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote figures/" + stem + ".png/.pdf")


# AI rank-ordered (best->worst fragility / largest->smallest loss)
AI_FORM = [82.9, 77.5, 51.7, 46.2]
AI_WHO_F = ["Foundation Models", "Hardware", "Cloud Providers", "LLM Wrappers"]
AI_BURST = [239.4, 132.7, 85.39, 20.83]
AI_WHO_B = ["Hardware", "Cloud", "Labs", "Apps"]
AI_BCR = [("Disclosure", 35.87), ("Lending caps", 57.40), ("Reserve", 7.97)]

b08 = pd.read_csv(os.path.join(BT, "backtest_burst.csv"))
f08 = pd.read_csv(os.path.join(BT, "backtest_formation.csv"))
p08 = pd.read_csv(os.path.join(BT, "backtest_policy.csv"))
B08_FORM = list(f08.sort_values("Formation_Prob_Pct", ascending=False)["Formation_Prob_Pct"])
B08_WHO_F = list(f08.sort_values("Formation_Prob_Pct", ascending=False)["Archetype"])
B08_BURST = list(b08.sort_values("Total_Loss_Severe_$B", ascending=False)["Total_Loss_Severe_$B"])
B08_WHO_B = list(b08.sort_values("Total_Loss_Severe_$B", ascending=False)["Archetype"])
B08_BCR = [(r["Intervention"].replace("Circular-deal disclosure & margin rules", "Disclosure")
            .replace("GPU-collateral lending caps", "Lending caps")
            .replace("Strategic compute reserve / backstop", "Reserve"),
            float(r["BCR_Base"])) for _, r in p08.iterrows() if "Combined" not in str(r["Intervention"])]


def main():
    x = np.arange(4)
    # ---- rank shape: formation + burst, both crises ----
    # Short panel titles: long ones collide mid-figure at this aspect.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].bar(x - 0.2, AI_FORM, 0.4, label="AI stack", color=AI_C, edgecolor="black")
    axes[0].bar(x + 0.2, B08_FORM, 0.4, label="2008 system", color=Y08_C, edgecolor="black")
    for i, (a, b) in enumerate(zip(AI_FORM, B08_FORM)):
        axes[0].text(i - 0.2, a + 1.0, f"{a:.1f}", ha="center", fontsize=7)
        axes[0].text(i + 0.2, b + 1.0, f"{b:.1f}", ha="center", fontsize=7)
    axes[0].set_xticks(x, ["Rank 1", "Rank 2", "Rank 3", "Rank 4"])
    axes[0].set_ylabel("Bubble-condition score (%)")
    axes[0].set_title("(a) Fragility: AI dispersed, 2008 uniformly high",
                      fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].bar(x - 0.2, AI_BURST, 0.4, label="AI stack", color=AI_C, edgecolor="black")
    axes[1].bar(x + 0.2, B08_BURST, 0.4, label="2008 system", color=Y08_C, edgecolor="black")
    for i, (a, b) in enumerate(zip(AI_BURST, B08_BURST)):
        axes[1].text(i - 0.2, a + 6, f"${a:.0f}B", ha="center", fontsize=7)
        axes[1].text(i + 0.2, b + 6, f"${b:.0f}B", ha="center", fontsize=7)
    axes[1].set_xticks(x, ["Rank 1", "Rank 2", "Rank 3", "Rank 4"])
    axes[1].set_ylabel("Severe loss ($B)")
    axes[1].set_title("(b) Loss ranking: the same falling staircase",
                      fontsize=9)
    axes[1].legend(fontsize=8)
    fig.suptitle("Same shape, different patients: rank-ordered fragility and loss",
                 fontsize=10, fontweight="bold")
    fig.text(0.01, 0.0, "Rank 1 = most fragile / largest loss. Occupants differ; "
             "order statistics match. " + FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "compare_rank_shape")

    # ---- remedies: same order, both crises ----
    tools = [t for t, _ in AI_BCR]
    ai_v = [v for _, v in AI_BCR]
    b08_v = [dict(B08_BCR).get(t, float("nan")) for t in tools]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(tools))
    ax.barh(y + 0.2, ai_v, 0.4, label="AI stack (Table 7.13)", color=AI_C, edgecolor="black")
    ax.barh(y - 0.2, b08_v, 0.4, label="2008 system (backtest)", color=Y08_C, edgecolor="black")
    for i, (a, b) in enumerate(zip(ai_v, b08_v)):
        ax.text(a + 1.2, i + 0.2, f"{a:.1f}", va="center", fontsize=8, fontweight="bold")
        ax.text(b + 1.2, i - 0.2, f"{b:.1f}", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(y, tools)
    ax.set_xlabel("Benefit-cost ratio (base case)")
    ax.set_title("Remedy ranking transfers: caps first in both crises")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.0, "Skin-in-the-game (caps) > transparency (disclosure) > backstop, "
             "in 2007 and 2026. " + FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "compare_remedies")

    # ---- tail risk + propagation fingerprint ----
    ai_net = 100 * float(pd.read_csv(os.path.join(TABLES, "table_7.14_debtrank_contagion.csv"))
                         ["Second_Round_Loss_$B"].sum()
                         / (float(pd.read_csv(os.path.join(TABLES, "table_7.11_burst_ranking.csv"))
                                  ["Total_Loss_Severe_$B"].sum())))
    b08_net = 100 * float(b08["Network_Loss_Severe_$B"].sum()
                          / b08["Total_Loss_Severe_$B"].sum())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    axes[0].bar([0, 1], [10.7, 7.87], 0.5, color=[AI_C, Y08_C], edgecolor="black")
    axes[0].bar([2.2, 3.2], [-10.8, -34.57], 0.5, color=[AI_C, Y08_C],
                edgecolor="black", hatch="//")
    axes[0].set_xticks([0, 1, 2.2, 3.2],
                       ["AI mean", "2008 mean", "AI VaR", "2008 VaR"])
    axes[0].set_ylabel("Return (%)")
    axes[0].set_title("(a) Similar pay, triple the tail")
    axes[1].bar(["AI stack", "2008 system"], [ai_net, b08_net], 0.55,
                color=[AI_C, Y08_C], edgecolor="black")
    for i, v in enumerate([ai_net, b08_net]):
        axes[1].text(i, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Second-round share of loss (%)")
    axes[1].set_title("(b) Buffers contain; leverage propagates")
    fig.suptitle("Risk fingerprint: compensation rhymes, tails do not",
                 fontsize=10, fontweight="bold")
    fig.text(0.01, 0.0, "Portfolio sim + burst network term, each vintage's own inputs. "
             + FOOT, fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "compare_tail_risk")

    # ---- summary table ----
    summ = pd.DataFrame([
        {"Metric": "Formation top score", "AI_stack": "82.9 (labs)",
         "System_2008": "74.9 (Investors)", "Reading": "AI more dispersed; 2008 uniformly high"},
        {"Metric": "Rank-1 severe loss", "AI_stack": "$239.4B (Hardware)",
         "System_2008": "$255.9B (Investors)", "Reading": "Same staircase; holders lose most"},
        {"Metric": "Network share of loss", "AI_stack": f"{ai_net:.1f}%",
         "System_2008": f"{b08_net:.1f}%", "Reading": "Thick buffers vs thin capital"},
        {"Metric": "GDP cost share", "AI_stack": "0.452% (scenario)",
         "System_2008": "4.3% (actual)", "Reading": "Amplifiers absent in AI frame"},
        {"Metric": "GDP channel mix", "AI_stack": "Capex-led (86%)",
         "System_2008": "Wealth-led", "Reading": "Two channels fit both mixes"},
        {"Metric": "Top remedy (BCR)", "AI_stack": "Lending caps (57.4)",
         "System_2008": "Lending caps (65.5)", "Reading": "Identical remedy order"},
        {"Metric": "Portfolio VaR 95%", "AI_stack": "-10.8%",
         "System_2008": "-34.6%", "Reading": "Similar mean; triple the tail"},
        {"Metric": "System HHI", "AI_stack": "3,851",
         "System_2008": f'{float(pd.read_csv(os.path.join(BT, "backtest_market.csv"))["Value_2008"].iloc[0]):,.0f}',
         "Reading": "Both highly concentrated"},
        {"Metric": "Rank correlation", "AI_stack": "16/16 OAT; 100% joint",
         "System_2008": "rho=0.80 vs actual", "Reading": "Ordinal machinery transfers"},
    ])
    summ.to_csv(os.path.join(TABLES, "compare_summary.csv"), index=False)
    with open(os.path.join(TABLES, "compare_summary.tex"), "w") as f:
        f.write(summ.to_latex(index=False))
    try:
        summ.to_excel(os.path.join(TABLES, "compare_summary.xlsx"), index=False)
    except Exception as e:
        print("  xlsx skipped:", e)
    print("  wrote tables/compare_summary.csv/.tex/.xlsx")


if __name__ == "__main__":
    main()
