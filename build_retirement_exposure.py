"""Retirement-holder exposure addendum: who holds the wealth-channel loss.

Reads the paper's own severe wealth-channel loss live from
tables/table_7.12_gdp_impact.csv and decomposes it with sourced,
non-pipeline inputs (ICI retirement levels, Fed DFA ownership shares,
Fed heterogeneous-MPC estimates). Nothing here changes the $139.1B
headline: the heterogeneous-MPC recompute is a labeled memo sensitivity,
and the exposure stock is a sourced floor, not a model estimate.

Outputs: tables/backtest/retirement_exposure.csv/.tex/.xlsx and
figures/retirement_exposure.png (embedded in the appendix).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_T = os.path.join(ROOT, "tables", "backtest")
OUT_F = os.path.join(ROOT, "figures")
GDP_CSV = os.path.join(ROOT, "tables", "table_7.12_gdp_impact.csv")

# ---- sourced inputs (levels, not estimates) -------------------------------
# Mag-7 share of S&P 500: ~34% Dec-2025 (DataTrek via MarketWatch),
# 33-35% through Aug-2026 press. 0.34 used; range noted in the PDF.
MAG7_SHARE = 0.34
# ICI Q2-2026: equity funds held in 401(k)s $3.7T; in IRAs $4.8T.
K401_EQUITY_FUNDS_T = 3.7
IRA_EQUITY_FUNDS_T = 4.8
# ICI context levels: 401(k) assets ~$10.1T (Q4-2025); all retirement
# $51.2T (Q2-2026); IRAs $19.9T; govt DB pensions $10.4T; target-date
# funds $4.86T (end-2025).
# Fed DFA (income-ranked): top 10% hold ~87% of equities.
DFA_TOP10 = 0.87
# Top-quintile share is not published as a point number; 0.90-0.93 is the
# band consistent with the sourced top-10% 87%. Stated, not estimated.
TOPQ_BAND = (0.90, 0.93)
# Fed FEDS Notes Aug-2025 (wealth heterogeneity): MPC 0.8c top quintile,
# 7.5c bottom 80%, aggregate ~3.4c.
MPC_TOPQ, MPC_REST = 0.008, 0.075
WEALTH_MPC_UNIFORM = 0.04  # paper's Table 7.12 assumption


def main():
    gdp = pd.read_csv(GDP_CSV)
    sev = gdp[gdp["Severity"] == "severe"].set_index("Channel")
    wealth_uniform = float(sev.loc["Equity wealth effect", "GDP_Loss_$B"])
    capex = float(sev.loc["AI capex cut", "GDP_Loss_$B"])
    combined = float(sev.loc["Combined", "GDP_Loss_$B"])
    mktcap_loss = wealth_uniform / WEALTH_MPC_UNIFORM

    fund_equity = K401_EQUITY_FUNDS_T + IRA_EQUITY_FUNDS_T
    mag7_floor = fund_equity * MAG7_SHARE

    hetero, ret_slice, ret_cons = [], [], []
    for q in TOPQ_BAND:
        eff = q * MPC_TOPQ + (1 - q) * MPC_REST
        hetero.append(mktcap_loss * eff)
        ret_slice.append(mktcap_loss * (1 - q))
        ret_cons.append(mktcap_loss * (1 - q) * MPC_REST)

    rows = [
        ("401(k) equity-fund assets ($T, ICI Q2-2026)", K401_EQUITY_FUNDS_T,
         "S: ICI retirement assets Q2-2026 ($3.7T equity funds in 401(k)s)"),
        ("IRA equity-fund assets ($T, ICI Q2-2026)", IRA_EQUITY_FUNDS_T,
         "S: ICI Q2-2026 ($4.8T equity funds in IRAs)"),
        ("Mag-7 share of S&P 500 (Dec-2025-Aug-2026)", MAG7_SHARE,
         "S: DataTrek/Morningstar via press, 33-35% range; 0.34 used"),
        ("Retirement fund-held Big Tech equity floor ($T)",
         round(mag7_floor, 2),
         "8.5T fund equity x 0.34 market-weight; excludes hybrids, DB "
         "direct holdings, taxable accounts: a floor"),
        ("Top-10% share of equities (Fed DFA)", DFA_TOP10,
         "S: Fed Distributional Financial Accounts via 22V/FH analysis"),
        ("Paper uniform wealth channel, severe ($B)", wealth_uniform,
         "Live: tables/table_7.12_gdp_impact.csv (MPC 0.04)"),
        ("Implied severe market-cap loss ($B)", round(mktcap_loss, 1),
         "19.1 / 0.04: the equity base the 4c MPC applies to"),
        ("Heterogeneous-MPC wealth channel memo ($B)",
         f"{min(hetero):.1f}-{max(hetero):.1f}",
         "Memo: same $477.5B base x (q x 0.008 + (1-q) x 0.075), "
         "q in [0.90, 0.93] anchored on DFA top-10% 87%"),
        ("High-MPC holders' balance-sheet slice ($B)",
         f"{min(ret_slice):.1f}-{max(ret_slice):.1f}",
         "Memo: (1-q) of the market-cap loss sits with 7.5c households"),
        ("That slice's consumption drag ($B)",
         f"{min(ret_cons):.1f}-{max(ret_cons):.1f}",
         "Memo: slice x 0.075; adequacy shock beyond this flow"),
        ("Uniform capex channel, severe ($B)", capex,
         "Live: Table 7.12 (unchanged by this addendum)"),
        ("Combined GDP, uniform ($B)", combined,
         "Live: Table 7.12 headline, unchanged"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value", "Provenance"])
    df.to_csv(os.path.join(OUT_T, "retirement_exposure.csv"), index=False)
    with open(os.path.join(OUT_T, "retirement_exposure.tex"), "w") as f:
        f.write(df.to_latex(index=False))
    try:
        df.to_excel(os.path.join(OUT_T, "retirement_exposure.xlsx"),
                    index=False)
    except Exception as e:
        print("  xlsx skipped:", e)
    print("  wrote tables/backtest/retirement_exposure")

    # NOTE: dollar signs are escaped (\$) throughout: a bare $ opens
    # mathtext and eats inter-word spaces (the "2.9TExposurefloor" bug).
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].bar(["401(k) equity\nfunds", "IRA equity\nfunds"],
                [K401_EQUITY_FUNDS_T, IRA_EQUITY_FUNDS_T],
                color=["#0072B2", "#D55E00"], edgecolor="black")
    axes[0].axhline(mag7_floor, color="#009E73", ls="--", lw=1.5)
    axes[0].text(0.5, mag7_floor + 0.10,
                 rf"Big Tech slice ~\${mag7_floor:.1f}T (x Mag-7 34%)",
                 fontsize=9, color="#009E73", fontweight="bold",
                 ha="center", va="bottom")
    axes[0].set_ylabel(r"Fund-held equity (\$T)")
    axes[0].set_title("Retirement equity base (ICI levels)")

    axes[1].bar(["Uniform 4c\n(paper)", "Heterogeneous MPC\n(memo)"],
                [wealth_uniform, float(np.mean(hetero))],
                yerr=[0, float(np.ptp(hetero)) / 2],
                capsize=4, color=["#0072B2", "#CC79A7"], edgecolor="black")
    axes[1].set_ylabel(r"Severe wealth-channel loss (\$B)")
    axes[1].set_title(r"Same \$477.5B loss, holder-weighted MPC")
    fig.suptitle(r"Who holds the wealth channel: \$2.9T exposure floor, "
                 r"\$6-7B consumption memo",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.0,
             r"Levels: ICI Q2-2026; Mag-7 34% (DataTrek/press range 33-35%); "
             r"DFA top-10% 87%; MPCs Fed FEDS Notes Aug-2025 (0.8c/7.5c). "
             r"Memo only: headline \$139.1B unchanged.",
             fontsize=6, color="0.35")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_F, "retirement_exposure." + ext),
                    bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print("  wrote figures/retirement_exposure.png/.pdf")
    print(f"  memo: hetero wealth ${min(hetero):.1f}-{max(hetero):.1f}B "
          f"vs uniform ${wealth_uniform:.1f}B; "
          f"Mag-7 floor ${mag7_floor:.2f}T")


if __name__ == "__main__":
    main()
