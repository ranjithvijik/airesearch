#!/usr/bin/env python3
"""
build_technical_report.py — DOCX report generator for ai_ecosystem_model.py and its outputs.

What it does
------------
1. STATIC ANALYSIS of ai_ecosystem_model.py (structure: classes/functions/lines; embedded
   data vintage; methodology constants) without executing the heavy run_pipeline.
2. LIGHTWEIGHT RE-ANALYSIS using ai_ecosystem_model.py's own classes (market concentration,
   game-theoretic equilibria, welfare/DWL, cooperative Shapley, circular
   deals) - executed quietly; Monte Carlo and figure rendering are NOT
   repeated (those artifacts are read from disk).
3. ARTIFACT ANALYSIS of the latest ai_ecosystem_model.py outputs: every results table
   (CSV) and every figure (PNG) is inventoried and embedded.
4. INTEGRITY AUDIT: cross-checks data totals, naming conventions, payoff
   invariants and artifact counts; each check is reported pass/warn/fail.
5. Generates a formatted, publication-ready .docx (Times New Roman serif,
   numbered headings, cover page, TOC field, headers/footers with page
   numbers, styled Word tables, embedded figures with captions).
6. ANALYTICAL INTERPRETATION: the report dynamically interprets the results
   it computes rather than restating them — antitrust reading of the HHI,
   the Walled-Garden Nash equilibrium as rational-but-destructive mutual
   defection, the meaning of each deadweight-loss component, Shapley values
   as bargaining power, and the circular-deals bubble score as systemic
   fragility — woven into each section as styled "Analytical insight"
   callouts bound to the actual numbers in the run.

Requirements (Python 3.9+ recommended; the code itself needs 3.7+ -- install
with pip):
  pip install numpy pandas python-docx Pillow
  - numpy / pandas: audit tables and headline recomputation.
  - python-docx: .docx assembly (import name: docx).
  - Pillow: figure downscaling to emailable JPEGs (import name: PIL).
  Plus the ai_ecosystem_model.py REQUIREMENTS above, because build_technical_report.py imports ai_ecosystem_model.py for its
  data providers and analysis classes.
Run:
    python3 build_technical_report.py                 # default: report in ai_ecosystem_model/ directory
    python3 build_technical_report.py --out my.docx   # custom output path
    python3 build_technical_report.py --ai_ecosystem_model-dir PATH   # where ai_ecosystem_model.py + outputs live
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import glob
import io
import logging
import os
import re
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger("build_technical_report")

RUN_LOG_PREFIX = "build_technical_report_run"
RUN_LOG_SUFFIX = ".log"
RUN_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
RUN_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_run_log_file(directory):
    """Attaches a per-run log file mirroring console output; returns its path.

    The file is named ``build_technical_report_run_YYYYMMDD_HHMMSS.log`` inside ``directory``
    (the folder receiving the .docx, so the log sits next to the report).
    The handler uses the shared run-log format at INFO level on the root
    logger, capturing build_technical_report records and any ai_ecosystem_model records emitted during the
    build. Idempotent per path (a same-second collision with an unowned
    file gains a _NN suffix instead of truncating it) and never raises:
    an unwritable location warns on stderr and returns None. Call only
    from main().
    """
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(directory, f"{RUN_LOG_PREFIX}_{stamp}")
        path = base + RUN_LOG_SUFFIX
        root = logging.getLogger()

        def _owned(p):
            ap = os.path.abspath(p)
            return any(isinstance(h, logging.FileHandler) and
                       os.path.abspath(getattr(h, "baseFilename", "")) == ap
                       for h in root.handlers)

        if _owned(path):
            return path
        n = 0
        while os.path.exists(path):
            n += 1
            path = f"{base}_{n:02d}{RUN_LOG_SUFFIX}"
            if _owned(path):
                return path
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(RUN_LOG_FORMAT, datefmt=RUN_LOG_DATEFMT))
        root.addHandler(handler)
        return path
    except Exception as e:
        print(f"WARNING: run log file disabled (cannot write log): {e}", file=sys.stderr)
        return None

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "ai_ecosystem_model.py"
TABLES_DIR = "tables"
PLOTS_DIR = "figures"
# Figure prefixes owned by the prediction/backtest block (stages 6-8). They
# live in figures/ alongside the model exhibits but are embedded in the
# prediction DOCX/HTML (or are backtest diagnostics), never in this report --
# the placement gate must not count them.
PREDICTION_FIGURE_PREFIXES = ("pred_", "backtest_", "compare_", "payoff_", "telecom_",
                                 "retirement_")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "AI_Ecosystem_Game_Theory_Technical_Report.docx")

# --------------------------------------------------------------------------
# Quiet import of ai_ecosystem_model.py (suppress banners/log noise during analysis)
# --------------------------------------------------------------------------

##############################################################################
# 1. RE-ANALYSIS LAYER -- quiet ai_ecosystem_model import plus headline recomputation (Monte Carlo and figure rendering are never repeated here).
##############################################################################

def load_gt(gt_dir: str):
    """Quietly imports ai_ecosystem_model.py from gt_dir (stdout/stderr suppressed) and returns the module, so the report reuses ai_ecosystem_model.py's own data providers and analysis classes.
    """
    sys.path.insert(0, gt_dir)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import ai_ecosystem_model as _gt
    return _gt


def _quiet(fn, *args, **kwargs):
    """Run fn while suppressing stdout/stderr; return (result, captured_out)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        res = fn(*args, **kwargs)
    return res


# --------------------------------------------------------------------------
# Lightweight analysis (mirrors ai_ecosystem_model.py's own run_pipeline, minus MC/viz)
# --------------------------------------------------------------------------
def analyze(ai_ecosystem_model, data):
    """Recomputes headline results from ai_ecosystem_model.py's own classes (no Monte Carlo, no figure rendering).

    Runs MarketStructureAnalyzer (concentration plus market power),
    GameTheoryFramework (payoff construction, pure-strategy Nash, welfare metrics,
    best-effort _classify_game taxonomy), CoordinationFailureAnalyzer,
    WelfareEconomicsAnalyzer, CooperativeGame (Shapley values plus
    grand-coalition surplus) and CircularDealsAnalyzer, and records the
    methodology constants the run actually used (DWL rates, MC
    iterations/noise/confidence, GLOBAL_SEED). Heavy artifacts (MC draws, PNGs)
    are read from disk by later stages, never recomputed here.
    """
    players = data["players"]
    deps = data["dependencies"]
    meta = data["metadata"]
    out = {"players": players, "deps": deps, "meta": meta}

    # Market structure
    ma = ai_ecosystem_model.MarketStructureAnalyzer(players)
    out["concentration"] = _quiet(ma.calculate_market_concentration)
    out["market_power"] = _quiet(ma.calculate_market_power)

    # Strategic games
    gf = ai_ecosystem_model.GameTheoryFramework(players)
    out["matrices"] = _quiet(gf.construct_payoff_matrices)
    nash = {}
    for name, matrix in out["matrices"].items():
        p1, p2 = [p.strip() for p in name.split("-")]
        eq_list = gf._find_pure_strategy_nash_real(matrix, p1, p2, verbose=False)
        welfare = gf._calculate_welfare_metrics_real(matrix, eq_list, p1, p2, verbose=False)
        nash[name] = {"players": (p1, p2), "matrix": matrix,
                      "nash_equilibria": eq_list, "welfare_metrics": welfare}
        # game-type classification (best-effort)
        try:
            payoffs_list = [[tuple(map(float, t)) for t in row] for row in matrix.tolist()]
            p1d, p2d = gf._find_dominant_strategies(payoffs_list)
            ne_pos = eq_list[0]["position"] if eq_list else (-1, -1)
            nash[name]["game_type"] = gf._classify_game(
                payoffs_list, ne_pos, welfare.get("pareto_position", (-1, -1)), p1d, p2d)
        except Exception:
            nash[name]["game_type"] = "n/a"
    gf.nash_equilibria = nash
    out["nash"] = nash

    ca = ai_ecosystem_model.CoordinationFailureAnalyzer(gf)
    _quiet(ca.identify_coordination_failures)
    out["coordination"] = ca.coordination_failures
    out["total_coordination_dwl"] = ca.total_coordination_dwl

    # Welfare
    wa = ai_ecosystem_model.WelfareEconomicsAnalyzer(gf, players, dependencies_df=deps)
    out["welfare"] = _quiet(wa.calculate_aggregate_welfare_loss)

    # Cooperative game
    coop = ai_ecosystem_model.CooperativeGame(data)
    _quiet(coop.calculate_shapley_values)
    try:
        coop_surplus = coop.characteristic_function(coop.players) - sum(coop.player_values.values())
    except Exception:
        coop_surplus = np.nan
    out["shapley"] = dict(sorted(coop.shapley_values.items(), key=lambda kv: -kv[1]))
    out["coop_surplus"] = coop_surplus
    out["grand_coalition"] = coop.characteristic_function(coop.players)

    # Circular deals
    circ = ai_ecosystem_model.CircularDealsAnalyzer()
    out["circular"] = circ.calc_metrics()

    # A8 order-robustness battery (live seeded joint check). Section 7.5
    # formats its prose from this dict, never from hardcoded percentages, so
    # the report cannot restate stale robustness numbers.
    try:
        _rev = players.set_index('Player_Category')['Current_Revenue_Billions']
        _shares = (_rev / _rev.sum()).to_dict()
        _bb = ai_ecosystem_model.BubbleBurstAnalyzer(
            players, circ, dependencies_df=deps, hhi_shares=_shares,
            output_dir='.')
        _gf_se = ai_ecosystem_model.GameTheoryFramework(players)
        _quiet(_gf_se.construct_payoff_matrices)
        _quiet(_gf_se.find_nash_equilibria)
        _se = ai_ecosystem_model.StructuralEstimationAnalyzer(
            _gf_se, _bb, output_dir='.')
        out["joint_order"] = _quiet(_se.joint_order_check)
    except Exception as e:
        out["joint_order"] = {"error": str(e)}

    out["dwl_pct"] = ai_ecosystem_model.DWL_PCT_TOPDOWN
    out["dwl_billions"] = ai_ecosystem_model.ABSTRACT_DWL_BILLIONS
    # Game-level DWL% actually used for payoff construction (may differ from
    # the top-down rate; see ai_ecosystem_model.py DWL naming conventions).
    out["game_dwl_pct"] = float(getattr(gf, "game_dwl_pct", np.nan))
    out["mc_iterations"] = int(ai_ecosystem_model.MONTE_CARLO_ITERATIONS)
    out["mc_noise"] = float(ai_ecosystem_model.PERTURBATION_STD)
    out["mc_confidence"] = float(ai_ecosystem_model.CONFIDENCE_LEVEL)
    out["global_seed"] = int(getattr(ai_ecosystem_model, "GLOBAL_SEED", 42))
    return out


# --------------------------------------------------------------------------
# Static / integrity analysis of ai_ecosystem_model.py source + artifacts
# --------------------------------------------------------------------------

##############################################################################
# 2. STATIC PROFILE & INTEGRITY AUDIT -- source metrics plus pass/warn/fail artifact checks.
##############################################################################

def code_profile(gt_path: str) -> dict:
    """Static regex profile of ai_ecosystem_model.py source: line/byte counts, class, top-level-function and method inventories, plus the CONSOLIDATED version banner (or 'n/a'). Pure text scan -- ai_ecosystem_model.py is never executed.
    """
    with open(gt_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    lines = src.splitlines()
    classes = sorted(set(re.findall(r"^class\s+(\w+)", src, re.M)))
    # top-level functions only (col 0 defs)
    funcs = sorted(set(re.findall(r"^def\s+(\w+)", src, re.M)))
    methods = sorted(set(re.findall(r"^    def\s+(\w+)", src, re.M)))
    n_bytes = os.path.getsize(gt_path)
    # Look for a version banner like 'CONSOLIDATED V8.0'
    version_m = re.search(r"CONSOLIDATED\s+(V[\d.]+)", src)
    return {"lines": len(lines), "bytes": n_bytes, "classes": classes,
            "functions": funcs, "methods": methods,
            "banner": version_m.group(1) if version_m else "n/a"}


def integrity_checks(ai_ecosystem_model, data, prof: dict, tables_dir: str, plots_dir: str) -> list:
    """Cross-validates data, naming, payoffs and artifacts; returns audit checks.

    Each check is a dict(code, title, status, detail) with status pass/warn/fail:
    player/dependency totals vs embedded data, table/figure naming conventions,
    payoff-matrix invariants (shapes, finite entries), and on-disk artifact counts
    vs the figure/table registries. Reported in the console log and Appendix A.
    """
    players, deps, meta = data["players"], data["dependencies"], data["metadata"]
    checks = []

    def add(code, title, ok, detail):
        """Records one audit check; '~'-prefixed details downgrade failure to WARN."""
        checks.append({"code": code, "title": title,
                       "status": "PASS" if ok else ("WARN" if detail and detail.startswith("~") else "FAIL"),
                       "detail": str(detail).lstrip("~")})

    rev_sum = float(players["Current_Revenue_Billions"].sum())
    val_sum = float(players["Market_Valuation_Billions"].sum())
    # D2 reconciles the COMMITTED booking taxonomy against metadata: the
    # tracked frame also carries prospective/vertical-integration rows that
    # are disclosed, not booked (Booking_Status column; default 'committed'
    # for frames predating the taxonomy).
    _status = deps["Booking_Status"] if "Booking_Status" in deps.columns else "committed"
    dep_sum = float(deps.loc[_status == "committed", "Dependency_Value_Billions"].sum())

    add("D1", "Players revenue total is positive and plausible (< $5T)",
        rev_sum > 0 and rev_sum < 5000, f"${rev_sum:,.2f}B across {len(players)} categories")
    add("D2", "Sum of committed dependency deals matches metadata circular exposure",
        abs(dep_sum - float(meta.get("circular_exposure_billions", 0))) < 1e-6,
        f"committed deals ${dep_sum:,.2f}B vs metadata ${float(meta.get('circular_exposure_billions', 0)):,.2f}B")
    add("D3", "Sum of category valuations matches metadata total market cap",
        abs(val_sum - float(meta.get("total_ai_market_cap_billions", 0))) < 1e-6,
        f"valuations ${val_sum:,.2f}B vs metadata ${float(meta.get('total_ai_market_cap_billions', 0)):,.2f}B")
    payoff_players = set(data["payoffs"]["Player"].unique())
    cat_players = set(players["Player_Category"])
    add("D4", "Payoff-matrix players match player categories (naming invariant)",
        payoff_players == cat_players,
        f"payoffs {sorted(payoff_players)} vs categories {sorted(cat_players)}")
    req_cols = {"Player_Category", "Current_Revenue_Billions", "Market_Valuation_Billions",
                "Operating_Margin", "Current_Profitability", "Strategic_Risk",
                "Circular_Dependency_Index"}
    missing = req_cols - set(players.columns)
    add("D5", "Players DataFrame has all required analysis columns",
        not missing, f"missing: {sorted(missing) or 'none'}")
    valid_prof = {"High Positive", "Break-even", "Mixed"}
    bad_prof = set(players["Current_Profitability"].dropna().unique()) - valid_prof
    add("D6", "Current_Profitability values are within the defined taxonomy",
        not bad_prof, f"unexpected values: {sorted(bad_prof) or 'none'}")

    # payoff invariant: DD cell == observed revenue pair
    gf = ai_ecosystem_model.GameTheoryFramework(players)
    matrices = _quiet(gf.construct_payoff_matrices)
    rev_map = players.set_index("Player_Category")["Current_Revenue_Billions"].to_dict()
    mismatches = []
    for name, m in matrices.items():
        p1, p2 = [p.strip() for p in name.split("-")]
        dd = m[1, 1]
        if abs(float(dd[0]) - float(rev_map[p1])) > 1e-6 or abs(float(dd[1]) - float(rev_map[p2])) > 1e-6:
            mismatches.append(f"{name}: DD={dd} vs rev=({rev_map[p1]},{rev_map[p2]})")
    add("G1", "Payoff invariant: Defect-Defect cell equals observed revenue for all games",
        not mismatches, f"{len(matrices)} matrices checked; " + ("; ".join(mismatches[:3]) or "no mismatch"))

    n_crit = (deps["Risk_Level"] == "Critical").sum()
    add("D7", "Dependency risk levels parse (Critical deals identified)",
        n_crit >= 1, f"{n_crit} Critical deal(s) of {len(deps)} total")

    # artifact inventory
    n_tables = len(glob.glob(os.path.join(tables_dir, "table_*.csv")))
    n_xlsx = len(glob.glob(os.path.join(tables_dir, "table_*.xlsx")))
    n_tex = len(glob.glob(os.path.join(tables_dir, "table_*.tex")))
    n_figs = len(glob.glob(os.path.join(plots_dir, "*.png")))
    add("A1", "Table artifacts present (CSV / Excel / LaTeX)",
        n_tables >= 1 and n_xlsx == n_tables and n_tex == n_tables,
        f"{n_tables} CSV, {n_xlsx} XLSX, {n_tex} TEX in {os.path.basename(tables_dir)}/")
    # A2 checks registry coverage (not a hardcoded count): every figure the
    # report places must exist on disk. Count-independent since Sep 2026
    # consolidation (27 -> 16 publication figures).
    _placed_figs = {fn for files in SECTION_FIGURES.values() for fn in files}
    _missing_figs = sorted(fn for fn in _placed_figs
                           if not os.path.exists(os.path.join(plots_dir, fn)))
    add("A2", "All report-placed figures exist on disk",
        not _missing_figs, f"{n_figs} PNG on disk; "
        f"{len(_placed_figs)} placed, missing: {_missing_figs or 'none'}")
    add("C1", "ai_ecosystem_model.py parses as importable module", True,
        f"{prof['lines']:,} lines, {len(prof['classes'])} classes, "
        f"{len(prof['functions'])} top-level functions, {len(prof['methods'])} methods, {prof['banner']}")

    # literature-diagnostics tables (table_7.x): presence + internal consistency
    t7 = sorted(glob.glob(os.path.join(tables_dir, "table_7.*.csv")))
    add("T7", "Literature-diagnostics tables present (7.1–7.9)",
        len(t7) >= 9, f"{len(t7)} table_7.*.csv file(s)")
    try:
        _gp = os.path.join(tables_dir, "table_7.2_gatekeeper_index.csv")
        if os.path.exists(_gp):
            _g = pd.read_csv(_gp)
            _gs = float(pd.to_numeric(_g.get("Gatekeeper_Share_Pct"), errors="coerce").sum())
            add("G4", "Gatekeeper shares sum to 100%",
                abs(_gs - 100.0) < 1.0, f"sum={_gs:.1f}%")
        else:
            add("G4", "Gatekeeper shares sum to 100%", False, "~table_7.2 absent (stale outputs)")
    except Exception as e:
        add("G4", "Gatekeeper shares sum to 100%", False, f"unreadable: {e}")
    try:
        _sp = os.path.join(tables_dir, "table_7.6_policy_synergy.csv")
        if os.path.exists(_sp):
            _s = pd.read_csv(_sp)
            _r = float(_s.get("Synergy_Ratio", pd.Series([np.nan])).iloc[0]) if len(_s) else np.nan
            add("G5", "Policy synergy ratio finite",
                pd.notna(_r) and np.isfinite(_r), f"ratio={_r:.3f}" if pd.notna(_r) else "NaN")
        else:
            add("G5", "Policy synergy ratio finite", False, "~table_7.6 absent (stale outputs)")
    except Exception as e:
        add("G5", "Policy synergy ratio finite", False, f"unreadable: {e}")
    try:
        _ap = os.path.join(tables_dir, "table_7.8_equilibrium_audit.csv")
        if os.path.exists(_ap):
            _a = pd.read_csv(_ap)
            _ok = bool(len(_a)) and bool(_a.get("Position_Match", pd.Series([False])).all()) \
                and bool(_a.get("Count_Match", pd.Series([False])).all())
            add("G2", "Independent Nash audit matches on all games",
                _ok, f"{len(_a)} games audited")
        else:
            add("G2", "Independent Nash audit matches on all games", False, "~table_7.8 absent (stale outputs)")
    except Exception as e:
        add("G2", "Independent Nash audit matches on all games", False, f"unreadable: {e}")
    try:
        _cp = os.path.join(tables_dir, "table_7.9_mc_convergence.csv")
        if os.path.exists(_cp):
            _c = pd.read_csv(_cp)
            _bad = [g for g, a in zip(_c.get("Game", []), _c.get("Assessment", []))
                    if a not in ("adequate", "check")]
            add("G3", "MC convergence adequate/check on all games",
                not _bad, "all adequate-or-better" if not _bad else f"needs draws: {_bad}")
        else:
            add("G3", "MC convergence adequate/check on all games", False, "~table_7.9 absent (stale outputs)")
    except Exception as e:
        add("G3", "MC convergence adequate/check on all games", False, f"unreadable: {e}")
    return checks


# ==========================================================================
# DOCX rendering helpers
# ==========================================================================
from docx import Document  # noqa: E402
from docx.shared import Pt, Inches, RGBColor  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

# Publication palette (matches ai_ecosystem_model.py RESEARCH_COLORS)
INK = RGBColor(0x1A, 0x1A, 0x1A)
PRIMARY = RGBColor(0x2C, 0x3E, 0x50)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
GOOD = RGBColor(0x1E, 0x7B, 0x34)
WARN = RGBColor(0xB8, 0x6E, 0x00)
BAD = RGBColor(0xB0, 0x30, 0x30)
ANALYTICAL_INSIGHT = RGBColor(0x18, 0x4A, 0x71)  # deep blue for analytical insights

FIGURE_CAPTIONS = {
    "figure_1.png": "Conceptual framework (a) and literature map (b)",
    "figure_2.png": "Theory-to-empirics run_pipeline (a) and five-stage game (b)",
    "figure_3.png": "2025 revenue with 2030 projection; measured HHI decomposition; market shares",
    "figure_4.png": "AI value chain with vertical integration (incl. data layer)",
    "figure_5.png": "Bilateral strategic payoff matrices (2x2 games)",
    "figure_6.png": "Core game: equilibrium matrix (a) and joint payoffs by cell (b)",
    "figure_7.png": "Welfare composite: Nash vs joint-surplus maximum, totals, DWL sources, per-game efficiency",
    "figure_8.png": "Monte Carlo convergence diagnostics",
    "figure_9.png": "Sensitivity response curve: DWL vs DWL% assumption (9 steps)",
    "figure_10.png": "Sobol sensitivity indices",
    "figure_11.png": "Disequilibrium trajectories (a) and stability scores (b)",
    "figure_12.png": "Policy intervention simulation",
    "figure_13.png": "Revenue projections to 2030",
    "figure_14.png": "Dependency network and systemic risk",
    "figure_15.png": "Portfolio risk analysis",
    "figure_16.png": "Shapley value allocation",
    "figure_17.png": "Company valuation dashboard: multiples, dependency, sustainability, ratings",
    "figure_18.png": "Company valuation screen: 23-company metrics table",
}

# (figure filename, its caption) placements per section
# (consolidated Sep 2026: 16 publication figures; composites carry (a)/(b) panels)
SECTION_FIGURES = {
    "conceptual": ["figure_1.png", "figure_2.png",
                   "figure_4.png"],
    "market": ["figure_3.png",
               "figure_13.png"],
    "games": ["figure_5.png",
              "figure_6.png", "figure_11.png"],
    "welfare": ["figure_7.png",
                "figure_12.png"],
    "robustness": ["figure_8.png", "figure_9.png",
                   "figure_10.png"],
    "risk": ["figure_14.png",
             "figure_15.png",
             "figure_16.png"],
    "valuation": ["figure_17.png",
                  "figure_18.png"],
}

# Which report section number each figure belongs to (for the register/verifier)
SECTION_NUM = {"conceptual": 2, "market": 3, "games": 4, "welfare": 5,
               "robustness": 6, "risk": 7, "valuation": 8}
FIGURE_SECTION = {fn: SECTION_NUM[sec]
                  for sec, files in SECTION_FIGURES.items() for fn in files}

# ai_ecosystem_model.py's own results tables, embedded verbatim in the matching narrative
# section (keys = ai_ecosystem_model table ids used in tables_index.csv).
GT_TABLES_BY_SECTION = {
    2: ["2.4", "8_5", "8_6"],                       # player categories + assumption register + validated inputs
    3: ["1.1", "1.3_proj"],                       # market structure, revenue projections
    4: ["3.1", "3.2", "3.4"],                    # payoff matrices, core-game equilibrium & DWL
    5: ["5.3", "5.4", "6.1_6.3"],                 # welfare, dynamic, policy
    6: ["4.1", "8_1", "8_2", "8_3", "8_4"],       # MC & sensitivity, robustness battery
    7: ["5.4_coop", "A.0_network", "A.1_network_risk",
        "A.2_portfolio_risk", "A.3_success_scores"],  # Shapley + appendix risk tables
}
# Clean display labels for ai_ecosystem_model table ids (underscored identifiers -> table numbers)
TABLE_DISPLAY = {
    "1.3_proj": "1.3", "5.4_coop": "5.4b",
    "6.1_6.3": "6.1\u20136.3", "A.0_network": "A.0",
    "A.1_network_risk": "A.1", "A.2_portfolio_risk": "A.2",
    "A.3_success_scores": "A.3",
    "8_1": "8.1", "8_2": "8.2", "8_3": "8.3", "8_4": "8.4", "8_5": "8.5", "8_6": "8.6",
    "8_7": "8.7",
    "7_14": "7.14", "7_15": "7.15",
    "7.16_market_share_evolution": "7.16",
    "7.17_lerner_markup_summary": "7.17",
    "7.18_shapley_vs_revenue_share": "7.18",
}



##############################################################################
# 3. DOCX PRIMITIVES -- tables, payoff exhibits, figures, fields, insight callouts.
##############################################################################

def set_cell_bg(cell, hex_fill):
    """Fills a Word table cell via raw w:shd XML -- python-docx exposes no cell-shading API, so this drops one level to the underlying OXML element.
    """
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def mark_header_row(table):
    """Marks a table's first row as a repeating header (w:tblHeader) so Word
    repeats it at the top of every page a long table spans.
    """
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        hdr = OxmlElement("w:tblHeader")
        hdr.set(qn("w:val"), "true")
        tcPr.append(hdr)


def _fmt(v):
    """Display formatter for table cells: None/NaN render as '', ±inf as ∞/-∞ (repeated-game tables carry genuine +inf critical deltas), integral floats as grouped integers, other floats to 2dp, everything else via str().
    """
    if v is None:
        return ""
    if isinstance(v, float) and (pd.isna(v) or np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        if np.isinf(v):
            return "\u221e" if v > 0 else "-\u221e"
        if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
            return f"{v:,.0f}"
        return f"{v:,.2f}"
    return str(v)


def add_word_table(doc, df, caption=None, note=None, max_cols=7, transpose_wide=True):
    """Renders a DataFrame as a styled Word table; returns the table (None if empty).

    Wide frames are transposed (first column becomes headers) so they fit the page;
    captions render navy/bold with keep_with_next, notes italic grey with a
    transposition disclaimer when applied. Header row is white-on-navy, body rows
    zebra-striped, and font size steps down (10/9/8pt) as column count grows.
    """
    if df is None or df.empty:
        return None
    work = df.copy()
    transposed = False
    if transpose_wide and work.shape[1] > max_cols and work.shape[0] <= work.shape[1]:
        try:
            work = work.set_index(work.columns[0]).T.reset_index()
            transposed = True
        except Exception:
            pass
    if work.shape[1] > max_cols * 1.5:
        pass

    if caption:
        cap = doc.add_paragraph()
        cap.paragraph_format.space_before = Pt(10)
        cap.paragraph_format.space_after = Pt(2)
        cap.paragraph_format.keep_with_next = True
        r = cap.add_run(caption)
        r.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = PRIMARY
    if note:
        pn = doc.add_paragraph()
        pn.paragraph_format.space_after = Pt(2)
        rn = pn.add_run(note)
        rn.italic = True
        rn.font.size = Pt(8.5)
        rn.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
        if transposed:
            rn.text += "  (Transposed so columns fit the page width.)"

    n_rows, n_cols = work.shape
    table = doc.add_table(rows=n_rows + 1, cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    mark_header_row(table)
    body_pt = 8 if n_cols > 7 else (8.5 if n_cols > 4 else 9.5)
    header_pt = body_pt + 0.5

    # header
    for j, col in enumerate(work.columns):
        cell = table.cell(0, j)
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(str(col))
        run.bold = True
        run.font.size = Pt(header_pt)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(cell, "2C3E50")

    # body
    for i in range(n_rows):
        for j in range(n_cols):
            val = work.iloc[i, j]
            cell = table.cell(i + 1, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(_fmt(val))
            run.font.size = Pt(body_pt)
            if i % 2 == 1:
                set_cell_bg(cell, "EDF1F5")
    # spacing after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)
    sp.add_run("").font.size = Pt(2)
    return table


def add_payoff_exhibit(doc, game_no, name, res, s1, s2):
    """Renders one bilateral 2x2 payoff matrix ($B) with NE/joint-surplus annotation.

    Builds a 3x3 exhibit grid (strategy labels on the margins, payoff pairs in the
    cells) under an 'Exhibit 4.N' caption, with the Nash cell(s) and the Pareto
    cell annotated so the equilibrium-vs-optimum gap is visible at a glance.
    """
    matrix = res.get("matrix")
    if matrix is None:
        return
    eq = res.get("nash_equilibria", [])
    w = res.get("welfare_metrics", {})
    ne_pos = tuple(eq[0]["position"]) if eq else None
    po_pos = tuple(w.get("pareto_position", (-1, -1)))
    p1, p2 = res["players"]
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(10)
    cap.paragraph_format.space_after = Pt(2)
    cap.paragraph_format.keep_with_next = True
    r = cap.add_run(f"Exhibit 4.{game_no}  Payoff matrix — {name} ($B; row: {p1}, column: {p2})")
    r.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = PRIMARY
    table = doc.add_table(rows=3, cols=3)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    mark_header_row(table)
    labels = [["", s2[0], s2[1]], [s1[0], None, None], [s1[1], None, None]]
    for i in range(3):
        for j in range(3):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            if i == 0 or j == 0:
                run = p.add_run(str(labels[i][j]))
                run.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_bg(cell, "2C3E50")
                continue
            try:
                a = float(matrix[i - 1, j - 1][0])
                b = float(matrix[i - 1, j - 1][1])
                txt = f"({a:,.1f}, {b:,.1f})"
            except Exception:
                txt = "—"
            tags = []
            if ne_pos == (i - 1, j - 1):
                tags.append("NE")
            if po_pos == (i - 1, j - 1):
                tags.append("Joint-Max (K-H)")
            if tags:
                txt += f" [{'/'.join(tags)}]"
            run = p.add_run(txt)
            run.font.size = Pt(9)
            if ne_pos == (i - 1, j - 1):
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_bg(cell, "1F4E79")
            elif po_pos == (i - 1, j - 1):
                set_cell_bg(cell, "D9EAD3")
            elif (i + j) % 2 == 0:
                set_cell_bg(cell, "EDF1F5")
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)
    sp.add_run("").font.size = Pt(2)


def add_figure(doc, path, title, number=None, width_in=None, cache_dir=None,
               caption_prefix=None, alt_text=None):
    """Embeds a figure PNG with its caption; returns the paragraph.

    Large PNGs are downscaled to JPEG in cache_dir before embedding so the .docx
    stays emailable; an unreadable image falls back to default dimensions instead
    of failing the build, and the caption is always emitted. caption_prefix
    overrides the "Figure N: " lead (supplemental figures use their own lead);
    alt_text (default: the full caption) is stored as the image's wp:docPr
    description so screen readers have something to announce -- both best
    effort, never fatal to the build.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        w, h = 1600, 1200
    if width_in is None:
        aspect = w / max(h, 1)
        if aspect > 1.5:
            width_in = 6.3
        elif aspect < 0.85:
            width_in = 4.6
        else:
            width_in = 5.8
    if cache_dir is not None:
        # build a downscaled JPEG so the docx stays reasonable
        key = os.path.basename(path)
        jpg_path = os.path.join(cache_dir, key.rsplit(".", 1)[0] + ".jpg")
        if not os.path.exists(jpg_path):
            from PIL import Image as _I
            with _I.open(path) as im:
                im = im.convert("RGB")
                scale = 1800 / max(im.size)
                if scale < 1:
                    im = im.resize((int(im.width * scale), int(im.height * scale)),
                                   _I.LANCZOS)
                im.save(jpg_path, "JPEG", quality=90, dpi=(300, 300))
        embed_path = jpg_path
    else:
        embed_path = path

    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.keep_with_next = True
    par.paragraph_format.space_before = Pt(8)
    run = par.add_run()
    shape = run.add_picture(embed_path, width=Inches(width_in))

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    if caption_prefix is None:
        label = f"Figure {number}: " if number else "Figure: "
    else:
        label = caption_prefix
    cr = cap.add_run(f"{label}{title}")
    cr.italic = True
    cr.font.size = Pt(9)
    cr.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    try:
        docPr = shape._inline.find(qn("wp:docPr"))
        if docPr is not None:
            docPr.set("descr", alt_text if alt_text else f"{label}{title}"[:250])
    except Exception:
        pass


def add_field(par, instr):
    """Inserts a Word field code (PAGE, NUMPAGES, TOC) as raw w:fldSimple XML -- the mechanism behind automatic page numbers and the refreshable table of contents.
    """
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), instr)
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    par._p.append(fld)


# ==========================================================================
# Analytical-insight rendering
# ==========================================================================
def _shade_paragraph(p, hex_fill, border_hex):
    """Applies a background fill plus border to a paragraph's OXML -- the shaded 'Analytical insight' callout boxes.
    """
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    pPr.append(shd)
    pBdr = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")       # ~2.25 pt rule
    left.set(qn("w:space"), "4")
    left.set(qn("w:color"), border_hex)
    pBdr.append(left)
    pPr.append(pBdr)


def add_insights(doc, insights, section_label="Insights"):
    """Renders (heading, body) insight pairs as shaded callouts under a section label; no-op on empty input so sections without insights stay clean.
    """
    if not insights:
        return
    # Section-level label paragraph
    lab = doc.add_paragraph()
    lab.paragraph_format.space_before = Pt(10)
    lab.paragraph_format.space_after = Pt(2)
    lab.paragraph_format.keep_with_next = True
    lr = lab.add_run(f"Analytical insights — {section_label}")
    lr.bold = True
    lr.font.size = Pt(10.5)
    lr.font.color.rgb = ANALYTICAL_INSIGHT
    for heading, body in insights:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.12)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(7)
        _shade_paragraph(p, "EAF1F8", "1F4E79")
        rh = p.add_run(f"{heading}.  ")
        rh.bold = True
        rh.font.size = Pt(10)
        rh.font.color.rgb = ANALYTICAL_INSIGHT
        rb = p.add_run(body)
        rb.font.size = Pt(10)
        rb.font.color.rgb = RGBColor(0x1A, 0x2B, 0x3C)


def _pct(x, default=0.0):
    """Converts a fraction to a percent float; anything unparseable yields default.
    """
    try:
        return float(x) * 100.0
    except Exception:
        return default


def _exhibit_value(gt_dir, filename, label_substr):
    """Read a ai_ecosystem_model.py 2-column summary table and return the first numeric value
    whose row label contains label_substr (case-insensitive), else None."""
    try:
        df = pd.read_csv(os.path.join(gt_dir, TABLES_DIR, filename))
        name_col, val_col = df.columns[0], df.columns[1]
        rows = df[df[name_col].astype(str).str.contains(
            label_substr, case=False, na=False)]
        if not rows.empty:
            v = pd.to_numeric(rows[val_col].iloc[0], errors="coerce")
            if pd.notna(v):
                return float(v)
    except Exception:
        pass
    return None


def add_module_analysis(doc, spec):
    """Renders one Module → Outputs → Inference → Implication block as a shaded
    4-row table; no-op on empty spec. "Module" names the ai_ecosystem_model.py analyzer(s),
    "Outputs" the exhibits, "Inference" what the numbers say, "Implication"
    what follows for policy or research.
    """
    if not spec:
        return
    rows = [("Module", spec.get("module", "")),
            ("Outputs", spec.get("outputs", "")),
            ("Inference", spec.get("inference", "")),
            ("Implication", spec.get("implication", ""))]
    t = doc.add_table(rows=4, cols=2)
    t.style = "Table Grid"
    t.autofit = True
    for i, (k, v) in enumerate(rows):
        c0, c1 = t.cell(i, 0), t.cell(i, 1)
        c0.width, c1.width = Inches(1.25), Inches(5.25)
        c0.text, c1.text = "", ""
        rk = c0.paragraphs[0].add_run(k)
        rk.bold = True
        rk.font.size = Pt(10)
        rk.font.color.rgb = PRIMARY
        set_cell_bg(c0, "EDF1F5")
        rv = c1.paragraphs[0].add_run(v)
        rv.font.size = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def build_insights(ai_ecosystem_model, analysis, gt_dir):
    """Composes data-bound analytical insight callouts per report section.

    Returns a dict keyed market/games/welfare/robustness/shapley/circular/
    valuation/policy, each a list of (heading, body) pairs whose numbers are drawn
    from the recomputed analysis (never hardcoded), so the narrative cannot drift
    from the run's actual results.
    """
    ins = {"market": [], "games": [], "welfare": [], "robustness": [],
           "shapley": [], "circular": [], "valuation": [], "policy": []}
    players = analysis["players"]
    conc = analysis["concentration"]
    nash = analysis["nash"]
    welfare = analysis["welfare"]
    deps = analysis["deps"]

    # ---------- market structure / antitrust -------------------------------
    hhi = float(conc.get("HHI", 0))
    cr4 = float(conc.get("CR4", 0)) * 100.0
    gini = float(conc.get("Gini", 0))
    tot_rev = float(conc.get("total_revenue_billions", 0))
    shares = conc.get("market_shares", {})
    cloud_s = _pct(shares.get("Cloud Providers", 0))
    hw_s = _pct(shares.get("Hardware", 0))
    if hhi > 2500:
        hhi_read = ("presumptively concentrated under the DOJ/FTC Merger Guidelines "
                    "(HHI above 2,500 on the 2010 vintage — and above 1,800 on the "
                    "current 2023 vintage — invites structural scrutiny), so unilateral "
                    "conduct by the largest archetypes — let alone further "
                    "consolidation — would be the first thing a competition "
                    "authority examines.")
    elif hhi > 1500:
        hhi_read = ("moderately concentrated by the Merger Guidelines (1,500–2,500 on "
                    "the 2010 vintage; 1,000–1,800 on 2023), meaning deals between "
                    "the leaders would face case-by-case review rather than a "
                    "presumption.")
    else:
        hhi_read = ("below the 1,500 'unconcentrated' threshold, though the coarse "
                    "four-player lens understates intra-category concentration.")
    ins["market"].append(
        ("Antitrust temperature", 
         f"The HHI of {hhi:,.0f} is {hhi_read} The top two archetypes — Cloud "
         f"Providers ({cloud_s:.1f}%) and Hardware ({hw_s:.1f}%) — account for "
         f"{cloud_s + hw_s:.1f}% of measured revenue (CR4 = {cr4:.1f}%), a structure "
         f"consistent with a 'gatekeeper' market in which access, not price, is the "
         f"competitive margin."))
    ins["market"].append(
        ("Inequality as concentration",
         f"A Gini coefficient of {gini:.3f} (with only four players) shows that the "
         f"distribution is dominated by the two infrastructure archetypes: the "
         f"revenue share gap between Cloud/Hardware and the model/wrapper layers is "
         f"the structural source of the bargaining asymmetries examined later."))
    mp = analysis.get("market_power") or {}
    if mp:
        best_cat = max(mp, key=lambda k: mp[k].get("lerner_index") or 0)
        best = mp[best_cat]
        lern = best.get("lerner_index") or 0
        mkup = best.get("markup_percent") or 0
        ins["market"].append(
            ("Where the pricing power sits",
             f"The highest implied Lerner index is {lern:.3f} ({best_cat}, ~{mkup:.0f}% "
             f"markup over marginal cost). A Lerner value in this range implies "
             f"inelastic demand and strong switching costs — precisely the "
             f"preconditions for the vertical hold-up games modeled in Section 4."))
    # ---------- games / walled-garden -------------------------------------
    core = nash.get("Hardware-Cloud Providers", {})
    core_w = core.get("welfare_metrics", {})
    core_eq = core.get("nash_equilibria", [])
    core_type = core.get("game_type", "n/a")
    if core_eq:
        s1, s2 = core_eq[0]["strategies"]
        ins["games"].append(
            ("The core game: rational, mutually assured enclosure",
             f"The Hardware–Cloud Providers equilibrium ({s1}, {s2}) is a classic "
             f"mutual-defection trap: each side walls off its customer base because "
             f"the other would capture the full ecosystem if it stayed open. It is "
             f"'{core_type}': individually rational, jointly destructive. Neither "
             f"player can unilaterally open up without conceding rents, so the "
             f"lock-in persists even though a cooperative (open/interoperable) "
             f"outcome would raise combined welfare by "
             f"${core_w.get('deadweight_loss', 0):,.1f}B "
             f"(efficiency {core_w.get('efficiency_ratio', 0):.1f}%)."))
    pd_games = [n for n, r in nash.items() if r.get("game_type") == "Prisoner's Dilemma"]
    eff_games = [n for n, r in nash.items()
                 if (r.get("welfare_metrics", {}).get("deadweight_loss", 1) or 1) < 1e-9]
    if pd_games or eff_games:
        ins["games"].append(
            ("Heterogeneous game landscape",
             f"Only {'the ' + ', '.join(pd_games) if pd_games else 'no game'} is a "
             f"pure Prisoner's Dilemma; "
             f"{len(eff_games)} of {len(nash)} games reach an efficient equilibrium "
             f"with zero deadweight loss. The strategic problem is therefore not a "
             f"uniform race-to-the-bottom but a targeted failure concentrated in the "
             f"infrastructure layer, which narrows the policy response to the "
             f"specific bottleneck rather than blanket regulation."))
    # ---------- secondary games: what the other five games say ----------------
    from collections import Counter
    gtypes = Counter(r.get("game_type", "n/a") for r in nash.values())
    if gtypes:
        dist = "; ".join(f"{t} ×{c}" for t, c in sorted(gtypes.items()))
        non_core = [(n, r) for n, r in nash.items() if n != "Hardware-Cloud Providers"]
        non_core_dwl = [(n, float(r.get("welfare_metrics", {}).get("deadweight_loss", 0)))
                        for n, r in non_core]
        worst2 = sorted(non_core_dwl, key=lambda x: -x[1])[:2]
        core_dwl = float(core_w.get("deadweight_loss", 0))
        live = [(n, d) for n, d in worst2 if d > 1e-9]
        if live:
            top_n, top_d = live[0]
            ratio = top_d / max(core_dwl, 1e-9)
            conc_txt = (f"The largest secondary loss is {top_n} (${top_d:,.1f}B, "
                        f"about {ratio:.0%} of the core game's "
                        f"${core_dwl:,.1f}B) — the welfare problem is concentrated, "
                        f"not diffuse.")
        else:
            conc_txt = ("No secondary game produces material loss — the entire "
                        "welfare problem sits in the core infrastructure game.")
        others = "; ".join(f"{n} (${d:,.1f}B)" for n, d in live[1:])
        ins["games"].append(
            ("Beyond the core: the rest of the game landscape",
             f"Across all six games the type distribution is {dist}. {conc_txt}"
             f"{(' Next: ' + others + '.') if others else ''} For publication "
             f"purposes this concentration is a feature: a single-bottleneck "
             f"diagnosis yields a falsifiable claim (fix the infrastructure game, "
             f"recover most of the welfare) rather than a vague 'markets are "
             f"inefficient' verdict."))
    # ---------- welfare / DWL ---------------------------------------------
    tot_dwl = float(welfare.get("total_dwl", 0))
    eff = float(welfare.get("aggregate_efficiency", 0))
    pareto = float(welfare.get("total_pareto_welfare", 0))
    comp = welfare.get("dwl_by_source", {}) or {}
    if comp and tot_dwl > 0:
        top_src = max(comp, key=lambda k: comp[k])
        top_val = float(comp[top_src])
        share = top_val / tot_dwl * 100.0
        src_labels = {"coordination_failures": "coordination failure",
                      "monopoly_pricing": "monopoly pricing",
                      "innovation_distortion": "innovation distortion",
                      "quality_degradation": "quality degradation",
                      "switching_costs": "switching costs"}
        ins["welfare"].append(
            ("What the deadweight loss is made of",
             f"{src_labels.get(top_src, top_src).capitalize()} is the single largest "
             f"component of the ${tot_dwl:,.1f}B aggregate DWL "
             f"({share:.0f}%). Because the mix differs by source, so does the remedy: "
             f"a coordination-dominated loss points to interoperability and "
             f"standard-setting, whereas a switching-cost-dominated loss points to "
             f"portability mandates. Treating all ${tot_dwl:,.1f}B as one 'market "
             f"failure' number would mislead policy design."))
    ins["welfare"].append(
        ("Efficiency gap and the value of coordination",
         f"The system operates at {eff:.1f}% of its potential ({pareto:,.1f}B vs. "
         f"observed {welfare.get('total_nash_welfare', 0):,.1f}B). Coordination "
         f"failures total "
         f"${analysis.get('total_coordination_dwl', 0):,.1f}B gross across the six "
         f"games — shared players are counted repeatedly, which is why this "
         f"exceeds the top-down aggregate. The loss is not an exogenous shock but "
         f"an equilibrium outcome that cooperation, standard-setting, or mediated "
         f"bargaining could in principle recover."))
    # ---------- policy: why the winner wins ---------------------------------
    pol = load_policy_table(gt_dir)
    if pol is not None and "Net_Benefit_$B_Base" in pol.columns:
        try:
            nb = pd.to_numeric(pol["Net_Benefit_$B_Base"], errors="coerce")
            ranked = pol.assign(_nb=nb).dropna(subset=["_nb"]).sort_values("_nb", ascending=False)
            if len(ranked):
                win = ranked.iloc[0]
                wnb = float(win["_nb"])
                wdwl = float(pd.to_numeric(pd.Series([win.get("DWL_Reduction_$B_Base")]),
                                           errors="coerce").iloc[0] or 0)
                wcost = float(pd.to_numeric(pd.Series([win.get("Cost_$B_Base")]),
                                            errors="coerce").iloc[0] or 0)
                wbcr = pd.to_numeric(pd.Series([win.get("BCR_Base")]), errors="coerce").iloc[0]
                recov = wdwl / max(tot_dwl, 1e-9) * 100.0
                fit = ("fits the diagnosis exactly: the loss is a coordination "
                       "failure in the infrastructure layer, and interoperability "
                       "attacks that layer directly rather than taxing or "
                       "splitting it")
                if "nteroperab" not in str(win.get("Policy", "")):
                    fit = ("is notable precisely because it does not target the "
                           "infrastructure bottleneck head-on — which invites the "
                           "committee question of whether the ranking survives "
                           "under the optimistic/pessimistic scenarios")
                ins["policy"].append(
                    ("Why this intervention wins",
                     f"{win.get('Policy')} leads on base-scenario net benefit "
                     f"(${wnb:,.1f}B): it recovers ${wdwl:,.1f}B of deadweight "
                     f"loss ({recov:.0f}% of the ${tot_dwl:,.1f}B aggregate) at a "
                     f"cost of ${wcost:,.1f}B"
                     f"{f' (benefit–cost ratio {wbcr:.1f})' if pd.notna(wbcr) else ''}. "
                     f"This {fit}. The policy lesson is proportionality: the cheapest "
                     f"effective remedy dominates because the DWL pool is large "
                     f"relative to any plausible intervention cost."))
                if len(ranked) > 1:
                    runner = ranked.iloc[1]
                    gap = wnb - float(runner["_nb"])
                    ins["policy"].append(
                        ("How decisive the ranking is",
                         f"The runner-up ({runner.get('Policy')}) trails by "
                         f"${gap:,.1f}B in net benefit. A {'wide' if gap > wnb * 0.25 else 'narrow'} "
                         f"margin means the recommendation is "
                         f"{'robust to scenario choice — check the optimistic/conservative '
                            'columns to confirm' if gap > wnb * 0.25 else 'sensitive: the '
                            'optimistic and conservative scenarios in Exhibit 5.2b should '
                            'decide, not the base case alone'}."))
        except Exception:
            pass
    # ---------- validation: equilibrium audit + MC convergence ------------------
    try:
        _ea_p = os.path.join(gt_dir, TABLES_DIR, "table_7.8_equilibrium_audit.csv")
        if os.path.exists(_ea_p):
            _ea = pd.read_csv(_ea_p)
            if len(_ea):
                _pm = int(_ea.get("Position_Match", pd.Series([False] * len(_ea))).sum())
                _cm = int(_ea.get("Count_Match", pd.Series([False] * len(_ea))).sum())
                ins["robustness"].append(
                    ("Equilibria independently cross-validated",
                     f"Every reported Nash equilibrium was re-enumerated by an "
                     f"independent brute-force implementation (direct dominance checks, "
                     f"no shared code with the solver): {_pm}/{len(_ea)} positions match, "
                     f"{_cm}/{len(_ea)} counts match. "
                     f"{'The game solutions are verified, not merely computed.' if _pm == len(_ea) and _cm == len(_ea) else 'MISMATCHES REQUIRE INVESTIGATION before citing any equilibrium.'}"))
        _mc_p = os.path.join(gt_dir, TABLES_DIR, "table_7.9_mc_convergence.csv")
        if os.path.exists(_mc_p):
            _mcc = pd.read_csv(_mc_p)
            if len(_mcc):
                _ad = (_mcc.get("Assessment", pd.Series(["?"] * len(_mcc))).tolist())
                _worst = [a for a in _ad if a not in ("adequate", "check")]
                ins["robustness"].append(
                    ("Simulation converged by the book",
                     f"Per-game Monte-Carlo standard errors and split-half consistency "
                     f"were computed from the stored draw distributions: "
                     f"{', '.join(f'{g}={a}' for g, a in zip(_mcc.get('Game', []), _ad))}. "
                     f"R-hat is deliberately not used — for IID draws it is ~1 by "
                     f"construction and proves nothing (Vehtari et al. 2021). "
                     f"{'All games adequate or check-level.' if not _worst else 'Games needing more draws: ' + ', '.join(_worst) + '.'}"))
    except Exception:
        pass
    # ---------- synergy: test the complementarity assumption ------------------
    try:
        _syn_p = os.path.join(gt_dir, TABLES_DIR, "table_7.6_policy_synergy.csv")
        if os.path.exists(_syn_p):
            _syn = pd.read_csv(_syn_p)
            if len(_syn):
                _r = _syn.iloc[0]
                _ratio = float(_r.get("Synergy_Ratio", np.nan))
                _reading = str(_r.get("Reading", "n/a"))
                if pd.notna(_ratio):
                    ins["policy"].append(
                        ("Complementarity tested, not assumed",
                         f"The combined intervention recovers "
                         f"${float(_r.get('Combined_DWL_Reduction_$B', 0)):,.1f}B against "
                         f"${float(_r.get('Sum_Standalone_DWL_Reduction_$B', 0)):,.1f}B summed "
                         f"standalone (ratio {_ratio:.3f}: {_reading}). "
                         f"{('The levers OVERLAP — the Combined ranking in Exhibit 5.2b wins '
                            'on absolute net benefit, but stacking all four buys less than the '
                            'sum of its parts. A committee should ask whether a two- or '
                            'three-lever bundle dominates the full package.') if _ratio < 0.95 else
                           ('The levers are genuinely complementary — the package is worth more '
                            'than its parts, strengthening the Combined recommendation.') if _ratio > 1.05 else
                           ('The package is roughly additive — no synergy premium, no overlap '
                            'penalty.')}"
                         f" This is the second-best logic of Section 2.3 made quantitative."))
    except Exception:
        pass
    # ---------- robustness ------------------------------------------------
    try:
        # Table 4.2 folded into 4.1 (Sep 2026): calibration rows live in table_4.1.csv.
        sens_df = pd.read_csv(os.path.join(gt_dir, TABLES_DIR, "table_4.1.csv"))
        # table_4.1.csv is a 2-column (Parameter, Value) table; "Elasticity" is a row label.
        name_col, val_col = sens_df.columns[0], sens_df.columns[1]
        el_row = sens_df[sens_df[name_col].astype(str).str.contains(
            "Elasticity", case=False, na=False)]
        if not el_row.empty:
            el = pd.to_numeric(el_row[val_col].iloc[0], errors="coerce")
            if pd.notna(el):
                ins["robustness"].append(
                    ("How sensitive the headline loss is",
                     f"The sensitivity elasticity is {float(el):.2f}: a 1% change in the "
                     f"market-implied DWL assumption moves the dollar deadweight loss by "
                     f"about {float(el):.2f}%. Results of this leverage should be read as "
                     f"orders of magnitude, not decimals — the directional story (which "
                     f"games fail, which sources dominate) is stable even if the point "
                     f"estimate of ${tot_dwl:,.1f}B carries wide uncertainty."))
    except Exception:
        pass
    try:
        mc_df = pd.read_csv(os.path.join(gt_dir, TABLES_DIR, "table_4.1.csv"))
        # table_4.1.csv is a 2-column (Parameter, Value) table; stability is a row label.
        name_col, val_col = mc_df.columns[0], mc_df.columns[1]
        stab_row = mc_df[mc_df[name_col].astype(str).str.contains(
            "Stability", case=False, na=False)]
        if not stab_row.empty:
            stab = pd.to_numeric(stab_row[val_col].iloc[0], errors="coerce")
            if pd.notna(stab):
                qual = ("highly robust" if stab >= 90 else
                        "moderately robust" if stab >= 70 else "sensitive to payoff noise")
                ins["robustness"].append(
                    ("Equilibrium stability under noise",
                     f"Across the Monte-Carlo draws the equilibrium reproduces in "
                     f"{stab:.1f}% of perturbed payoffs ({qual}). Low stability would "
                     f"signal knife-edge payoffs where small shocks — a price change, "
                     f"a new chip generation — flip the game; high stability says the "
                     f"defect-lock-in is structural, not incidental."))
    except Exception:
        pass
    # ---------- Shapley / bargaining --------------------------------------
    shp = analysis.get("shapley") or {}
    if shp:
        rev_map = players.set_index("Player_Category")["Current_Revenue_Billions"].to_dict()
        tot_shp = sum(shp.values()) or 1e-9
        tot_rev_all = sum(rev_map.values()) or 1e-9
        rows = []
        for k, v in shp.items():
            rows.append((k, v, v / tot_shp * 100.0, rev_map.get(k, 0) / tot_rev_all * 100.0))
        top = max(rows, key=lambda r: r[1])
        ratio = (top[1] / tot_shp) / max(top[3] / 100.0, 1e-9)
        ins["shapley"].append(
            ("Shapley values as bargaining power",
             f"{top[0]} captures the largest share of cooperative surplus "
             f"({top[2]:.1f}% of the Shapley allocation) — about {ratio:.2f}× its "
             f"revenue share ({top[3]:.1f}%). A Shapley share far above size means "
             f"the player is a bottleneck: others' gains depend on its participation, "
             f"which is the formal counterpart of the informal 'you need me more than "
             f"I need you' power the market-structure results implied."))
        gap_rows = sorted(rows, key=lambda r: r[2] - r[3])
        low = gap_rows[0]
        if low[2] - low[3] < -5:
            ins["shapley"].append(
                ("Who is underpaid by the current equilibrium",
                 f"{low[0]} receives only {low[2]:.1f}% of cooperative surplus despite "
                 f"a {low[3]:.1f}% revenue share — the largest negative gap. Under the "
                 f"observed non-cooperative equilibrium this player is effectively "
                 f"subsidising the ecosystem's growth without capturing the "
                 f"corresponding marginal value, a standing invitation to integrate "
                 f"vertically or defect."))
    surplus = analysis.get("coop_surplus")
    if surplus is not None and np.isfinite(surplus):
        ins["shapley"].append(
            ("The price of non-cooperation",
             f"Grand-coalition cooperation would create roughly "
             f"${surplus:,.0f}B of surplus over standalone play — the amount the "
             f"ecosystem leaves on the table by insisting on bilateral, adversarial "
             f"relationships rather than joint investment."))
    # ---------- circular deals / fragility --------------------------------
    cm = analysis.get("circular") or {}
    bubble = cm.get("bubble_score")
    if bubble is not None:
        if bubble >= 0.7:
            bub_read = "elevated fragility: growth is substantially self-referential"
        elif bubble >= 0.5:
            bub_read = "material fragility: a meaningful share of demand is circular"
        else:
            bub_read = "moderate fragility: circularity is present but not dominant"
        ins["circular"].append(
            ("Bubble score and self-referential demand",
             f"The circular-deals bubble score of {float(bubble):.3f} signals {bub_read}. "
             f"Score construction weights concentrated, Critical deals most heavily, "
             f"so it is best read as a fragility index: how much of today's revenue "
             f"depends on counterparties paying each other rather than on final "
             f"customers."))
    try:
        _fr_p = os.path.join(gt_dir, TABLES_DIR, "table_7.5_fragility_composite.csv")
        if os.path.exists(_fr_p):
            _fr = pd.read_csv(_fr_p)
            if len(_fr):
                _f = _fr.iloc[0]
                _cols = [c for c in _fr.columns if c not in ("Composite", "Band")]
                _vals = {c: float(_f.get(c, 0) or 0) for c in _cols}
                _top = max(_vals, key=_vals.get)
                _chan = ("concentrated circular exposure" if "ircular" in _top
                         else "upstream dependence and market structure"
                         if "ingle" in _top or "oncentration" in _top
                         else "realized inefficiency")
                ins["circular"].append(
                    (f"Fragility composite: {_f.get('Band', 'n/a')}, and where it comes from",
                     f"The five-channel fragility composite is "
                     f"{float(_f.get('Composite', 0)):.3f} "
                     f"({_f.get('Band', 'n/a')}). Its largest channel is "
                     f"{_top} — so the fragility story is about {_chan}, "
                     f"consistent with the bubble-score reading above rather than a "
                     f"separate alarm."))
    except Exception:
        pass
    # NVIDIA-centricity of the deal fabric
    if deps is not None and len(deps):
        text_all = (deps["Dependent_Player"].astype(str) + " " +
                    deps["Dependency_On"].astype(str))
        nv_mask = text_all.str.contains("NVIDIA|Nvidia", regex=True)
        nv_val = float(deps.loc[nv_mask, "Dependency_Value_Billions"].sum())
        tot_dep = float(deps["Dependency_Value_Billions"].sum())
        if tot_dep > 0:
            ins["circular"].append(
                ("A single-point-of-failure fabric",
                 f"{nv_val / tot_dep * 100:.0f}% of circular deal value "
                 f"(${nv_val:,.0f}B of ${tot_dep:,.0f}B) touches NVIDIA — through "
                 f"GPU supply, vendor financing, or equity. Concentration of that "
                 f"kind means a shock to one balance sheet (a demand pause, a "
                 f"financing retrenchment, a chip-design slip) propagates through "
                 f"the debt and equity loops rather than being absorbed by a "
                 f"diversified supplier base."))

    # ---------- valuation / risk-adjustment ---------------------------------
    tot_val = float(players["Market_Valuation_Billions"].sum())
    circ_exposure = (float(deps["Dependency_Value_Billions"].sum())
                     if deps is not None and len(deps) else 0.0)
    if tot_val > 0:
        circ_share = circ_exposure / tot_val * 100.0
        ins["valuation"].append(
            ("Valuations are not yet circularity-adjusted",
             f"Market values sum to ${tot_val:,.0f}B across the four archetypes, of "
             f"which ${circ_exposure:,.0f}B ({circ_share:.1f}%) is circular exposure — "
             f"revenue that is effectively vendor-financed rather than end-customer "
             f"demand. Standard DCF and multiples do not penalise future cash flows for "
             f"interlocking dependency, so headline valuations overstate pure equity "
             f"value; the circular-risk adjustments in this section restate them on a "
             f"de-levered, stress-tested basis."))
    return ins


def _read_2col(gt_dir, filename):
    """Read a 2-column Metric/Value (or Parameter/Value) table into a dict."""
    try:
        df = pd.read_csv(os.path.join(gt_dir, TABLES_DIR, filename))
        if len(df.columns) >= 2:
            return dict(zip(df[df.columns[0]].astype(str),
                            df[df.columns[1]].astype(str)))
    except Exception:
        pass
    return {}


def build_module_analyses(analysis, gt_dir):
    """Module-by-module analysis blocks: Module → Outputs → Inference → Implication.

    Numbers are read from the run's own tables (never hardcoded), mirroring
    build_insights. Keys: market/games/welfare/policy/robustness/risk/
    circular/valuation.
    """
    mod = {}
    players = analysis["players"]
    conc = analysis["concentration"]
    nash = analysis["nash"]
    welfare = analysis["welfare"]
    deps = analysis["deps"]

    hhi = float(conc.get("HHI", 0))
    tot_rev = float(conc.get("total_revenue_billions", 0))
    shares = conc.get("market_shares", {}) or {}
    top2 = sum(sorted([float(v) for v in shares.values()], reverse=True)[:2]) * 100.0
    mod["market"] = {
        "module": "MarketStructureAnalyzer, ElasticitySensitivityAnalyzer, RevenueProjection",
        "outputs": "Tables 1.1 (structure), 1.3 (2030 projections), 7.1 (concentration); Figure 3 (revenue, HHI, shares).",
        "inference": (f"HHI {hhi:,.0f} on ${tot_rev:,.0f}B revenue with the top two archetypes at "
                      f"{top2:.1f}%: a gatekeeper structure where access, not price, is the competitive margin."),
        "implication": ("Screen consolidation first and price conduct second: interoperability and "
                        "portability remedies target the access margin directly, while breakup analysis "
                        "starts from the measured, not assumed, bottleneck.")}

    core = nash.get("Hardware-Cloud Providers", {}) or {}
    core_w = core.get("welfare_metrics", {}) or {}
    core_eq = core.get("nash_equilibria", []) or []
    s_txt = (f"{core_eq[0]['strategies'][0]} / {core_eq[0]['strategies'][1]}" if core_eq else "n/a")
    n_pd = sum(1 for r in nash.values() if r.get("game_type") == "Prisoner's Dilemma")
    mod["games"] = {
        "module": "GameTheoryFramework, CoordinationFailureAnalyzer",
        "outputs": "Tables 3.1, 3.2 (payoff matrices), 3.4 (equilibrium & DWL); Figures 5 (all matrices), 6 (core game).",
        "inference": (f"Core equilibrium ({s_txt}) wastes ${float(core_w.get('deadweight_loss', 0)):,.1f}B; "
                      f"{n_pd} of {len(nash)} games are pure Prisoner's Dilemmas — failure is concentrated "
                      f"in the infrastructure layer, not diffuse."),
        "implication": ("Regulate the bottleneck, not the ecosystem: a single-game diagnosis gives a falsifiable "
                        "claim (fix the core game, recover most welfare) instead of blanket platform rules.")}

    tot_dwl = float(welfare.get("total_dwl", 0))
    eff = float(welfare.get("aggregate_efficiency", 0))
    comp = welfare.get("dwl_by_source", {}) or {}
    top_src, top_v = (max(comp.items(), key=lambda kv: float(kv[1])) if comp else ("n/a", 0))
    mod["welfare"] = {
        "module": "WelfareEconomicsAnalyzer",
        "outputs": "Tables 3.4 (core + aggregate), 5.3 (decomposition); Figure 7 (welfare composite).",
        "inference": (f"${tot_dwl:,.1f}B aggregate loss at {eff:.1f}% efficiency, led by "
                      f"{str(top_src).replace('_', ' ')} (${float(top_v):,.1f}B)."),
        "implication": ("Match remedy to source: switching-cost dominance argues portability mandates; "
                        "coordination share argues standards and interop — one DWL number would misdirect both.")}

    pol = load_policy_table(gt_dir)
    win_txt = "n/a"
    try:
        if pol is not None and "Net_Benefit_$B_Base" in pol.columns:
            nb = pd.to_numeric(pol["Net_Benefit_$B_Base"], errors="coerce")
            w = pol.assign(_nb=nb).dropna(subset=["_nb"]).sort_values("_nb", ascending=False).iloc[0]
            bcr = pd.to_numeric(pd.Series([w.get("BCR_Base")]), errors="coerce").iloc[0]
            win_txt = (f"{w.get('Policy')} (net ${float(w['_nb']):,.1f}B"
                       f"{f', BCR {float(bcr):.1f}' if pd.notna(bcr) else ''})")
    except Exception:
        pass
    mod["policy"] = {
        "module": "PolicyInterventionAnalyzer",
        "outputs": "Table 6.1/6.3 (simulation summary); Figure 12 (cost-benefit).",
        "inference": f"Base-case winner: {win_txt}. The DWL pool dwarfs any plausible intervention cost.",
        "implication": ("Proportionality governs: fund the cheapest effective lever first, and re-check the "
                        "ranking under optimistic/conservative scenarios before legislating.")}

    t41 = _read_2col(gt_dir, "table_4.1.csv")
    stab_txt = t41.get("Nash Equilibrium Stability (%)", "n/a")
    elas_txt = t41.get("Elasticity", "n/a")
    top_st, top_st_v = "n/a", "n/a"
    try:
        s48 = pd.read_csv(os.path.join(gt_dir, TABLES_DIR, "table_8.4.csv"))
        if len(s48) and "ST" in s48.columns:
            r = s48.sort_values("ST", ascending=False).iloc[0]
            top_st, top_st_v = r["Parameter"], f"{float(r['ST']):.2f}"
    except Exception:
        pass
    mod["robustness"] = {
        "module": "MonteCarloSimulator, SensitivityAnalysis, RobustnessBattery, StructuralEstimationAnalyzer",
        "outputs": "Table 4.1 (MC + calibration), Tables 8.1–8.4 (validation battery); Figures 8–10.",
        "inference": (f"Equilibria reproduce in {stab_txt}% of perturbed payoffs; DWL elasticity {elas_txt}; "
                      f"top total-order driver {top_st} (ST={top_st_v}). Ranks hold while levels move."),
        "implication": ("Cite ranks and orders of magnitude, not point estimates; any welfare claim that "
                        "collapses outside the ±20% band is not publication-grade.")}

    shp = analysis.get("shapley") or {}
    shp_txt = "n/a"
    if shp:
        top_k = max(shp, key=shp.get)
        rev_map = players.set_index("Player_Category")["Current_Revenue_Billions"].to_dict()
        t_shp, t_rev = sum(shp.values()) or 1e-9, sum(rev_map.values()) or 1e-9
        shp_txt = (f"{top_k} takes {shp[top_k] / t_shp * 100:.1f}% of cooperative surplus on a "
                   f"{rev_map.get(top_k, 0) / t_rev * 100:.1f}% revenue share")
    mod["risk"] = {
        "module": "NetworkRiskAnalysis, PortfolioRiskAnalysis, SuccessProbabilityModel, CooperativeGame",
        "outputs": "Tables 5.4b (Shapley), A.1–A.3 (network/portfolio/success); Figures 14–16.",
        "inference": f"{shp_txt} — the bottleneck's bargaining power exceeds its size.",
        "implication": ("Bargaining and merger review should price bottleneck participation, and stress tests "
                        "should shock the central node first: contagion starts where substitution is hardest.")}

    n_deals = len(deps) if deps is not None else 0
    dep_tot = float(deps["Dependency_Value_Billions"].sum()) if deps is not None and len(deps) else 0
    form_txt, burst_txt, gdp_txt = "n/a", "n/a", "n/a"
    try:
        f10 = pd.read_csv(os.path.join(gt_dir, TABLES_DIR, "table_7.10_bubble_formation.csv"))
        if len(f10):
            r = f10.sort_values("Probability_Pct", ascending=False).iloc[0]
            form_txt = f"{r['Archetype']} {float(r['Probability_Pct']):.1f}%"
    except Exception:
        pass
    try:
        f11 = pd.read_csv(os.path.join(gt_dir, TABLES_DIR, "table_7.11_burst_ranking.csv"))
        if len(f11):
            r = f11.sort_values("Severe_Burst_Loss_$B", ascending=False).iloc[0]
            burst_txt = f"{r['Archetype']} ${float(r['Severe_Burst_Loss_$B']):,.1f}B"
    except Exception:
        pass
    try:
        f12 = pd.read_csv(os.path.join(gt_dir, TABLES_DIR, "table_7.12_gdp_impact.csv"))
        sev = f12[f12["Scenario"].astype(str).str.contains("Severe", na=False)]
        if len(sev):
            gdp_txt = (f"${float(sev['GDP_Cost_$B'].iloc[0]):,.1f}B "
                       f"({float(sev['GDP_Cost_Pct'].iloc[0]):.3f}% of GDP)")
    except Exception:
        pass
    mod["circular"] = {
        "module": "CircularDealsAnalyzer, BubbleBurstAnalyzer, DebtRankClearingEngine, CreditMonitor",
        "outputs": "Tables 7.10–7.13 (formation, burst, GDP, mitigation); deal-level dependency frame.",
        "inference": (f"{n_deals} deals, ${dep_tot:,.0f}B circular exposure; formation peaks at {form_txt}; "
                      f"severe burst hits {burst_txt}; GDP cost {gdp_txt}."),
        "implication": ("Monitor frontier labs where round-tripping is densest, and pre-position disclosure "
                        "and lending-cap tools: burst-mitigation BCRs above 6 make waiting expensive.")}

    val_txt = "n/a"
    try:
        vdf = pd.read_csv(os.path.join(gt_dir, TABLES_DIR, "enhanced_valuation_metrics.csv"))
        if len(vdf):
            vc = vdf["screen_flag"].value_counts()
            nb = int(vc.get("Positive", 0)); ns = int(vc.get("Negative", 0)); nh = int(vc.get("Neutral", 0))
            nv = vdf.loc[vdf["company_name"] == "NVIDIA"].iloc[0]
            val_txt = (f"{len(vdf)} companies screen {nb} Positive / {ns} Negative / {nh} Neutral; "
                       f"NVIDIA {float(nv['market_cap']):,.0f}B at {float(nv['pe_ratio']):.1f}x")
    except Exception:
        pass
    mod["valuation"] = {
        "module": "EnhancedValuationAnalyzer, ValuationMetricsCalculator",
        "outputs": "enhanced_valuation_metrics.csv; valuation dashboard and table figures.",
        "inference": f"{val_txt}; loss-makers screen N/A rather than a fabricated multiple.",
        "implication": ("Multiples discipline the story: infrastructure cash flows justify large caps while "
                        "private-round pricing (Anthropic, Databricks, xAI) flags where repricing risk lives.")}
    return mod



##############################################################################
# 4. DOCUMENT THEME & ASSEMBLY -- Times serif styles plus the build_report sections.
##############################################################################

def configure_styles(doc):
    """Applies the publication theme: Times New Roman throughout, 11pt/1.15 Normal in ink, navy bold H1-H3 with paper spacing.
    """
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.font.color.rgb = INK
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)

    def style_font(name, size, bold=True, color=PRIMARY, italic=False):
        """Applies the Times house style (size/weight/color) to one Word style."""
        st = styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.italic = italic
        st.font.color.rgb = color
        return st

    h1 = style_font("Heading 1", 16)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(8)
    h2 = style_font("Heading 2", 13)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)
    h3 = style_font("Heading 3", 11.5, bold=True, color=ACCENT, italic=False)
    h3.paragraph_format.space_before = Pt(8)
    h3.paragraph_format.space_after = Pt(4)


def add_h1(doc, text):
    """Appends a styled level-1 chapter heading (e.g. '4. Strategic Interaction').
    """
    h = doc.add_heading(text, level=1)
    return h


def add_h2(doc, text):
    """Appends a styled level-2 section heading within a chapter.
    """
    return doc.add_heading(text, level=2)


def add_h3(doc, text):
    """Appends a styled level-3 (subsection) heading.
    """
    return doc.add_heading(text, level=3)


def para(doc, text="", size=11, bold=False, italic=False, align=None,
         color=None, space_after=6):
    """Appends a run-formatted paragraph (size/bold/italic/alignment/color/spacing); empty text yields a spacer.
    """
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        if color:
            r.font.color.rgb = color
    return p


def bullet(doc, text, level=0, size=10.5):
    """Appends a bulleted paragraph (level 0/1 selects the bullet style).
    """
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.font.size = Pt(size)
    return p


# ==========================================================================
# Document assembly
# ==========================================================================
def build_report(ai_ecosystem_model, analysis, prof, checks, gt_dir, out_path):
    """Assembles the full paper DOCX; returns {path, placement, issues}.

    Lays down headers/footers (running head, PAGE/NUMPAGES), cover page, live
    abstract, refreshable TOC field, numbered sections 1-9 plus appendices (every
    ai_ecosystem_model table and figure embedded with captions), then front matter and word count.
    Each embedded table/figure appends a (label, section, kind) entry to the
    placement log consumed by verify_placement(); build problems accumulate in
    issues instead of aborting the document.
    """
    players, deps, meta = analysis["players"], analysis["deps"], analysis["meta"]
    conc = analysis["concentration"]
    nash = analysis["nash"]
    welfare = analysis["welfare"]
    _core_ne = _core_ne_strategies(nash)

    doc = Document()
    configure_styles(doc)
    for section in doc.sections:
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.top_margin = Inches(0.9)
        section.bottom_margin = Inches(0.9)

    # ---- headers & footers ------------------------------------------------
    header = doc.sections[0].header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("AI Ecosystem Game Theory Analysis — Technical Report")
    hr.font.size = Pt(8.5)
    hr.italic = True
    hr.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    footer = doc.sections[0].footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    tabs = fp.paragraph_format.tab_stops
    tabs.add_tab_stop(Inches(3.25), WD_TAB_ALIGNMENT.CENTER)
    tabs.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    r0 = fp.add_run("AI Ecosystem Game Theory — Technical Report\t")
    r0.font.size = Pt(8.5)
    r0.italic = True
    r0.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    r1 = fp.add_run("Page ")
    r1.font.size = Pt(9)
    add_field(fp, "PAGE")
    r2 = fp.add_run(" of ")
    r2.font.size = Pt(9)
    add_field(fp, "NUMPAGES")
    r3 = fp.add_run("\t" + str(meta.get("data_vintage", "")))
    r3.font.size = Pt(8.5)
    r3.italic = True
    r3.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    for r in fp.runs:
        if not r.font.size:
            r.font.size = Pt(9)

    # temp dir for downscaled JPEG embeds (keeps the .docx size sane)
    tmpdir_ctx = tempfile.TemporaryDirectory(prefix="gt_rpt_")
    jpeg_cache_dir = tmpdir_ctx.name

    # ---- cover page -------------------------------------------------------
    for _ in range(5):
        para(doc, "", space_after=0)
    para(doc, "AI ECOSYSTEM GAME THEORY ANALYSIS", 26, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, color=PRIMARY, space_after=2)
    para(doc, "Market Structure, Strategic Interaction, and Welfare", 16, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, color=ACCENT, space_after=18)
    para(doc, "Technical & Validation Report", 14, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)
    para(doc, "Ranjith Keerikkattil", 13, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, color=PRIMARY, space_after=24)
    meta_lines = [
        ("Report date", datetime.now().strftime("%B %d, %Y")),
        ("Data vintage", meta.get("data_vintage", "n/a")),
        ("Data module version", players.attrs.get("data_version", "n/a")),
        ("Script", f"{MODEL_FILE} ({prof['lines']:,} lines; {prof['banner']})"),
        ("Analysis engine", "ai_ecosystem_model.py embedded data + analysis classes (recomputed for this report)"),
        ("Scope", "4 player archetypes · 6 bilateral games · circular-deals & welfare analysis"),
        ("Units", "Monetary values in $B unless noted; September 2026 vintage"),
    ]
    mt = doc.add_table(rows=len(meta_lines), cols=2)
    mt.style = "Table Grid"
    for i, (k, v) in enumerate(meta_lines):
        c0, c1 = mt.cell(i, 0), mt.cell(i, 1)
        c0.text = ""
        rk = c0.paragraphs[0].add_run(k)
        rk.bold = True
        rk.font.size = Pt(10)
        set_cell_bg(c0, "EDF1F5")
        c1.text = ""
        rv = c1.paragraphs[0].add_run(str(v))
        rv.font.size = Pt(10)
    para(doc, "", space_after=0)
    para(doc, "WORKING PAPER", 12, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, color=PRIMARY, space_after=6)
    para(doc, "Prepared for journal submission. All market figures are dated "
              "Q2/Q3 2026 and cross-validated from primary sources (company filings, SEC, "
              "tier-1 financial media).", 9, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, color=RGBColor(0x60, 0x60, 0x60))
    doc.add_page_break()

    # ---- abstract (numbers recomputed live from ai_ecosystem_model.py outputs) ----------------
    add_h1(doc, "Abstract")
    _a_rev = float(players["Current_Revenue_Billions"].sum())
    para(doc,
         f"This paper presents a computational analysis of strategic interaction "
         f"in the artificial-intelligence ecosystem through game theory, welfare "
         f"economics, and market-concentration analysis. Four player archetypes — "
         f"Hardware, Cloud Providers, Foundation Models, and LLM Wrappers, with "
         f"aggregate revenue of ${_a_rev:,.1f}B (Q2/Q3 2026 vintage) — interact as six "
         f"bilateral 2×2 games solved for pure-strategy Nash equilibria by "
         f"best-response analysis, with welfare measured against the joint-surplus maximum (Kaldor–Hicks benchmark).")
    para(doc,
         f"The market is highly concentrated (HHI {conc['HHI']:,.0f}, CR4 "
         f"{conc['CR4']*100:.1f}%). Aggregate welfare at the Nash outcome is "
         f"${welfare.get('total_nash_welfare', 0):,.1f}B against a potential "
         f"${welfare.get('total_pareto_welfare', 0):,.1f}B — a deadweight loss of "
         f"${welfare.get('total_dwl', 0):,.1f}B "
         f"({welfare.get('aggregate_efficiency', 0):.1f}% efficiency). The core "
         f"Hardware–Cloud Providers game is a distributive obstruction: the Nash outcome differs from a joint-surplus maximum that requires compensating transfers; "
         f"equilibrium stability and policy scenarios are evaluated with Monte-Carlo "
         f"simulation and sensitivity analysis, and every figure and table below is "
         f"an artifact of the ai_ecosystem_model.py run_pipeline, re-verified by this generator.")
    para(doc, "Keywords: game theory; Nash equilibrium; deadweight loss; market "
              "concentration; HHI; Lerner index; Monte Carlo; AI ecosystem; "
              "antitrust; interoperability.", 10, italic=True)
    doc.add_page_break()

    # ---- TOC --------------------------------------------------------------
    add_h1(doc, "Contents")
    toc_p = doc.add_paragraph()
    add_field(toc_p, 'TOC \\o "1-2" \\h \\z \\u')
    para(doc, "If the table of contents does not display, select it and press F9 in Word "
              "(or set Tools → Options → Print → Update fields).", 8.5, italic=True,
         color=RGBColor(0x80, 0x80, 0x80))
    doc.add_page_break()
    # Anchor: Lists of Tables/Figures/Abbreviations are assembled at the end of
    # the build (they need the full placement log) and relocated here.
    front_anchor = para(doc, "", space_after=0)

    # ======================================================================
    # 1. Executive summary
    # ======================================================================
    add_h1(doc, "1.  Executive Summary")
    tot_rev = float(players["Current_Revenue_Billions"].sum())
    para(doc, f"This report documents the AI-ecosystem game-theory analysis produced by "
              f"{MODEL_FILE}. It synthesizes an estimated aggregate market revenue of "
              f"${tot_rev:,.1f}B across four player archetypes (Hardware, Cloud Providers, "
              f"Foundation Models, and LLM Wrappers) whose strategic interaction is modeled "
              f"as six bilateral 2×2 games, with equilibrium outcomes, welfare losses, "
              f"circular-dependency risk, and policy implications evaluated quantitatively.")
    es = [
        ("Market structure",
         f"Total revenue ${tot_rev:,.1f}B; HHI {conc['HHI']:,.0f} (highly concentrated); "
         f"CR4 {conc['CR4'] * 100:.1f}%; Gini {conc['Gini']:.3f}. "
         f"Cloud Providers ({conc['market_shares'].get('Cloud Providers', 0) * 100:.1f}%) and "
         f"Hardware ({conc['market_shares'].get('Hardware', 0) * 100:.1f}%) dominate."),
        ("Strategic games",
         f"{len(nash)} bilateral games analyzed. The core Hardware–Cloud Providers game is a "
         f"coordination failure: the Nash equilibrium ({_core_ne[0]}, {_core_ne[1]}) is not "
         f"a joint-surplus maximum (Kaldor–Hicks benchmark), with a game-level deadweight loss of "
         f"${_nash_dwl(nash, 'Hardware-Cloud Providers'):.1f}B."),
        ("Welfare",
         f"Aggregate welfare at the Nash outcome is ${welfare.get('total_nash_welfare', 0):,.1f}B "
         f"vs. a potential (Pareto) ${welfare.get('total_pareto_welfare', 0):,.1f}B, implying a "
         f"deadweight loss of ${welfare.get('total_dwl', 0):,.1f}B "
         f"({welfare.get('total_dwl', 0) / max(welfare.get('total_pareto_welfare', 1), 1e-9) * 100:.1f}% of potential) "
         f"and aggregate efficiency of {welfare.get('aggregate_efficiency', 0):.1f}%."),
        ("Circular dependency",
         f"{len(analysis['deps'])} circular deals totaling ${analysis['deps']['Dependency_Value_Billions'].sum():,.1f}B; "
         f"systemic criticality score {analysis['circular'].get('bubble_score', 0):.3f}."),
        ("Cooperation",
         f"Shapley allocation across the four players; cooperative surplus of the grand "
         f"coalition ≈ ${analysis['coop_surplus']:,.1f}B."),
    ]
    for title, text in es:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(f"{title}: ")
        r.bold = True
        r.font.size = Pt(10.5)
        r2 = p.add_run(text)
        r2.font.size = Pt(10.5)
    try:
        _pol = load_policy_table(gt_dir)
        if _pol is not None and "Net_Benefit_$B_Base" in _pol.columns:
            _nb = pd.to_numeric(_pol["Net_Benefit_$B_Base"], errors="coerce")
            if _nb.notna().any():
                _best = _pol.loc[_nb.idxmax()]
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.space_after = Pt(4)
                r = p.add_run("Best policy (base scenario): ")
                r.bold = True
                r.font.size = Pt(10.5)
                r2 = p.add_run(f"{_best['Policy']} — net benefit "
                               f"${float(_best['Net_Benefit_$B_Base']):,.1f}B.")
                r2.font.size = Pt(10.5)
    except Exception:
        pass

    # Headline metrics table
    add_h2(doc, "1.1  Headline metrics")
    rows = {
        "Aggregate market revenue": f"${tot_rev:,.1f}B",
        "HHI / CR4 / Gini": f"{conc['HHI']:,.0f} / {conc['CR4']*100:.1f}% / {conc['Gini']:.3f}",
        "Market-implied DWL": f"{analysis['dwl_pct']*100:.2f}%  (${analysis['dwl_billions']:,.1f}B)",
        "Aggregate welfare (Nash)": f"${welfare.get('total_nash_welfare', 0):,.1f}B",
        "Aggregate efficiency": f"{welfare.get('aggregate_efficiency', 0):.2f}%",
        "Coordination-failure DWL": f"${analysis['total_coordination_dwl']:,.2f}B",
        "Circular exposure": f"${deps['Dependency_Value_Billions'].sum():,.1f}B",
        "Core game NE": f"Hardware: {_core_ne[0]}; Cloud: {_core_ne[1]}",
    }
    hd = pd.DataFrame({"Metric": list(rows.keys()), "Value": list(rows.values())})
    add_word_table(doc, hd)

    add_h2(doc, "1.2  Research questions")
    for rq in [
        ("RQ1 — Market concentration and power",
         f"How concentrated is the AI ecosystem and how much pricing power follows? "
         f"Measured: HHI {conc['HHI']:,.0f} against both the 2010 (2,500) and 2023 "
         f"(1,800) highly-concentrated thresholds, with Lerner-implied markups per "
         f"archetype (Exhibit 3.3)."),
        ("RQ2 — Nash equilibria and strategic behavior",
         f"What are the equilibria of the six bilateral games, and do they coincide "
         f"with Pareto optima? The core Hardware–Cloud game is a coordination "
         f"failure (Nash versus joint-surplus maximum, DWL "
         f"${_nash_dwl(nash, 'Hardware-Cloud Providers'):.1f}B); Section 4 reports "
         f"all six (Exhibits 4.1–4.7)."),
        ("RQ3 — Structural inefficiency sources",
         f"How is the ${welfare.get('total_dwl', 0):,.1f}B aggregate deadweight loss "
         f"attributed across switching costs, monopoly pricing, innovation "
         f"distortion, quality degradation, and coordination failures (Exhibit 5.2)?"),
        ("RQ4 — Policy effectiveness",
         "Which interventions — interoperability mandates, antitrust enforcement, "
         "innovation subsidies, data-sharing mandates, or combinations — deliver the "
         "highest net benefit under conservative, base, and optimistic scenarios "
         "(Exhibit 5.2b and Section 5 deep dives)?"),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(rq[0] + ".  ")
        r.bold = True
        r.font.size = Pt(10.5)
        r2 = p.add_run(rq[1])
        r2.font.size = Pt(10.5)

    add_h2(doc, "1.3  Research objectives")
    for b in [
        "Model the four-layer AI ecosystem as six bilateral 2×2 games with market-derived payoffs.",
        "Quantify concentration (HHI, CR4, Gini) and market power (elasticity-based Lerner indices).",
        "Decompose aggregate deadweight loss by structural source with quantitative attribution.",
        f"Validate robustness with {ai_ecosystem_model.MONTE_CARLO_ITERATIONS:,} Monte-Carlo draws and "
        "sensitivity sweeps around the DWL assumption.",
        "Rank costed policy interventions across scenarios and state the evidence basis "
        "for each recommendation.",
    ]:
        bullet(doc, b)
    doc.add_page_break()

    # ======================================================================
    # 2. Data, sources and methodology
    # ======================================================================
    add_h1(doc, "2.  Data, Sources and Methodology")
    add_h2(doc, "2.1  Embedded data and vintage")
    para(doc, f"The analysis uses ai_ecosystem_model.py's embedded dataset (data module "
              f"{players.attrs.get('data_version', 'n/a')}, validation status: "
              f"{players.attrs.get('validation_status', 'n/a')}; last updated "
              f"{players.attrs.get('last_updated', 'n/a')}). All monetary figures are in "
              f"billions of U.S. dollars and reflect the Q2/Q3 2026 reporting cycle.")
    md = pd.DataFrame({
        "Category": players["Player_Category"],
        "Representative players": players["Key_Companies"],
        "Revenue ($B)": players["Current_Revenue_Billions"],
        "Market value ($B)": players["Market_Valuation_Billions"],
        "Op. margin": [f"{m*100:.1f}%" for m in players["Operating_Margin"]],
        "Circ. index": players["Circular_Dependency_Index"],
    })
    add_word_table(doc, md, caption="Exhibit 2.1  Player archetypes and market data (computed by build_technical_report.py)")

    srcs = meta.get("data_sources", {})
    if isinstance(srcs, dict) and srcs:
        sd = pd.DataFrame({"Parameter": list(srcs.keys()), "Source": list(srcs.values())})
        add_word_table(doc, sd, caption="Exhibit 2.2  Data sources and macro parameters",
                       max_cols=4)
    para(doc, f"Macro context: {meta.get('analyst_notes', '')}", 10, italic=True)

    add_h2(doc, "2.2  Methodological choices")
    for b in [
        "Players are the four archetypes Hardware, Cloud Providers, Foundation Models and LLM Wrappers; "
        "each bilateral interaction is a 2×2 game with strategies defined per player.",
        "Payoffs are derived from market data: the Defect–Defect cell equals observed revenue; "
        "cooperation expands the total to an efficient benchmark; temptation and sucker payoffs are "
        "calibrated from margins and dependency exposure.",
        f"Deadweight loss is market-implied: DWL% ≈ min(40%, max(5%, circular-exposure ÷ revenue × 25%)) "
        f"= {analysis['dwl_pct']*100:.2f}%, i.e. ${analysis['dwl_billions']:,.1f}B on potential (Pareto) welfare.",
        f"Nash equilibria are found by best-response analysis; welfare is measured as the distance from "
        f"the maximum-total joint-surplus outcome (Kaldor–Hicks benchmark); coordination failures are Nash outcomes that miss the "
        f"Pareto optimum.",
        f"Robustness is assessed with {ai_ecosystem_model.MONTE_CARLO_ITERATIONS:,} Monte-Carlo draws at "
        f"{ai_ecosystem_model.PERTURBATION_STD*100:.0f}% log-normal payoff noise and a {ai_ecosystem_model.CONFIDENCE_LEVEL*100:.0f}% "
        f"confidence interval, plus sensitivity analysis around the DWL assumption.",
    ]:
        bullet(doc, b)

    add_h2(doc, "2.3  Equilibrium concepts used in this report")
    _core_t = nash.get("Hardware-Cloud Providers", {}).get("game_type", "n/a")
    for b in [
        "Nash equilibrium (Nash, 1950): a strategy profile from which no player can "
        "profitably deviate unilaterally. All six games are solved by exhaustive "
        "best-response checks over the four cells — no equilibrium is assumed.",
        "Prisoner's Dilemma: temptation > cooperation > defection > sucker payoffs, "
        "so mutual defection is the equilibrium although mutual cooperation pays more. "
        f"The core Hardware–Cloud game is classified '{_core_t}'; Section 4 classifies "
        "all six.",
        "Joint-surplus maximum and distributive obstruction: a true Pareto improvement leaves no "
        "player can gain without another losing; a coordination failure is Nash ≠ "
        "party worse off; a Kaldor–Hicks benchmark requires compensation. Positive Nash–joint-max gaps measure forgone joint surplus.",
        "Second-best (Lipsey & Lancaster, 1956): with several simultaneous distortions, "
        "fixing one in isolation need not improve welfare — the rationale for ranking "
        "combined as well as standalone interventions in Section 5.",
    ]:
        bullet(doc, b)

    add_h2(doc, "2.4  Market-power and welfare measurement")
    _mp = analysis.get("market_power") or {}
    _lerner_txt = ""
    if _mp:
        _bk = max(_mp, key=lambda k: _mp[k].get("lerner_index") or 0)
        _lerner_txt = (f" The highest implied Lerner index in this run is "
                       f"{(_mp[_bk].get('lerner_index') or 0):.3f} ({_bk}).")
    _comp = welfare.get("dwl_by_source", {}) or {}
    _comp_txt = ", ".join(f"{k.replace('_', ' ')} {v / max(sum(_comp.values()), 1e-9) * 100:.0f}%"
                          for k, v in sorted(_comp.items(), key=lambda kv: -kv[1]))
    for b in [
        "Concentration: HHI is the sum of squared percentage market shares (0–10,000). "
        "Both DOJ/FTC vintages are used throughout — 2010: 1,500/2,500; current 2023: "
        "1,000/1,800 (highly concentrated, +100pts presumption). This market exceeds "
        "both high cutoffs, so the verdict does not depend on vintage.",
        "Market power: the Lerner index L = (P − MC)/P = 1/|elasticity| measures pricing "
        f"power on a 0–1 scale (0 perfect competition, 1 monopoly).{_lerner_txt}",
        "Deadweight loss (Harberger, 1954): the surplus triangle lost when price exceeds "
        "marginal cost, growing with the square of the markup — which is why the "
        "high-markup archetypes dominate the loss. In this run the aggregate loss splits "
        f"as {_comp_txt}.",
        "Switching costs and lock-in (Katz & Shapiro, 1985; Farrell & Klemperer, 2007): "
        "platform incompatibilities and data gravity sustain pricing power even in "
        "equilibrium, so part of the loss persists under any conduct remedy — the "
        "structural interpretation developed in Sections 5–6.",
    ]:
        bullet(doc, b)

    add_h2(doc, "2.5  Computation, reproducibility and figure provenance")
    game_pct = analysis.get("game_dwl_pct", np.nan)
    for b in [
        f"Two distinct deadweight-loss rates are used (ai_ecosystem_model.py DWL naming conventions). "
        f"The top-down rate DWL_PCT_TOPDOWN = {analysis['dwl_pct']*100:.2f}% (dependency-value "
        f"ratio) feeds aggregate welfare, policy and sensitivity analysis. The game-level rate "
        f"{game_pct*100:.2f}% (concentration/dependency geometric mean) is a recorded diagnostic "
        f"only — payoff levels derive from pair efficiency, margins and dependencies. "
        f"They answer different questions and must not be conflated.",
        f"Monte Carlo is vectorized (batched log-normal draws, numpy best-response masks; "
        f"statistically identical to the per-draw loop) with run-scoped generators seeded "
        f"{analysis['global_seed']} — each game gets an independent noise stream and repeated "
        f"runs reproduce exactly.",
        "Figures render headless-safe (Agg backend), use the Okabe–Ito colorblind-safe palette "
        "in DejaVu Sans, carry journal-style (a)/(b) panel tags on data panels, and are saved as "
        "300-DPI PNG plus vector PDF; panel inventories come from ai_ecosystem_model.py's figure_captions.md "
        "when present.",
        "Convergence is assessed with Monte-Carlo standard errors and split-half "
        "consistency from stored draws (table_7.9); R-hat is not reported because it "
        "is uninformative for IID Monte Carlo. Equilibria are cross-validated by an "
        "independent brute-force enumeration (table_7.8).",
        "A five-part robustness battery (ai_ecosystem_model.py RobustnessBattery, Tables 8.1–8.5) backs "
        "every headline: (R1) Geweke mean-stability screens with Holm correction across "
        "games; (R2) full-sample reruns at two independent seeds with drift measured in "
        "pooled standard errors; (R3) perturbation-scale robustness at 2.5/5/10% noise with "
        "Spearman rank preservation; (R4) Saltelli variance decomposition (N=2048, 500 "
        "bootstraps) of game DWL over calibrated primitives; (R5) an explicit assumption "
        "register mapping each modeling choice to the check that disciplines it. All "
        "stochastic steps derive from seeded generators, so the battery reproduces exactly.",
    ]:
        bullet(doc, b)

    add_h2(doc, "2.6  Literature review")
    para(doc, "Recent (2025–2026) work on AI markets falls into five streams that "
              "jointly motivate this report's design. Each stream is summarized below "
              "with its direct bearing on the research questions; full citations are "
              "in References. Where the literature disagrees, the disagreement is "
              "stated explicitly and carried forward as a robustness check or limitation.")
    add_h3(doc, "Scaling, concentration, and market structure")
    para(doc, "Korinek and Vipra (2025) argue that AI markets concentrate even without "
              "anticompetitive conduct, through three jointly-operating features: very "
              "large fixed training costs, cost advantages that grow with size, and high "
              "customer switching costs — with concentration threatening an unprecedented "
              "accumulation of economic power. Their framework is the direct antecedent "
              "of this report's measurement strategy: the HHI/CR4/Gini battery (Section 3, "
              "RQ1) and the Lerner-based market-power assessment (Exhibit 3.3) operationalize "
              "exactly those three channels.")
    para(doc, "Hagiu and Wright (2025) analyze the vertical AI technology stack and data "
              "feedback loops around three questions — concentration in core AI services, "
              "AI's impact on existing market structures, and emerging policy challenges — "
              "identifying risks of new gatekeeper types alongside disruption of old "
              "platforms. Their layered view matches this report's four-archetype model, "
              "and their gatekeeper analysis anticipates the bottleneck finding of Section 4. "
              "Korinek and Vipra (2025) trace how scaling concentrates the AI stack and "
              "how upstream power moves downstream prices, industry structure, and "
              "welfare — the theoretical "
              "counterpart of the deadweight-loss decomposition in Section 5 (RQ3), "
              "with frontier scale economies as the limiting case.")
    para(doc, "The contestability counterpoint matters for RQ1: if power-law capability "
              "scaling plus fast-follower dynamics keep training costs contestable, "
              "concentration findings weaken. This report does not adjudicate that "
              "debate; instead the sensitivity analysis (Section 6) and the DWL-elasticity "
              "reading test how much the conclusions move if contestability is stronger than "
              "the base case assumes.")
    add_h3(doc, "Competition policy and antitrust")
    para(doc, "A restraint counter-argument runs the other way: preemptive "
              "regulatory approaches to platform–laboratory partnerships and investments "
              "in the generative-AI ecosystem risk false positives, counselling "
              "doctrinal humility. That sets the evidentiary bar Section 5 must clear — "
              "ranked, costed, scenario-bound interventions rather than precautionary "
              "ones. Institutional reviews pull the other way: the OECD (2025) documents "
              "multifaceted, evolving concentration risks across stack layers including "
              "tying and algorithmic collusion, while the World Bank (2026) finds end-user "
              "outcomes largely predetermined by control of critical upstream "
              "infrastructure — consistent with this report's infrastructure-bottleneck "
              "diagnosis. The operative legal regime for the interoperability remedy is the "
              "EU Digital Markets Act (European Commission, 2022), cited as precedent in "
              "the policy deep dives.")
    add_h3(doc, "Game theory and algorithmic collusion")
    para(doc, "Bichler et al. (2025) survey when pricing algorithms converge to Nash "
              "equilibria (notably in potential games) versus collusive outcomes, and which "
              "algorithm properties tilt the balance. Their convergence conditions inform "
              "this report's stability reading in Section 6: high Monte-Carlo stability is "
              "interpreted as structural lock-in rather than coincidence. This report's "
              "one-shot bilateral design is a static-game analysis, and the "
              "repeated-game gap it leaves is recorded as future work (Section 9.3).")
    add_h3(doc, "Circular investment and systemic fragility")
    para(doc, "Realized revenue and adoption support part of AI valuations, while "
              "capex outruns monetization in some layers and private valuations concentrate "
              "in few firms — the dot-com precedent for transformative technology alongside "
              "overvaluation (Ofek, 2003; Greenwood et al., 2019). That 'localized fragility' "
              "framing disciplines Section 7: the "
              "bubble score is presented as a fragility index, not a crash prediction. "
              "The financing channel provides the transmission mechanism behind the "
              "single-point-of-failure reading of the NVIDIA-centered deal fabric: the "
              "dependency frame and credit monitor in Section 7 trace how a repricing "
              "travels from circular claims to equity and GDP.")
    add_h3(doc, "Circular-dealing measurement and vendor-financing contracts")
    para(doc, "Round-trip chains run through multiple stack layers, which supports "
              "this report's archetype-stack design over pairwise analysis; the "
              "committed-value exposure frame books contracted circularity deal by deal, "
              "with conditional tranches parked outside it. Two contract patterns "
              "organize the channel: an integration-plus-circular-investment pattern "
              "(vertical integration plus circular investment; apparent sales that "
              "reverse on a demand shock) and a GPU-collateralised leverage pattern "
              "(supplier resale guarantees, herd entry, and chain-reaction "
              "fire sales of homogeneous collateral). The fire-sale cascade is the "
              "statically-modeled second round of this report's DebtRank reverberation (Section 7, "
              "Exhibit 7.8); dynamic clearing with endogenous leverage remains open. The "
              "run_pipeline documents the leverage stack it would detonate — $675.40B in "
              "committed exposure across 17 edges ($1,188.33B tracked over 19 rows), the 2026 debt run_pipeline, Oracle's "
              "single-day bond sale and off-balance-sheet vehicle, and the record "
              "CDS print with downgrade to BBB- — all recorded in the September 2026 "
              "market-data vintage; "
              "the Oracle repricing corroborates the burst ranking. The ex-post agenda "
              "is priced as the complement rather than the rival of prevention: prosecute "
              "proximate-cause financial engineering, "
              "convert stranded data centres to a public cloud, protect workers, and structurally separate "
              "via a Glass-Steagall for AI and utility-style regulation — the ex-post complement to this "
              "report's ex-ante BCR-ranked tools.")
    add_h3(doc, "Interoperability and policy mechanisms")
    para(doc, "The interoperability remedy rests on cross-sector precedent: mandated "
              "infrastructure unbundling (telecoms) and open-banking standardisation show "
              "mandates can unlock downstream entry, while the payments experience shows "
              "mandates without standardisation fragment into competing API standards "
              "(Ofcom, discussion paper on mandated interoperability in digital markets). "
              "That caveat is priced into Section 5: the API-mandate deep dive carries "
              "implementation cost and scenario bounds rather than assuming frictionless "
              "adoption. Data-portability and shared-access arguments (cf. OECD, "
              "2025) motivate the data-sharing "
              "mandate alternative ranked alongside it.")
    para(doc, "Taken together, the literature supports measuring concentration before "
              "asserting power (Korinek and Vipra; Hagiu and Wright), tracing upstream "
              "power to downstream welfare (Korinek and Vipra), testing stability "
              "rather than assuming it (Bichler et al.), treating circularity as fragility "
              "rather than growth, and costing remedies against scenarios "
              "rather than asserting them. The mapping below records where each "
              "stream binds on this report.", 10, italic=True)
    _litmap = pd.DataFrame([
        {"Literature stream": "Scaling → concentration",
         "Key sources": "Korinek & Vipra (2025)",
         "Report locus": "§3, RQ1",
         "What this report adds": "HHI/CR4/Gini plus Lerner-implied markups on Q2/Q3 2026 data"},
        {"Literature stream": "Vertical stack & gatekeepers",
         "Key sources": "Hagiu & Wright (2025); World Bank (2026); OECD (2025)",
         "Report locus": "§4, RQ2",
         "What this report adds": "Six solved bilateral games; bottleneck located, not assumed"},
        {"Literature stream": "Upstream power → welfare",
         "Key sources": "Korinek & Vipra (2025)",
         "Report locus": "§5, RQ3",
         "What this report adds": "$-level DWL decomposition by source"},
        {"Literature stream": "Equilibrium convergence/stability",
         "Key sources": "Bichler et al. (2025)",
         "Report locus": "§6",
         "What this report adds": "Monte-Carlo stability of each game's NE"},
        {"Literature stream": "Bubble precedent & fragility",
         "Key sources": "Ofek (2003); Greenwood et al. (2019)",
         "Report locus": "§7",
         "What this report adds": "Bubble score as fragility index; NVIDIA-share of deal value"},
        {"Literature stream": "Circularity measurement",
         "Key sources": "FTC (2025); BIS (2026)",
         "Report locus": "§7",
         "What this report adds": "Committed-value exposure frame; fire-sale round noted as unmodeled"},
        {"Literature stream": "Remedies & restraint",
         "Key sources": "DMA (2022); Ofcom",
         "Report locus": "§5.1, RQ4",
         "What this report adds": "Costed, scenario-bound ranking with BCRs"},
    ])
    add_word_table(doc, _litmap, caption="Exhibit 2.6  Literature-to-report mapping (compiled by build_technical_report.py)",
                   max_cols=5)

    # ------------------------------------------------------------------
    # placement toolkit: logs every embedded ai_ecosystem_model artifact so we can verify
    # afterwards that tables/figures landed in the correct sections.
    placement = []
    tbl_meta = load_table_meta(gt_dir)
    added_gt = set()

    def embed(fn, cache=True):
        """Embeds one ai_ecosystem_model.py figure with its registry caption; logs misses to placement."""
        fp = os.path.join(gt_dir, PLOTS_DIR, fn)
        title, panels = caption_for(fn, cap_map)
        if panels and panels.lower().startswith("panels: schematic"):
            title = f"{title} (schematic)"
        elif panels:
            title = f"{title} [{panels}]"
        if not os.path.exists(fp):
            placement.append({"kind": "figure", "ok": False, "file": fn,
                              "label": f"Figure {figure_number(fn) or '?'}: {title}",
                              "section": FIGURE_SECTION.get(fn, None)})
            return
        add_figure(doc, fp, title,
                   number=figure_number(fn),
                   cache_dir=jpeg_cache_dir if cache else None)
        placement.append({"kind": "figure", "ok": True, "file": fn,
                          "label": f"Figure {figure_number(fn) or '?'}: {title}",
                          "section": FIGURE_SECTION.get(fn, None)})

    def gt_tbl(section_no, tid):
        """Embed one ai_ecosystem_model.py-generated table verbatim under its own table id."""
        info = tbl_meta.get(tid)
        if not info:
            placement.append({"kind": "table", "ok": False, "file": f"table {tid}",
                              "label": f"Table {TABLE_DISPLAY.get(tid, tid)}", "section": section_no})
            return False
        try:
            df = pd.read_csv(info["path"])
        except Exception as e:
            placement.append({"kind": "table", "ok": False, "file": info["stem"],
                              "label": f"Table {TABLE_DISPLAY.get(tid, tid)}", "section": section_no})
            return False
        display = TABLE_DISPLAY.get(tid, tid)
        caption = f"Table {display}: {info['desc'] or tid}"
        add_word_table(doc, df, caption=caption)
        added_gt.add(tid)
        placement.append({"kind": "table", "ok": True, "file": os.path.basename(info["path"]),
                          "label": caption, "section": section_no})
        return True

    def place_section_gt(section_no):
        """Embeds every not-yet-placed ai_ecosystem_model.py table registered to one report section."""
        for tid in GT_TABLES_BY_SECTION.get(section_no, []):
            if tid not in added_gt:
                gt_tbl(section_no, tid)

    def embed_supp(stem, code, title, section_no):
        """Embeds one figures/supplemental/ figure with a Supplemental caption.

        Missing files render an italic placeholder and log nothing, so report
        builds against older ai_ecosystem_model.py outputs (no supplemental/ directory) keep
        working; present files are placement-logged with a supplemental/ file
        tag that verify_placement counts separately from the 1-18 registry.
        """
        fp = os.path.join(gt_dir, PLOTS_DIR, "supplemental", stem + ".png")
        caption = f"Supplemental {code} — {title}"
        if not os.path.exists(fp):
            para(doc, f"[{caption} unavailable — rerun ai_ecosystem_model.py to generate "
                      f"figures/supplemental/{stem}.png.]", 9, italic=True, color=BAD)
            return False
        add_figure(doc, fp, title, number=None,
                   caption_prefix=f"Supplemental {code} — ",
                   alt_text=caption,
                   cache_dir=jpeg_cache_dir)
        placement.append({"kind": "figure", "ok": True,
                          "file": f"supplemental/{stem}.png",
                          "label": caption, "section": section_no})
        return True
    # ------------------------------------------------------------------

    insights = build_insights(ai_ecosystem_model, analysis, gt_dir)
    mod = build_module_analyses(analysis, gt_dir)
    # Best-available figure captions: ai_ecosystem_model.py's figure_captions.md (with panel
    # tags) when present, else the static caption table compiled into build_technical_report.py.
    cap_map = load_figure_captions(gt_dir)

    add_h2(doc, "2.7  Assumptions register")
    para(doc, "Table 8.5 records every material modeling assumption, where it is used, "
              "what justifies it, which robustness check disciplines it, and what breaks "
              "if it is violated. Assumptions marked “not directly tested” are carried "
              "forward as Limitations in §9.2 rather than asserted as findings.")
    gt_tbl(2, "8_5")
    add_h2(doc, "2.8  Validated inputs register")
    para(doc, "Table 8.6 lists every validated input value and modeling assumption the "
              "analysis consumes -- archetype revenues, valuations, margins and risks; "
              "macro and market parameters; bubble-block scenario constants and tax rates; "
              "demand elasticities with literature anchors; policy scenarios with reduction "
              "rates, costs and sources; all 19 circular deals; and the audited scope "
              "decisions (inclusions, updates, exclusions). It is built live from the "
              "canonical ai_ecosystem_model.py structures, so the values below are the values the models ran on.")
    # The full 90+-row register would dominate the body: show a 10-row excerpt
    # here and point at Appendix A for the complete source-documented table.
    # (The excerpt caption keeps the Table 8.6 mirror, so dashboard/report
    # cross-links keep resolving.)
    _86 = tbl_meta.get("8_6")
    _86df = None
    if _86 is not None:
        try:
            _86df = pd.read_csv(_86["path"])
        except Exception:
            _86df = None
    if _86df is not None and len(_86df) > 10:
        add_word_table(doc, _86df.head(10),
                       caption=f"Table 8.6 (excerpt) — Validated inputs register "
                               f"(first 10 of {len(_86df)} rows)")
        para(doc, f"The complete {len(_86df)}-row register, with a source or verification "
                  f"note per input, appears as Table 8.6 in Appendix A.", 10.5)
    else:
        gt_tbl(2, "8_6")
    place_section_gt(2)  # Table 2.4: player categories (+ 8.5 placed above; 8.6 excerpt above)
    add_h3(doc, "Figures — framework and design")
    for fn in SECTION_FIGURES["conceptual"]:
        embed(fn)
    doc.add_page_break()

    # ======================================================================
    # 3. Market structure
    # ======================================================================
    add_h1(doc, "3.  Market Structure Results")
    para(doc, "Concentration and market-power metrics computed from the revenue distribution "
              "across the four archetypes.")
    conc_rows = {
        "Total market revenue": f"${conc['total_revenue_billions']:,.2f}B",
        "HHI": f"{conc['HHI']:,.2f}  (highly concentrated: >2,500 in 2010, >1,800 in 2023 guidelines)",
        "CR4": f"{conc['CR4']*100:.2f}%",
        "Gini coefficient": f"{conc['Gini']:.4f}",
    }
    cd = pd.DataFrame({"Metric": list(conc_rows.keys()), "Value": list(conc_rows.values())})
    add_word_table(doc, cd, caption="Exhibit 3.1  Concentration metrics (computed by build_technical_report.py)")
    # NOTE: sort on the numeric shares BEFORE formatting: the display column
    # holds strings, and lexicographic order misranks single-digit shares
    # ("9.xx" sorts above "46.xx" as text).
    _share_items = sorted(conc["market_shares"].items(), key=lambda kv: kv[1], reverse=True)
    shares = pd.DataFrame({
        "Player category": [k for k, _ in _share_items],
        "Market share (%)": [f"{v*100:.2f}" for _, v in _share_items],
    }).reset_index(drop=True)
    add_word_table(doc, shares, caption="Exhibit 3.2  Market shares (computed by build_technical_report.py)")
    mp = analysis["market_power"]
    if mp:
        mpr = pd.DataFrame([{
            "Player category": k,
            "Demand elasticity": f"{v.get('derived_demand_elasticity', np.nan):.2f}",
            "Lerner index": f"{v.get('lerner_index', np.nan):.3f}",
            "Implied markup": f"{v.get('markup_percent', np.nan):.1f}%",
            "Power level": v.get("market_power_interpretation", "n/a"),
        } for k, v in mp.items()])
        add_word_table(doc, mpr, caption="Exhibit 3.3  Market power (Lerner-based, computed by build_technical_report.py)")
    para(doc, "Table 7.17 restates the same pricing power with its elasticity-driven "
              "uncertainty bands: the interval between the conservative and aggressive "
              "Lerner estimates is the robustness range for any markup claim, and the "
              "figure plots the base index against those bands.", 10.5)
    gt_tbl(3, "7.17_lerner_markup_summary")
    embed_supp("s2_lerner_markup_bands", "S2", "Lerner Index with Uncertainty Bands", 3)
    _gate_p = os.path.join(gt_dir, TABLES_DIR, "table_7.2_gatekeeper_index.csv")
    if os.path.exists(_gate_p):
        try:
            _gate = pd.read_csv(_gate_p)
            if len(_gate):
                _gshow = _gate.head(5).copy()
                _gshow["Inbound ($B)"] = pd.to_numeric(
                    _gshow.get("Inbound_Dependency_$B"), errors="coerce").map(
                    lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
                _gshow["Gatekeeper share"] = pd.to_numeric(
                    _gshow.get("Gatekeeper_Share_Pct"), errors="coerce").map(
                    lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
                add_word_table(doc, _gshow[["Archetype", "Dependents_Count",
                                            "Inbound ($B)", "Gatekeeper share"]],
                               caption="Exhibit 3.4  Gatekeeper index: who the ecosystem "
                                       "cannot route around (ai_ecosystem_model.py diagnostics)")
                _gtop = _gate.iloc[0]
                para(doc, f"Top gatekeeper: {_gtop['Archetype']} "
                          f"({_gtop['Gatekeeper_Share_Pct']:.1f}% of dependency value). "
                          f"Upstream control predetermines downstream outcomes — the "
                          f"World Bank (2026) thesis, named here rather than assumed.",
                     10.5)
        except Exception as e:
            para(doc, f"Gatekeeper table unreadable: {e}", 9, italic=True, color=BAD)
    add_insights(doc, insights["market"], "market structure")
    add_module_analysis(doc, mod.get("market"))
    place_section_gt(3)  # ai_ecosystem_model Tables 1.1 (market structure) & 1.3 (revenue projections)
    para(doc, "Table 7.16 converts the 2025 observed and 2030 projected revenue levels "
              "into market shares: the levels say how big the market gets, the shares "
              "say who gains weight under diminishing-growth compounding. The figure "
              "plots the two vintages side by side.", 10.5)
    gt_tbl(3, "7.16_market_share_evolution")
    embed_supp("s1_market_share_levels", "S1", "Revenue Levels 2025 vs 2030", 3)
    for fn in SECTION_FIGURES["market"]:
        embed(fn)
    doc.add_page_break()

    # ======================================================================
    # 4. Strategic interaction
    # ======================================================================
    add_h1(doc, "4.  Strategic Interaction and Equilibrium")
    para(doc, f"{len(nash)} bilateral games are solved for pure-strategy Nash equilibria; "
              f"the table below reports each game's equilibrium, classification, efficiency "
              f"and deadweight loss.")
    rows4 = []
    for name, res in nash.items():
        w = res.get("welfare_metrics", {})
        eq = res.get("nash_equilibria", [])
        p1, p2 = res["players"]
        s1 = gf_strategy_of(ai_ecosystem_model, players, p1)
        s2 = gf_strategy_of(ai_ecosystem_model, players, p2)
        rows4.append({
            "Game": name,
            "Game type": res.get("game_type", "n/a"),
            "NE strategies": (f"{eq[0]['strategies'][0]} / {eq[0]['strategies'][1]}"
                              if eq else "—"),
            "NE payoffs ($B)": (f"({eq[0]['payoffs'][0]:.1f}, {eq[0]['payoffs'][1]:.1f})"
                                if eq else "—"),
            "Efficiency (%)": f"{w.get('efficiency_ratio', np.nan):.1f}",
            "DWL ($B)": f"{w.get('deadweight_loss', np.nan):.1f}",
            "Coord. failure": (
                "Yes" if analysis["coordination"].get(name, {}).get("is_coordination_failure")
                else "No"),
        })
    gd = pd.DataFrame(rows4)
    add_word_table(doc, gd, caption="Exhibit 4.1  Nash equilibria across all bilateral games (computed by build_technical_report.py)",
                   max_cols=6)
    for gi, (gname, res) in enumerate(nash.items(), start=2):
        p1, p2 = res["players"]
        add_payoff_exhibit(doc, gi, gname, res,
                           gf_strategy_of(ai_ecosystem_model, players, p1),
                           gf_strategy_of(ai_ecosystem_model, players, p2))

    # Corrected welfare benchmark audit. These tables are emitted by the
    # ai_ecosystem_model.py audit run_pipeline (run_welfare_benchmark_audits) and distinguish
    # Pareto improvements from Kaldor-Hicks joint-surplus maxima requiring
    # compensating transfers. They are captioned as Tables (not Exhibits) so
    # the placement log and dashboard Report tab can mirror them by number.
    add_h2(doc, "4.3  Welfare benchmark, transfers and equilibrium selection")
    _t35 = load_audit_table(gt_dir, "table_3.5_welfare_benchmark_audit.csv")
    _t36 = load_audit_table(gt_dir, "table_3.6_equilibrium_multiplicity.csv")
    _t37 = load_audit_table(gt_dir, "table_3.7_repeated_game_sustainability.csv")
    if _t35 is not None and not _t35.empty:
        _kh = _t35[_t35.get("classification", pd.Series(dtype=str)).astype(str).str.contains("Kaldor-Hicks", na=False)]
        _joint = pd.to_numeric(_t35.get("joint_gain"), errors="coerce").sum()
        _transfer = pd.to_numeric(_t35.get("minimum_transfer"), errors="coerce").sum()
        para(doc, f"The payoff audit corrects a material terminology error: {_kh.shape[0]} of "
                  f"{_t35.shape[0]} joint-surplus benchmarks are Kaldor–Hicks improvements, "
                  f"not Pareto improvements. Their gross joint gain is ${_joint:,.2f}B, but "
                  f"downstream parties require ${_transfer:,.2f}B in compensating transfers to "
                  f"accept the transition from the reported Nash profile. The result is a "
                  f"distributive-obstruction diagnosis rather than a generic coordination claim.", 10.5)
        add_word_table(doc, _t35, caption="Table 3.5 — Welfare-benchmark audit (Kaldor-Hicks transfers)", max_cols=8)
    else:
        para(doc, "Corrected welfare-benchmark audit unavailable; run ai_ecosystem_model.py to generate Table 3.5.", 9, italic=True, color=BAD)
    if _t36 is not None and not _t36.empty:
        para(doc, "Equilibrium multiplicity is reported using weak best responses; the solver and independent auditor must agree on the count before any selection claim is made.", 10.5)
        add_word_table(doc, _t36, caption="Table 3.6 — Equilibrium multiplicity and risk-dominance audit", max_cols=8)
    if _t37 is not None and not _t37.empty:
        para(doc, "Repeated-game analysis reports whether grim-trigger punishment can sustain the joint-surplus profile without transfers. Infinite critical discount factors denote cases where Nash reversion is not a credible punishment.", 10.5)
        add_word_table(doc, _t37, caption="Table 3.7 — Repeated-game sustainability (critical discount factors)", max_cols=8)
    para(doc, "Supplemental Figure S5 plots Table 3.5's bottom line by game: the height "
              "of each bar is the joint gain from moving off Nash, the color is the "
              "welfare verdict, and the annotation is the minimum transfer that unlocks "
              "the Kaldor-Hicks cases.", 10.5)
    embed_supp("s5_joint_gains_transfers", "S5", "Joint Gains and Minimum Transfers by Game", 4)
    add_insights(doc, insights["games"], "strategic interaction")
    add_module_analysis(doc, mod.get("games"))
    place_section_gt(4)  # ai_ecosystem_model payoff matrices, joint-max audit, multiplicity and repeated-game tables
    for fn in SECTION_FIGURES["games"]:
        embed(fn)
    add_h2(doc, "Discussion: what the equilibria mean for theory")
    _eff_n = sum(1 for r in nash.values()
                 if float(r.get("welfare_metrics", {}).get("deadweight_loss", 0)) < 1e-9)
    _eff_clause = (f"{_eff_n} of {len(nash)} games are efficient at the equilibrium"
                   if _eff_n else
                   f"none of the {len(nash)} games reaches an efficient equilibrium")
    for b in [
        f"Asymmetric multi-layer platforms behave differently from textbook symmetric "
        f"oligopoly: Nash equilibrium exists in every bilateral game here, yet "
        f"{_eff_clause} — the infrastructure game included. Existence theorems "
        f"transfer; welfare conclusions do not.",
        "Efficient local equilibria coexisting with an inefficient aggregate is the "
        "empirical face of second-best economics: with switching costs, markups, and "
        "lock-in operating simultaneously, the system can be locally stable and "
        "globally wasteful. Policy must therefore target the distortion portfolio "
        "(Section 5), not merely 'more competition' in the abstract.",
        "A repeated-game lens qualifies the one-shot result: trigger strategies can "
        "sustain cooperation where stage games trap players, so the observed "
        "Walled-Garden persistence may partly reflect punishment equilibria rather "
        "than stage-game necessity — see Limitations (9.2) and Future work (9.3).",
    ]:
        bullet(doc, b)
    doc.add_page_break()

    # ======================================================================
    # 5. Welfare and policy
    # ======================================================================
    add_h1(doc, "5.  Welfare, Deadweight Loss and Policy")
    wl = welfare
    wrows = {
        "Observed (Nash) welfare": f"${wl.get('total_nash_welfare', 0):,.2f}B",
        "Potential joint-surplus welfare": f"${wl.get('total_pareto_welfare', 0):,.2f}B",
        "Aggregate deadweight loss": f"${wl.get('total_dwl', 0):,.2f}B",
        "Aggregate efficiency": f"{wl.get('aggregate_efficiency', 0):.2f}%",
        "Market-implied DWL assumption": f"{analysis['dwl_pct']*100:.2f}%",
    }
    wd = pd.DataFrame({"Metric": list(wrows.keys()), "Value": list(wrows.values())})
    add_word_table(doc, wd, caption="Exhibit 5.1  Aggregate welfare (computed by build_technical_report.py)")

    dwl_src = wl.get("dwl_by_source", {})
    if dwl_src:
        dsd = pd.DataFrame({
            "Source": [k.replace("_", " ").title() for k in dwl_src.keys()],
            "DWL ($B)": [f"{v:,.2f}" for v in dwl_src.values()],
            "Share (%)": [f"{v / max(sum(dwl_src.values()), 1e-9) * 100:.1f}"
                          for v in dwl_src.values()],
        })
        add_word_table(doc, dsd, caption="Exhibit 5.2  Deadweight-loss decomposition (computed by build_technical_report.py)")

    # policy interventions (ai_ecosystem_model.py v4.0 enhancement module output)
    pol = load_policy_table(gt_dir)
    best_policy = None
    if pol is not None and "Net_Benefit_$B_Base" in pol.columns:
        try:
            pol["_nb"] = pd.to_numeric(pol["Net_Benefit_$B_Base"], errors="coerce")
            ranked = pol.dropna(subset=["_nb"]).sort_values("_nb", ascending=False)
            if len(ranked):
                best_policy = ranked.iloc[0]
                show = ranked.head(5).copy()
                show["DWL reduction ($B)"] = pd.to_numeric(
                    show.get("DWL_Reduction_$B_Base"), errors="coerce").map(
                    lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
                show["Cost ($B)"] = pd.to_numeric(
                    show.get("Cost_$B_Base"), errors="coerce").map(
                    lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
                show["Net benefit ($B)"] = show["_nb"].map(lambda v: f"{v:,.1f}")
                bcr = pd.to_numeric(show.get("BCR_Base"), errors="coerce")
                show["BCR"] = bcr.map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
                add_word_table(doc, show[["Policy", "DWL reduction ($B)", "Cost ($B)",
                                          "Net benefit ($B)", "BCR"]],
                               caption="Exhibit 5.2b  Policy interventions ranked by base-scenario "
                                       "net benefit (ai_ecosystem_model.py enhancement output)")
        except Exception as e:
            para(doc, f"Policy table unreadable: {e}", 9, italic=True, color=BAD)
    if best_policy is not None:
        para(doc, f"Highest net-benefit intervention (base scenario): "
                  f"{best_policy['Policy']} — net benefit ${float(best_policy['_nb']):,.1f}B.",
             10.5)
    add_insights(doc, insights["policy"], "policy choice")
    add_module_analysis(doc, mod.get("policy"))

    # per-policy deep dives (mechanism + scenario arithmetic from the CSV)
    _mech = {
        "interoperability": "mandate open, standardized APIs to cut proprietary lock-in",
        "antitrust": "challenge high-concentration conduct and impose market-conduct remedies",
        "innovation": "subsidize alternative architectures, competing chips, and open foundations",
        "data sharing": "require sharing of anonymized data to diffuse model/data advantages",
        "combined": "deploy all four levers jointly (tested sub-additive, ratio 0.547: the levers overlap rather than complement)",
    }
    if pol is not None and len(pol):
        add_h2(doc, "5.1  Policy deep dives")
        for _, _row in pol.iterrows():
            _nm = str(_row.get("Policy", "n/a"))
            add_h3(doc, _nm)
            _key = next((k for k in _mech if k in _nm.lower()), None)
            para(doc, f"Mechanism: {_mech[_key]}." if _key else
                 "Mechanism: as specified in the ai_ecosystem_model.py policy module.")
            _sc = []
            for _s in ["Conservative", "Base", "Optimistic"]:
                try:
                    _sc.append({
                        "Scenario": _s,
                        "DWL reduction ($B)": f"{float(_row.get(f'DWL_Reduction_$B_{_s}')):,.1f}",
                        "Cost ($B)": f"{float(_row.get(f'Cost_$B_{_s}')):,.1f}",
                        "Net benefit ($B)": f"{float(_row.get(f'Net_Benefit_$B_{_s}')):,.1f}",
                        "BCR": f"{float(_row.get(f'BCR_{_s}')):.2f}",
                    })
                except Exception:
                    continue
            if _sc:
                add_word_table(doc, pd.DataFrame(_sc),
                               caption=f"Scenarios — {_nm} (ai_ecosystem_model.py enhancement output)",
                               max_cols=6)
            _src = str(_row.get("Literature_Source", "")).strip()
            if _src and _src.lower() != "nan":
                para(doc, f"Stated precedent/analogue: {_src}.", 9.5, italic=True,
                     color=RGBColor(0x60, 0x60, 0x60))

    # coordination detail
    add_h2(doc, "5.2  Coordination failures")
    cf_rows = []
    for name, c in analysis["coordination"].items():
        sev = c.get("severity")
        if sev is None or str(sev).lower() in ("none", "nan"):
            sev_disp = "—"
        else:
            sev_disp = str(sev)
        cf_rows.append({
            "Game": name,
            "Coordination failure": "Yes" if c.get("is_coordination_failure") else "No",
            "Severity": sev_disp,
            "DWL ($B)": f"{c.get('dwl_billions', 0):,.2f}",
            "Efficiency (%)": f"{c.get('efficiency_ratio', 0):.1f}" if c.get(
                "is_coordination_failure") else "100.0",
        })
    if cf_rows:
        add_word_table(doc, pd.DataFrame(cf_rows),
                       caption="Exhibit 5.3  Coordination-failure diagnostics (computed by build_technical_report.py)")
    para(doc, f"Total deadweight loss attributable to coordination failures: "
              f"${analysis['total_coordination_dwl']:,.2f}B (gross sum across games — "
              f"shared players are counted repeatedly, so this exceeds the top-down "
              f"aggregate by construction; see the discussion below).", 10.5)

    add_insights(doc, insights["welfare"], "welfare and deadweight loss")
    add_module_analysis(doc, mod.get("welfare"))
    place_section_gt(5)  # ai_ecosystem_model Tables 3.4, 5.3, 5.4, 6.1-6.3 (coordination/welfare/dynamic/policy)
    for fn in SECTION_FIGURES["welfare"]:
        embed(fn)
    add_h2(doc, "Discussion: who bears the loss")
    _dc = wl.get("dwl_by_source", {}) or {}
    _dt = sum(_dc.values()) or 1e-9
    _cons = (_dc.get("switching_costs", 0) + _dc.get("monopoly_pricing", 0)) / _dt * 100.0
    _dyn = (_dc.get("innovation_distortion", 0) + _dc.get("quality_degradation", 0)) / _dt * 100.0
    for b in [
        f"Roughly {_cons:.0f}% of the loss (switching costs plus monopoly pricing) is "
        f"consumer-facing — higher prices and locked-in users — while {_dyn:.0f}% "
        f"(innovation distortion plus quality degradation) is dynamic: slower progress "
        f"and worse products. The first is visible in bills; the second in the "
        f"products that never ship.",
        "Paradoxically the dominant archetypes absorb the largest absolute losses too: "
        "the core-game DWL sits inside Hardware–Cloud revenues, so enclosure taxes "
        "its architects as well as its subjects — NVIDIA and the hyperscalers leave "
        "money on the table by defending the walls, which is why interoperability "
        "can be framed as mutual gain rather than punishment.",
    ]:
        bullet(doc, b)
    doc.add_page_break()

    # ======================================================================
    # 6. Robustness
    # ======================================================================
    add_h1(doc, "6.  Robustness: Monte Carlo and Sensitivity")
    para(doc, f"Stability of the equilibrium outcomes was tested with {ai_ecosystem_model.MONTE_CARLO_ITERATIONS:,} "
              f"perturbed payoffs per game (log-normal noise, σ = {ai_ecosystem_model.PERTURBATION_STD*100:.0f}%), and "
              f"the headline DWL assumption was swept ±20% in the sensitivity analysis. "
              f"A five-part statistical battery (R1–R5, Tables 8.1–8.5) then subjects the "
              f"simulation itself to formal checks.")
    def _rb_csv(fn):
        """Reads one robustness-battery CSV for §6 prose; None when absent."""
        try:
            return pd.read_csv(os.path.join(gt_dir, TABLES_DIR, fn))
        except Exception:
            return None
    _t81, _t82, _t83, _t84 = (_rb_csv(f"table_8.{i}.csv") for i in (1, 2, 3, 4))
    add_h2(doc, "6.1  Simulation diagnostics and replicability (R1–R3)")
    if _t81 is not None and not _t81.empty:
        _stat = int((_t81.get("Holm_verdict", _t81.get("Verdict", pd.Series(dtype=str))) == "stationary").sum())
        _bad = sorted(set(_t81.loc[_t81.get("Holm_verdict", _t81.get("Verdict", pd.Series(dtype=str))) != "stationary", "Game"])) if "Game" in _t81.columns else []
        para(doc, f"Geweke mean-stability screens (first-10% versus last-50% window means, "
                  f"Holm-corrected across games) read stationary in {_stat}/{len(_t81)} games (Table 8.1). "
                  + (f"The exception is {', '.join(_bad)}: its early-window mean differs at the 5% level, "
                     f"so game-specific inference there carries a stationarity caveat — the pooled "
                     f"headlines do not depend on it." if _bad else "No game rejects stationarity."))
    if _t82 is not None and not _t82.empty:
        _rep = int((_t82["Verdict"] == "replicated").sum()) if "Verdict" in _t82.columns else 0
        _maxd = float(_t82["Drift_in_SE"].max()) if "Drift_in_SE" in _t82.columns else float("nan")
        para(doc, f"Full-sample reruns at two independent seeds replicate {_rep}/{len(_t82)} game-seeds "
                  f"within 3 pooled standard errors (Table 8.2; worst drift {_maxd:.2f} SE): conclusions "
                  f"do not depend on the random seed.")
    if _t83 is not None and "Verdict" in _t83.columns:
        try:
            _core = _t83[_t83["Game"].str.contains("Hardware-Cloud Providers", na=False)].sort_values("Sigma")
            _span = (f"the core game goes from {_core['Stability_Pct'].iloc[0]:.1f}% to "
                     f"{_core['Stability_Pct'].iloc[-1]:.1f}% stability") if len(_core) else "stability falls with noise"
        except Exception:
            _span = "stability falls with noise"
        para(doc, f"Perturbation-scale robustness (Table 8.3): {_t83['Verdict'].iloc[0]}. "
                  f"Stability falls mechanically as noise rises ({_span}), but the cross-game loss "
                  f"ranking — and in particular the dominance of the Hardware–Cloud game — survives "
                  f"at every noise level.")
    for _tid in ["4.1", "8_1", "8_2", "8_3"]:
        gt_tbl(6, _tid)
    add_h2(doc, "6.2  What drives game DWL: Saltelli decomposition (R4)")
    if _t84 is not None and not _t84.empty:
        _top = _t84.sort_values("ST", ascending=False).iloc[0]
        para(doc, f"Variance-based Saltelli decomposition of core-game DWL over five calibrated "
                  f"primitives (independent Uniform ±20%, N=2048, 500 bootstraps; Table 8.4, Figure 10) "
                  f"finds {_top['Parameter']} the largest total-order driver (ST={_top['ST']:.2f}, "
                  f"95% CI [{_top['ST_95_lo']:.2f}, {_top['ST_95_hi']:.2f}]). Total-order indices "
                  f"materially exceed first-order ones throughout: DWL variance is driven more by "
                  f"interactions between primitives — chiefly equilibrium-regime switches — than by "
                  f"any single input. That interaction dominance is itself the substantive result: "
                  f"point-calibration precision matters less than getting the strategic structure right.")
    gt_tbl(6, "8_4")
    _se_p = os.path.join(gt_dir, TABLES_DIR, "table_8.7_structural_estimation.csv")
    if os.path.exists(_se_p):
        try:
            _se = pd.read_csv(_se_p)
            if len(_se):
                _seshow = _se[["Block", "Parameter", "Base", "Ident_Low",
                                "Ident_High", "Status"]].copy()
                add_word_table(doc, _seshow,
                               caption="Exhibit 6.x  Breakdown-frontier calibration: identified "
                                       "intervals for A3 rate maps and A8 logit weights")
                _narrow = _se[_se["Status"] == "interior"]
                _narrow_txt = (f"{len(_narrow)} has an interior (two-sided) identified interval"
                               if len(_narrow) == 1 else
                               f"{len(_narrow)} have interior (two-sided) identified intervals")
                _narrow_names = (", ".join(_narrow["Parameter"].tolist())
                                 if len(_narrow) else "none")
                para(doc, f"Nine coefficients calibrated by coordinate bisection; "
                          f"{_narrow_txt} ({_narrow_names}) and "
                          f"the rest are open on at least one side. Wide intervals mean the "
                          f"published Nash positions and formation order do not hang on the "
                          f"judgment values; the two-sided intervals mark where "
                          f"re-estimation would matter most.",
                     10.5)
        except Exception as e:
            para(doc, f"Estimation table unreadable: {e}", 9, italic=True, color=BAD)
    add_insights(doc, insights["robustness"], "robustness")
    add_module_analysis(doc, mod.get("robustness"))
    for fn in SECTION_FIGURES["robustness"]:
        embed(fn)
    doc.add_page_break()

    # ======================================================================
    # 7. Risk, stability and cooperation
    # ======================================================================
    add_h1(doc, "7.  Risk, Stability and Cooperation")
    add_h2(doc, "7.1  Cooperative game: Shapley allocation")
    if analysis["shapley"]:
        shp = pd.DataFrame({
            "Player category": list(analysis["shapley"].keys()),
            "Shapley value ($B)": [f"{v:,.1f}" for v in analysis["shapley"].values()],
            "Share (%)": [f"{v / max(sum(analysis['shapley'].values()), 1e-9) * 100:.1f}"
                          for v in analysis["shapley"].values()],
        })
        add_word_table(doc, shp, caption="Exhibit 7.1  Shapley value allocation (computed by build_technical_report.py)")
    para(doc, f"Grand-coalition value ${analysis['grand_coalition']:,.1f}B vs. sum of standalone "
              f"values ${sum(analysis['players']['Current_Revenue_Billions']):,.1f}B → cooperative "
              f"surplus ${analysis['coop_surplus']:,.1f}B.", 10.5)
    gt_tbl(7, "5.4_coop")  # ai_ecosystem_model Table 5.4b: Shapley allocation (ai_ecosystem_model.py output)
    para(doc, "Table 7.18 sets each archetype's Shapley share against its revenue share: "
              "positive gaps mark players whose cooperative value exceeds their standalone "
              "weight — the cooperative counterfactual to the Nash outcomes of Section 4. "
              "The figure plots both shares with the gap annotated.", 10.5)
    gt_tbl(7, "7.18_shapley_vs_revenue_share")
    embed_supp("s3_shapley_vs_revenue", "S3", "Shapley Share vs Revenue Share", 7)
    add_insights(doc, insights["shapley"], "cooperation and bargaining power")
    add_module_analysis(doc, mod.get("risk"))
    # NOTE: Figure 16 (Shapley values) is embedded once, with the other risk
    # figures, at the end of Section 7.3 (see SECTION_FIGURES['risk']).

    add_h2(doc, "7.2  Circular deals")
    cm = analysis["circular"]
    crows = {
        "Total deal value": f"${cm.get('total_value', 0):,.1f}B",
        "Circular concentration (≥0.8 factor)": f"{cm.get('circ_conc', 0)*100:.1f}%",
        "Systemic share (Critical)": f"{cm.get('systemic', 0)*100:.1f}%",
        "Network density": f"{cm.get('density', 0):.3f}",
        "Bubble score": f"{cm.get('bubble_score', 0):.3f}",
    }
    cdf = pd.DataFrame({"Metric": list(crows.keys()), "Value": list(crows.values())})
    add_word_table(doc, cdf, caption="Exhibit 7.2  Circular-deals metrics (computed by build_technical_report.py)")
    add_insights(doc, insights["circular"], "circular deals and systemic fragility")
    add_module_analysis(doc, mod.get("circular"))

    add_h2(doc, "7.3  Five-pillar fragility screen (run_pipeline composite)")
    para(doc, "The five pillars adapt standard bubble-diagnostic themes — narrative "
              "tilt, multiple exuberance and expansion, a circular-hype proxy, and "
              "capex intensity — to segment level. Bands are descriptive labels, not "
              "forecasts; the ranking across archetypes is the finding.")
    _fp_p = os.path.join(gt_dir, TABLES_DIR, "table_7.7_five_pillar_screen.csv")
    if os.path.exists(_fp_p):
        try:
            _fp = pd.read_csv(_fp_p)
            if len(_fp):
                _fshow = _fp.copy()
                for _c in ["P1_Narrative_Tilt", "P2_Multiple_Exuberance",
                           "P3_Multiple_Expansion", "P4_Circular_Hype_Proxy",
                           "P5_Capex_Intensity", "Composite"]:
                    if _c in _fshow.columns:
                        _fshow[_c] = pd.to_numeric(_fshow[_c], errors="coerce").map(
                            lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
                _cols = [c for c in ["Archetype", "P1_Narrative_Tilt",
                                     "P2_Multiple_Exuberance", "P3_Multiple_Expansion",
                                     "P4_Circular_Hype_Proxy", "P5_Capex_Intensity",
                                     "Composite", "Classification"] if c in _fshow.columns]
                add_word_table(doc, _fshow[_cols],
                               caption="Exhibit 7.3  Five-pillar segment screen "
                                       "(ai_ecosystem_model.py diagnostics; NaN = uninformative pillar)")
                _note = str(_fp.get("Data_Note", pd.Series([""])).iloc[0])
                if _note and _note != "—":
                    para(doc, f"Data note: {_note}. Constants are excluded from "
                              f"composites rather than scored.", 9, italic=True,
                         color=RGBColor(0x60, 0x60, 0x60))
                try:
                    _inf = _fp[_fp["Archetype"].isin(["Hardware", "Cloud Providers"])]
                    _dn = _fp[_fp["Archetype"].isin(["Foundation Models", "LLM Wrappers"])]
                    if len(_inf) and len(_dn):
                        _im = pd.to_numeric(_inf["Composite"], errors="coerce").mean()
                        _dm = pd.to_numeric(_dn["Composite"], errors="coerce").mean()
                        para(doc, f"Paper H3 test (infrastructure shows weaker bubble "
                                  f"signals): infra mean {_im:.3f} vs downstream {_dm:.3f} — "
                                  f"{'supported' if _im < _dm else 'CONTRADICTED'}. "
                                  + ("The core layers screen worse than the application "
                                     "layers, against the paper's expectation — flag for "
                                     "committee discussion."
                                     if not _im < _dm else "Consistent with bottleneck-rent "
                                     "fundamentals."),
                             10.5)
                except Exception:
                    pass
        except Exception as e:
            para(doc, f"Five-pillar table unreadable: {e}", 9, italic=True, color=BAD)

    add_h2(doc, "7.4  Network, portfolio and stability outputs")
    place_section_gt(7)  # ai_ecosystem_model Tables A.0-A.3 (centrality, network risk, portfolio, success scores)
    for fn in SECTION_FIGURES["risk"]:
        embed(fn)

    add_h2(doc, "7.5  Bubble formation probability (Table 7.10)")
    _bf_p = os.path.join(gt_dir, TABLES_DIR, "table_7.10_bubble_formation.csv")
    if os.path.exists(_bf_p):
        try:
            _bf = pd.read_csv(_bf_p)
            if len(_bf):
                add_word_table(doc, _bf,
                               caption="Exhibit 7.4  Bubble-conditions probability by archetype "
                                       "(scenario-calibrated logistic composite, not a forecast)")
                _top = _bf.iloc[0]
                _jc = analysis.get("joint_order", {}) if isinstance(analysis, dict) else {}
                if _jc.get("error"):
                    _jc_txt = ("Rank-order validation (seeded joint check) unavailable "
                               f"in this build: {_jc['error']}.")
                else:
                    _jc_txt = (
                        f"Rank-order validation ({_jc['n_draws']:.0f} joint perturbations of all "
                        f"judgment weights, seed {_jc['seed']:.0f}): formation order exact in "
                        f"{_jc['formation_exact_pct']:.0f}% of draws (mean Kendall tau "
                        f"{_jc['formation_tau']:.2f}, FM top in {_jc['fm_top_pct']:.0f}%); burst order "
                        f"exact in {_jc['burst_exact_pct']:.0f}% (tau {_jc['burst_tau']:.2f}) with "
                        f"Hardware first in {_jc['hw_first_pct']:.0f}% of draws -- a decisive "
                        f"Hardware lead on base totals, preserved in "
                        f"{_jc['oat_formation_matches']:.0f}/{_jc['oat_formation_runs']:.0f} formation and "
                        f"{_jc['oat_burst_matches']:.0f}/{_jc['oat_burst_runs']:.0f} burst one-at-a-time "
                        f"runs (halving/doubling plus 0.25x/4x extremes). Levels move ~±25pp across draws: the rank "
                        f"structure, not the point estimates, is the finding.")
                para(doc, f"Highest bubble-conditions probability: {_top['Archetype']} "
                          f"({_top['Formation_Prob_Pct']:.1f}%, band {_top['Prob_Low_Pct']:.1f}-"
                          f"{_top['Prob_High_Pct']:.1f}%). Circular exposure dominates the "
                          f"composite by mechanism design (R5/A8). {_jc_txt}",
                     10.5)
        except Exception as e:
            para(doc, f"Formation table unreadable: {e}", 9, italic=True, color=BAD)

    add_h2(doc, "7.6  Burst impact ranking and GDP-at-risk (Tables 7.11-7.12)")
    _br_p = os.path.join(gt_dir, TABLES_DIR, "table_7.11_burst_ranking.csv")
    if os.path.exists(_br_p):
        try:
            _br = pd.read_csv(_br_p)
            if len(_br):
                _bshow = _br[["Archetype", "Burst_Rank", "Total_Loss_Severe_$B",
                               "Equity_Impact_Severe_Pct", "Revenue_Impact_Severe_Pct"]].copy()
                add_word_table(doc, _bshow,
                               caption="Exhibit 7.5  Burst impact ranking (descending severe loss; "
                                       "direct circular impairment + one network round)")
                _t1 = _br.iloc[0]
                _sev = pd.to_numeric(_br["Total_Loss_Severe_$B"], errors="coerce").sum()
                para(doc, f"Worst hit: {_t1['Archetype']} (rank 1, -${_t1['Total_Loss_Severe_$B']:,.1f}B, "
                          f"{_t1['Equity_Impact_Severe_Pct']:.1f}% of market cap). Severe bust destroys "
                          f"${_sev:,.1f}B across archetypes; mild-scenario losses are an order of "
                          f"magnitude smaller (see full table).",
                     10.5)
        except Exception as e:
            para(doc, f"Ranking table unreadable: {e}", 9, italic=True, color=BAD)
    _gdp_p = os.path.join(gt_dir, TABLES_DIR, "table_7.12_gdp_impact.csv")
    if os.path.exists(_gdp_p):
        try:
            _gdp = pd.read_csv(_gdp_p)
            if len(_gdp):
                _gshow = _gdp[_gdp["Channel"] != "Transmission rule (any economy)"].copy()
                add_word_table(doc, _gshow,
                               caption="Exhibit 7.6  US GDP-at-risk (AI-capex + wealth channels; "
                                       "BEA $30.8T baseline, no multiplier)")
                _sevc = _gshow[(_gshow["Channel"] == "Combined") & (_gshow["Severity"] == "severe")]
                if len(_sevc):
                    para(doc, f"Severe burst costs {_sevc['GDP_Share_Pct'].iloc[0]:.3f}% of US GDP "
                              f"(${_sevc['GDP_Loss_$B'].iloc[0]:,.1f}B), led by the capex channel. "
                              f"Non-US economies apply the transmission rule (last table row) to "
                              f"their own GDP -- no foreign calibration is asserted.",
                         10.5)
        except Exception as e:
            para(doc, f"GDP table unreadable: {e}", 9, italic=True, color=BAD)

    add_h2(doc, "7.7  Bubble-mitigation interventions (Table 7.13)")
    _bp_p = os.path.join(gt_dir, TABLES_DIR, "table_7.13_bubble_policy.csv")
    if os.path.exists(_bp_p):
        try:
            _bp = pd.read_csv(_bp_p)
            if len(_bp):
                _bpshow = _bp[["Intervention", "Loss_Reduction_$B_Base", "Cost_$B_Base",
                                "Net_Benefit_$B_Base", "BCR_Base"]].copy()
                add_word_table(doc, _bpshow,
                               caption="Exhibit 7.7  Burst-mitigation options (denominator = expected "
                                       "severe burst loss, NOT DWL -- rates are not comparable to §5)")
                _bb = _bp.sort_values("BCR_Base", ascending=False).iloc[0]
                para(doc, f"Highest benefit-cost ratio: {_bb['Intervention']} (BCR {_bb['BCR_Base']:.1f}); "
                          f"largest net benefit: { _bp.sort_values('Net_Benefit_$B_Base', ascending=False).iloc[0]['Intervention']}. "
                          f"All four are scenario assumptions (R5/A8-A10), i.e. policy options, not predictions.",
                     10.5)
        except Exception as e:
            para(doc, f"Bubble-policy table unreadable: {e}", 9, italic=True, color=BAD)

    add_h2(doc, "7.8  Second-round contagion and credit monitor (Tables 7.14-7.15)")
    _dr_p = os.path.join(gt_dir, TABLES_DIR, "table_7.14_debtrank_contagion.csv")
    if os.path.exists(_dr_p):
        try:
            _dr = pd.read_csv(_dr_p)
            if len(_dr):
                add_word_table(doc, _dr,
                               caption="Exhibit 7.8  DebtRank second-round contagion (Battiston et al. "
                                       "2012; equity = validated market caps; shock = severe direct losses)")
                _d2 = pd.to_numeric(_dr["Second_Round_Loss_$B"], errors="coerce").sum()
                _dd = pd.to_numeric(_dr["Direct_Loss_Severe_$B"], errors="coerce").sum()
                para(doc, f"Full reverberation adds ${_d2:,.1f}B to ${_dd:,.1f}B of direct "
                          f"losses (multiplier {(_d2 + _dd) / _dd:.3f} when direct losses are "
                          f"positive). Foundation Models absorb the largest second round; "
                          f"large equity buffers keep the multiplier near one.",
                     10.5)
                embed_supp("s4_debtrank_losses", "S4",
                           "Direct vs Second-Round Losses (DebtRank)", 7)
        except Exception as e:
            para(doc, f"DebtRank table unreadable: {e}", 9, italic=True, color=BAD)
    _cm_p = os.path.join(gt_dir, TABLES_DIR, "table_7.15_credit_monitor.csv")
    if os.path.exists(_cm_p):
        try:
            _cm = pd.read_csv(_cm_p)
            if len(_cm):
                add_word_table(doc, _cm,
                               caption="Exhibit 7.9  Credit monitor (sourced spreads and run_pipeline; "
                                       "computed hazard, ratios, haircuts and live watchlist)")
                para(doc, "Sourced facts (Oracle CDS print, BBB- action, $570B debt run_pipeline) "
                          "anchor the ledger; the hazard uses a stated 40% recovery, haircuts "
                          "are stated scenario parameters, and the watchlist reproduces the live "
                          "burst-rank order. No firm-level spread is invented.",
                     10.5)
        except Exception as e:
            para(doc, f"Credit table unreadable: {e}", 9, italic=True, color=BAD)
    doc.add_page_break()

    # ======================================================================
    # 8. Valuation
    # ======================================================================
    add_h1(doc, "8.  Valuation Integration")
    para(doc, "The enhanced valuation module expands the four category rows into company-level "
              "financials and derives 50+ valuation metrics (P/E, EV/Revenue, ROIC, circular-risk "
              "adjustments, sustainability scores and ratings) from the same market data.")
    val = load_valuation_table(gt_dir)
    if val is not None and len(val):
        try:
            keep = [c for c in ["company_name", "category", "market_cap", "pe_ratio",
                                "ev_revenue", "circular_risk_rating", "valuation_rating",
                                "circular_adjusted_value"] if c in val.columns]
            order = pd.to_numeric(val.get("market_cap"), errors="coerce")
            top = val.assign(_mc=order).sort_values("_mc", ascending=False).head(12)[keep]
            ren = {"company_name": "Company", "category": "Category",
                   "market_cap": "Market cap ($B)", "pe_ratio": "P/E",
                   "ev_revenue": "EV/Revenue", "circular_risk_rating": "Circular risk",
                   "valuation_rating": "Rating",
                   "circular_adjusted_value": "Circ.-adj. value ($B)"}
            top = top.rename(columns=ren)
            for c in ["Market cap ($B)", "P/E", "EV/Revenue", "Circ.-adj. value ($B)"]:
                if c in top.columns:
                    top[c] = pd.to_numeric(top[c], errors="coerce").map(
                        lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
            add_word_table(doc, top, caption="Exhibit 8.1  Top companies by market cap with "
                                             "key valuation metrics (ai_ecosystem_model.py enhancement output)")
            _val_sum = float(pd.to_numeric(val.get("market_cap"),
                                           errors="coerce").sum())
            para(doc, f"Full company-level file: enhanced_valuation_metrics.csv "
                      f"({len(val)} companies, summing to ${_val_sum:,.0f}B). The "
                      f"balance to the archetype total is the judgment-valued "
                      f"long tail carried in the category rows, not in any filed "
                      f"company.", 9, italic=True,
                 color=RGBColor(0x60, 0x60, 0x60))
        except Exception as e:
            para(doc, f"Valuation table unreadable: {e}", 9, italic=True, color=BAD)
    add_insights(doc, insights["valuation"], "valuation and circular-risk adjustment")
    add_module_analysis(doc, mod.get("valuation"))
    for fn in SECTION_FIGURES["valuation"]:
        embed(fn)
    doc.add_page_break()

    # ======================================================================
    # 9. Conclusion
    # ======================================================================
    add_h1(doc, "9.  Conclusion")
    add_h2(doc, "9.1  What the analysis establishes")
    _tot_dwl = float(welfare.get("total_dwl", 0))
    _pareto = float(welfare.get("total_pareto_welfare", 0))
    _eff = float(welfare.get("aggregate_efficiency", 0))
    _core = nash.get("Hardware-Cloud Providers", {})
    _core_w = _core.get("welfare_metrics", {})
    _core_eq = _core.get("nash_equilibria", [])
    _core_s = (f"{_core_eq[0]['strategies'][0]} / {_core_eq[0]['strategies'][1]}"
               if _core_eq else "n/a")
    _comp = welfare.get("dwl_by_source", {}) or {}
    _top_src = max(_comp, key=lambda k: _comp[k]) if _comp else "n/a"
    _stab = _exhibit_value(gt_dir, "table_4.1.csv", "Stability")
    _elas = _exhibit_value(gt_dir, "table_4.1.csv", "Elasticity")
    _pol_df = load_policy_table(gt_dir)
    _pol_txt = "n/a"
    try:
        if _pol_df is not None and "Net_Benefit_$B_Base" in _pol_df.columns:
            _nb = pd.to_numeric(_pol_df["Net_Benefit_$B_Base"], errors="coerce")
            if _nb.notna().any():
                _w = _pol_df.loc[_nb.idxmax()]
                _pol_txt = (f"{_w['Policy']} (net benefit "
                            f"${float(_w['Net_Benefit_$B_Base']):,.1f}B)")
    except Exception:
        pass
    _bub = analysis.get("circular", {}).get("bubble_score")
    _bub_txt = (f"{float(_bub):.3f}" if _bub is not None else "n/a")
    for b in [
        f"The AI ecosystem is a concentrated oligopoly (HHI {conc['HHI']:,.0f}; "
        f"CR4 {conc['CR4']*100:.1f}%) whose infrastructure layer plays a mutual-"
        f"defection game: the Hardware–Cloud Providers Nash equilibrium "
        f"({_core_s}) destroys "
        f"${float(_core_w.get('deadweight_loss', 0)):,.1f}B relative to the joint-surplus maximum (a Kaldor–Hicks benchmark).",
        f"In aggregate the system attains {_eff:.1f}% of its ${_pareto:,.1f}B "
        f"potential, leaving ${_tot_dwl:,.1f}B of deadweight loss dominated by "
        f"{str(_top_src).replace('_', ' ')}. Coordination failures total "
        f"${analysis.get('total_coordination_dwl', 0):,.1f}B gross across games "
        f"(shared players counted repeatedly, hence above the top-down total) — "
        f"an equilibrium outcome recoverable in principle, priced by the "
        f"cooperative surplus at "
        f"${analysis.get('coop_surplus', 0):,.0f}B.",
        f"The costed remedy is {_pol_txt}. Circularity tempers the optimism: a "
        f"bubble score of {_bub_txt} means part of the measured market is "
        f"self-referential demand that cooperation alone cannot make real.",
        f"These magnitudes are orders-of-magnitude robust"
        f"{f' (equilibrium stability {_stab:.0f}%' if _stab is not None else ' (stability n/a'}"
        f"{f', DWL elasticity {_elas:.2f}' if _elas is not None else ''}): the "
        f"location of the failure (the infrastructure bottleneck) is stable even "
        f"where the dollar point-estimates carry wide uncertainty.",
    ]:
        bullet(doc, b)

    add_h2(doc, "9.2  Limitations")
    for b in [
        "Four archetypes aggregate away intra-category concentration (notably among "
        "hyperscalers and GPU suppliers), so the HHI understates firm-level market power; "
        "firm-level games would sharpen but also complicate the diagnosis.",
        "The core model is one-shot bilateral 2×2 games. Repeated interaction, reputation, "
        "and endogenous investment (fabs, interconnect, model R&D) are outside the frame, "
        "so the analysis cannot speak to dynamic deterrence or the timing of defection.",
        f"The top-down DWL assumption carries leverage (elasticity {_elas:.2f}): the "
        "dollar totals should be read as magnitudes with uncertainty bands, not "
        "decimals — only the ranking of games, sources, and policies is asserted firmly."
        if _elas is not None else
        "The top-down DWL assumption carries leverage: dollar totals should be read as "
        "magnitudes with uncertainty bands — only the ranking of games, sources, and "
        "policies is asserted firmly.",
        "Embedded data are vintage Q2/Q3 2026 and static; Monte-Carlo stability and "
        "figures are run_pipeline artifacts reused from disk rather than recomputed here, "
        "so this report inherits the provenance of the latest ai_ecosystem_model.py run.",
        f"Per-game Nash–joint-surplus gaps sum to a gross "
        f"${analysis.get('total_coordination_dwl', 0):,.2f}B across the six games — shared "
        f"players are counted repeatedly — against the ${_tot_dwl:,.1f}B top-down aggregate "
        f"the policy and welfare conclusions rest on. The two are different accountings, "
        f"not rival estimates; the top-down figure is defended by the R2–R4 battery, but "
        f"a reader demanding a reconciled bottom-up aggregate will not find it here.",
        "Temptation and sucker adjustments are pair-level absolutes rather than share-scaled, "
        "which amplifies relative swings for small players, and the Saltelli decomposition shows "
        "DWL variance dominated by equilibrium-regime interactions rather than any single input. "
        "Both points argue for reading dollar levels as magnitudes and rankings as the firm claims.",
    ]:
        bullet(doc, b)

    add_h2(doc, "9.3  Future work")
    for b in [
        "Repeated-game extensions with calibrated discount factors to test whether the "
        "Walled-Garden lock-in survives punishment strategies or unravels into openness.",
        "Firm-level bilateral games (e.g., NVIDIA–Microsoft, OpenAI–AWS) to locate the "
        "bottleneck precisely and price remedies per relationship rather than per archetype.",
        "Endogenous policy response: modelling how the ranked interventions change payoffs "
        "and re-solve the games, instead of scoring them against fixed welfare.",
        "Live data feeds replacing the embedded vintage, with the integrity-audit battery "
        "in Section 10 as the regression gate for each refresh.",
    ]:
        bullet(doc, b)
    doc.add_page_break()

    # ======================================================================
    # 10. Integrity audit
    # ======================================================================
    add_h1(doc, "10.  Source & Output Integrity Audit")
    para(doc, "Automated checks run by this report generator against ai_ecosystem_model.py, its embedded data, "
              "and the latest on-disk artifacts.")
    status_color = {"PASS": GOOD, "WARN": WARN, "FAIL": BAD}
    chk = pd.DataFrame({
        "ID": [c["code"] for c in checks],
        "Check": [c["title"] for c in checks],
        "Status": [c["status"] for c in checks],
        "Detail": [c["detail"] for c in checks],
    })
    t = add_word_table(doc, chk, caption="Exhibit 10.1  Automated integrity checks (computed by build_technical_report.py)", max_cols=6)
    # color the Status column
    try:
        for i, c in enumerate(checks):
            cell = t.cell(i + 1, 2)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.color.rgb = status_color.get(c["status"], INK)
                    r.bold = True
    except Exception:
        pass

    add_h2(doc, "10.4  Artifact inventory")
    inv = []
    for fn in sorted(os.listdir(os.path.join(gt_dir, TABLES_DIR))):
        fp = os.path.join(gt_dir, TABLES_DIR, fn)
        inv.append({"Artifact": fn, "Type": "table", "Size": f"{os.path.getsize(fp):,} B"})
    for fn in sorted(os.listdir(os.path.join(gt_dir, PLOTS_DIR))):
        fp = os.path.join(gt_dir, PLOTS_DIR, fn)
        inv.append({"Artifact": fn, "Type": "figure", "Size": f"{os.path.getsize(fp):,} B"})
    if inv:
        inv_df = pd.DataFrame(inv)
        add_word_table(doc, inv_df, caption="Exhibit 10.2  Output inventory (computed by build_technical_report.py)", max_cols=4)

    # Placement register: where each ai_ecosystem_model.py table/figure was embedded in this report
    add_h2(doc, "10.5  Artifact placement register")
    reg_rows = []
    for entry in placement:
        reg_rows.append({
            "Kind": entry.get("kind", "").title(),
            "Label": entry.get("label", ""),
            "Source file": entry.get("file", ""),
            "Report section": (f"Section {entry['section']}" if entry.get("section")
                               else "—"),
            "Embedded": "Yes" if entry.get("ok", False) else "MISSING",
        })
    if reg_rows:
        reg_df = pd.DataFrame(reg_rows)
        add_word_table(doc, reg_df, caption="Exhibit 10.3  Placement of ai_ecosystem_model.py outputs in this report",
                       max_cols=6)
    doc.add_page_break()

    # ======================================================================
    # References (only works cited in the report body)
    # ======================================================================
    add_h1(doc, "References")
    for ref in [
        "Aldasoro, I., Doerr, S., & Rees, D. (2026). Financing the AI boom: from "
        "cash flows to debt. BIS Bulletin, No. 120. Bank for International Settlements.",
        "Baker, J. B. (2019). The antitrust paradigm. Harvard University Press.",
        "Bichler, M., et al. (2025). Algorithmic pricing and algorithmic collusion. "
        "Business & Information Systems Engineering, 67(6), 971–979.",
        "European Commission. (2022). Regulation (EU) 2022/1925 on contestable and fair "
        "markets in the digital sector (Digital Markets Act). Official Journal of the "
        "European Union.",
        "Farrell, J., & Klemperer, P. (2007). Coordination and lock-in: Competition "
        "with switching costs and network effects. In M. Armstrong & R. Porter (Eds.), "
        "Handbook of industrial organization (Vol. 3, pp. 1967–2072). Elsevier.",
        "Federal Trade Commission. (2025). Partnerships between cloud service "
        "providers and AI developers (staff report, 6(b) study).",
        "U.S. Department of Justice & Federal Trade Commission. (2023). Merger "
        "guidelines (2023 vintage HHI thresholds).",
        "Harberger, A. C. (1954). Monopoly and resource allocation. American Economic "
        "Review, 44(2), 77–87.",
        "Katz, M. L., & Shapiro, C. (1985). Network externalities, competition, and "
        "compatibility. American Economic Review, 75(3), 424–440.",
        "Lipsey, R. G., & Lancaster, K. (1956). The general theory of second best. "
        "Review of Economic Studies, 24(1), 11–32.",
        "Hagiu, A., & Wright, J. (2025). Artificial intelligence and competition "
        "policy. International Journal of Industrial Organization, 103(PA). "
        "https://doi.org/10.1016/j.ijindorg.2025.103134",
        "Korinek, A., & Vipra, J. (2025). Concentrating intelligence: Scaling and "
        "market structure in artificial intelligence. Economic Policy, 40(121), "
        "225–256. https://doi.org/10.1093/epolic/eiae057",
        "Nash, J. F. (1950). Equilibrium points in n-person games. Proceedings of the "
        "National Academy of Sciences, 36(1), 48–49.",
        "OECD. (2025). Artificial intelligence and competitive dynamics in downstream "
        "markets. OECD Publishing.",
        "Ofcom. Mandated interoperability in digital markets. Discussion paper.",
        "Ofek, E., & Richardson, M. (2003). DotCom mania: The rise and fall of "
        "Internet stock prices. Journal of Finance, 58(3), 1113–1137.",
        "Greenwood, R., Shleifer, A., & You, Y. (2019). Bubbles for Fama. "
        "Journal of Financial Economics, 131(1), 20–43.",
        "World Bank. (2026). Competition in AI markets.",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        r = p.add_run(ref)
        r.font.size = Pt(10)
    doc.add_page_break()

    # ======================================================================
    # Appendix A: full table set (sorted in logical numeric order, then A.x)
    # ======================================================================
    add_h1(doc, "Appendix A.  Complete Table Set")
    idx_csv = os.path.join(gt_dir, TABLES_DIR, "tables_index.csv")
    index_df = None
    if os.path.exists(idx_csv):
        try:
            index_df = pd.read_csv(idx_csv)
        except Exception:
            index_df = None
    appendix_entries = []  # (table_id, description, csv_path)
    if index_df is not None and len(index_df):
        for _, row in index_df.iterrows():
            file_field = str(row.get("File", ""))
            if not file_field or "(alias)" in file_field or not file_field.startswith("table_"):
                continue
            stem = file_field.split(".csv")[0]  # e.g. 'table_1.3.proj'
            p = os.path.join(gt_dir, TABLES_DIR, f"{stem}.csv")
            if os.path.exists(p):
                appendix_entries.append((str(row.get("Table", "")),
                                         str(row.get("Description", "")), p))
    else:  # fallback: alphabetical scan
        for p in sorted(glob.glob(os.path.join(gt_dir, TABLES_DIR, "table_*.csv"))):
            base = os.path.basename(p)[len("table_"):-len(".csv")]
            appendix_entries.append((base, "", p))
    appendix_entries.sort(key=lambda e: _table_sort_key(e[0]))
    if not appendix_entries:
        para(doc, "No generated tables found — run ai_ecosystem_model.py first.", italic=True, color=BAD)
    gt_section_of = {tid: sec for sec, ids in GT_TABLES_BY_SECTION.items() for tid in ids}
    for tid, desc, p in appendix_entries:
        try:
            df = pd.read_csv(p)
            display = TABLE_DISPLAY.get(tid, tid)
            where = f" (also in Section {gt_section_of[tid]})" if tid in gt_section_of else " (appendix only)"
            caption = f"Table {display}: {desc}{where}" if desc else f"Table {display}{where}"
            add_word_table(doc, df, caption=caption)
        except Exception as e:
            para(doc, f"Could not read {os.path.basename(p)}: {e}", 9, italic=True, color=BAD)
    doc.add_page_break()

    # ======================================================================
    # Appendix B: list of figures (all embedded in their topical sections)
    # ======================================================================
    add_h1(doc, "Appendix B.  Figure Register")
    para(doc, "Every figure is embedded in its topical section above; this register lists the "
              "complete set with file names, captions and the section where each appears.")
    fig_rows = []
    for fn in sorted(os.listdir(os.path.join(gt_dir, PLOTS_DIR))):
        if fn.endswith(".png"):
            title, panels = caption_for(fn, cap_map)
            fig_rows.append({
                "Figure": figure_number(fn) or "",
                "File": fn,
                "Caption": title,
                "Panels": (panels.replace("Panels: ", "") if panels else "—"),
                "Section": (f"Section {FIGURE_SECTION[fn]}" if fn in FIGURE_SECTION else "—"),
            })
    if fig_rows:
        fig_df = pd.DataFrame(fig_rows)
        add_word_table(doc, fig_df, caption="Exhibit B.1  Figure register (computed by build_technical_report.py)",
                       max_cols=5)

    # ---- placement completion: order every body exhibit by document position --
    # gt_tbl()/embed() log as they place, but the ~30 inline add_word_table
    # exhibits (3.1, 5.1, 3.5-3.7, 7.16-7.18, ...) never do -- without this
    # pass the List of Tables would omit them.
    placement[:] = complete_placement_order(doc, placement)

    # ---- front matter assembled late, displayed early -----------------------
    # Lists of Tables/Figures need the complete placement log, so they are
    # built in a scratch document and moved before front_anchor. (A body-index
    # skim does NOT work: python-docx inserts new content before the trailing
    # sectPr, so absolute indices shift — verified by experiment.)
    fdoc = Document()
    add_h1(fdoc, "List of Tables")
    lot = [{"Table": e["label"], "Section": f"Section {e['section']}"}
           for e in placement
           if e.get("kind") == "table" and e.get("ok") and e.get("section")]
    if lot:
        add_word_table(fdoc, pd.DataFrame(lot), max_cols=4)
    else:
        para(fdoc, "No tables logged.", italic=True)
    add_h1(fdoc, "List of Figures")
    lof = [{"Figure": e["label"], "Section": f"Section {e['section']}"}
           for e in placement
           if e.get("kind") == "figure" and e.get("ok") and e.get("section")]
    if lof:
        add_word_table(fdoc, pd.DataFrame(lof), max_cols=4)
    else:
        para(fdoc, "No figures logged.", italic=True)
    add_h1(fdoc, "List of Abbreviations")
    add_word_table(fdoc, pd.DataFrame([
        {"Abbreviation": "AI", "Definition": "Artificial Intelligence"},
        {"Abbreviation": "BCR", "Definition": "Benefit–Cost Ratio"},
        {"Abbreviation": "CI", "Definition": "Confidence Interval"},
        {"Abbreviation": "CR4", "Definition": "Concentration Ratio (top 4 archetypes)"},
        {"Abbreviation": "DWL", "Definition": "Deadweight Loss"},
        {"Abbreviation": "HHI", "Definition": "Herfindahl–Hirschman Index"},
        {"Abbreviation": "K-H", "Definition": "Kaldor–Hicks (joint-surplus benchmark requiring transfers)"},
        {"Abbreviation": "LLM", "Definition": "Large Language Model"},
        {"Abbreviation": "MC", "Definition": "Monte Carlo (simulation)"},
        {"Abbreviation": "NE", "Definition": "Nash Equilibrium"},
        {"Abbreviation": "TTM", "Definition": "Trailing Twelve Months (valuation inputs)"},
        {"Abbreviation": "JM", "Definition": "Joint-Surplus Maximum (Kaldor–Hicks Benchmark)"},
    ]), max_cols=4)
    for _el in list(fdoc.element.body):
        if _el.tag.endswith("}sectPr"):
            continue
        front_anchor._element.addprevious(_el)

    # ---- word count (body text + tables, excluding embedded figures) --------
    try:
        _words = sum(len(p.text.split()) for p in doc.paragraphs)
        _words += sum(len(c.text.split()) for t in doc.tables
                      for r in t.rows for c in r.cells)
        para(doc, f"Word count: ~{_words:,} words (body text and tables; "
                  f"embedded figures excluded).", 9, italic=True,
             align=WD_ALIGN_PARAGRAPH.CENTER,
             color=RGBColor(0x60, 0x60, 0x60))
    except Exception:
        pass

    # closing note
    para(doc, "", space_after=2)
    para(doc, "— End of report —", 10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    # enable field update on open (TOC/page numbers)
    try:
        settings = doc.settings.element
        uf = OxmlElement("w:updateFields")
        uf.set(qn("w:val"), "true")
        settings.append(uf)
    except Exception:
        pass

    doc.save(out_path)
    issues = verify_placement(out_path, placement, gt_dir)
    tmpdir_ctx.cleanup()
    return {"path": out_path, "placement": placement, "issues": issues}



##############################################################################
# 5. POST-BUILD VERIFICATION & ENTRY POINT -- table/figure loaders, placement audit, main().
##############################################################################

def _core_ne_strategies(nash):
    """Live core-game Nash strategies: [hardware_strategy, cloud_strategy].

    Reads the solved Hardware-Cloud Providers equilibrium from the run's own
    analysis (never hardcoded); ['n/a', 'n/a'] when the game or equilibrium
    is absent so the report degrades gracefully instead of failing the build.
    """
    try:
        eq = (nash.get("Hardware-Cloud Providers", {}) or {}).get("nash_equilibria", []) or []
        if eq:
            return [str(eq[0]["strategies"][0]), str(eq[0]["strategies"][1])]
    except Exception:
        pass
    return ["n/a", "n/a"]


def _nash_dwl(nash, key):
    """Safe DWL reader: game key -> deadweight_loss $B float; NaN if the game or metric is absent.
    """
    try:
        return float(nash[key]["welfare_metrics"].get("deadweight_loss", np.nan))
    except Exception:
        return np.nan


def gf_strategy_of(ai_ecosystem_model, players_df, player):
    """Return strategy tuple for a player the way GameTheoryFramework would."""
    defined = {
        "Hardware": ("Open Access", "Walled Garden"),
        "Cloud Providers": ("Interoperable Stack", "Proprietary Stack"),
        "Foundation Models": ("Collaborate API", "Exclusive Model"),
        "LLM Wrappers": ("Deep Integration", "Independent App"),
    }
    return defined.get(player, ("Cooperate", "Defect"))


def load_table_meta(gt_dir):
    """Load tables_index.csv -> {gt_table_id: {'desc':..., 'path':..., 'stem':...}}."""
    meta = {}
    idx_csv = os.path.join(gt_dir, TABLES_DIR, "tables_index.csv")
    if not os.path.exists(idx_csv):
        return meta
    try:
        index_df = pd.read_csv(idx_csv)
    except Exception:
        return meta
    for _, row in index_df.iterrows():
        tid = str(row.get("Table", "")).strip()
        file_field = str(row.get("File", ""))
        if not tid or not file_field or "(alias)" in file_field:
            continue
        stem = file_field.split(".csv")[0].strip()  # e.g. 'table_1.3.proj'
        p = os.path.join(gt_dir, TABLES_DIR, f"{stem}.csv")
        if os.path.exists(p):
            meta[tid] = {"desc": str(row.get("Description", "")).strip(),
                         "path": p, "stem": stem}
    return meta


def load_audit_table(gt_dir: str, filename: str):
    """Load a corrected welfare/multiplicity audit table when ai_ecosystem_model.py
    has generated it. Missing audit artifacts are nonfatal for backward
    compatibility with legacy ai_ecosystem_model.py runs."""
    path = os.path.join(gt_dir, TABLES_DIR, filename)
    try:
        return pd.read_csv(path) if os.path.exists(path) else None
    except Exception:
        return None


def load_policy_table(gt_dir):
    """Read policy_interventions_enhanced.csv; return DataFrame or None."""
    p = os.path.join(gt_dir, TABLES_DIR, "policy_interventions_enhanced.csv")
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def load_valuation_table(gt_dir):
    """Read enhanced_valuation_metrics.csv; return DataFrame or None."""
    p = os.path.join(gt_dir, TABLES_DIR, "enhanced_valuation_metrics.csv")
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def load_figure_captions(gt_dir):
    """Parse figures/figure_captions.md (written by ai_ecosystem_model.py) into
    {png_filename: {'desc': ..., 'panels': ...}}. Returns {} when absent
    (older ai_ecosystem_model.py runs) — callers fall back to FIGURE_CAPTIONS."""
    out = {}
    path = os.path.join(gt_dir, PLOTS_DIR, "figure_captions.md")
    if not os.path.exists(path):
        return out
    try:
        text = open(path, encoding="utf-8").read()
    except Exception:
        return out
    # Blocks look like: ## Figure 3 — Market Evolution\n- Files: `a.png`, `a.pdf`\n- Panels: (a), (b)
    for m in re.finditer(r"## Figure \d+ — ([^\n]+)\n- Files: `([^`]+)`[^\n]*\n- (Panels:[^\n]+)", text):
        desc, fn, panels = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        out[fn] = {"desc": desc, "panels": panels}
    return out


def caption_for(fn, cap_map):
    """Best-available caption + panel note for a figure file."""
    info = cap_map.get(fn)
    if info:
        return info["desc"], info["panels"]
    return FIGURE_CAPTIONS.get(fn, fn), ""


def complete_placement_order(doc, placement):
    """Orders the placement log by document position, adding unlogged body exhibits.

    gt_tbl()/embed() log as they place, but the ~30 inline add_word_table
    exhibits (3.1, 5.1, 3.5-3.7, 7.16-7.18, ...) never do -- so List of Tables
    would omit them. Walks the body XML in document order (paragraphs AND
    tables, which doc.paragraphs alone cannot interleave): tracks the current
    numbered section from H1s; a Table/Exhibit caption paragraph immediately
    followed by a table adopts the already-logged entry at its document
    position (keeps verify_placement's cursor monotonic) or appends a new one.
    Caption text is read from w:t nodes only -- exactly what Paragraph.text
    sees on the reopened document -- so walk labels can never skew from what
    verify_placement matches (raw itertext would also sweep in field
    instructions, deleted text, and tail text). The caption-then-table
    adjacency requirement keeps prose mentions ("Table 2 summarizes...") out.
    Stops at the appendices, whose tables keep the existing appendix-only
    treatment. Returns the ordered entry list; placement itself is untouched.
    """
    # Note: appendix numbers lead with a letter (Table A.0), so the
    # caption class must accept letters as well as digits/dots.
    _cap_re = re.compile(r"^(Table|Exhibit)\s+[A-Za-z\d.]+\b")
    _h1_re = re.compile(r"^(\d+)\.\s+\S")
    _known = {e["label"]: e for e in placement}
    _ordered, _sec, _pending = [], None, None

    def _adopt_figure(_t):
        """Adopts an already-logged figure caption at its document position.

        Exact-label match only, so prose mentions can never create entries;
        unknown captions (should not happen -- every figure goes through
        embed()) are ignored rather than invented.
        """
        _e = _known.pop(_t, None)
        if _e is not None:
            _ordered.append(_e)
            return True
        return False

    for _child in doc.element.body:
        if _child.tag == qn("w:tbl"):
            if _pending is not None:
                _e = _known.pop(_pending[0], None)
                if _e is None:
                    _e = {"kind": "table", "ok": True, "file": "",
                          "label": _pending[0], "section": _pending[1]}
                _ordered.append(_e)
            _pending = None
            continue
        if _child.tag != qn("w:p"):
            _pending = None
            continue
        _t = "".join((t.text or "") for t in _child.iter(qn("w:t"))).strip()
        _style = ""
        try:
            _pPr = _child.find(qn("w:pPr"))
            _pStyle = _pPr.find(qn("w:pStyle")) if _pPr is not None else None
            if _pStyle is not None:
                _style = _pStyle.get(qn("w:val")) or ""
        except Exception:
            pass
        if _style.startswith("Heading"):
            if _t.startswith("Appendix"):
                break
            _m = _h1_re.match(_t)
            if _m:
                _sec = int(_m.group(1))
            _pending = None
            continue
        if _t and _cap_re.match(_t):
            _pending = (_t, _sec)
            continue
        if _t and (_t.startswith("Figure ") or _t.startswith("Supplemental ")):
            _adopt_figure(_t)
            _pending = None
            continue
        if _t:
            _pending = None
    _ordered.extend(_known.values())  # defensive: logged but never seen as a paragraph
    return _ordered


def verify_placement(doc_path, expected, gt_dir):
    """Verify every logged ai_ecosystem_model table/figure caption appears in the docx after its
    section's heading. Returns a list of issue strings (empty == all good)."""
    issues = []
    try:
        doc = Document(doc_path)
    except Exception as e:
        return [f"cannot open generated docx: {e}"]
    texts = [p.text.strip() for p in doc.paragraphs]

    # heading index for numbered sections (1..9) and appendices
    heading_idx = {}
    for i, t in enumerate(texts):
        m = re.match(r"^(\d+)\.\s+\S", t)  # e.g. "4.  Strategic Interaction ..."
        if m:
            heading_idx[int(m.group(1))] = i
    n_media = len(doc.inline_shapes)
    n_png = len([p for p in glob.glob(os.path.join(gt_dir, PLOTS_DIR, "*.png"))
                 if not os.path.basename(p).startswith(PREDICTION_FIGURE_PREFIXES)])
    # Supplemental figures live in figures/supplemental/ (outside the 1-18
    # registry glob): each successfully embedded one accounts for one more
    # inline shape, identified by its supplemental/ file tag in placement.
    n_supp = sum(1 for e in expected
                 if e.get("kind") == "figure" and e.get("ok")
                 and str(e.get("file", "")).startswith("supplemental/"))
    if n_media != n_png + n_supp:
        issues.append(f"image count mismatch: {n_media} embedded vs {n_png} PNGs on disk"
                      f" + {n_supp} supplemental embeds")

    cursor = 0
    for entry in expected:
        label = entry.get("label", "")
        sec = entry.get("section")
        kind = entry.get("kind", "")
        if not entry.get("ok", True):
            issues.append(f"[MISSING SOURCE] {kind}: {label}")
            continue
        # find label after cursor
        found = -1
        for j in range(cursor, len(texts)):
            if texts[j] == label:
                found = j
                break
        if found < 0:
            hint = ""
            try:
                close = difflib.get_close_matches(label, texts, n=1, cutoff=0.5)
                if close:
                    hint = f" (closest paragraph: '{close[0][:160]}')"
            except Exception:
                pass
            issues.append(f"[NOT FOUND] {kind} '{label}' missing from docx{hint}")
            continue
        cursor = found + 1
        if sec is not None and sec in heading_idx:
            if heading_idx[sec] > found:
                issues.append(f"[SECTION] '{label}' (section {sec}) appears before its heading")
    return issues


def _table_sort_key(tid):
    """Sort key for ai_ecosystem_model table ids: numeric sections (1.x…) first, then A.x."""
    tid = str(tid).strip()
    m = re.match(r"^(\d+)\.(\d+)(.*)$", tid)
    if m:
        return (0, int(m.group(1)), int(m.group(2)), m.group(3))
    a = re.match(r"^[Aa]\.(\d+)(.*)$", tid)
    if a:
        return (1, int(a.group(1)), 0, a.group(2))
    return (2, 0, 0, tid)


def figure_number(fn):
    """Maps a figure filename to its report Figure number: figure_N.png -> N (1-18 cover the publication set); None when unknown.
    """
    m = re.match(r"figure_(\d+)\.png", fn)
    if m:
        return m.group(1)
    return None


def main():
    """Four-stage driver with a placement gate.

    [1/4] quietly imports ai_ecosystem_model.py; [2/4] recomputes headlines via analyze();
    [3/4] profiles source and runs integrity_checks(); [4/4] assembles the DOCX
    and runs verify_placement() over the placement log. Any placement issue exits
    1 so a misfiled table/figure can never ship silently; success prints the path
    and reminds the user to refresh the TOC field (Ctrl+A, F9) in Word.
    """
    ap = argparse.ArgumentParser(description="Generate a publication DOCX report for ai_ecosystem_model.py")
    ap.add_argument("--ai_ecosystem_model-dir", default=SCRIPT_DIR, help="directory containing ai_ecosystem_model.py + outputs")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output .docx path")
    args = ap.parse_args()
    gt_dir = args.ai_ecosystem_model_dir
    tables_dir = os.path.join(gt_dir, TABLES_DIR)
    plots_dir = os.path.join(gt_dir, PLOTS_DIR)

    logging.basicConfig(level=logging.INFO, format=RUN_LOG_FORMAT, datefmt=RUN_LOG_DATEFMT)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or SCRIPT_DIR
    run_log_path = setup_run_log_file(out_dir)

    print("=" * 72)
    print("ai_ecosystem_model.py REPORT GENERATOR")
    print(f"GT directory : {gt_dir}")
    print(f"Tables dir   : {tables_dir}")
    print(f"Plots dir    : {plots_dir}")
    print(f"Output       : {args.out}")
    if run_log_path:
        print(f"Run log      : {run_log_path}")
    print("=" * 72)
    logger.info(f"Report run started: {' '.join(sys.argv)}"
                + (f" (log: {run_log_path})" if run_log_path else ""))

    if not os.path.exists(os.path.join(gt_dir, MODEL_FILE)):
        sys.exit(f"ERROR: {MODEL_FILE} not found in {gt_dir}")
    if not os.path.exists(tables_dir):
        print(f"WARNING: {tables_dir} missing — run ai_ecosystem_model.py first so tables/figures exist.")
    if not os.path.exists(plots_dir):
        print(f"WARNING: {plots_dir} missing — run ai_ecosystem_model.py first so figures exist.")

    print("[1/4] Loading ai_ecosystem_model.py (quietly)...")
    ai_ecosystem_model = load_gt(gt_dir)

    print("[2/4] Recomputing headline analysis from ai_ecosystem_model.py classes...")
    data = ai_ecosystem_model.load_all_data()
    analysis = analyze(ai_ecosystem_model, data)
    logger.info(f"Headline analysis recomputed: {len(analysis.get('nash', {}))} games, "
                f"{len(analysis.get('shapley', {}))} Shapley values")

    print("[3/4] Static code profile + integrity audit...")
    prof = code_profile(os.path.join(gt_dir, MODEL_FILE))
    checks = integrity_checks(ai_ecosystem_model, data, prof, tables_dir, plots_dir)
    for c in checks:
        print(f"      [{c['status']:4s}] {c['code']}: {c['title']} — {c['detail']}")
    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARN"]
    logger.info(f"Integrity audit: {len(checks)} checks, {len(fails)} FAIL, {len(warns)} WARN")
    for c in fails + warns:
        logger.warning(f"Audit {c['status']} {c['code']}: {c['title']} — {c['detail']}")

    print("[4/4] Assembling DOCX report...")
    result = build_report(ai_ecosystem_model, analysis, prof, checks, gt_dir, args.out)
    out_path = result["path"]
    size = os.path.getsize(out_path)
    print(f"\n✓ Report written: {out_path} ({size/1024:.0f} KB)")
    print("  Tip: open in Word and press Ctrl+A then F9 to refresh the Table of Contents.")
    logger.info(f"Report written: {out_path} ({size/1024:.0f} KB)")

    # Placement verification
    issues = result.get("issues", [])
    total = len(result.get("placement", []))
    print(f"\nPlacement verification: {total} ai_ecosystem_model artifacts logged.")
    if issues:
        print("  ISSUES FOUND:")
        for iss in issues:
            print(f"    ✗ {iss}")
            logger.warning(f"Placement issue: {iss}")
        logger.info(f"Report run FAILED placement gate ({len(issues)} issues)"
                    + (f" (log: {run_log_path})" if run_log_path else ""))
        sys.exit(1)
    else:
        print("  All tables/figures verified at the correct sections: OK.")
        logger.info(f"Report run finished: {total} artifacts placed, no issues"
                    + (f" (log: {run_log_path})" if run_log_path else ""))


if __name__ == "__main__":
    main()
