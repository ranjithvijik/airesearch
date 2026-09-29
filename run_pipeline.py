#!/usr/bin/env python3
"""End-to-end run_pipeline: ai_ecosystem_model.py -> build_technical_report.py -> build_interactive_dashboard.py -> tests -> build_retirement_exposure.py -> pdflatex -> build_prediction_enhancement.py -> build_crisis_backtest.py -> build_compare_2008_ai.py -> build_prediction_artifacts.py.

Runs the full chain that produces the technical report, the dashboard, the
test verdict, the article PDF, and the prediction/backtest distributables,
verifying each stage's artifacts before the next stage consumes them. Fails
fast with the missing artifact named.

Order note: the test suite runs BEFORE pdflatex on purpose -- a red suite
means the outputs are unverified, and no article PDF should be typeset from
them. Use --skip-tests to build the PDF regardless (not recommended). The
prediction block (stages 6-9) runs after the PDF because no tested artifact
consumes its outputs; use --skip-prediction to stop after the PDF.

Requirements (Python 3.7+): this file itself is standard library only
(argparse/logging/os/re/shutil/subprocess/sys/datetime) -- no pip install
for the orchestrator. But each stage it drives needs its own stack, so the
full chain needs the union of all ten files' requirements -- install once
with pip:
  pip install numpy pandas matplotlib seaborn networkx scipy openpyxl python-docx Pillow beautifulsoup4
  - ai_ecosystem_model stage: numpy / pandas / matplotlib / seaborn / networkx / scipy
    (data layer, figure engine, network metrics; openpyxl optional --
    without it .xlsx exports are skipped while CSV/LaTeX still land).
  - build_technical_report stage: numpy / pandas / python-docx (import name: docx) / Pillow
    (import name: PIL), plus the ai_ecosystem_model stack transitively (build_technical_report imports ai_ecosystem_model).
  - dashboard stage: standard library only, except the Games tab which
    imports ai_ecosystem_model and needs the ai_ecosystem_model stack (numpy/pandas/matplotlib).
  - tests stage: numpy / pandas / matplotlib / openpyxl / python-docx.
  - pdf stage: not a pip package -- a LaTeX distribution providing
    `pdflatex` (e.g. TeX Live or MacTeX).
  - build_retirement_exposure stage: numpy / pandas / matplotlib (reads only
    tables/table_7.12_gdp_impact.csv plus sourced constants).
  - build_prediction_enhancement stage: numpy / pandas / matplotlib.
  - build_crisis_backtest stage: numpy / pandas / matplotlib, plus the
    ai_ecosystem_model stack transitively (build_crisis_backtest imports ai_ecosystem_model).
  - build_compare_2008_ai stage: numpy / pandas / matplotlib.
  - build_prediction_artifacts stage: not a pip package -- `pandoc`, plus
    python-docx and beautifulsoup4 (import name: bs4).

Usage:
    python3 run_pipeline.py                 # full chain
    python3 run_pipeline.py --from build_technical_report      # resume: build_technical_report -> dashboard -> tests -> pdf -> prediction block
    python3 run_pipeline.py --skip-tests    # build everything except the suite
    python3 run_pipeline.py --skip-pdf      # skip the article PDF (prediction stages still run)
    python3 run_pipeline.py --skip-prediction  # stop after the PDF

A run_pipeline_run_YYYYMMDD_HHMMSS.log file is written next to this file on every
run, mirroring all stage output for the session.
"""
import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")
PLOTS = os.path.join(HERE, "figures")
SUPP = os.path.join(PLOTS, "supplemental")
ARTICLE = os.path.join(HERE, "article")
TEX_NAME = "ai_circularity_article.tex"
GENERATED_TEX = "generated_a8_intervals.tex"
PDF_NAME = "ai_circularity_article.pdf"
DOCX_NAME = "AI_Ecosystem_Game_Theory_Technical_Report.docx"
DASH_NAME = "dashboard.html"
PRED_DOCX_NAME = "AI_Bubble_Burst_Prediction_Report.docx"
PRED_HTML_NAME = "prediction.html"
BACKTEST_T = os.path.join(TABLES, "backtest")

STAGES = ("ai_ecosystem_model", "build_technical_report", "dashboard", "tests",
          "build_retirement_exposure", "pdf",
          "build_prediction_enhancement", "build_crisis_backtest",
          "build_compare_2008_ai", "build_prediction_artifacts")
PREDICTION_STAGES = ("build_prediction_enhancement", "build_crisis_backtest",
                     "build_compare_2008_ai", "build_prediction_artifacts")

RUN_LOG_PREFIX = "run_pipeline_run"
RUN_LOG_SUFFIX = ".log"
RUN_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
RUN_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("run_pipeline")


def setup_run_log_file(directory):
    """Attaches a per-run log file mirroring console output; returns its path.

    The file is named ``run_pipeline_run_YYYYMMDD_HHMMSS.log`` inside
    ``directory`` (the run_pipeline folder, so the log sits next to the tables,
    figures, report, and dashboard it verifies). The handler uses the shared
    run-log format at INFO level on the root logger, capturing run_pipeline
    records and every stage's streamed child output. Idempotent per path
    (a same-second collision with an unowned file gains a _NN suffix
    instead of truncating it) and never raises: an unwritable location
    warns on stderr and returns None. Call only from main().
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


def ensure_console():
    """Attaches a stdout console handler if the root logger has none yet.

    Keeps `python3 run_pipeline.py` chatty when run as a script, while leaving
    the test suite's own logging configuration untouched.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(RUN_LOG_FORMAT, datefmt=RUN_LOG_DATEFMT))
        root.addHandler(console)


def clear_bytecode_cache(directory):
    """Removes a stale __pycache__ so stages never run outdated bytecode.

    Python validates cached bytecode by source mtime/size at coarse
    granularity, so rapid successive edits can leave a cache entry that
    shadows newer source -- a stage then runs obsolete code and fails
    spuriously. The cache is purely regenerable, so deleting it is always
    safe. Never raises.
    """
    shutil.rmtree(os.path.join(directory, "__pycache__"), ignore_errors=True)


def run(cmd, cwd, stage):
    """Runs one stage, streaming output to the session log; raises on failure.

    Every line of child output is logged so the run_pipeline log is a complete
    record; a nonzero exit raises RuntimeError naming the stage and command.
    Python stages run with a cleared __pycache__ so stale bytecode can never
    shadow edited source (see clear_bytecode_cache).
    """
    if os.path.basename(cmd[0]).startswith("python"):
        clear_bytecode_cache(cwd)
    logger.info(f"[{stage}] $ {' '.join(cmd)} (cwd={cwd})")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info(f"[{stage}] {line.rstrip()}")
    rc = proc.wait()
    proc.stdout.close()
    if rc != 0:
        raise RuntimeError(f"stage '{stage}' failed with exit code {rc}: {' '.join(cmd)}")
    logger.info(f"[{stage}] done (exit 0)")


def missing(paths):
    """Returns the subset of paths that are absent or empty files."""
    bad = []
    for p in paths:
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            bad.append(os.path.relpath(p, HERE))
    return bad


def check_gt_outputs():
    """Verifies the ai_ecosystem_model.py artifacts every downstream stage consumes."""
    want = [os.path.join(TABLES, "tables_index.csv"),
            os.path.join(TABLES, "table_3.5_welfare_benchmark_audit.csv"),
            os.path.join(TABLES, "table_3.6_equilibrium_multiplicity.csv"),
            os.path.join(TABLES, "table_3.7_repeated_game_sustainability.csv"),
            os.path.join(TABLES, "table_7.16.market.share.evolution.csv"),
            os.path.join(TABLES, "table_7.17.lerner.markup.summary.csv"),
            os.path.join(TABLES, "table_7.18.shapley.vs.revenue.share.csv")]
    want += [os.path.join(PLOTS, f"figure_{n}.png") for n in range(1, 19)]
    want += [os.path.join(SUPP, f"s{i}_{s}.png") for i, s in
             ((1, "market_share_levels"), (2, "lerner_markup_bands"),
              (3, "shapley_vs_revenue"), (4, "debtrank_losses"),
              (5, "joint_gains_transfers"))]
    bad = missing(want)
    if bad:
        raise RuntimeError("ai_ecosystem_model.py outputs incomplete -- missing/empty:\n  "
                           + "\n  ".join(bad))
    logger.info(f"[ai_ecosystem_model] verified {len(want)} artifacts "
                f"(index + audits + 7.16-18, figures 1-18, supplemental S1-S5)")


def stage_gt():
    """Runs ai_ecosystem_model.py, then verifies its tables and figures landed on disk."""
    run([sys.executable, os.path.join(HERE, "ai_ecosystem_model.py")], HERE, "ai_ecosystem_model")
    check_gt_outputs()


def stage_rpt():
    """Runs build_technical_report.py, then verifies the DOCX report exists and is non-empty."""
    run([sys.executable, os.path.join(HERE, "build_technical_report.py")], HERE, "build_technical_report")
    out = os.path.join(HERE, DOCX_NAME)
    bad = missing([out])
    if bad:
        raise RuntimeError(f"build_technical_report.py finished but the report is missing: {bad[0]}")
    logger.info(f"[build_technical_report] verified {DOCX_NAME} ({os.path.getsize(out) // 1024} KB)")


def stage_dashboard():
    """Runs build_interactive_dashboard.py, then verifies dashboard.html exists."""
    run([sys.executable, os.path.join(HERE, "build_interactive_dashboard.py")], HERE, "dashboard")
    out = os.path.join(HERE, DASH_NAME)
    bad = missing([out])
    if bad:
        raise RuntimeError(f"dashboard build finished but {DASH_NAME} is missing")
    logger.info(f"[dashboard] verified {DASH_NAME} ({os.path.getsize(out) // 1024} KB)")


def stage_tests():
    """Runs the full test_ai_ecosystem_model.py suite; any failure blocks the article PDF."""
    run([sys.executable, "-m", "unittest", "test_ai_ecosystem_model", "-v"], HERE, "tests")
    logger.info("[tests] suite green -- outputs verified, article may build")


def check_retirement_outputs():
    """Verifies the retirement-exposure addendum the article appendix prints."""
    want = [os.path.join(BACKTEST_T, "retirement_exposure.csv"),
            os.path.join(PLOTS, "retirement_exposure.png")]
    bad = missing(want)
    if bad:
        raise RuntimeError("build_retirement_exposure.py outputs incomplete -- missing/empty:\n  "
                           + "\n  ".join(bad))
    logger.info("[build_retirement_exposure] verified retirement_exposure.csv + figure")


def stage_retirement():
    """Runs build_retirement_exposure.py, then verifies its table and figure landed on disk."""
    run([sys.executable, os.path.join(HERE, "build_retirement_exposure.py")], HERE,
        "build_retirement_exposure")
    check_retirement_outputs()


def check_pdf_prerequisites():
    """Fails fast listing what is missing: tex, figures, or the compiler."""
    want = [os.path.join(ARTICLE, TEX_NAME),
            os.path.join(ARTICLE, GENERATED_TEX),
            os.path.join(PLOTS, "figure_3.png"),
            os.path.join(PLOTS, "figure_7.png"),
            os.path.join(PLOTS, "retirement_exposure.png"),
            os.path.join(SUPP, "s5_joint_gains_transfers.png")]
    bad = missing(want)
    if bad:
        raise RuntimeError("cannot typeset -- missing inputs:\n  " + "\n  ".join(bad)
                           + "\n(run the ai_ecosystem_model stage first)")
    if shutil.which("pdflatex") is None:
        raise RuntimeError("cannot typeset -- `pdflatex` not found on PATH "
                           "(install TeX Live or MacTeX, or use --skip-pdf)")


def stage_pdf():
    """Compiles the article twice and checks the log for errors.

    Two passes resolve the table of contents and cross-references. Afterwards
    the TeX log is scanned: any '!'-prefixed error line or undefined-reference
    warning fails the stage (a PDF may still be written by nonstopmode, but it
    would be inaccurate -- hence the gate).
    """
    check_pdf_prerequisites()
    for i in (1, 2):
        run(["pdflatex", "-interaction=nonstopmode", TEX_NAME], ARTICLE,
            f"pdflatex-{i}")
    pdf = os.path.join(ARTICLE, PDF_NAME)
    bad = missing([pdf])
    if bad:
        raise RuntimeError("pdflatex finished but no PDF was produced")
    log_path = os.path.join(ARTICLE, "ai_circularity_article.log")
    problems = []
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("!"):
                    problems.append(line.strip())
                elif "undefined references" in line.lower() and "Rerun" not in line:
                    problems.append(line.strip())
                elif re.search(r"LaTeX Warning.*undefined", line):
                    problems.append(line.strip())
    except OSError as e:
        raise RuntimeError(f"cannot read TeX log for verification: {e}")
    seen = list(dict.fromkeys(problems))
    if seen:
        raise RuntimeError("PDF written but TeX log reports problems:\n  "
                           + "\n  ".join(seen[:20]))
    logger.info(f"[pdf] verified {PDF_NAME} ({os.path.getsize(pdf) // 1024} KB), "
                f"log clean (2 passes, no errors, no undefined references)")


PRED_FIGURES = ("bubble_policy_bcr", "burst_waterfall", "cascade_stack",
                "company_losses", "concentration", "equity_revenue_impact",
                "expected_losses", "five_pillar", "formation_bands", "fragility",
                "gatekeeper", "gauge_backtest", "gdp_waterfall", "geography_bars",
                "layer_composition", "ml_horserace", "open_weights", "robustness_survival",
                "scenario_ladder", "screen_flags", "stability_pockets",
                "timing_call", "timing_window", "trapped_vs_efficiency",
                "valuation_scatter")


def check_enhancement_outputs():
    """Verifies the prediction-exhibit figures the article prints."""
    want = [os.path.join(PLOTS, f"pred_{s}.png") for s in PRED_FIGURES]
    bad = missing(want)
    if bad:
        raise RuntimeError("build_prediction_enhancement.py outputs incomplete -- missing/empty:\n  "
                           + "\n  ".join(bad))
    logger.info(f"[build_prediction_enhancement] verified {len(want)} article figures (pred_*)")


def stage_enhancement():
    """Runs build_prediction_enhancement.py, then verifies its figures landed on disk."""
    run([sys.executable, os.path.join(HERE, "build_prediction_enhancement.py")], HERE,
        "build_prediction_enhancement")
    check_enhancement_outputs()


def check_backtest_outputs():
    """Verifies the 2008-crisis + telecom backtest tables and figures the compare stage consumes."""
    want = [os.path.join(BACKTEST_T, f"backtest_{s}.csv") for s in
            ("actuals", "burst", "formation", "games", "joint", "market", "mc",
             "policy", "portfolio", "saltelli", "shapley", "stability", "verdict",
             "welfare")]
    want += [os.path.join(BACKTEST_T, "valuation_2008.csv")]
    want += [os.path.join(BACKTEST_T, f"telecom_{s}.csv") for s in
             ("actuals", "burst", "formation", "games", "joint", "market", "mc",
              "policy", "portfolio", "saltelli", "shapley", "stability", "verdict",
              "welfare")]
    want += [os.path.join(BACKTEST_T, "valuation_telecom.csv")]
    want += [os.path.join(PLOTS, f"backtest_{s}.png") for s in
             ("2008", "games", "gdp_channels", "hhi_compare", "rank_validation",
              "saltelli", "stability_compare", "timeline")]
    want += [os.path.join(PLOTS, f"payoff_matrices_{s}.png") for s in
             ("2008", "telecom")]
    want += [os.path.join(PLOTS, f"telecom_{s}.png") for s in
             ("overview", "games", "rank", "saltelli")]
    bad = missing(want)
    if bad:
        raise RuntimeError("build_crisis_backtest.py outputs incomplete -- missing/empty:\n  "
                           + "\n  ".join(bad))
    logger.info(f"[build_crisis_backtest] verified {len(want)} artifacts "
                f"(30 backtest tables, 14 backtest figures)")


def stage_backtest():
    """Runs build_crisis_backtest.py, then verifies its tables and figures landed on disk."""
    run([sys.executable, os.path.join(HERE, "build_crisis_backtest.py")], HERE,
        "build_crisis_backtest")
    check_backtest_outputs()


def check_compare_outputs():
    """Verifies the 2008-vs-AI comparison summary and exhibits."""
    want = [os.path.join(TABLES, "compare_summary.csv")]
    want += [os.path.join(PLOTS, f"compare_{s}.png") for s in
             ("rank_shape", "remedies", "tail_risk")]
    bad = missing(want)
    if bad:
        raise RuntimeError("build_compare_2008_ai.py outputs incomplete -- missing/empty:\n  "
                           + "\n  ".join(bad))
    logger.info(f"[build_compare_2008_ai] verified {len(want)} artifacts "
                f"(compare_summary.csv, 3 compare figures)")


def stage_compare():
    """Runs build_compare_2008_ai.py, then verifies its summary and figures landed on disk."""
    run([sys.executable, os.path.join(HERE, "build_compare_2008_ai.py")], HERE,
        "build_compare_2008_ai")
    check_compare_outputs()


def check_artifact_prerequisites():
    """Fails fast listing what is missing: pandoc, the reference DOCX, the tex, or printed figures."""
    if shutil.which("pandoc") is None:
        raise RuntimeError("cannot build prediction artifacts -- `pandoc` not found on PATH "
                           "(install pandoc, or use --skip-prediction)")
    want = [os.path.join(HERE, DOCX_NAME),
            os.path.join(ARTICLE, TEX_NAME),
            os.path.join(ARTICLE, GENERATED_TEX)]
    want += [os.path.join(PLOTS, f"pred_{s}.png") for s in
             ("bubble_policy_bcr", "valuation_scatter")]
    want += [os.path.join(PLOTS, f"backtest_{s}.png") for s in
             ("2008", "rank_validation")]
    bad = missing(want)
    if bad:
        raise RuntimeError("cannot build prediction artifacts -- missing inputs:\n  "
                           + "\n  ".join(bad)
                           + "\n(run the model/enhancement/backtest stages first)")
    logger.info("[build_prediction_artifacts] prerequisites present "
                "(pandoc, reference DOCX, tex, printed figures)")


def stage_artifacts():
    """Runs build_prediction_artifacts.py, then verifies the DOCX and HTML landed on disk."""
    run([sys.executable, os.path.join(HERE, "build_prediction_artifacts.py")], HERE,
        "build_prediction_artifacts")
    out_docx = os.path.join(HERE, PRED_DOCX_NAME)
    out_html = os.path.join(HERE, PRED_HTML_NAME)
    bad = missing([out_docx, out_html])
    if bad:
        raise RuntimeError("build_prediction_artifacts.py finished but outputs are missing: "
                           + ", ".join(bad))
    logger.info(f"[build_prediction_artifacts] verified {PRED_DOCX_NAME} "
                f"({os.path.getsize(out_docx) // 1024} KB) + {PRED_HTML_NAME} "
                f"({os.path.getsize(out_html) // 1024} KB)")


def parse_args(argv=None):
    """Parses CLI flags: resume point, skipping tests, the PDF, or the prediction block."""
    parser = argparse.ArgumentParser(
        description="End-to-end run_pipeline: ai_ecosystem_model -> build_technical_report -> dashboard -> tests -> build_retirement_exposure -> pdf -> build_prediction_enhancement -> build_crisis_backtest -> build_compare_2008_ai -> build_prediction_artifacts.")
    parser.add_argument("--from", dest="start", choices=STAGES, default="ai_ecosystem_model",
                        help="resume the chain at this stage (default: ai_ecosystem_model)")
    parser.add_argument("--skip-tests", action="store_true",
                        help="build the PDF even if the suite is red (not recommended)")
    parser.add_argument("--skip-pdf", action="store_true",
                        help="skip the article PDF (prediction stages still run)")
    parser.add_argument("--skip-prediction", action="store_true",
                        help="stop after the PDF, do not run stages 6-9")
    return parser.parse_args(argv)


def main(argv=None):
    """Runs the requested stages in order, failing fast on the first error.

    Resume mode (--from X) still verifies X's input artifacts before running
    X, so a resume onto stale outputs fails with the missing file named
    instead of a confusing downstream crash.
    """
    args = parse_args(argv)
    ensure_console()
    log_path = setup_run_log_file(HERE)
    logger.info(f"run_pipeline start (stages from '{args.start}'; "
                f"skip_tests={args.skip_tests}, skip_pdf={args.skip_pdf}, "
                f"skip_prediction={args.skip_prediction})"
                + (f"; session log: {log_path}" if log_path else ""))
    order = STAGES[STAGES.index(args.start):]
    try:
        if "ai_ecosystem_model" in order:
            stage_gt()
        if "build_technical_report" in order:
            if "ai_ecosystem_model" not in order:
                check_gt_outputs()  # resume: refuse to build on stale tables
            stage_rpt()
        if "dashboard" in order:
            stage_dashboard()
        if "tests" in order and not args.skip_tests:
            stage_tests()
        elif "tests" in order:
            logger.warning("tests skipped by --skip-tests; PDF is unverified")
        if "build_retirement_exposure" in order:
            if "ai_ecosystem_model" not in order:
                check_gt_outputs()  # resume: refuse to build on stale inputs
            stage_retirement()
        if "pdf" in order and not args.skip_pdf:
            stage_pdf()
        if "build_prediction_enhancement" in order and not args.skip_prediction:
            if "ai_ecosystem_model" not in order:
                check_gt_outputs()  # resume: refuse to build on stale tables
            stage_enhancement()
        if "build_crisis_backtest" in order and not args.skip_prediction:
            if "ai_ecosystem_model" not in order:
                check_gt_outputs()  # resume: refuse to run on stale inputs
            stage_backtest()
        if "build_compare_2008_ai" in order and not args.skip_prediction:
            if "build_crisis_backtest" not in order:
                check_backtest_outputs()  # resume: refuse without backtest tables
            if "ai_ecosystem_model" not in order:
                check_gt_outputs()
            stage_compare()
        if "build_prediction_artifacts" in order and not args.skip_prediction:
            check_artifact_prerequisites()  # always: pandoc + reference DOCX + tex + figures
            stage_artifacts()
        elif "build_prediction_artifacts" in order:
            logger.warning("prediction stages skipped by --skip-prediction; "
                           "distributables are stale")
        logger.info("run_pipeline COMPLETE: " + " -> ".join(
            s for s in order if not (s == "tests" and args.skip_tests)
            and not (s == "pdf" and args.skip_pdf)
            and not (s in PREDICTION_STAGES and args.skip_prediction)))
    except RuntimeError as e:
        logger.error(f"run_pipeline FAILED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
