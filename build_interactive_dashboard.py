#!/usr/bin/env python3
"""Build a single-file interactive HTML dashboard from ai_ecosystem_model.py outputs.

Reads tables/*.csv (+ live game matrices via ai_ecosystem_model import) and emits
one self-contained dashboard.html: inline JSON data, vanilla JS, inline SVG
charts, zero external dependencies (works from file:// offline).

Dashboard sections (tab registry TABS, injected as TABSDEF):
  Overview (KPIs) | Pipeline Tour (guided walkthrough of every ai_ecosystem_model.py module,
  parsed live from ai_ecosystem_model.py section banners + class docstrings) | Report (the
  build_technical_report.py DOCX parsed into browsable sections with insight callouts and mirror
  tables that jump to their live tab) | Market | Games | Welfare |
  Monte Carlo | Risk & Network | Policy | Revenue & Valuation | Diagnostics
  (Tables 7.x/8.x) | Figures (print-resolution PNGs with lightbox viewer).

Chart library (JS_CORE): vbar/hbar/donut/line render inline SVG with wrapped
(not truncated) labels, dynamic padding, gradient fills and hover tooltips;
all tables are sortable/filterable with CSV export.

Usage:
    python3 build_interactive_dashboard.py [--tables-dir D] [--plots-dir P] [--out O]
                               [--ai_ecosystem_model-file G] [--report R]
Defaults resolve to ./tables, ./figures, ./dashboard.html
relative to the current working directory; --ai_ecosystem_model-file defaults to ai_ecosystem_model.py next to
the tables dir and --report to the build_technical_report.py DOCX next to it.

Requirements (Python 3.9+ recommended; the code itself needs 3.7+):
  standard library only (argparse/csv/html/io/json/logging/os/re/sys/datetime)
  -- no pip install needed for the build itself. One exception: the Games tab
  solves live 2x2 matrices via a ai_ecosystem_model.py import, so that tab additionally needs
  the ai_ecosystem_model.py stack (numpy/pandas/matplotlib); every other tab builds from the
  CSVs alone.
"""
import argparse
import csv
import html
import io
import json
import logging
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

logger = logging.getLogger("dashboard")

RUN_LOG_PREFIX = "build_interactive_dashboard_run"
RUN_LOG_SUFFIX = ".log"
RUN_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
RUN_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_run_log_file(directory):
    """Attaches a per-run log file mirroring console output; returns its path.

    The file is named ``build_interactive_dashboard_run_YYYYMMDD_HHMMSS.log`` inside
    ``directory`` (the folder receiving dashboard.html, so the log sits next
    to the build receipt). The handler uses the shared run-log format at INFO
    level on the root logger, capturing dashboard records and any ai_ecosystem_model records
    emitted while solving the live game payloads. Idempotent per path (a
    same-second collision with an unowned file gains a _NN suffix instead
    of truncating it) and never raises: an unwritable location warns on
    stderr and returns None. Call only from main().
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



##############################################################################
# 1. CSV LOADING & PARSING HELPERS -- tolerant number parsing plus row-shape utilities.
##############################################################################

def read_csv_rows(path):
    """Reads a CSV (BOM-tolerant utf-8-sig) into a list of row dicts; the single file-read primitive behind every dashboard table.
    """
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def read_csv_rows_optional(path):
    """Like read_csv_rows, but returns [] instead of raising when the file is
    absent -- for tables that only exist on newer ai_ecosystem_model.py runs (e.g. the
    corrected welfare-benchmark/multiplicity/repeated-game audits and the
    enhanced dependency breakdowns), so older runs still build a dashboard.
    """
    if not os.path.exists(path):
        return []
    try:
        return read_csv_rows(path)
    except Exception:
        return []


def num(x):
    """Parse numbers tolerantly ('41.27%', '100,000', '$B' strings -> float/None)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(',', '').replace('%', '').replace('$', '').replace('B', '')
    try:
        return float(s)
    except ValueError:
        return None


def kv_table(rows):
    """Metric/Value[/Interpretation] tables -> {metric: value}."""
    out = {}
    for r in rows:
        vals = list(r.values())
        if len(vals) >= 2 and vals[0]:
            out[str(vals[0]).strip()] = vals[1]
    return out


def games_payload():
    """Live payoff matrices plus NE/welfare/taxonomy per game, solved exactly via ai_ecosystem_model import.

    Constructs GameTheoryFramework from embedded data, solves all six games, and
    returns JSON-ready dicts (matrix pairs, NE positions, welfare subset, viz
    description as game type) plus the canonical player order for relabeling.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import ai_ecosystem_model as gtmod
    data = gtmod.load_all_data()
    gf = gtmod.GameTheoryFramework(data.get('players'))
    gf.construct_payoff_matrices()
    gf.find_nash_equilibria()
    plt.close('all')
    games = []
    for name, eq in gf.nash_equilibria.items():
        M = eq['matrix']
        mat = [[(float(M[r, c][0]), float(M[r, c][1])) for c in range(2)] for r in range(2)]
        p1s = gf.player_strategies[eq['players'][0]]
        p2s = gf.player_strategies[eq['players'][1]]
        vd = None
        try:
            vd = gf._prepare_matrix_visualization_data(name)
        except Exception:
            vd = None
        games.append({
            'name': name, 'p1': eq['players'][0], 'p2': eq['players'][1],
            'p1_strats': list(p1s), 'p2_strats': list(p2s),
            'matrix': mat,
            'ne': [list(d['position']) for d in eq['nash_equilibria']],
            'welfare': {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in eq['welfare_metrics'].items()
                        if k in ('deadweight_loss', 'efficiency_ratio',
                                 'pareto_optimal_welfare', 'nash_welfare',
                                 'pareto_position', 'dwl_percent')},
            'game_type': (vd.get('description') if vd else None),
        })
    order = data.get('players')['Player_Category'].tolist()
    return games, order


def parse_pair_cell(s):
    """Parses '(a, b)' payoff-pair cells into (float, float); returns (None, None) when the cell does not match.
    """
    m = re.match(r'\(\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)\s*\)', str(s or ''))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)



##############################################################################
# 2. PAYLOAD ASSEMBLY -- ai_ecosystem_model tables plus live game solutions into one JSON-serializable dict.
##############################################################################

def extract_walkthrough(gt_path):
    """Parse ai_ecosystem_model.py into run_pipeline stages: section banners plus classes with docstrings.

    Returns {stages, modules}: stages are the numbered '# N. ...' banner titles
    in file order; each module records its name, enclosing stage, docstring
    (flattened, capped) and public method names. Used by the Pipeline Tour tab.
    Returns empty lists (never raises) when ai_ecosystem_model.py cannot be read or parsed."""
    import ast as _ast
    try:
        with open(gt_path, encoding='utf-8') as f:
            src = f.read()
    except Exception:
        return {'stages': [], 'modules': []}
    try:
        tree = _ast.parse(src)
    except Exception:
        return {'stages': [], 'modules': []}
    banners = []
    for i, ln in enumerate(src.splitlines()):
        m = re.match(r'#\s*(\d{1,2})\.\s+(\S.*)$', ln)
        if m:
            banners.append((i + 1, m.group(1) + '. ' + m.group(2).strip()))
    mods = []
    for node in tree.body:
        if isinstance(node, _ast.ClassDef):
            doc = (_ast.get_docstring(node) or '').strip().replace('\n', ' ')
            meths = [n.name for n in node.body
                     if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                     and not n.name.startswith('_')]
            stage = next((t for ln, t in reversed(banners) if ln < node.lineno), 'Core')
            mods.append({'stage': stage, 'name': node.name,
                         'doc': re.sub(r'\s+', ' ', doc)[:700], 'methods': meths})
    return {'stages': [t for _, t in banners], 'modules': mods}


def extract_report(report_path, max_rows=25, max_cells=6000):
    """Parse the build_technical_report.py DOCX into interactive sections (narrative plus tables).

    Walks body elements in document order so captions stay attached to their
    tables: 'Table/Figure/Exhibit N...' captions preceding a table mark it as
    a mirror of a ai_ecosystem_model.py CSV (rendered with a jump link to the live tab);
    other tables are embedded with data (capped). Shaded paragraphs become
    insight callouts. Returns {sections} or {absent: reason} (never raises)."""
    try:
        import docx as _docx
    except Exception:
        return {'absent': 'python-docx not installed'}
    if not os.path.exists(report_path):
        return {'absent': 'report not built yet (run build_technical_report.py first)'}
    try:
        doc = _docx.Document(report_path)
    except Exception as e:
        return {'absent': 'cannot open: %s' % e}
    from docx.text.paragraph import Paragraph as _Para
    from docx.table import Table as _Tbl
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

    def shaded(p):
        """True when a report paragraph carries Word shading (insight callout)."""
        try:
            pPr = p._p.pPr
            return pPr is not None and pPr.find(W + 'shd') is not None
        except Exception:
            return False

    def cap_kind(t):
        """Parses a caption lead ('Table 7.14') into (kind, number); None otherwise."""
        m = re.match(r'^(Table|Figure|Exhibit)\s+([\d.]+)\b', t)
        return (m.group(1), m.group(2)) if m else None

    sections, cur, pending, cells = [], None, None, [0]

    def ensure_cur():
        """Returns the active report section, creating Front matter when none exists."""
        nonlocal cur
        if cur is None:
            cur = {'title': 'Front matter', 'blocks': []}
            sections.append(cur)
        return cur

    for child in doc.element.body:
        tag = child.tag
        if tag == W + 'p':
            p = _Para(child, doc)
            t = p.text.strip()
            if not t:
                continue
            st = p.style.name if p.style is not None else ''
            if st.startswith('Heading'):
                lvl = st.rsplit(' ', 1)[-1]
                if lvl == '1':
                    cur = {'title': t[:160], 'blocks': []}
                    sections.append(cur)
                else:
                    cur = ensure_cur()
                    cur['blocks'].append({'h': t[:160], 'paras': [], 'tables': []})
                pending = None
            elif cap_kind(t) and len(t) < 300:
                pending = t[:200]
            else:
                cur = ensure_cur()
                if not cur['blocks'] or 'paras' not in cur['blocks'][-1]:
                    cur['blocks'].append({'h': '', 'paras': [], 'tables': []})
                blk = cur['blocks'][-1]
                if sum(len(b['paras']) for b in cur['blocks']) < 1500:
                    blk['paras'].append({'t': t[:1200], 'insight': shaded(p)})
                pending = None
        elif tag == W + 'tbl':
            if cells[0] > max_cells:
                pending = None
                continue
            tbl = _Tbl(child, doc)
            rows = [[c.text.strip() for c in r.cells] for r in tbl.rows]
            if not rows or not rows[0]:
                pending = None
                continue
            cells[0] += sum(len(r) for r in rows)
            kind = cap_kind(pending or '')
            cur = ensure_cur()
            if not cur['blocks'] or 'paras' not in cur['blocks'][-1]:
                cur['blocks'].append({'h': '', 'paras': [], 'tables': []})
            entry = {'caption': pending or 'Table',
                     'cols': [c[:80] for c in rows[0][:10]],
                     'rows': [[v[:120] for v in r[:10]] for r in rows[1:max_rows + 1]]}
            if kind and kind[0] == 'Table':
                entry['mirror'] = kind[1]
            if re.search(r'burst|bubble|GDP-at-risk|mitigation|formation probability|'
                         r'contagion|second-round|credit monitor',
                         pending or '', re.I):
                entry['gotab'] = 'burst'
            if len(rows) - 1 > max_rows:
                entry['truncated'] = '%d rows total; first %d shown' % (len(rows) - 1, max_rows)
            cur['blocks'][-1]['tables'].append(entry)
            pending = None
    return {'sections': sections}


def assemble(tables_dir, plots_dir, gt_path=None, report_path=None):
    """Builds the JSON-serializable payload injected into dashboard.html.

    Derives human 'Table N -- ...' titles from tables_index.csv (synthesizing
    7.x/8.x titles from filenames where the index omits them), extracts the KPI
    banner (market value, HHI plus band, aggregate/core DWL, efficiency, sim
    count), collects diagnostics tables, reattaches canonical player order to
    unlabeled rows (revenue, success, network), appends live per-game solutions
    via games_payload(), and links on-disk PNGs. Attaches the Pipeline Tour
    (extract_walkthrough) plus the Report viewer payload (extract_report).
    Expects a completed ai_ecosystem_model run.
    """
    def load_figcaps():
        """Human figure titles from figures/figure_captions.md ('## Figure N — Title').

        Falls back to a prettified filename so captions never render raw paths.
        """
        caps = {}
        try:
            with open(os.path.join(plots_dir, 'figure_captions.md'), encoding='utf-8') as fh:
                for line in fh:
                    m = re.match(r'##\s+Figure\s+(\d+)\s+[—–-]\s*(.+)', line.strip())
                    if m:
                        n = int(m.group(1))
                        caps[f'figure_{n}.png'] = f'Figure {n} — {m.group(2).strip()}'
        except Exception:
            pass
        return caps

    def R(f):
        """Reads one tables-dir CSV into row dicts for the dashboard payload.

        Required tables fail loudly with the file name and the fix (run ai_ecosystem_model.py
        first) instead of a bare traceback from deep inside csv parsing.
        """
        p = os.path.join(tables_dir, f)
        try:
            return read_csv_rows(p)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"dashboard needs {p} — run ai_ecosystem_model.py first so tables/ exists") from None

    def Ropt(f):
        """Like R(), but tolerant of the file being absent (newer-ai_ecosystem_model.py-only tables)."""
        return read_csv_rows_optional(os.path.join(tables_dir, f))
    titles = {}
    try:
        for r in R('tables_index.csv'):
            f = (r.get('File') or '').split('/')[0]
            if f.endswith('.csv') and 'alias' not in (r.get('Status') or ''):
                numid = (r.get('Table') or '').strip().split('_')[0]
                titles[f] = 'Table ' + numid + ' — ' + (r.get('Description') or '').strip()
    except Exception:
        pass
    # table_7.x diagnostics are saved by run_all, not save_all_tables, so the
    # index omits them: synthesize "Table 7.N — <name>" titles from filenames.
    # Dotted stems (table_7.16.market.share.evolution.csv, as produced by
    # TableGenerator) are prettified the same way when the index is stale.
    def _pretty_table_title(numid, raw):
        """Turns a filename fragment into a title-cased table description."""
        desc = raw.replace('_', ' ').replace('.', ' ').title()
        for a, b in (('Mc ', 'MC '), ('Dwl', 'DWL'), ('Hhi', 'HHI')):
            desc = desc.replace(a, b)
        return 'Table ' + numid + ' — ' + desc.strip()
    try:
        import glob as _glob
        for p in sorted(_glob.glob(os.path.join(tables_dir, 'table_7.*.csv'))):
            f = os.path.basename(p)
            if f not in titles:
                m = re.match(r'table_(7\.\d+)_(.+)\.csv$', f)
                if m:
                    titles[f] = _pretty_table_title(m.group(1), m.group(2))
                    continue
                m = re.match(r'table_(7\.\d+)\.(.+)\.csv$', f)
                if m:
                    titles[f] = _pretty_table_title(m.group(1), m.group(2))
    except Exception:
        pass
    titles.update({
        'policy_interventions_enhanced.csv': 'Policy Intervention Ranges',
        'enhanced_valuation_metrics.csv': 'Company Valuation Screen',
        'elasticity_sensitivity_analysis.csv': 'Elasticity Sensitivity Ranges',
        'circular_metrics.csv': 'Circular Exposure Metrics',
        'table_3.5_welfare_benchmark_audit.csv': 'Table 3.5 — Welfare-Benchmark Audit (Kaldor-Hicks Transfers)',
        'table_3.6_equilibrium_multiplicity.csv': 'Table 3.6 — Equilibrium Multiplicity & Risk-Dominance Audit',
        'table_3.7_repeated_game_sustainability.csv': 'Table 3.7 — Repeated-Game Sustainability (Critical Discount Factors)',
        'industry_dependencies_enhanced.csv': 'Deal-Level Dependency Register (Enhanced)',
        'category_level_dependencies.csv': 'Category-Level Dependency Aggregates',
        'table_3.1.csv': 'Table 3.1 — Canonical Payoff Exhibit (Cloud Game)',
        'table_3.2.csv': 'Table 3.2 — Canonical Payoff Exhibit (Hardware Game)',
        'table_4.1.csv': 'Table 4.1 — Monte Carlo & Sensitivity Summary',
        'table_5.4.csv': 'Table 5.4 — Cooperation Time-Preference Screen',
        'table_5.4.coop.csv': 'Table 5.4 (Coop) — Shapley Value Allocation',
        'table_A.1.network.risk.csv': 'Table A.1 — Network Risk Summary',
        'table_6.4.bcr.break.even.csv': 'Table 6.4 — BCR Break-Even Conditions',
        'table_7.1b.within.layer.hhi.csv': 'Table 7.1b — Within-Layer Concentration',
        'table_A.4.layer.exposure.csv': 'Table A.4 — Exposure by Layer',
        'table_8.6.csv': 'Table 8.6 — Validated Inputs Register',
        'table_7.14_debtrank_contagion.csv': 'Table 7.14 — DebtRank Contagion',
        'table_7.15_credit_monitor.csv': 'Table 7.15 — Credit Monitor',
        'table_8.7_structural_estimation.csv': 'Table 8.7 — Structural Estimation',
        'pred_A_scenario_ledger.csv': 'Prediction A — Scenario Ledger (Mild/Severe/Activation)',
        'pred_B_trigger_watchlist.csv': 'Prediction B — Trigger Watchlist',
        'pred_C_valuation_watchlist.csv': 'Prediction C — Valuation Watchlist (23-Company Screen)',
        'pred_D_robustness_ledger.csv': 'Prediction D — Robustness Ledger',
        'pred_E_formation_drivers.csv': 'Prediction E — Formation Drivers',
        'pred_F_early_warning.csv': 'Prediction F — Early-Warning Dashboard',
        'pred_tables_index.csv': 'Prediction Tables Index',
        'pred_G_burst_timing.csv': 'Prediction G — Burst-Timing Window',
        'pred_H_company_losses.csv': 'Prediction H — Company Burst Losses',
        'pred_I_price_gauges.csv': 'Prediction I — Price Gauges',
        'pred_J_gauge_backtest.csv': 'Prediction J — Gauge Backtest',
        'pred_K_expected_losses.csv': 'Prediction K — Expected Burst Losses',
        'pred_L_timing_rationale.csv': 'Prediction L — Timing Rationale',
        'pred_M_ml_horserace.csv': 'Prediction M — ML Horse-Race',
        'compare_summary.csv': '2008-vs-AI Comparison Summary',
    })
    t11rows = R('table_1.1.csv')
    t11 = kv_table(t11rows)
    hhi_interp = next((r.get('Interpretation') for r in t11rows if r.get('Metric') == 'HHI'), None)
    t34 = kv_table(R('table_3.4.csv'))
    t41 = kv_table(R('table_4.1.csv'))
    # Price-clock vintage: the weekly closes behind the pred_I/pred_J
    # gauges, surfaced in the payload meta and the Overview footer so the
    # gauge exhibits always carry their data vintage.
    _pxv = Ropt('price_history_weekly.csv')
    _px_dates = [r.get('Date', '') for r in _pxv if r.get('Date')]
    data = {
        'meta': {
            'tables_dir': os.path.basename(tables_dir.rstrip('/')),
            'n_tables': len([f for f in os.listdir(tables_dir) if f.endswith('.csv')]),
            'titles': titles,
            'price_vintage': (f"{min(_px_dates)} to {max(_px_dates)} ({len(_px_dates)} weeks)"
                              if _px_dates else ''),
        },
        'kpi': {
            'market': num(t11.get('Total Market Value ($B)')),
            'hhi': num(t11.get('HHI')),
            'hhi_band': hhi_interp or '',
            'dwl': num(t34.get('Aggregate Deadweight Loss ($B)')),
            'eff': num(t34.get('Aggregate Efficiency (%)')),
            'core_dwl': num(t34.get('Core Game Deadweight Loss ($B)')),
            'sims': t41.get('Number of Simulations'),
        },
        'players': R('table_2.4.csv'),
        'hhi_parts': R('table_7.1_concentration_decomposition.csv'),
        'welfare': R('table_5.3.csv'),
        'mc': R('table_7.9_mc_convergence.csv'),
        'policy': R('table_6.1.6.3.csv'),
        'policy_ranges': R('policy_interventions_enhanced.csv'),
        'bcr_break_even': Ropt('table_6.4.bcr.break.even.csv'),
        'revenue': R('table_1.3.proj.csv'),
        'network': R('table_A.0.network.csv'),
        'portfolio': R('table_A.2.portfolio.risk.csv'),
        'success': R('table_A.3.success.scores.csv'),
        'circular': R('circular_metrics.csv'),
        'burst_formation': R('table_7.10_bubble_formation.csv'),
        'burst_ranking': R('table_7.11_burst_ranking.csv'),
        'burst_gdp': R('table_7.12_gdp_impact.csv'),
        'burst_policy': R('table_7.13_bubble_policy.csv'),
        'debtrank': R('table_7.14_debtrank_contagion.csv'),
        'credit': R('table_7.15_credit_monitor.csv'),
        'valuation': R('enhanced_valuation_metrics.csv'),
        'nash_compare': R('table_3.4.csv'),
        'payoff_31': R('table_3.1.csv'),
        'payoff_32': R('table_3.2.csv'),
        'welfare_calib': R('table_4.1.csv'),
        'coop': R('table_5.4.csv'),
        'shapley': R('table_5.4.coop.csv'),
        'netrisk': R('table_A.1.network.risk.csv'),
        'layer_exposure': Ropt('table_A.4.layer.exposure.csv'),
        'elasticity': R('elasticity_sensitivity_analysis.csv'),
        'inputs_register': R('table_8.6.csv'),
        'welfare_audit': Ropt('table_3.5_welfare_benchmark_audit.csv'),
        'equilibrium_multiplicity': Ropt('table_3.6_equilibrium_multiplicity.csv'),
        'repeated_game': Ropt('table_3.7_repeated_game_sustainability.csv'),
        'industry_deps': Ropt('industry_dependencies_enhanced.csv'),
        'category_deps': Ropt('category_level_dependencies.csv'),
        'diag': {},
        'games': [],  # filled below via games_payload()
        'figures': sorted(f for f in os.listdir(plots_dir) if f.endswith('.png')) if os.path.isdir(plots_dir) else [],
        'figcaps': load_figcaps(),
        'plots_rel': os.path.basename(plots_dir.rstrip('/')),
    }
    data['walkthrough'] = extract_walkthrough(
        gt_path or os.path.join(os.path.dirname(tables_dir.rstrip('/')), 'ai_ecosystem_model.py'))
    data['report'] = extract_report(
        report_path or os.path.join(os.path.dirname(tables_dir.rstrip('/')),
                                    'AI_Ecosystem_Game_Theory_Technical_Report.docx'))
    for f in sorted(os.listdir(tables_dir)):
        if (f.startswith('table_7.') or f.startswith('table_8.')) and f.endswith('.csv'):
            data['diag'][f] = R(f)
    # Prediction-block ledgers land after the dashboard stage in run_pipeline
    # order, so read them tolerantly: present-and-nonempty tables join the
    # Diagnostics tab, absent ones are skipped and the dashboard still builds.
    for f in sorted(os.listdir(tables_dir)):
        if f.endswith('.csv') and (f.startswith('pred_') or f == 'compare_summary.csv'):
            rows = Ropt(f)
            if rows:
                data['diag'][f] = rows
    # revenue rows are unlabeled: attach canonical player order
    rev_order = ['Hardware', 'Cloud Providers', 'Foundation Models', 'LLM Wrappers']
    for i, r in enumerate(data['revenue']):
        r['_player'] = rev_order[i] if i < len(rev_order) else f'Player {i + 1}'
    games, live_order = games_payload()
    data['games'] = games
    # success + network CSVs lost their index: reattach live player order
    for key in ('success', 'network'):
        for i, r in enumerate(data[key]):
            r['_player'] = live_order[i] if i < len(live_order) else f'Player {i + 1}'
    return data



##############################################################################
# 3. ENTRY POINT -- inject payload into the template and write dashboard.html.
##############################################################################

def main():
    """Assembles the payload and writes the self-contained dashboard.html.

    Serializes with json.dumps(allow_nan=False) -- fail-fast on non-finite values
    rather than emitting invalid JSON -- escapes '</' for safe inline <script>
    use, and injects the payload into PAGE_HEAD plus the JS bundles. Zero
    dependencies, works from file:// offline. Prints byte size and
    table/game/figure counts as a build receipt.
    """
    ap = argparse.ArgumentParser(description='Build single-file dashboard from ai_ecosystem_model.py outputs')
    ap.add_argument('--tables-dir', default=os.path.join(os.getcwd(), 'tables'))
    ap.add_argument('--plots-dir', default=os.path.join(os.getcwd(), 'figures'))
    ap.add_argument('--ai_ecosystem_model-file', default=None, help='ai_ecosystem_model.py to parse for the Pipeline Tour (default: next to --tables-dir)')
    ap.add_argument('--report', default=None, help='.docx from build_technical_report.py for the Report tab (default: next to --tables-dir)')
    ap.add_argument('--out', default=os.path.join(os.getcwd(), 'dashboard.html'))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=RUN_LOG_FORMAT, datefmt=RUN_LOG_DATEFMT)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or os.getcwd()
    run_log_path = setup_run_log_file(out_dir)
    logger.info(f"Dashboard build started: {' '.join(sys.argv)}"
                + (f" (log: {run_log_path})" if run_log_path else ""))
    data = assemble(args.tables_dir, args.plots_dir,
                    gt_path=args.ai_ecosystem_model_file, report_path=args.report)
    logger.info(f"Payload assembled: {data['meta']['n_tables']} tables, "
                f"{len(data['games'])} games, {len(data['figures'])} figures, "
                f"{len(data['walkthrough']['modules'])} walkthrough modules")
    payload = json.dumps(data, allow_nan=False).replace('</', '<\\/')
    page = PAGE_HEAD.replace('__DATA__', payload).replace('__TABS__', json.dumps(TABS))
    page += JS_CORE + JS_TABS + '\n</body>\n</html>\n'
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(page)
    rep = data.get('report', {})
    repinfo = ('%d report sections' % len(rep.get('sections', []))) if 'sections' in rep else ('report: %s' % rep.get('absent', 'n/a'))
    print(f'Dashboard written: {args.out} ({len(page) / 1024:.0f} KB, '
          f'{data["meta"]["n_tables"]} tables, {len(data["games"])} games, '
          f'{len(data["figures"])} figures linked, '
          f'{len(data["walkthrough"]["modules"])} modules, {repinfo})')
    if run_log_path:
        print(f'Run log: {run_log_path}')
    logger.info(f"Dashboard written: {args.out} ({len(page) / 1024:.0f} KB, "
                f"{data['meta']['n_tables']} tables, {len(data['games'])} games, "
                f"{len(data['figures'])} figures linked, "
                f"{len(data['walkthrough']['modules'])} modules, {repinfo})"
                + (f" (log: {run_log_path})" if run_log_path else ""))


# === transparency ===

##############################################################################
# 4. FRONT-END TEMPLATE -- inline CSS/JS: tabs, SVG charts, tables, modal viewer (zero dependencies).
##############################################################################

OKABE = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#882255', '#2C3E50']

PAGE_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Ecosystem Game Theory — Interactive Dashboard</title>
<style>
:root{--ink:#1B2A3A;--mut:#5D6D7E;--line:#DDE3E8;--bg:#EEF1F4;--card:#FFFFFF;--acc:#0072B2;--acc2:#009E73;--rad:12px;--fs-xs:11px;--fs-sm:12px;--fs-md:13px;--fs-base:14px;--fs-h4:14px;--fs-h3:16px;--fs-h1:24px;--fs-kpi:30px;--f-ui:"Inter","SF Pro Display",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif;--f-mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
*{box-sizing:border-box}body{font-family:"Inter","SF Pro Display",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif;margin:0;background:linear-gradient(180deg,#E8EDF1 0%,var(--bg) 240px);color:var(--ink);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.vh{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
th[data-k]{cursor:pointer}th[data-k]:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
header{background:linear-gradient(120deg,#1B2A3A 0%,#243B55 60%,#14425F 100%);color:#fff;padding:22px 26px;box-shadow:0 2px 10px rgba(0,0,0,.25)}
header h1{margin:0;font-size:var(--fs-h1);font-weight:800;letter-spacing:-.3px}header p{margin:6px 0 0;font-size:var(--fs-sm);color:#B0BEC9}
nav{display:flex;flex-wrap:wrap;gap:4px;background:#22303F;padding:8px 14px;position:sticky;top:0;z-index:5;box-shadow:0 1px 6px rgba(0,0,0,.2)}
nav button{background:none;border:0;color:#CBD4DC;padding:9px 14px;font-size:13px;cursor:pointer;border-radius:8px;transition:background .15s,color .15s}
nav button:hover{background:rgba(255,255,255,.1);color:#fff}
nav button.on{background:#fff;color:var(--ink);font-weight:700;box-shadow:0 1px 4px rgba(0,0,0,.25)}
main{padding:20px 24px;max-width:1320px;margin:0 auto}
.tab{display:none}.tab.on{display:block;animation:fade .25s ease}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.grid{display:grid;gap:16px}.g4{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(430px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--rad);padding:16px 18px;box-shadow:0 2px 8px rgba(27,42,58,.07)}
.card h3{margin:0 0 10px;font-size:var(--fs-h3);font-weight:700;letter-spacing:.1px;display:flex;align-items:center;gap:8px}.card h3 small{color:var(--mut);font-weight:400;font-size:var(--fs-sm)}
.card h3 .dot{width:9px;height:9px;border-radius:50%;background:linear-gradient(135deg,var(--acc),var(--acc2));flex:none}
.kpi{font-size:var(--fs-kpi);font-weight:800;letter-spacing:-.8px;font-variant-numeric:tabular-nums}.kpi small{font-size:var(--fs-sm);color:var(--mut);font-weight:400}
.kpi-sub{font-size:var(--fs-sm);color:var(--mut);margin-top:3px}
.sec-desc{color:var(--mut);font-size:var(--fs-base);margin:0 0 14px;max-width:900px}
.insight{background:linear-gradient(90deg,#E8F4FD,#E9F7F1);border-left:4px solid var(--acc);border-radius:0 8px 8px 0;padding:10px 14px;margin:0 0 14px;font-size:var(--fs-base)}
.insight b{color:var(--ink)}
details.xpl{margin-top:10px;font-size:var(--fs-md);color:var(--mut)}
details.xpl summary{cursor:pointer;color:var(--acc);font-weight:600;font-size:var(--fs-md)}
details.xpl summary:hover{text-decoration:underline}
details.xpl div{margin-top:6px;line-height:1.55}
svg rect{transition:opacity .15s}svg rect:hover{opacity:.82}
svg path{transition:opacity .15s}svg path:hover{opacity:.85}
#tip{position:fixed;pointer-events:none;background:#1B2A3A;color:#fff;font-size:var(--fs-sm);padding:6px 10px;border-radius:6px;display:none;z-index:50;box-shadow:0 2px 8px rgba(0,0,0,.3);white-space:nowrap}
#modal{position:fixed;inset:0;background:rgba(15,25,35,.85);display:none;align-items:center;justify-content:center;z-index:40;padding:24px}
#modal.on{display:flex}#modal img{max-width:94vw;max-height:88vh;background:#fff;border-radius:8px}
#modal .cap{color:#D6DBDF;text-align:center;margin-top:8px;font-size:var(--fs-md)}
.dl{float:right;font-size:var(--fs-sm);color:var(--acc);cursor:pointer;text-decoration:underline;background:none;border:0;padding:0}
table.dt thead th{position:sticky;top:0}table.dt tbody tr{transition:background .12s}table.dt tbody tr:hover{background:#E4F0F8 !important}
table.dt{border-collapse:collapse;width:100%;font-size:var(--fs-base);font-variant-numeric:tabular-nums}table.dt th,table.dt td{border:1px solid var(--line);padding:7px 10px;text-align:left}
table.dt th{background:#E7ECEF;cursor:pointer;white-space:nowrap;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#44566A}table.dt th:hover{background:#D6DEE4}
table.dt tr:nth-child(even){background:#F8F9F9}.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:11px;font-weight:700;letter-spacing:.3px;color:#fff}
.mono{font-family:var(--f-mono);font-size:.92em}
svg text{letter-spacing:.1px}
@media print{nav,#tip,#modal,.dl{display:none !important}body{background:#fff}main{max-width:none}.card{box-shadow:none;break-inside:avoid}.tab{display:block !important}}
.b-ok{background:#009E73}.b-warn{background:#E69F00}.b-bad{background:#D55E00}.b-info{background:#0072B2}
select,input[type=text]{font-size:13px;padding:6px 9px;border:1px solid var(--line);border-radius:6px;margin:0 8px 10px 0}
.btnrow{margin:2px 0 12px}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:4px}
svg text{font-family:inherit}.ax{stroke:#5D6D7E;stroke-width:1}.gl{stroke:#EAEDED;stroke-width:1}
.foot{color:var(--mut);font-size:var(--fs-sm);margin:22px 0 30px}
.mgrid{display:grid;grid-template-columns:130px 1fr 130px;gap:6px;align-items:center;margin:8px 0}
.mgrid .rlab{font-size:var(--fs-sm);text-align:right;color:var(--mut)}.mgrid .clab{font-size:var(--fs-sm);color:var(--mut)}
.cell{border-radius:6px;padding:10px 6px;text-align:center;color:#fff;font-size:12px;font-weight:bold;border:1px solid rgba(27,42,58,.22);text-shadow:0 1px 2px rgba(0,0,0,.25)}
.ne{outline:3px solid #F39C12;outline-offset:-3px}
.legend{font-size:12px;color:var(--mut)}.figgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.figgrid figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px}
.figgrid img{width:100%;height:auto;display:block;background:#fff}.figgrid figcaption{font-size:var(--fs-sm);font-weight:600;color:var(--ink);padding:8px 2px 2px}
a{color:var(--acc)}
.lgnd{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 2px 2px;font-size:12px;color:var(--mut)}
.lgnd i{display:inline-block;width:14px;height:4px;border-radius:2px;margin-right:6px;vertical-align:middle}
.stage-strip{display:flex;gap:8px;overflow-x:auto;padding:4px 2px 12px;margin-bottom:6px}
.stage-chip{flex:0 0 auto;max-width:230px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px 12px;font-size:12px;color:var(--mut);cursor:pointer}
.stage-chip b{display:block;color:var(--ink);font-size:var(--fs-md)}.stage-chip.on{border-color:var(--acc);box-shadow:0 0 0 2px rgba(0,114,178,.25)}
.mchip{display:inline-block;background:#EEF1F4;border:1px solid var(--line);border-radius:6px;padding:1px 8px;margin:2px 4px 2px 0;font-size:var(--fs-sm)}
.mchip.mono{color:#243B55}.stage-badge{display:inline-block;background:#243B55;color:#fff;border-radius:8px;font-size:var(--fs-xs);padding:2px 8px;margin-left:8px;vertical-align:middle}
.tourbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;background:#1B2A3A;color:#fff;border-radius:10px;padding:10px 14px;margin:0 0 14px;font-size:var(--fs-base)}
.tourbar button{background:#fff;border:0;border-radius:6px;padding:6px 14px;font-size:var(--fs-md);font-weight:700;cursor:pointer;color:#1B2A3A}
.tourbar button:disabled{opacity:.4;cursor:default}.tourbar .pos{font-variant-numeric:tabular-nums;color:#B0BEC9}
.tour-on{outline:3px solid #0072B2 !important;outline-offset:2px;box-shadow:0 4px 18px rgba(0,114,178,.3)}
.repnav{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 14px}
.repnav button{background:#fff;border:1px solid var(--line);border-radius:16px;padding:5px 13px;font-size:var(--fs-md);cursor:pointer;color:var(--ink)}
.repnav button:hover{border-color:var(--acc);color:var(--acc)}
.repsec h4{margin:16px 0 6px;font-size:var(--fs-h4);color:#243B55}.repsec p{margin:6px 0;font-size:var(--fs-base);max-width:960px}
.viewbtn{background:none;border:0;color:var(--acc);cursor:pointer;text-decoration:underline;font-size:12px;padding:0}
details.mod{margin:8px 0;border:1px solid var(--line);border-radius:8px;background:#fff}
details.mod summary{cursor:pointer;padding:9px 13px;font-size:13px;font-weight:600;list-style:none}
details.mod summary::-webkit-details-marker{display:none}
details.mod summary:hover{color:var(--acc)}details.mod .body{padding:0 13px 12px}
</style></head>
<body>
<header><h1>AI Ecosystem Game Theory — Interactive Dashboard</h1></header>
<nav id="tabs"></nav>
<main id="main"></main>
<div id="tip"></div>
<div id="modal" onclick="this.classList.remove('on')"></div>
<script>
"use strict";
const DATA = __DATA__;
const PALETTE = ['#0072B2','#D55E00','#009E73','#E69F00','#56B4E9','#CC79A7','#882255','#2C3E50'];
const TABSDEF = __TABS__;
</script>
"""

"""Tab registry (id, label) injected into the page as TABSDEF; the JS tab bar
renders from it, so adding a tab here is enough to add a dashboard section."""
TABS = [
    ('overview', 'Overview'),
    ('walk', 'Pipeline Tour'),
    ('report', 'Report'),
    ('inputs', 'Inputs & Assumptions'),
    ('market', 'Market'),
    ('games', 'Games'),
    ('welfare', 'Welfare'),
    ('mc', 'Monte Carlo'),
    ('risk', 'Risk & Network'),
    ('burst', 'Bubble Burst'),
    ('policy', 'Policy'),
    ('revenue', 'Revenue & Valuation'),
    ('diag', 'Diagnostics'),
    ('figs', 'Figures'),
]

JS_CORE = """<script>
"use strict";
/* ---------- helpers ---------- */
const $=s=>document.querySelector(s);
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&ai_ecosystem_model;','"':'&quot;'}[c]));}
function num(x){if(x==null||x==='')return NaN;const v=parseFloat(String(x).replace(/[, $B%]/g,''));return isFinite(v)?v:NaN;}
function fmt(x,d){const v=num(x);return isFinite(v)?v.toLocaleString('en-US',{maximumFractionDigits:d==null?2:d}):esc(x);}
function badge(t,cls){return '<span class="badge '+cls+'">'+esc(t)+'</span>';}
/* ---------- tabs ---------- */
let TABS=[];
function initTabs(tabs){TABS=tabs;$('#tabs').setAttribute('role','tablist');
$('#tabs').innerHTML=tabs.map((t,i)=>'<button role="tab" data-t="'+t[0]+'" aria-selected="'+(i?'false':'true')+'"'+(i?'':' class="on"')+'>'+t[1]+'</button>').join('');
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>showTab(b.dataset.t));
$('#main').innerHTML=tabs.map(t=>'<section class="tab" role="tabpanel" id="tab-'+t[0]+'"></section>').join('');
$('#tab-'+tabs[0][0]).classList.add('on');}
function showTab(id){document.querySelectorAll('#tabs button').forEach(b=>{const on=b.dataset.t===id;b.classList.toggle('on',on);b.setAttribute('aria-selected',on?'true':'false');});
document.querySelectorAll('.tab').forEach(s=>s.classList.toggle('on',s.id==='tab-'+id));
try{if(history.replaceState)history.replaceState(null,'','#'+id);}catch(e){}}
/* ---------- sortable table ---------- */
function dtable(rows,cols,opts){
 // rows: array of objects; cols: [key,label,numeric?]
 opts=opts||{};
 const id='t'+Math.floor(Math.random()*1e9);
 let order={k:null,d:1},filter='';
 function render(){
  let r=rows.filter(row=>!filter||JSON.stringify(row).toLowerCase().includes(filter));
  if(order.k){r=r.slice().sort((a,b)=>{const x=a[order.k],y=b[order.k];const xn=num(x),yn=num(y);
   if(isFinite(xn)&&isFinite(yn))return (xn-yn)*order.d;return String(x).localeCompare(String(y))*order.d;});}
  const el=document.getElementById(id);
  el.innerHTML='<thead><tr>'+cols.map(c=>'<th data-k="'+esc(c[0])+'" tabindex="0" title="Sort by '+esc(c[1])+'">'+esc(c[1])+(order.k===c[0]?(order.d>0?' ▲':' ▼'):'')+'</th>').join('')+'</tr></thead><tbody>'+
   r.map(row=>'<tr>'+cols.map(c=>{const v=row[c[0]];const cls=c[2]?' class="num"':'';
    return '<td'+cls+'>'+(c[2]?fmt(v):esc(v))+'</td>';}).join('')+'</tr>').join('')+'</tbody>';
  el.querySelectorAll('th').forEach(th=>{const sort=()=>{const k=th.dataset.k;
   order=order.k===k?{k,d:-order.d}:{k,d:1};render();};
   th.onclick=sort;th.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();sort();}};});
 }
 const f=opts.filter?'<label class="vh" for="'+id+'f">Filter rows</label><input type="text" placeholder="Filter…" aria-label="Filter table rows" id="'+id+'f">':'';
 setTimeout(()=>{render();if(opts.filter){document.getElementById(id+'f').oninput=e=>{filter=e.target.value.toLowerCase();render();}}},0);
 return f+'<div style="overflow-x:auto"><table class="dt" id="'+id+'"></table></div>';
}
function colsOf(rows){if(!rows.length)return [];const keys=Object.keys(rows[0]).filter(k=>!k.startsWith('_'));
 return keys.map(k=>{const vals=rows.map(r=>r[k]);const numeric=vals.some(v=>isFinite(num(v))&&String(v).trim()!=='');
  const label=k.replace(/_/g,' ');
  return [k,label,numeric];});}
function TT(f){return (DATA.meta&&DATA.meta.titles&&DATA.meta.titles[f])||f;}
/* ---------- svg charts (labels wrap, never truncate) ---------- */
function svgOpen(w,h){return '<svg viewBox="0 0 '+w+' '+h+'" width="100%" role="img">';}
function wrapLines(s,n){const w=String(s==null?'':s).split(/\\s+/).filter(Boolean);const L=[];let cur='';
 w.forEach(x=>{const t=cur?cur+' '+x:x;if(t.length>n&&cur){L.push(cur);cur=x;}else cur=t;});
 if(cur)L.push(cur);return L.slice(0,3);}
function grad(id,c){return '<defs><linearGradient id="'+id+'" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="'+c+'"/><stop offset="1" stop-color="'+c+'" stop-opacity=".5"/></linearGradient></defs>';}
function vbar(elid,labels,vals,opts){
 opts=opts||{};const W=560;let pt=16,pr=14;const pl=66;
 const wrapped=labels.map(l=>wrapLines(l,14));
 const pb=wrapped.some(w=>w.length>1)?62:46;const H=300+ (pb-46);
 const mx=Math.max.apply(null,vals.concat([0]))||1,mn=Math.min.apply(null,vals.concat([0]));
 const y0=H-pb-(0-mn)/(mx-mn)*(H-pt-pb);
 let s=svgOpen(W,H);
 for(let g=0;g<=4;g++){const yv=mx+(mn-mx)*g/4;const y=pt+(H-pt-pb)*g/4;
  s+='<line class="gl" x1="'+pl+'" y1="'+y.toFixed(1)+'" x2="'+(W-pr)+'" y2="'+y.toFixed(1)+'"/>';
  s+='<text x="'+(pl-6)+'" y="'+(y+4).toFixed(1)+'" font-size="11" text-anchor="end" fill="#5D6D7E">'+fmt(yv)+'</text>';}
 const bw=(W-pl-pr)/vals.length;
 vals.forEach((v,i)=>{const h=Math.abs(v)/(mx-mn)*(H-pt-pb);const y=v>=0?y0-h:y0;
  const c=(opts.colors||PALETTE)[i%PALETTE.length];const gid='g-'+elid+'-'+i;
  s+=grad(gid,c);
  s+='<rect x="'+(pl+i*bw+bw*0.18).toFixed(1)+'" y="'+y.toFixed(1)+'" width="'+(bw*0.64).toFixed(1)+'" height="'+Math.max(h,1).toFixed(1)+'" fill="url(#'+gid+')" rx="3"><title>'+esc(labels[i])+': '+fmt(v)+'</title></rect>';
  const cx=(pl+i*bw+bw/2).toFixed(1);
  wrapped[i].forEach((ln,k)=>{s+='<text x="'+cx+'" y="'+(H-pb+16+k*12)+'" font-size="11" text-anchor="middle">'+esc(ln)+'</text>';});});
 s+='<line class="ax" x1="'+pl+'" y1="'+pt+'" x2="'+pl+'" y2="'+(H-pb)+'"/><line class="ax" x1="'+pl+'" y1="'+y0+'" x2="'+(W-pr)+'" y2="'+y0+'"/></svg>';
 document.getElementById(elid).innerHTML=s;
}
function hbar(elid,labels,vals,opts){
 opts=opts||{};const W=560,pt=10,pb=10;
 const wrapped=labels.map(l=>wrapLines(l,30));
 const maxlen=Math.max.apply(null,wrapped.map(w=>Math.max.apply(null,w.map(x=>x.length).concat([1]))));
 const pl=Math.min(300,Math.max(120,10+6.6*maxlen));
 const rows=vals.map((v,i)=>({v:v,lines:wrapped[i],rh:wrapped[i].length>1?42:28}));
 let H=pt+pb;rows.forEach(r=>{r.y=H;H+=r.rh;});
 const vfmt=vals.map(v=>fmt(v));const maxvw=Math.max.apply(null,vfmt.map(t=>t.length).concat([1]));
 const pr=14+6.8*maxvw;
 const mx=Math.max.apply(null,vals.concat([0]))||1;
 let s=svgOpen(W,H);
 rows.forEach((r,i)=>{const w=Math.max(r.v,0)/mx*(W-pl-pr);
  const c=(opts.colors||PALETTE)[i%PALETTE.length];const gid='gh-'+elid+'-'+i;
  s+=grad(gid,c);
  r.lines.forEach((ln,k)=>{s+='<text x="'+(pl-8)+'" y="'+(r.y+15+k*13)+'" font-size="11" text-anchor="end">'+esc(ln)+'</text>';});
  const by=r.y+(r.rh-18)/2;
  s+='<rect x="'+pl+'" y="'+by.toFixed(1)+'" width="'+Math.max(w,1).toFixed(1)+'" height="18" fill="url(#'+gid+')" rx="3"><title>'+esc(labels[i])+': '+fmt(r.v)+'</title></rect>';
  s+='<text x="'+(pl+w+7).toFixed(1)+'" y="'+(by+13).toFixed(1)+'" font-size="11" fill="#1B2A3A">'+esc(vfmt[i])+'</text>';});
 document.getElementById(elid).innerHTML=s+'</svg>';
}
function donut(elid,labels,vals,opts){
 opts=opts||{};const S=240,R=88,r=58;const tot=vals.reduce((a,b)=>a+b,0)||1;let a0=-Math.PI/2;
 let s=svgOpen(S,S+ (opts.legend===false?0:24*labels.length+8));
 const cx=S/2,cy=S/2;
 vals.forEach((v,i)=>{const a1=a0+v/tot*2*Math.PI;const big=(a1-a0)>Math.PI?1:0;
  const x0=cx+R*Math.cos(a0),y0=cy+R*Math.sin(a0),x1=cx+R*Math.cos(a1),y1=cy+R*Math.sin(a1);
  const xi1=cx+r*Math.cos(a1),yi1=cy+r*Math.sin(a1),xi0=cx+r*Math.cos(a0),yi0=cy+r*Math.sin(a0);
  const c=(opts.colors||PALETTE)[i%PALETTE.length];
  s+='<path d="M'+x0.toFixed(1)+' '+y0.toFixed(1)+' A'+R+' '+R+' 0 '+big+' 1 '+x1.toFixed(1)+' '+y1.toFixed(1)+' L'+xi1.toFixed(1)+' '+yi1.toFixed(1)+' A'+r+' '+r+' 0 '+big+' 0 '+xi0.toFixed(1)+' '+yi0.toFixed(1)+' Z" fill="'+c+'"><title>'+esc(labels[i])+': '+fmt(v)+' ('+(100*v/tot).toFixed(1)+'%)</title></path>';
  a0=a1;});
 s+='<text x="'+cx+'" y="'+(cy+5)+'" font-size="13" font-weight="bold" text-anchor="middle">'+esc(opts.center||'')+'</text>';
 if(opts.legend!==false){let y=S+18;labels.forEach((l,i)=>{const c=(opts.colors||PALETTE)[i%PALETTE.length];
  const lines=wrapLines(l+' — '+fmt(vals[i]),34);
  s+='<rect x="14" y="'+(y-10)+'" width="12" height="12" fill="'+c+'" rx="2"><title>'+esc(l)+': '+fmt(vals[i])+'</title></rect>';
  lines.forEach((ln,k)=>{s+='<text x="32" y="'+y+'" font-size="10.5">'+esc(ln)+'</text>';y+=13;});
  y+=8;});
  s=s.replace(/viewBox="0 0 (\\d+) (\\d+)"/,'viewBox="0 0 $1 '+Math.max(S,y+6)+'"');}
 document.getElementById(elid).innerHTML=s+'</svg>';
}
function line(elid,series,opts){
 // series: [{label,color,pts:[[x,y],...]}]
 opts=opts||{};const W=560,H=300,pl=60,pb=40,pt=14,pr=14;
 const xs=series.flatMap(s=>s.pts.map(p=>p[0])),ys=series.flatMap(s=>s.pts.map(p=>p[1]));
 const x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs),y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys.concat([0]));
 const X=x=>pl+(x-x0)/((x1-x0)||1)*(W-pl-pr),Y=y=>H-pb-(y-y0)/(((y1-y0))||1)*(H-pt-pb);
 let s=svgOpen(W,H);
 for(let g=0;g<=4;g++){const y=pt+(H-pt-pb)*g/4;s+='<line class="gl" x1="'+pl+'" y1="'+y+'" x2="'+(W-pr)+'" y2="'+y+'"/>';}
 series.forEach(sr=>{s+='<polyline fill="none" stroke="'+sr.color+'" stroke-width="2" points="'+sr.pts.map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ')+'"><title>'+esc(sr.label)+'</title></polyline>';});
 series.forEach(sr=>{sr.pts.forEach(p=>{s+='<circle cx="'+X(p[0]).toFixed(1)+'" cy="'+Y(p[1]).toFixed(1)+'" r="3" fill="'+sr.color+'"><title>'+esc(sr.label)+': '+fmt(p[1])+'</title></circle>';});});
 s+='<line class="ax" x1="'+pl+'" y1="'+pt+'" x2="'+pl+'" y2="'+(H-pb)+'"/><line class="ax" x1="'+pl+'" y1="'+(H-pb)+'" x2="'+(W-pr)+'" y2="'+(W-pr)+'"/></svg>';
 let lg='<div class="lgnd">'+series.map(sr=>'<span><i style="background:'+sr.color+'"></i>'+esc(sr.label)+'</span>').join('')+'</div>';
 document.getElementById(elid).innerHTML=s+lg;
}
/* ---------- floating tooltip, lightbox, csv export ---------- */
function armTip(){
 const tip=document.getElementById('tip');
 document.addEventListener('mousemove',e=>{
  const t=e.target&&e.target.tagName;
  const ttl=e.target&&e.target.querySelector?e.target.querySelector(':scope > title'):null;
  if((t==='rect'||t==='path'||t==='circle')&&ttl&&ttl.textContent){tip.textContent=ttl.textContent;
   tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+12)+'px';}
  else tip.style.display='none';});
}
function openFig(src,cap){
 const m=document.getElementById('modal');
 m.innerHTML='<div><img src="'+esc(src)+'" alt="'+esc(cap)+'"><div class="cap">'+esc(cap)+' — click anywhere to close</div></div>';
 m.classList.add('on');
}
function dlCSV(name,rows){
 if(!rows||!rows.length)return;
 const keys=Object.keys(rows[0]);
 const q=v=>'"'+String(v==null?'':v).replace(/"/g,'""')+'"';
 const csv=keys.map(q).join(',')+'\\n'+rows.map(r=>keys.map(k=>q(r[k])).join(',')).join('\\n');
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
 a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000);
}
function dlBtn(name,varr){return '<button class="dl" onclick="dlCSV(&quot;'+name+'&quot;,'+varr+')">CSV ↓</button>';}
function insight(html){return '<div class="insight">'+html+'</div>';}
function secDesc(t){return '<p class="sec-desc">'+t+'</p>';}
function xpl(t){return '<details class="xpl"><summary>How to read this</summary><div>'+t+'</div></details>';}
function welfareParts(){return (DATA.welfare||[]).filter(r=>String(r['Source of Deadweight Loss']).trim().toLowerCase()!=='total');}
</script>
"""

JS_TABS = """<script>
"use strict";
function kpi(v,u){return '<div class="kpi">'+v+' <small>'+u+'</small></div>';}
function card(t,inner,sub){return '<div class="card"><h3><span class="dot"></span>'+t+(sub?' <small>'+sub+'</small>':'')+'</h3>'+inner+'</div>';}
function buildOverview(){
 const k=DATA.kpi;
 const top=welfareParts().slice().sort((a,b)=>num(b['Estimated Contribution ($B)'])-num(a['Estimated Contribution ($B)']))[0];
 const best=DATA.policy.slice().sort((a,b)=>num(b['Benefit-Cost Ratio'])-num(a['Benefit-Cost Ratio']))[0];
 $('#tab-overview').innerHTML=secDesc('Headline results of the AI-ecosystem game-theoretic analysis. Hover any chart element for exact values; click table headers to sort.')
 +insight('<b>Takeaways:</b> the largest DWL source is <b>'+esc(top['Source of Deadweight Loss'])+'</b> ($'+fmt(top['Estimated Contribution ($B)'])+'B); the highest-leverage policy is <b>'+esc(best['Policy Scenario'])+'</b> (BCR '+fmt(best['Benefit-Cost Ratio'],1)+'); market HHI of '+fmt(k.hhi,0)+' reads <b>'+esc(k.hhi_band)+'</b> under 2023 guidelines.')
 +'<div class="grid g4">'
 +card('Total Market Value',kpi('$'+fmt(k.market)+'B','revenue'))
 +card('Market HHI',kpi(fmt(k.hhi,0),esc(k.hhi_band||'')))
 +card('Aggregate DWL',kpi('$'+fmt(k.dwl)+'B','deadweight loss'))
 +card('Aggregate Efficiency',kpi(fmt(k.eff,1)+'%','of potential'))
 +card('Core Game DWL',kpi('$'+fmt(k.core_dwl)+'B','HW vs Cloud'))
 +card('MC Simulations',kpi(esc(k.sims||''),'per game'))+'</div>'
 +'<div class="grid g2" style="margin-top:14px">'
 +card('DWL Decomposition ($B)','<div id="c-wel"></div>'+dlBtn('welfare.csv','DATA.welfare')+xpl("Each slice attributes part of the $416.77B aggregate DWL to one economic channel — switching costs, innovation distortion, quality degradation, coordination failure, monopoly pricing. Components are rescaled to sum exactly to the total, so read them as loss attribution, not independent estimates. Hover a slice for its value and share; the largest slice names the paper's first policy target."),'table_5.3')
 +card('Policy Net Benefit — Base ($B)','<div id="c-pol"></div>'+xpl("Net annual benefit equals DWL reduction minus implementation cost under the base scenario. Bars rank tools by absolute gain; ratios are compared on the Policy tab. Conservative pairs low reduction with high cost; optimistic does the reverse — switch scenarios there to re-rank the tools."),'table_6.1.6.3')+'</div>'
 +(DATA.meta&&DATA.meta.price_vintage?'<div class="foot">Price-clock vintage (weekly closes): '+esc(DATA.meta.price_vintage)+'.</div>':'');
 donut('c-wel',welfareParts().map(r=>r['Source of Deadweight Loss']),welfareParts().map(r=>num(r['Estimated Contribution ($B)'])),{center:'$'+fmt(k.dwl)+'B'});
 hbar('c-pol',DATA.policy.map(r=>r['Policy Scenario']),DATA.policy.map(r=>num(r['Net Annual Benefit ($B)'])));
}
function buildInputs(){
 const rows=DATA.inputs_register||[];
 const cats=[];rows.forEach(r=>{if(cats.indexOf(r['Category'])<0)cats.push(r['Category']);});
 $('#tab-inputs').innerHTML=secDesc('Every validated input value and modeling assumption the analysis consumes — Table 8.6, built live from the canonical ai_ecosystem_model.py structures so these are the values the models ran on. Type to filter rows; click headers to sort.')
 +cats.map((c,i)=>card(esc(c),'<div id="inp-'+i+'"></div>'+xpl("One Category slice of Table 8.6: ID, parameter, value, unit and verification source per row. Type in the filter box to narrow rows; click any header to sort. Every value here is the exact September 2026 vintage the models ran on, so this register is the audit trail for all headline numbers."),'table_8.6.csv')).join('');
 cats.forEach((c,i)=>{const sub=rows.filter(r=>r['Category']===c);
  document.getElementById('inp-'+i).innerHTML=dtable(sub,colsOf(sub),{filter:true});});
}
function buildMarket(){
 const parts=DATA.hhi_parts;
 const lead=parts.slice().sort((a,b)=>num(b['HHI_Points'])-num(a['HHI_Points']))[0];
 $('#tab-market').innerHTML=secDesc('Concentration and player structure. HHI thresholds (2023): 1,000 moderate / 1,800 highly concentrated.')
 +insight('<b>'+esc(lead['Archetype'])+'</b> contributes the most HHI points ('+fmt(lead['HHI_Points'],1)+', '+fmt(lead['HHI_Contribution_Pct'],1)+'% of the index) — concentration is a two-leader story.')
 +'<div class="grid g2">'
 +card('HHI Points by Player','<div id="c-hhi"></div>'+xpl("HHI is the sum of squared market shares times 10,000 — here 3,916 on $1,023B segment revenue. Each bar is one archetype's point contribution, and the bars sum exactly to the headline index. 2023 guidelines read above 1,000 as moderate and above 1,800 as highly concentrated, so this market clears the high bar by a wide margin."),'sums to '+fmt(DATA.kpi.hhi,0))
 +card(TT('table_2.4.csv'),'<div id="t-play"></div>'+xpl("One row per archetype: observed revenue, market share of the four-archetype total, and strategic-risk label. The risk labels feed the portfolio simulation and the stability scores on the Risk tab — this table is where market structure hands off to strategy."),'table_2.4.csv')+'</div>';
 hbar('c-hhi',parts.map(r=>r['Archetype']),parts.map(r=>num(r['HHI_Points'])));
 $('#t-play').innerHTML=dtable(DATA.players,colsOf(DATA.players),{});
 const el=DATA.elasticity||[];
 if(el.length){
  const div=document.createElement('div');
  div.innerHTML='<div class="grid g2" style="margin-top:14px">'
  +card('Base Lerner Index by Archetype','<div id="c-el"></div>'+xpl("Lerner index L = −1/ε from literature demand elasticities: the price-cost margin implied by each archetype's demand curve. Higher bars mean more pricing power — inelastic demand (small |ε|) converts directly into markup. Hover bars for exact margins."),'elasticity_sensitivity_analysis.csv')
  +card(TT('elasticity_sensitivity_analysis.csv'),dlBtn('elasticity.csv','DATA.elasticity')+'<div id="t-el"></div>'+xpl("Elasticity ranges with literature sources, mapped to conservative/base/aggressive Lerner indices and markups. Intervals, not points — the paper reports bounds. Use the conservative/aggressive columns as the robustness range for any markup claim."),'elasticity_sensitivity_analysis.csv')+'</div>';
  $('#tab-market').appendChild(div);
  hbar('c-el',el.map(r=>r['Player_Category']),el.map(r=>num(r['Base_Lerner'])));
  $('#t-el').innerHTML=dtable(el,colsOf(el),{});
 }
}
function shade(v,mx){const t=Math.max(0,Math.min(1,v/(mx||1)));const r=Math.round(44+ (213-44)*(1-t)),g=Math.round(62+(180-62)*(1-t)),b=Math.round(80+(226-80)*(1-t));return 'rgb('+r+','+g+','+b+')';}
function cellInk(v,mx){const t=Math.max(0,Math.min(1,v/(mx||1)));const r=44+(213-44)*(1-t),g=62+(180-62)*(1-t),b=80+(226-80)*(1-t);return (0.299*r+0.587*g+0.114*b)>130?'#1B2A3A':'#ffffff';}
let CUR_GAME=0;
function parsePair(s){const m=String(s==null?'':s).match(/\\(?\\s*(-?[\\d.,]+)\\s*,\\s*(-?[\\d.,]+)\\s*\\)?/);if(!m)return [null,null];return [parseFloat(m[1].replace(/,/g,'')),parseFloat(m[2].replace(/,/g,''))];}
function payoffExhibit(rows){
 const keys=Object.keys(rows[0]||{});
 if(keys.length<3)return '';
 const rk=keys[0],ck=keys.slice(1);
 const matrix=rows.map(r=>ck.map(c=>{const p=parsePair(r[c]);return [p[0]==null||!isFinite(p[0])?0:p[0],p[1]==null||!isFinite(p[1])?0:p[1]];}));
 const rl=rows.map(r=>String(r[rk]).replace(/^P1:\\s*/,''));
 const cl=ck.map(c=>String(c).replace(/^P2:\\s*/,'').replace(/\\s*\\(P1,\\s*P2\\)\\s*$/,''));
 const mx=Math.max.apply(null,matrix.map(row=>row.map(p=>p[0]+p[1])).flat().concat([1]));
 let h='<div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:4px;font-size:12px;align-items:stretch"><div></div>'
  +cl.map(c=>'<div style="text-align:center;color:var(--ink);font-weight:600;font-size:11px;align-self:center">'+esc(c)+'</div>').join('');
 matrix.forEach((row,r)=>{h+='<div style="color:var(--ink);font-weight:600;font-size:11px;align-self:center">'+esc(rl[r])+'</div>';
  row.forEach(p=>{const tot=p[0]+p[1];h+='<div class="cell" style="background:'+shade(tot,mx)+';color:'+cellInk(tot,mx)+';padding:8px 4px;font-size:12px" title="P1 $'+fmt(p[0])+'B, P2 $'+fmt(p[1])+'B — joint $'+fmt(tot)+'B">'+fmt(p[0],0)+'/'+fmt(p[1],0)+'<br><span style="font-size:10px;font-weight:normal;opacity:.85">P1 / P2, $B</span></div>';});});
 return h+'</div>';
}
function buildGames(){
 const gs=DATA.games;
 $('#tab-games').innerHTML=secDesc('All six bilateral games recomputed live at build time. Select a game to drill into its payoff matrix, Nash equilibrium, welfare, and Monte-Carlo robustness.')
 +'<div class="toolbar"><label>Game: <select id="gsel">'+gs.map((g,i)=>'<option value="'+i+'">'+esc(g.name)+'</option>').join('')+'</select></label>'
 +'<span class="legend">Drill into one game below, or compare all six at a glance further down.</span></div><div id="gdetail"></div><div id="gcomp"></div>';
 $('#gsel').onchange=e=>{CUR_GAME=+e.target.value;showGame();};
 showGame();
 renderGcomp();
 const nc=(DATA.nash_compare||[]).map(r=>({Metric:r['Metric'],Value:r['Value']}));
 if(nc.length){
  const div=document.createElement('div');
  div.innerHTML='<div class="grid g2" style="margin-top:14px">'+card(TT('table_3.4.csv'),'<div id="t-nc"></div>'+xpl("Core-game equilibrium, coordination diagnosis, and DWL: Nash/Pareto strategies and payoffs, efficiency ratio, aggregate loss, and the five-channel attribution. Read the Nash row for the stable outcome, the Pareto row for the benchmark, and the DWL row for what non-cooperation costs — the gap between those two rows is the number the policy remedies price against."),'table_3.4.csv')+'</div>';
  $('#tab-games').appendChild(div);
  $('#t-nc').innerHTML=dtable(nc,colsOf(nc),{});
 }
 const ex=[['table_3.1.csv',DATA.payoff_31],['table_3.2.csv',DATA.payoff_32]].filter(x=>x[1]&&x[1].length);
 if(ex.length){
  const div=document.createElement('div');
  const EXPL={'table_3.1.csv':"Filed Cloud-game matrix in $B, read P1 / P2 per cell with joint-payoff shading (darker = larger joint payoff; hover any cell for exact values). Rows are Cloud Providers strategies — Interoperable vs Proprietary Stack; columns are P2's choice — Collaborate API vs Exclusive Model. The Nash cell is Interoperable Stack + Exclusive Model ($462.8B / $130.5B): given Exclusivity, Cloud earns more from Interoperability, and given Interoperability, P2 earns more from Exclusivity. Compare it with the live drill-down above, where ★ marks equilibria and ◆ the best joint cell.",'table_3.2.csv':"Filed Hardware-game matrix in $B — the core game behind the paper's headline DWL. Rows are Hardware strategies — Open Access vs Walled Garden; columns are P2's stack choice — Interoperable vs Proprietary. The Nash cell is Open Access + Proprietary Stack ($487.9B / $606.8B); the Pareto optimum is Walled Garden + Interoperable Stack, and the $58.6B gap between their joint totals is the core-game deadweight loss. Darker shading marks larger joint payoffs; hover any cell for exact values."};
  div.innerHTML='<div class="grid g2" style="margin-top:14px">'+ex.map(x=>card(TT(x[0]),payoffExhibit(x[1])+xpl(EXPL[x[0]]||"Canonical payoff exhibit in $B (P1 row player / P2 column player). Darker cells carry larger joint payoffs; hover any cell for exact values."),x[0])).join('')+'</div>';
  $('#tab-games').appendChild(div);
 }
}
const TYPE_COLORS={"Prisoner's Dilemma":"#D55E00","Efficient NE":"#009E73","Dominant Strategy NE (Inefficient)":"#E69F00","Coordination Failure":"#CC79A7","Stag Hunt":"#0072B2"};
function pickGame(i){CUR_GAME=i;const s=document.getElementById('gsel');if(s)s.value=String(i);showGame();const d=document.getElementById('gdetail');if(d&&d.scrollIntoView)d.scrollIntoView({block:"nearest"});}
function bestResponses(g){
 const p1=[[false,false],[false,false]],p2=[[false,false],[false,false]];
 for(let c=0;c<2;c++){const m=Math.max(g.matrix[0][c][0],g.matrix[1][c][0]);for(let r=0;r<2;r++)if(g.matrix[r][c][0]===m)p1[r][c]=true;}
 for(let r=0;r<2;r++){const m=Math.max(g.matrix[r][0][1],g.matrix[r][1][1]);for(let c=0;c<2;c++)if(g.matrix[r][c][1]===m)p2[r][c]=true;}
 return {p1:p1,p2:p2};
}
function miniMatrix(g){
 const mx=Math.max.apply(null,g.matrix.flat().map(p=>p[0]+p[1]));
 const isNE=(r,c)=>g.ne.some(p=>p[0]===r&&p[1]===c);
 let h='<div style="display:grid;grid-template-columns:1fr 1fr;gap:3px">';
 for(let r=0;r<2;r++)for(let c=0;c<2;c++){const p=g.matrix[r][c];
  h+='<div class="cell'+(isNE(r,c)?' ne':'')+'" style="background:'+shade(p[0]+p[1],mx)+';color:'+cellInk(p[0]+p[1],mx)+';padding:5px 2px;font-size:11px" title="P1 '+fmt(p[0])+', P2 '+fmt(p[1])+(isNE(r,c)?' — NE':'')+'">'+fmt(p[0],0)+'/'+fmt(p[1],0)+(isNE(r,c)?' ★':'')+'</div>';}
 return h+'</div>';
}
function mcRow(name){return (DATA.mc||[]).find(r=>r["Game"]===name);}
function renderGcomp(){
 const gs=DATA.games;
 const cards=gs.map((g,i)=>{
  const mc=mcRow(g.name);const col=TYPE_COLORS[g.game_type]||"#5D6D7E";
  return '<div class="card" style="cursor:pointer" onclick="pickGame('+i+')"><h3><span class="dot" style="background:'+col+'"></span>'+esc(g.name)+'</h3>'+miniMatrix(g)
  +'<p class="legend">DWL $'+fmt(g.welfare.deadweight_loss)+'B · Eff '+fmt(g.welfare.efficiency_ratio,0)+'%'+(mc?' · MC $'+fmt(mc["Mean_DWL_$B"])+'B':'')+'<br>'+esc(g.game_type||"unclassified")+'</p></div>';}).join("");
 const rows=gs.map((g,i)=>{const mc=mcRow(g.name);
  return '<tr style="cursor:pointer" onclick="pickGame('+i+')" title="Drill into '+esc(g.name)+'"><td>'+esc(g.name)+'</td><td>'+esc(g.game_type||"—")+'</td><td>'+(g.ne.length?g.ne.map(p=>"("+p[0]+","+p[1]+")").join(", "):"mixed")+'</td><td class="num">'+fmt(g.welfare.deadweight_loss)+'</td><td class="num">'+fmt(g.welfare.efficiency_ratio,1)+'</td><td class="num">'+(mc?fmt(mc["Mean_DWL_$B"]):"—")+'</td></tr>';}).join("");
 document.getElementById("gcomp").innerHTML='<h3 style="margin:20px 0 10px">All six games at a glance <small style="color:var(--mut);font-weight:normal">click any card or row to drill in</small></h3>'
 +'<div class="grid g4" style="margin-bottom:14px">'+cards+'</div>'
 +card("Cross-game comparison",'<table class="dt"><thead><tr><th>Game</th><th>Type</th><th>NE</th><th>DWL ($B)</th><th>Eff (%)</th><th>MC mean ($B)</th></tr></thead><tbody>'+rows+'</tbody></table>'+xpl("Every bilateral game on one screen: equilibrium cell, deadweight loss, efficiency, and Monte-Carlo mean side by side. Only Hardware–Cloud Providers carries base-game DWL; the others lose welfare only under perturbation. Click any card or row to drill into that game above — start here to choose which game deserves attention, then read the drill-down for mechanism."),"all games");
}
function showGame(){
 const g=DATA.games[CUR_GAME];const mx=Math.max.apply(null,g.matrix.flat().map(p=>p[0]+p[1]));
 const isNE=(r,c)=>g.ne.some(p=>p[0]===r&&p[1]===c);
 const br=bestResponses(g);
 const tot=g.matrix.map(row=>row.map(p=>p[0]+p[1]));
 const pmax=Math.max.apply(null,tot.flat());
 const isPO=(r,c)=>tot[r][c]===pmax;
 let cells='';
 for(let r=0;r<2;r++)for(let c=0;c<2;c++){const p=g.matrix[r][c];
  const marks=(isNE(r,c)?" ★":"")+(isPO(r,c)?" ◆":"");
  const brm=(br.p1[r][c]?"1":"")+(br.p1[r][c]&&br.p2[r][c]?"/":"")+(br.p2[r][c]?"2":"");
  cells+='<div class="cell'+(isNE(r,c)?' ne':'')+'" style="background:'+shade(p[0]+p[1],mx)+';color:'+cellInk(p[0]+p[1],mx)+'" title="total '+fmt(p[0]+p[1])+'">P1 '+fmt(p[0])+'<br>P2 '+fmt(p[1])+(marks?'<br>'+marks:'')+'<br><span style="font-size:10px;font-weight:normal">BR:'+(brm||"—")+'</span></div>';}
 const w=g.welfare;
 const mcrow=(DATA.mc||[]).find(r=>r['Game']===g.name);
 $('#gdetail').innerHTML='<div class="grid g2"><div>'+card(esc(g.p1)+' vs '+esc(g.p2)+' <small>'+esc(g.game_type||'')+'</small>',
  '<div class="mgrid"><div></div><div class="clab" style="text-align:center">'+esc(g.p2)+': '+esc(g.p2_strats[0])+' (left) / '+esc(g.p2_strats[1])+' (right)</div><div></div>'
  +'<div class="rlab">'+esc(g.p1)+'<br>'+esc(g.p1_strats[0])+'<br>'+esc(g.p1_strats[1])+'</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">'+cells+'</div><div></div></div>'
  +'<p class="legend">Cell shading = joint payoff. ★ Nash equilibrium · ◆ Pareto-optimal cell · BR:1/2 = best response of P1/P2.</p>'
  +xpl("Rows are P1 strategies, columns are P2 strategies; each cell shows both players payoffs in $B, shaded by joint payoff (darker = larger total). A starred cell is a Nash equilibrium: neither player can gain by deviating alone, which is exactly where both best-response flags coincide — BR:1 marks P1's best reply to each P2 column, BR:2 marks P2's best reply to each P1 row. The diamond marks the highest joint-payoff cell. Deadweight loss equals Pareto-optimal total minus Nash total for the selected equilibrium; the welfare panel at right converts that gap into efficiency and DWL share. Hover any cell for exact payoffs."))+'</div>'
 +'<div>'+card('Equilibrium & Welfare',
  '<table class="dt"><tbody>'
  +'<tr><th>Nash equilibrium</th><td>'+(g.ne.length?g.ne.map(p=>'('+p[0]+','+p[1]+')').join(', '):'none (mixed only)')+'</td></tr>'
  +'<tr><th>Deadweight loss</th><td class="num">$'+fmt(w.deadweight_loss)+'B</td></tr>'
  +'<tr><th>Efficiency</th><td class="num">'+fmt(w.efficiency_ratio,1)+'%</td></tr>'
  +'<tr><th>Pareto welfare</th><td class="num">$'+fmt(w.pareto_optimal_welfare)+'B</td></tr>'
  +'<tr><th>Game type</th><td><span class="badge" style="background:'+(TYPE_COLORS[g.game_type]||'#5D6D7E')+'">'+esc(g.game_type||'unclassified')+'</span></td></tr>'
  +'<tr><th>NE joint welfare</th><td class="num">$'+fmt(w.nash_welfare)+'B</td></tr>'
  +'<tr><th>Pareto cell</th><td>('+w.pareto_position+')</td></tr>'
  +'<tr><th>DWL share of Pareto</th><td class="num">'+fmt(num(w.dwl_percent)*100,2)+'%</td></tr>'
  +(mcrow?'<tr><th>MC mean DWL</th><td class="num">$'+fmt(mcrow['Mean_DWL_$B'])+'B (MCSE '+fmt(mcrow['MCSE_DWL_$B'],4)+', '+esc(mcrow['Assessment'])+')</td></tr>':'')
  +'</tbody></table>')+'</div></div>';
}
function buildWelfare(){
 $('#tab-welfare').innerHTML=secDesc('Where the $'+fmt(DATA.kpi.dwl)+'B aggregate deadweight loss comes from. Components are normalized to sum exactly to the total.')
 +'<div class="grid g2">'
 +card('DWL Decomposition','<div id="c-wd"></div>'+xpl("Same attribution as the Overview donut: five channels rescaled to sum exactly to the $416.77B aggregate. Hover slices for values and shares — switching costs is the largest slice, which is why portability remedies lead the policy ranking."),'shares sum to total')
 +card(TT('table_5.3.csv'),dlBtn('welfare.csv','DATA.welfare')+'<div id="t-wd"></div>'+xpl("Five-channel attribution of the aggregate deadweight loss, rescaled to sum exactly to the total. Percentages are shares of the aggregate DWL, so the rows rank where the welfare loss has an address — match each row to its remedy on the Policy tab."),'table_5.3.csv')+'</div>';
 donut('c-wd',welfareParts().map(r=>r['Source of Deadweight Loss']),welfareParts().map(r=>num(r['Estimated Contribution ($B)'])),{center:'$'+fmt(DATA.kpi.dwl)+'B'});
 $('#t-wd').innerHTML=dtable(DATA.welfare,colsOf(DATA.welfare),{});
 const wc=(DATA.welfare_calib||[]).map(r=>({Metric:r['Parameter'],Value:r['Value']}));
 if(wc.length){
  const div=document.createElement('div');
  div.innerHTML='<div class="grid g2" style="margin-top:14px">'+card(TT('table_4.1.csv'),'<div id="t-wc"></div>'+xpl("Monte Carlo stability summary plus DWL calibration: the market-estimated DWL percentage, the ±20% sweep range, and the elasticity used for the local component. The sweep's 1.41 elasticity says how fast DWL moves with the calibration — small input errors stay small in the headline."),'table_4.1.csv')+'</div>';
  $('#tab-welfare').appendChild(div);
  $('#t-wc').innerHTML=dtable(wc,colsOf(wc),{});
 }
}
function buildMC(){
 const allOk=DATA.mc.every(r=>r['Assessment']==='adequate');
 const worst=DATA.mc.slice().sort((a,b)=>num(b['MCSE_Rel_Pct'])-num(a['MCSE_Rel_Pct']))[0];
 $('#tab-mc').innerHTML=secDesc('100,000 perturbed-payoff draws per game (5% log-normal noise). Diagnostics: Monte-Carlo standard error and split-half consistency.')
 +insight(allOk?'<b>All '+DATA.mc.length+' games adequate:</b> relative MCSE and split-half drift under 5%. Largest relative MCSE is <b>'+esc(worst['Game'])+'</b> ('+fmt(worst['MCSE_Rel_Pct'],2)+'% — still negligible).':'<b>Mixed convergence:</b> check games not marked adequate before citing their means.')
 +'<div class="grid g2">'
 +card('Mean Simulated DWL by Game ($B)','<div id="c-mc"></div>'+xpl("Mean DWL over 100,000 draws per game with 5% log-normal payoff noise; draws with no pure Nash equilibrium are excluded. Longer bars mean larger simulated losses, not worse convergence — convergence is judged only by relative MCSE and split-half drift in the diagnostics table."),'table_7.9')
 +card(TT('table_7.9_mc_convergence.csv'),dlBtn('mc_convergence.csv','DATA.mc')+'<div id="t-mc"></div>'+xpl("One row per game: valid draws, mean simulated DWL, Monte-Carlo standard error with its relative size, split-half drift, and the adequacy verdict. N Valid of 100,000 means every draw found a pure Nash equilibrium. Adequate requires both relative MCSE and drift under 5% — check the verdict column before citing any game's mean."),'table_7.9_mc_convergence.csv')+'</div>';
 hbar('c-mc',DATA.mc.map(r=>r['Game']),DATA.mc.map(r=>num(r['Mean_DWL_$B'])));
 $('#t-mc').innerHTML=dtable(DATA.mc,colsOf(DATA.mc),{});
}
function buildRisk(){
 const pf={};DATA.portfolio.forEach(r=>{pf[r['Metric']]=r['Value'];});
 const circ=DATA.circular[0]||{};
 $('#tab-risk').innerHTML=secDesc('Equal-weight portfolio simulation, dependency-network centrality, player stability tiers, and circular-deal exposure.')
 +'<div class="grid g4">'
 +card('Portfolio Mean Return',kpi(esc(pf['Mean Return (%)'])+'%','Sharpe '+esc(pf['Sharpe Ratio']))+xpl("Equal-weight portfolio over the four archetypes with dependency-driven correlations — diversification is limited because the same circular deals link all four legs. Sharpe is mean divided by std (no risk-free adjustment); positive means the simulated portfolio beats zero on a risk-adjusted basis."))
 +card('Portfolio VaR 95%',kpi(esc(pf['VaR 95% (%)'])+'%','CVaR '+esc(pf['CVaR 95% (%)'])+'%')+xpl("VaR 95% is the 5th percentile of simulated portfolio returns; CVaR is the average return at or below VaR — the expected loss in the bad tail. CVaR exceeds VaR by construction; both are percentages of portfolio value, so scale them by exposure before comparing with dollar losses elsewhere."))
 +card('Circular Value',kpi('$'+fmt(circ['total_value'])+'B','dependency'))
 +card('Bubble Score',kpi(fmt(circ['bubble_score'],3),'systemic '+fmt(circ['systemic'],3))+xpl("Exposure of ecosystem revenue to circular investment deals. Higher bubble and systemic scores mean more fragility to financing-chain stress. Compare with the Burst tab: scores anticipate vulnerability, Table 7.11 realizes it as dollar losses."))+'</div>'
 +'<div class="grid g2" style="margin-top:14px">'
 +card('Stability Scores','<div id="c-su"></div>'+xpl("Weighted score by archetype: strategic risk 0.3, profitability 0.3, revenue 0.2, network degree 0.1, PageRank 0.1 (network parts max-normalized). Tiers: above 0.7 High, above 0.4 Moderate, otherwise Low. Taller bars are more stable archetypes — stability here means resilience of position, not safety of investment."),'table_A.3')
 +card('Network Centrality','<div id="c-nw"></div>'+xpl("Degree is normalized (in+out) centrality and PageRank is relative upstream importance on the dependency network. Higher values mark archetypes the ecosystem cannot route around — the nodes whose impairment propagates furthest in the burst ranking."),'table_A.0')+'</div>'
 +'<div class="grid g2" style="margin-top:14px">'
 +card(TT('table_A.3.success.scores.csv'),'<div id="t-su"></div>'+xpl("Stability score per archetype with its tier. Scores top at 1.0 for the leading archetype because the network components are max-normalized; tier cutoffs 0.7 and 0.4 are descriptive labels, not estimated thresholds."),'table_A.3.success.scores.csv')
 +card(TT('table_A.0.network.csv'),'<div id="t-nw"></div>'+xpl("Dependency-network centrality per archetype: normalized degree, betweenness, PageRank, and eigenvector scores, rounded to three decimals. Higher values mark harder-to-route-around positions — betweenness flags brokers, eigenvector flags connection to other central nodes. Compare columns before rows: each metric answers a different question."),'table_A.0.network.csv')+'</div>';
 hbar('c-su',DATA.success.map(r=>r['_player']||r['Player']||r['Category']||''),DATA.success.map(r=>num(r['Score']||r['stability_score'])));
 const deg=DATA.network.map(r=>({l:r['_player']||r['Player']||r['Category']||r['Node']||'',v:num(r['Degree'])}));
 hbar('c-nw',deg.map(d=>d.l),deg.map(d=>d.v));
 $('#t-su').innerHTML=dtable(DATA.success,colsOf(DATA.success),{});
 $('#t-nw').innerHTML=dtable(DATA.network,colsOf(DATA.network),{});
 const nr=(DATA.netrisk||[]).map(r=>({Metric:r['Metric'],Value:r['Value']}));
 if(nr.length){
  const div=document.createElement('div');
  div.innerHTML='<div class="grid g2" style="margin-top:14px">'
  +card(TT('table_A.1.network.risk.csv'),'<div id="t-nr"></div>'+xpl("Network risk ledger: total vs critical dependency dollars, the critical-risk ratio, dependency HHI, density, and the identified critical node. The critical-risk ratio is critical dollars over total dependency dollars — the share of the network that fails together under stress."),'table_A.1.network.risk.csv')
  +card(TT('table_A.4.layer.exposure.csv'),'<div id="t-lay"></div>'+xpl("Committed exposure by multilayer layer (Table A.4): procurement enters DebtRank at face, equity enters the valuation haircut, the guarantee is contingent; prospective and vertical-integration rows are memo lines. Layers are distinct exposure types, not additive units of one risk."),'table_A.4.layer.exposure.csv')
  +card(TT('table_5.4.csv'),'<div id="t-cp"></div>'+xpl("Cooperation screen: short- vs long-term payoffs per archetype with time-preference and risk labels. Read archetype by archetype: where long exceeds short, patience sustains cooperation; where it does not, the non-cooperative Nash of the Games tab binds."),'table_5.4.csv')+'</div>'
  +'<div class="grid g2" style="margin-top:14px">'
  +card('Shapley Allocation (% of Total)','<div id="c-sh"></div>'+xpl("Each archetype's share of the grand-coalition value under exact Shapley allocation — the fair split of what full cooperation would create. Hardware and Cloud Providers jointly command nearly four-fifths, so cooperation surplus concentrates with infrastructure; shares sum to 100% by construction."),'table_5.4.coop.csv')
  +card(TT('table_5.4.coop.csv'),dlBtn('shapley.csv','DATA.shapley')+'<div id="t-sh"></div>'+xpl("Shapley allocation: original standalone values, Shapley values, gains from cooperation, and shares. Gain from cooperation is Shapley value minus standalone value — every archetype gains, which is the efficiency axiom in action and the cooperative counterfactual to the Nash outcomes."),'table_5.4.coop.csv')+'</div>';
  $('#tab-risk').appendChild(div);
  $('#t-nr').innerHTML=dtable(nr,colsOf(nr),{});
  $('#t-lay').innerHTML=dtable(DATA.layer_exposure||[],colsOf(DATA.layer_exposure||[]),{});
  $('#t-cp').innerHTML=dtable(DATA.coop||[],colsOf(DATA.coop||[]),{});
  $('#t-sh').innerHTML=dtable(DATA.shapley||[],colsOf(DATA.shapley||[]),{});
  hbar('c-sh',(DATA.shapley||[]).map(r=>r['Player']),(DATA.shapley||[]).map(r=>num(r['Share of Total (%)'])));
 }
}
function buildBurst(){
 const f=DATA.burst_formation||[],rk=DATA.burst_ranking||[],gdp=DATA.burst_gdp||[],pol=DATA.burst_policy||[];
 if(!rk.length){$('#tab-burst').innerHTML=secDesc('Bubble formation odds, burst impact ranking, GDP-at-risk and mitigation.')+insight('Burst tables absent — run the current ai_ecosystem_model.py first.');return;}
 const top=rk[0];
 const sev=gdp.find(r=>r['Channel']==='Combined'&&r['Severity']==='severe')||{};
 const best=pol.slice().sort((a,b)=>num(b['BCR_Base'])-num(a['BCR_Base']))[0]||{};
 $('#tab-burst').innerHTML=secDesc('From circular deals to burst: formation odds, who loses most, US GDP-at-risk, and what governments can do. All scenario arithmetic — see table notes for assumptions.')
 +insight('<b>'+esc(top['Archetype'])+'</b> is hit worst (rank 1, -$'+fmt(top['Total_Loss_Severe_$B'])+'B severe); a severe burst costs <b>'+fmt(sev['GDP_Share_Pct'],3)+'% of US GDP</b> ($'+fmt(sev['GDP_Loss_$B'])+'B). Best mitigation BCR: <b>'+esc(best['Intervention']||'—')+'</b> ('+fmt(best['BCR_Base'],1)+').')
 +'<div class="grid g4">'
 +card('Worst hit',kpi(esc(top['Archetype']),'rank 1 of '+rk.length))
 +card('Severe burst loss',kpi('$'+fmt(rk.reduce((a,r)=>a+num(r['Total_Loss_Severe_$B']),0))+'B','all archetypes'))
 +card('Formation (top)',kpi(fmt(top['Formation_Prob_Pct'],1)+'%' ,esc(top['Archetype'])))
 +card('US GDP at risk',kpi(fmt(sev['GDP_Share_Pct'],3)+'%','severe, combined'))+'</div>'
 +'<div class="grid g2" style="margin-top:14px">'
 +card('Formation probability (%)','<div id="c-bf"></div>'+xpl("Scenario-calibrated logistic composite of circular exposure, capex intensity, HHI share and exuberance — the odds that bubble conditions, not a realized crash, describe each archetype. The band is a stated z±0.4 sensitivity, not a confidence interval: order is the validated claim (exact in 65% of 300 joint weight draws, tau 0.88, FM top in 90%) while levels swing ~±25pp. Foundation Models tops the odds; Wrappers trails."),'table_7.10')
 +card('Severe loss by archetype ($B)','<div id="c-br"></div>'+xpl("Direct circular-capital impairment (70% of circular capital) plus one dependency-network propagation round under a 30% AI-capex cut, worst first. Equity impact is capped at 95% — no negative equity in a scenario. Validated: order exact in 100% of 300 joint draws (tau 1.00) with Hardware first in every draw ($239.4B vs $132.7B severe) — the HW-vs-Cloud top call is decisive and the rest of the order is firm."),'table_7.11')+'</div>'
 +'<div class="grid g2" style="margin-top:14px">'
 +card(TT('table_7.11_burst_ranking.csv'),'<div id="t-br"></div>'+xpl("Full ranking with mild-scenario losses, equity impact shares and revenue impact shares. Click headers to sort: sort by equity impact to see who hurts most relative to size, by total loss for absolute incidence. Direct vs network columns split own-impairment from propagated loss."),'table_7.11_burst_ranking.csv')
 +card('US GDP-at-risk','<div id="t-gdp"></div>'+xpl("AI-capex channel (one-for-one, no multiplier) plus equity-wealth channel at 4 cents on the dollar (MPC 0.04) — $120.0B + $19.1B = $139.1B severe, 0.452% of BEA GDP. No multiplier is applied, so the combined row is a floor. US-only calibration; other economies apply the per-$100B transmission rule to their own GDP and capex shares."),'table_7.12_gdp_impact.csv')+'</div>'
 +'<div class="grid g2" style="margin-top:14px">'
 +card(TT('table_7.13_bubble_policy.csv'),dlBtn('bubble_policy.csv','DATA.burst_policy')+'<div id="t-bp"></div>'+xpl("Mitigation options — disclosure and margin rules, GPU-collateral lending caps, a strategic compute reserve — in the same BCR machinery as §5 policy but denominated in severe burst loss, so rates here are NOT comparable to DWL-based BCRs. Compare tools by BCR for efficiency and by net benefit for scale; all are scenario assumptions with stated bases."),'table_7.13_bubble_policy.csv')
 +card('Mitigation net benefit, base ($B)','<div id="c-bp"></div>'+xpl("Base-case loss reduction minus implementation cost. The combined package wins on net benefit even where single tools win on ratio — Lending caps lead on efficiency (BCR 57.4), disclosure on sequencing value. Hover bars for exact values."),'')+'</div>'
 +'<div class="grid g2" style="margin-top:14px">'
 +card(TT('table_7.14_debtrank_contagion.csv'),'<div id="t-dr"></div>'+xpl("Full DebtRank reverberation over the first-order floor: equity buffers are validated market caps, the shock is the severe direct loss per archetype. Read Initial vs Final Distress for how far stress travels, DebtRank for each node's contagion contribution, and the multiplier 1.032 — the second round adds $15.2B to $478.3B of direct losses, concentrated in Foundation Models, because large equity buffers absorb most of the shock."),'table_7.14_debtrank_contagion.csv')
 +card(TT('table_7.15_credit_monitor.csv'),'<div id="t-cm"></div>'+xpl("Sourced spreads and run_pipeline with a stated-recovery hazard, haircut scenarios, and the live burst-rank watchlist — no firm-level spread is invented. Hazard is approximately spread ÷ (1 − recovery) at the stated 40% recovery; the $570B run_pipeline reads 1.43× one year of Big Four capex. Use the watchlist as the market-price cross-check on the burst ranking above."),'table_7.15_credit_monitor.csv')+'</div>';
 hbar('c-bf',f.map(r=>r['Archetype']),f.map(r=>num(r['Formation_Prob_Pct'])));
 hbar('c-br',rk.map(r=>r['Archetype']),rk.map(r=>num(r['Total_Loss_Severe_$B'])));
 hbar('c-bp',pol.map(r=>r['Intervention']),pol.map(r=>num(r['Net_Benefit_$B_Base'])));
 $('#t-br').innerHTML=dtable(rk,colsOf(rk),{});
 $('#t-gdp').innerHTML=dtable(gdp,colsOf(gdp),{});
 $('#t-bp').innerHTML=dtable(pol,colsOf(pol),{});
 const dr=DATA.debtrank||[],cm=DATA.credit||[];
 const tdr=document.getElementById('t-dr'),tcm=document.getElementById('t-cm');
 if(dr.length&&tdr)tdr.innerHTML=dtable(dr,colsOf(dr),{});
 if(cm.length&&tcm)tcm.innerHTML=dtable(cm,colsOf(cm),{});
}
let CUR_SCEN='Base';
function buildPolicy(){
 $('#tab-policy').innerHTML=secDesc('Intervention scenarios with literature-sourced reduction rates and costs. Net benefit = DWL reduction − cost; BCR = reduction ÷ cost.')
 +'<div class="toolbar"><label>Scenario: <select id="psel"><option>Conservative</option><option selected>Base</option><option>Optimistic</option></select></label></div>'
 +'<div class="grid g2">'+card('Net Benefit by Policy ($B)','<div id="c-pn"></div>'+xpl("DWL reduction minus implementation cost for the selected scenario — the absolute scale of each remedy. Reduction rates and costs come from the cited literature per intervention; the Combined Intervention assumes complementarity. Switch scenarios above to re-rank the tools: antitrust leads on ratio at base, interoperability on absolute gain."),'table_6.1.6.3')
 +card('Benefit–Cost Ratio','<div id="c-bcr"></div>'+xpl("DWL reduction divided by cost — the efficiency of each dollar spent. Above 1 means the intervention pays for itself in recovered welfare; every evaluated tool clears that bar at base case. Compare scenarios with the selector above: ratios compress under Conservative and stretch under Optimistic."),'')+'</div>'
 +card(TT('policy_interventions_enhanced.csv'),dlBtn('policy.csv','DATA.policy_ranges')+'<div id="t-pol"></div>'+xpl("One row per intervention with conservative, base, and optimistic DWL-reduction rates, implementation costs, net benefits, and benefit-cost ratios, each with its literature source. The base case uses the stated base rate, or the range midpoint where none is stated — read the source column before citing any single number."),'policy_interventions_enhanced.csv')
 +card(TT('table_6.4.bcr.break.even.csv'),'<div id="t-bcr6"></div>'+xpl("Break-even audit for both policy families (Table 6.4): denominator, base reduction, cost, base and conservative BCRs, the break-even reduction rate that exactly pays the base cost, and headroom. Welfare and burst families share the BCR form but never the denominator — do not compare their ratios rate-to-rate."),'table_6.4.bcr.break.even.csv');
 $('#psel').onchange=e=>{CUR_SCEN=e.target.value;showPolicy();};
 showPolicy();
 $('#t-bcr6').innerHTML=dtable(DATA.bcr_break_even||[],colsOf(DATA.bcr_break_even||[]),{});
}
function showPolicy(){
 const suf={'Conservative':'_Conservative','Base':'_Base','Optimistic':'_Optimistic'}[CUR_SCEN];
 const rows=DATA.policy_ranges.length?DATA.policy_ranges:DATA.policy;
 const lk=(r,b)=>r['Net_Benefit_$B'+suf]!==undefined?r['Net_Benefit_$B'+suf]:r['Net Annual Benefit ($B)'];
 const lb=r=>r['Policy']||r['Policy Scenario'];
 hbar('c-pn',rows.map(lb),rows.map(r=>num(lk(r))));
 const bcr=r=>r['BCR'+suf]!==undefined?r['BCR'+suf]:r['Benefit-Cost Ratio'];
 hbar('c-bcr',rows.map(lb),rows.map(r=>num(bcr(r))));
 $('#t-pol').innerHTML=dtable(rows,colsOf(rows),{});
}
function buildRevenue(){
 const rev=DATA.revenue;
 const top=rev.slice().sort((a,b)=>num(b['Projected Revenue 2030 ($B)'])-num(a['Projected Revenue 2030 ($B)']))[0];
 $('#tab-revenue').innerHTML=secDesc('Diminishing-growth projections (CAGR −2pp/year) and company valuation screen with circular-deal adjustments.')
 +insight('<b>'+esc(top['_player'])+'</b> is projected largest in 2030 at $'+fmt(top['Projected Revenue 2030 ($B)'])+'B (CAGR '+fmt(top['CAGR (%)'],1)+'%).')
 +'<div class="grid g2">'
 +card('Revenue 2025 vs 2030 ($B)','<div id="c-rv"></div>'+xpl("Diminishing-growth projection: each archetype CAGR decays 2 percentage points per year from 2025 observed revenue. Dark bars are observed 2025 revenue; light bars are 2030 model outputs, not forecasts — treat the gap as the earnings growth multiples must deliver, and hover bars for exact values."),'grouped bars')
 +card(TT('enhanced_valuation_metrics.csv'),dlBtn('valuation.csv','DATA.valuation')+'<div id="t-val"></div>'+xpl("Per-company multiples (market cap, P/E, EV/revenue, ROIC and its spread over WACC) with circular-deal adjustments; circular_adjusted_value discounts dependency exposure. Screen flags summarize the screen (Positive where cash flows cover the price, Negative where multiples need earnings to arrive, Neutral otherwise; N/A trailing multiples for loss-makers rather than forced flags) --- triage labels, not investment ratings. Type to filter rows; click headers to sort."),'enhanced_valuation_metrics.csv')+'</div>';
 const cats=rev.map(r=>r['_player']);
 const s2025=rev.map(r=>num(r['Revenue 2025 ($B)'])),s2030=rev.map(r=>num(r['Projected Revenue 2030 ($B)']));
 const W=560,H=300,pl=150,pb=60,pt=14;const mx=Math.max.apply(null,s2030.concat([0]))||1;
 const vreserve=10+7*Math.max.apply(null,s2030.map(v=>fmt(v).length).concat([4]));
 const plotW=W-pl-vreserve;
 let s='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 cats.forEach((c,i)=>{const w1=s2025[i]/mx*plotW,w0=s2030[i]/mx*plotW;
  s+='<text x="'+(pl-8)+'" y="'+(pt+i*62+34)+'" font-size="11" text-anchor="end">'+esc(c)+'</text>';
  s+='<rect x="'+pl+'" y="'+(pt+i*62+6)+'" width="'+w0.toFixed(1)+'" height="16" fill="#56B4E9"><title>2030 '+fmt(s2030[i])+'</title></rect>';
  s+='<rect x="'+pl+'" y="'+(pt+i*62+26)+'" width="'+w1.toFixed(1)+'" height="16" fill="#0072B2"><title>2025 '+fmt(s2025[i])+'</title></rect>';
  s+='<text x="'+(pl+Math.max(w0,w1)+6)+'" y="'+(pt+i*62+34)+'" font-size="11">'+fmt(s2030[i])+'</text>';});
 s+='<rect x="'+pl+'" y="'+(H-24)+'" width="12" height="12" fill="#0072B2"/><text x="'+(pl+16)+'" y="'+(H-14)+'" font-size="11">2025</text>';
 s+='<rect x="'+(pl+70)+'" y="'+(H-24)+'" width="12" height="12" fill="#56B4E9"/><text x="'+(pl+86)+'" y="'+(H-14)+'" font-size="11">2030</text></svg>';
 document.getElementById('c-rv').innerHTML=s;
 const revTab=$('#tab-revenue');
 const rdiv=document.createElement('div');
 rdiv.innerHTML='<div class="grid g2" style="margin-top:14px">'+card(TT('table_1.3.proj.csv'),'<div id="t-rv"></div>'+xpl("CAGR, observed 2025 revenue, and 2030 model revenue per archetype under diminishing growth (CAGR decays 2pp per year as the base compounds). Sort by 2030 revenue to see the projected ranking — the leader is the demand baseline the games are calibrated against."),'table_1.3.proj.csv')+'</div>';
 revTab.appendChild(rdiv);
 $('#t-rv').innerHTML=dtable(DATA.revenue,colsOf(DATA.revenue),{});
 $('#t-val').innerHTML=dtable(DATA.valuation,colsOf(DATA.valuation),{filter:true});
}
const DIAG_NOTES={
 "table_7.1_concentration_decomposition.csv":"Each archetype's HHI point contribution; contributions sum to the headline HHI. Read the largest contributor first — concentration here is a two-leader story. (Korinek & Vipra 2025)",
 "table_7.2_gatekeeper_index.csv":"Share of total dependency value targeting each archetype: who the ecosystem cannot route around. The top-ranked archetype is the gatekeeper whose impairment propagates furthest. (Hagiu & Wright 2025)",
 "table_7.3_contestability_screen.csv":"NE rent premia |p1-p2|/(p1+p2): large premia concentrated in one game cut against the contestability view. Small, diffuse premia would have favored it.",
 "table_7.4_stability_screen.csv":"Games whose NE is both highly stable under perturbation and materially loss-making: structural lock-in pockets where the bad outcome is also the sticky one. (Bichler et al. 2025; OECD 2025)",
 "table_7.5_fragility_composite.csv":"Mean of 0-1 fragility pillars (higher means more fragile); bands are descriptive labels, not forecasts. Compare archetypes relatively — the ranking, not the level, is the finding. (Pipeline composite)",
 "table_7.6_policy_synergy.csv":"Tests combined-vs-standalone DWL reduction: ratio above 1.05 reads as complementarity, below 0.95 as overlap. At 0.547 the package is sub-additive — sequence the tools rather than stacking them. (Lipsey & Lancaster 1956)",
 "table_7.7_five_pillar_screen.csv":"Five-pillar fragility screen per archetype (run_pipeline composite adapting standard bubble-diagnostic themes). Pillars that fail together across archetypes mark stack-wide, not idiosyncratic, risk.",
 "table_7.8_equilibrium_audit.csv":"Independent brute-force re-enumeration of every base-game NE: positions and counts must all match. Any mismatch fails the build — a match means the solver and the audit agree definitionally.",
 "table_7.9_mc_convergence.csv":"Per-game Monte-Carlo diagnostics: MCSE and split-half gap, both under 5% reading adequate. Check the verdict column before citing any mean simulated DWL.",
 "table_8.7_structural_estimation.csv":"Breakdown-frontier calibration: largest coefficient intervals preserving all six Nash position sets (A3) and the exact formation order (A8). Rows marked interior are two-sided identified sets; open ends hit the searched range. Narrower intervals mark where re-estimation matters most.",
 "table_8.1.csv":"Geweke mean-stability screens per game with Holm correction; Holm_verdict is binding. A pass means reported distributions are sampler equilibria, not transients.",
 "table_8.2.csv":"Full-sample reruns at two independent seeds; drift within 3 pooled SE reads replicated. Seed-independence is what lets ranks, not just levels, be cited.",
 "table_8.3.csv":"Stability and mean DWL per game at 2.5/5/10% noise; verdict is cross-sigma rank preservation. Watch whether ranks survive as noise rises — that gradient is the robustness claim.",
 "table_8.4.csv":"Saltelli variance decomposition (N=2048, 500 bootstraps): first/total-order indices with 95% CIs. The top total-order driver is the variable the model is most sensitive to — check it is the variable the paper emphasizes.",
 "table_8.5.csv":"Assumption register: each modeling choice, its justification, its check, and its risk (A8-A10 cover the bubble block). Read the risk column as the paper's own list of how it could be wrong.",
 "table_7.10_bubble_formation.csv":"Bubble-conditions probability per archetype (scenario-calibrated logistic; band is a stated sensitivity, not a CI). Cite the order across archetypes; treat levels as scenario markers.",
 "table_7.11_burst_ranking.csv":"Burst ranking, worst first: direct circular impairment plus one dependency-network round; equity capped at 95%. The HW-vs-Cloud top call is decisive ($239.4B vs $132.7B) — everything below it is firm.",
 "table_7.12_gdp_impact.csv":"US GDP-at-risk by channel and severity (BEA $30.8T, no multiplier) plus a portable per-$100B rule. The combined row is a floor; port the rule, not the level, to other economies.",
 "table_7.13_bubble_policy.csv":"Burst-mitigation options denominated in severe burst loss (NOT DWL): same BCR machinery, different denominator. Never compare these BCRs with DWL-based ones.",
 "table_7.14_debtrank_contagion.csv":"DebtRank reverberation ledger: direct loss, initial vs final distress, and each node's contagion contribution. The 1.032 multiplier says the second round adds $15.2B to $478.3B of direct losses — compare Final vs Initial Distress to see how far stress travels.",
 "table_7.15_credit_monitor.csv":"Sourced credit monitor: Oracle CDS print with stated-recovery hazard, run_pipeline scale, haircut scenarios, and the live burst-rank watchlist. Hazard is approximately spread divided by (1 minus recovery); no firm-level spread is invented.",
 "table_8.6.csv":"Validated-inputs register: ID, parameter, value, unit and verification source per row. Type in the filter box to narrow rows; every value here is the exact vintage the models ran on.",
"pred_A_scenario_ledger.csv":"Section 10 scenarios: mild/severe/activation ledger with direct, second-round, and GDP losses.",
"pred_B_trigger_watchlist.csv":"Section 9 triggers: observable plus tripwire per trigger.",
"pred_C_valuation_watchlist.csv":"Section 11 watchlist: 23-company screen, compact.",
"pred_D_robustness_ledger.csv":"Section 8 stress record: burst vs formation survival split.",
"pred_E_formation_drivers.csv":"Section 6 bubble scores: driver decomposition plus bands.",
"pred_F_early_warning.csv":"Sections 6-7: five-screen early-warning dashboard per archetype.",
"pred_tables_index.csv":"Registry of prediction ledgers: linked figures and use per table.",
"pred_G_burst_timing.csv":"Hazard-implied burst window per archetype plus the first-tip system row; slow/fast ends vary the stated 12-quarter lag, not the data.",
"pred_H_company_losses.csv":"23-company severe/mild split on market-cap and deal-flow bases; Severe_mid is the mean and the haircut is capped at wipeout.",
"pred_I_price_gauges.csv":"Convexity t-stats plus hazard scores for live names and frozen backtests; verdicts use the stated 2.0 fire line.",
"pred_J_gauge_backtest.csv":"Hazard crash-capture over SPY 2000-2026 with stated weights: capture share vs the 20% baseline.",
"pred_K_expected_losses.csv":"Per-player expected burst loss as a severity blend (P_burst x severe + rest x mild); driver names the dominant allocation basis.",
"pred_L_timing_rationale.csv":"Why-ledger behind the conditional burst-timing call: hazard base plus stated trigger, gauge, and precedent adjustments.",
"pred_M_ml_horserace.csv":"Walk-forward scoreboard, fitted L2 logit vs stated hazard: no stable winner, so no ML upgrade ships.",
"compare_summary.csv":"2008-vs-AI comparison: formation scores, worst-first losses, and mechanism readings side by side."};
function buildDiag(){
 const names=Object.keys(DATA.diag);
 $('#tab-diag').innerHTML=secDesc('Literature-grounded structural screens (§2.6 streams) plus the R1–R5 statistical robustness battery: stationarity, multi-seed replicability, noise robustness, Saltelli decomposition, and the assumption register.')
 +names.map(f=>card(TT(f),' <div id="d-'+f.replace(/[^A-Za-z0-9]/g,'')+'"></div>'+(DIAG_NOTES[f]?xpl(DIAG_NOTES[f]):''),f)).join('');
 names.forEach(f=>{document.getElementById('d-'+f.replace(/[^A-Za-z0-9]/g,'')).innerHTML=dtable(DATA.diag[f],colsOf(DATA.diag[f]),{});});
}
function buildFigs(){
 const base=DATA.plots_rel+'/';
 $('#tab-figs').innerHTML=secDesc('Static publication figures at print resolution. Thumbnails are small — <b>click any figure to inspect it full-size</b>.')
 +'<div class="figgrid">'+DATA.figures.map(f=>{var cap=(DATA.figcaps&&DATA.figcaps[f])||('Figure '+f.replace(/^figure_/,'').replace(/\\.png$/,''));return '<figure><img loading="lazy" src="'+esc(base+f)+'" alt="'+esc(cap)+'" style="cursor:zoom-in" onclick="openFig(this.src,this.alt)"><figcaption>'+esc(cap)+'</figcaption></figure>';}).join('')+'</div>';
}
const MODULE_TABS={EmbeddedDataSource:'market',MarketDataSource:'market',DependencyEnhancer:'market',ElasticitySensitivityAnalyzer:'policy',PolicyInterventionAnalyzer:'policy',LiteratureDiagnostics:'diag',NetworkRiskAnalysis:'risk',MarketStructureAnalyzer:'market',DataValidator:'market',CoordinationFailureAnalyzer:'games',MonteCarloSimulator:'mc',WelfareEconomicsAnalyzer:'welfare',SensitivityAnalysis:'diag',RobustnessBattery:'diag',RevenueProjection:'revenue',PortfolioRiskAnalysis:'risk',SuccessProbabilityModel:'risk',CooperativeGame:'games',TableGenerator:'diag',GameTheoryFramework:'games',CompanyFinancials:'revenue',ValuationMetricsCalculator:'revenue',VisualizationEngine:'figs',EnhancedValuationAnalyzer:'revenue',CircularDeal:'risk',CircularDealsAnalyzer:'risk',BubbleBurstAnalyzer:'burst',DebtRankClearingEngine:'burst',CreditMonitor:'burst',StructuralEstimationAnalyzer:'diag',TestFramework:'diag'};
let WALK_STAGE=null,TOUR=-1;
function walkVisible(){return (DATA.walkthrough.modules||[]).filter(m=>!WALK_STAGE||m.stage===WALK_STAGE);}
function buildWalk(){
 const w=DATA.walkthrough||{stages:[],modules:[]};
 if(!w.modules.length){$('#tab-walk').innerHTML=secDesc('Step-by-step guide to the ai_ecosystem_model.py analysis run_pipeline.')+insight('ai_ecosystem_model.py could not be parsed at build time — the tour needs the source file.');return;}
 const stages=w.stages||[];
 let h=secDesc('Step-by-step guide to the ai_ecosystem_model.py analysis run_pipeline, in execution order. Pick a stage to filter, or take the guided tour through every module.')
 +'<div class="stage-strip"><div class="stage-chip" data-s="" style="'+(WALK_STAGE?'':'font-weight:700')+'"><b>All stages</b>'+w.modules.length+' modules</div>'
 +stages.map(s=>{const n=w.modules.filter(m=>m.stage===s).length;if(!n)return '';
   return '<div class="stage-chip'+(WALK_STAGE===s?' on':'')+'" data-s="'+esc(s)+'"><b>'+esc(s.split('--')[0])+'</b>'+n+' module'+(n>1?'s':'')+'</div>';}).join('')+'</div>'
 +'<div class="tourbar"><span>Guided tour</span><button id="t-start">Start</button><button id="t-prev">← Prev</button><button id="t-next">Next →</button><button id="t-exit">Exit</button><span class="pos" id="t-pos"></span><span id="t-name"></span></div>'
 +'<div id="walkmods"></div>';
 $('#tab-walk').innerHTML=h;
 document.querySelectorAll('#tab-walk .stage-chip').forEach(c=>c.onclick=()=>{WALK_STAGE=c.dataset.s||null;TOUR=-1;buildWalk();});
 document.getElementById('t-start').onclick=()=>tourGo(0);
 document.getElementById('t-prev').onclick=()=>tourGo(TOUR-1);
 document.getElementById('t-next').onclick=()=>tourGo(TOUR+1);
 document.getElementById('t-exit').onclick=()=>tourGo(-1);
 renderWalkMods();tourGo(-1);
}
function renderWalkMods(){
 const vis=walkVisible();
 document.getElementById('walkmods').innerHTML=vis.map((m)=>{const idx=(DATA.walkthrough.modules||[]).indexOf(m);
  const tab=MODULE_TABS[m.name];
  return '<details class="mod" id="mod-'+idx+'"><summary><span class="mono">'+esc(m.name)+'</span><span class="stage-badge">'+esc(m.stage.split('--')[0])+'</span></summary><div class="body">'
  +'<p>'+esc(m.doc||'No description.')+'</p>'
  +(m.methods&&m.methods.length?'<div>'+m.methods.map(f=>'<span class="mchip mono">'+esc(f)+'</span>').join('')+'</div>':'')
  +(tab?'<p><button class="viewbtn" onclick="showTab(&quot;'+tab+'&quot;)">See it live in the dashboard →</button></p>':'')
  +'</div></details>';}).join('');
}
function tourGo(i){
 const vis=walkVisible();
 document.querySelectorAll('#walkmods .tour-on').forEach(e=>e.classList.remove('tour-on'));
 if(i<0||i>=vis.length){TOUR=-1;document.getElementById('t-pos').textContent='';document.getElementById('t-name').textContent='tour stopped — press Start';return;}
 TOUR=i;const m=vis[i];const idx=(DATA.walkthrough.modules||[]).indexOf(m);
 const el=document.getElementById('mod-'+idx);
 if(el){el.open=true;el.classList.add('tour-on');el.scrollIntoView({block:'center',behavior:'smooth'});}
 document.getElementById('t-pos').textContent=(i+1)+' / '+vis.length;
 document.getElementById('t-name').textContent=m.name+' — '+m.stage.split('--')[0];
}
function tabFor(mirror){const parts=String(mirror||'').split('.');const n=parseInt(parts[0],10);
 if(n===7&&parts.length>1&&parseInt(parts[1],10)>=10)return 'burst';
 if(n===1||n===2)return 'market';if(n===3||n===4)return 'games';if(n===5)return 'welfare';if(n===6)return 'policy';if(n===7||n===8)return 'diag';return null;}
function repTable(entry){
 const cols=(entry.cols||[]).map((c,i)=>{const k='c'+i;const vals=(entry.rows||[]).map(r=>r[i]);
  const numeric=vals.some(v=>isFinite(num(v))&&String(v).trim()!=='');return [k,c||('col '+(i+1)),numeric];});
 const rows=(entry.rows||[]).map(r=>{const o={};cols.forEach((c,i)=>{o[c[0]]=r[i];});return o;});
 return dtable(rows,cols,{});
}
function buildReport(){
 const r=DATA.report||{};
 if(r.absent){$('#tab-report').innerHTML=secDesc('The build_technical_report.py technical report, browsable section by section.')+insight('Report unavailable: '+esc(r.absent)+'.');return;}
 const secs=r.sections||[];
 if(!secs.length){$('#tab-report').innerHTML=secDesc('The build_technical_report.py technical report.')+insight('No sections parsed from the report.');return;}
 let h=secDesc('The full build_technical_report.py technical report — narrative, insight callouts and tables — browsable by section. Mirror tables link back to their live dashboard tab.')
 +'<div class="repnav">'+secs.map((s,i)=>'<button data-s="repsec-'+i+'" title="'+esc(s.title)+'">'+esc(s.title.length>42?s.title.slice(0,42)+'…':s.title)+'</button>').join('')+'</div>';
 secs.forEach((s,i)=>{
  h+='<div class="card repsec" id="repsec-'+i+'"><h3><span class="dot"></span>'+esc(s.title)+'</h3>';
  (s.blocks||[]).forEach(b=>{
   if(b.h)h+='<h4>'+esc(b.h)+'</h4>';
   (b.paras||[]).forEach(p=>{h+=p.insight?insight(esc(p.t)):'<p>'+esc(p.t)+'</p>';});
   (b.tables||[]).forEach((t,ti)=>{
    const tab=t.gotab||(t.mirror?tabFor(t.mirror):null);
    h+='<p style="margin-bottom:2px"><b>'+esc(t.caption)+'</b>'+(t.truncated?' <small style="color:var(--mut)">'+esc(t.truncated)+'</small>':'')
    +(tab?' <button class="viewbtn" onclick="showTab(&quot;'+tab+'&quot;)">live table →</button>':'')+'</p>';
    h+='<div id="rt-'+i+'-'+ti+'"></div>';});
  });
  h+='</div>';
 });
 $('#tab-report').innerHTML=h;
 document.querySelectorAll('#tab-report .repnav button').forEach(b=>b.onclick=()=>{const el=document.getElementById(b.dataset.s);if(el)el.scrollIntoView({block:'start',behavior:'smooth'});});
 secs.forEach((s,i)=>{(s.blocks||[]).forEach((b,bi)=>{(b.tables||[]).forEach((t,ti)=>{
  const el=document.getElementById('rt-'+i+'-'+ti);if(el)el.innerHTML=repTable(t);});});});
}
function init(){
 const tabs=(typeof TABSDEF!=='undefined'&&TABSDEF&&TABSDEF.length)?TABSDEF:[["overview","Overview"],["market","Market"],["games","Games"],["welfare","Welfare"],["mc","Monte Carlo"],["risk","Risk & Network"],["policy","Policy"],["revenue","Revenue & Valuation"],["diag","Diagnostics"],["figs","Figures"]];
 initTabs(tabs);
 armTip();
 buildOverview();buildWalk();buildReport();buildInputs();buildMarket();buildGames();buildWelfare();buildMC();buildRisk();buildBurst();buildPolicy();buildRevenue();buildDiag();buildFigs();
 try{const h=(location.hash||'').replace('#','');if(h&&document.getElementById('tab-'+h))showTab(h);}catch(e){}
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
</script>
"""


if __name__ == "__main__":
    main()
