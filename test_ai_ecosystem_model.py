#!/usr/bin/env python3
"""Detailed unit + integration tests for ai_ecosystem_model.py, build_technical_report.py, build_interactive_dashboard.py and run_pipeline.py.

Unit tests exercise every analysis module against the embedded data with small,
fast parameters (tiny Monte-Carlo counts, few sensitivity steps). Integration
tests verify cross-file continuity: table artifacts on disk,
build_technical_report.py integrity gates, and the dashboard payload/build
(the full suite runs in ~45-50 minutes, dominated by the end-to-end model run).

Run:
    cd airesearch && python3 -m unittest test_ai_ecosystem_model -v
Requires a completed `python3 ai_ecosystem_model.py` run for the integration (artifact) tests.

Coverage (over ai_ecosystem_model.py, build_technical_report.py, build_interactive_dashboard.py and run_pipeline.py -- target: 100%):
    pip install coverage
    cd airesearch && coverage run -m unittest test_ai_ecosystem_model -v
    coverage report -m
See .coveragerc for scope (test_ai_ecosystem_model.py itself is omitted from the report;
`if __name__ == "__main__":` blocks are excluded -- exercised only when a
file runs as a script, while in-process main() tests cover the logic).

A test_ai_ecosystem_model_run_YYYYMMDD_HHMMSS.log file is written next to this file on every
run (via setUpModule below), mirroring all logger output for the session.

Requirements (Python 3.9+ recommended; the code itself needs 3.7+ -- install
with pip):
  pip install numpy pandas matplotlib openpyxl python-docx
  - numpy / pandas: array asserts and DataFrame fixtures throughout.
  - matplotlib: figure-builder tests (Agg backend, closed after each test).
  - openpyxl: Excel round-trip asserts (import name: openpyxl).
  - python-docx: report build/parse asserts (import name: docx).
  Plus the ai_ecosystem_model, build_technical_report, build_interactive_dashboard and run_pipeline
  modules, imported from this directory -- their own REQUIREMENTS above apply transitively.
"""
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import ai_ecosystem_model
import build_technical_report
import build_interactive_dashboard as dash
import build_prediction_enhancement as enh
import build_retirement_exposure as retmod
import build_crisis_backtest as btmod
import build_compare_2008_ai as cmpmod
import build_prediction_artifacts as artmod
import run_pipeline

TABLES = os.path.join(HERE, 'tables')
PLOTS = os.path.join(HERE, 'figures')

TEST_LOG_PREFIX = "test_ai_ecosystem_model_run"
TEST_LOG_SUFFIX = ".log"
TEST_LOG_PATH = None


def setUpModule():
    """Attaches the session log file before the first test in this module runs.

    Writes test_ai_ecosystem_model_run_YYYYMMDD_HHMMSS.log next to this file (same folder as
    the tables/ and figures/ under test), mirroring every logger record at
    INFO level for the session. Fires under both `python test_ai_ecosystem_model.py` and
    `python -m unittest`, and never on a bare import: collection alone runs
    no tests, so no log file appears. Never raises; teardown closes it.
    """
    global TEST_LOG_PATH
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        TEST_LOG_PATH = os.path.join(HERE, f"{TEST_LOG_PREFIX}_{stamp}{TEST_LOG_SUFFIX}")
        handler = logging.FileHandler(TEST_LOG_PATH, mode="w", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger("test_ai_ecosystem_model").info(
            f"Test session started: {' '.join(sys.argv)} (log: {TEST_LOG_PATH})")
    except Exception as e:
        print(f"WARNING: test session log disabled: {e}", file=sys.stderr)
        TEST_LOG_PATH = None


def tearDownModule():
    """Flushes, closes and detaches the session log file after the last test."""
    global TEST_LOG_PATH
    if not TEST_LOG_PATH:
        return
    try:
        logging.getLogger("test_ai_ecosystem_model").info(f"Test session finished (log: {TEST_LOG_PATH})")
        root = logging.getLogger()
        for h in [h for h in root.handlers
                  if isinstance(h, logging.FileHandler)
                  and os.path.abspath(getattr(h, "baseFilename", "")) == os.path.abspath(TEST_LOG_PATH)]:
            try:
                h.flush()
            finally:
                root.removeHandler(h)
                h.close()
    except Exception as e:
        print(f"WARNING: could not close test session log: {e}", file=sys.stderr)


def solve_framework():
    """GameTheoryFramework with all six games solved (shared fixture)."""
    data = ai_ecosystem_model.load_all_data()
    gf = ai_ecosystem_model.GameTheoryFramework(data['players'])
    gf.construct_payoff_matrices()
    gf.find_nash_equilibria()
    return data, gf


class TestHelpers(unittest.TestCase):
    """TestHelpers."""
    def test_make_rng_deterministic(self):
        a = ai_ecosystem_model.make_rng(7).normal(size=50)
        b = ai_ecosystem_model.make_rng(7).normal(size=50)
        c = ai_ecosystem_model.make_rng(8).normal(size=50)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_hhi_bands(self):
        t = ai_ecosystem_model.HHI_THRESHOLDS['2023']
        self.assertEqual(ai_ecosystem_model.hhi_band(0), 'Unconcentrated')
        self.assertEqual(ai_ecosystem_model.hhi_band(t['moderate']), 'Unconcentrated')  # strict >
        self.assertEqual(ai_ecosystem_model.hhi_band(t['moderate'] + 1), 'Moderately Concentrated')
        self.assertEqual(ai_ecosystem_model.hhi_band(t['high']), 'Moderately Concentrated')
        self.assertEqual(ai_ecosystem_model.hhi_band(t['high'] + 1), 'Highly Concentrated')
        self.assertEqual(ai_ecosystem_model.hhi_band(10000), 'Highly Concentrated')

    def test_resolve_category_known_and_unknown(self):
        self.assertEqual(ai_ecosystem_model.resolve_category('NVIDIA'), 'Hardware')
        self.assertTrue(isinstance(ai_ecosystem_model.resolve_category('Some Unknown Corp XYZ'), str))

    def test_estimate_market_dwl_pct_bounded(self):
        v = ai_ecosystem_model.estimate_market_dwl_pct()
        self.assertGreaterEqual(v, 0)
        self.assertLessEqual(v, 100)

    def test_dashboard_style_contract(self):
        # The shared palette + grid default the dashboard consumes.
        import matplotlib.pyplot as plt
        pal = ai_ecosystem_model.set_dashboard_style()
        self.assertTrue({'green', 'amber', 'red', 'blue', 'purple',
                         'teal', 'gray'} <= set(pal))
        self.assertTrue(plt.rcParams['axes.grid'])

    def test_critical_paths_sorted_and_empty(self):
        # Critical-only edges come back value-sorted; an empty graph
        # yields [] instead of failing.
        players = pd.DataFrame({
            'Player_Category': ['A', 'B', 'C'],
            'Current_Revenue_Billions': [10.0, 20.0, 30.0],
            'Strategic_Risk': ['Medium'] * 3})
        deps = pd.DataFrame({
            'Dependent_Player': ['A', 'B', 'A'],
            'Dependency_On': ['B', 'C', 'C'],
            'Dependency_Value_Billions': [30.0, 10.0, 50.0],
            'Risk_Level': ['Critical', 'Critical', 'High']})
        paths = ai_ecosystem_model.NetworkRiskAnalysis(
            {'players': players, 'dependencies': deps}).get_critical_paths()
        self.assertEqual([p['value'] for p in paths], [30.0, 10.0])
        self.assertEqual(paths[0]['from_node'], 'A')
        self.assertEqual(paths[0]['to_node'], 'B')
        bare = ai_ecosystem_model.NetworkRiskAnalysis(
            {'players': pd.DataFrame(), 'dependencies': pd.DataFrame()})
        self.assertEqual(bare.get_critical_paths(), [])


class TestEmbeddedData(unittest.TestCase):
    """TestEmbeddedData."""
    @classmethod
    def setUpClass(cls):
        cls.data = ai_ecosystem_model.load_all_data()

    def test_players_frame(self):
        p = self.data['players']
        self.assertGreater(len(p), 0)
        for col in ('Player_Category', 'Current_Revenue_Billions'):
            self.assertIn(col, p.columns)
        self.assertTrue((p['Current_Revenue_Billions'] > 0).all())

    def test_dependencies_reference_categories(self):
        deps = self.data['dependencies']
        cats = set(self.data['players']['Player_Category'])
        self.assertGreater(len(deps), 0)
        self.assertTrue((deps['Dependency_Value_Billions'] > 0).all())

    def test_metadata_present(self):
        self.assertIn('metadata', self.data)
        self.assertTrue(bool(self.data['metadata']))

    def test_payoff_shapes(self):
        # Embedded payoffs are a strategy-level frame (4 players x 3 strategies)...
        sp = ai_ecosystem_model.EmbeddedDataSource().get_strategic_payoffs()
        self.assertEqual(len(sp), 12)
        self.assertIn('Strategy', sp.columns)
        # ...while solved games hold true 2x2 matrices.
        _, gf = solve_framework()
        for name, m in gf.payoff_matrices.items():
            self.assertEqual(np.asarray(m).shape, (2, 2), f'{name} not 2x2')

    def test_self_test_passes(self):
        self.assertIsNot(ai_ecosystem_model.run_data_self_test(), False)


class TestMarketStructure(unittest.TestCase):
    """TestMarketStructure."""
    @classmethod
    def setUpClass(cls):
        cls.data = ai_ecosystem_model.load_all_data()
        cls.ma = ai_ecosystem_model.MarketStructureAnalyzer(cls.data['players'])
        cls.conc = cls.ma.calculate_market_concentration()
        cls.mp = cls.ma.calculate_market_power()

    def test_hhi_valid(self):
        self.assertGreaterEqual(self.conc['HHI'], 0)
        self.assertLessEqual(self.conc['HHI'], 10000)

    def test_shares_sum_to_one(self):
        shares = list(self.conc['market_shares'].values())
        self.assertAlmostEqual(sum(shares), 1.0, places=6)

    def test_hhi_matches_shares(self):
        calc = sum(s ** 2 for s in self.conc['market_shares'].values()) * 10000
        self.assertAlmostEqual(self.conc['HHI'], calc, places=1)

    def test_gini_bounded(self):
        self.assertGreaterEqual(self.conc['Gini'], 0)
        self.assertLessEqual(self.conc['Gini'], 1)


class TestDependencyEnhancer(unittest.TestCase):
    """TestDependencyEnhancer."""
    def test_enhanced_files(self):
        data = ai_ecosystem_model.load_all_data()
        with tempfile.TemporaryDirectory() as td:
            de = ai_ecosystem_model.DependencyEnhancer(output_dir=td)
            cat_df, ind_df = de.create_enhanced_files(data['dependencies'])
            self.assertGreater(len(cat_df), 0)
            self.assertGreater(len(ind_df), 0)
            for col in ('Dependent_Category', 'Dependency_Category'):
                self.assertIn(col, ind_df.columns)

    def test_enhanced_booking_taxonomy(self):
        data = ai_ecosystem_model.load_all_data()
        with tempfile.TemporaryDirectory() as td:
            de = ai_ecosystem_model.DependencyEnhancer(output_dir=td)
            enhanced_df, category_df = de.create_enhanced_files(data['dependencies'])
            # Enhanced register carries all 19 tracked rows with status ...
            self.assertIn('Booking_Status', enhanced_df.columns)
            self.assertEqual(len(enhanced_df), 19)
            # ... while the category aggregation sums committed rows only.
            self.assertAlmostEqual(
                float(category_df['Total_Dependency_Value_Billions'].sum()), 675.40, places=2)
            self.assertNotIn(500.0, list(category_df['Total_Dependency_Value_Billions']))
            # Reconciliation columns ride along on every tracked row.
            for col in ('Booking_Status', 'Layer', 'Overlap_Flag'):
                self.assertIn(col, enhanced_df.columns, col)
            agg = enhanced_df.loc[enhanced_df['Dependency_Value_Billions'] == 249.2].iloc[0]
            self.assertIn('may overlap', agg['Overlap_Flag'])
            self.assertEqual(agg['Layer'], 'committed-procurement')


class TestNewRevisionTables(unittest.TestCase):
    """Tables 6.4 (BCR break-even), 7.1b (within-layer HHI), A.4 (layer exposure)."""
    @staticmethod
    def _tg():
        return ai_ecosystem_model.TableGenerator(output_dir='/tmp')

    def test_within_layer_hhi_values_and_bands(self):
        df = self._tg()._generate_table_7_1b_within_layer_hhi()
        self.assertEqual(len(df), 4)
        by_layer = df.set_index('Layer')
        self.assertAlmostEqual(by_layer.loc['Hardware', 'HHI_Points'], 4056.5, places=1)
        self.assertAlmostEqual(by_layer.loc['Cloud Providers', 'HHI_Points'], 4237.6, places=1)
        self.assertAlmostEqual(by_layer.loc['Foundation Models', 'HHI_Points'], 5283.4, places=1)
        self.assertAlmostEqual(by_layer.loc['LLM Wrappers', 'HHI_Points'], 5632.4, places=1)
        self.assertTrue((df['DOJ_Band'] == 'Highly concentrated').all())
        # Exclusion rule: no attributed-revenue company enters any layer.
        est = ai_ecosystem_model.ESTIMATED_REVENUE_NAMES
        self.assertNotIn(by_layer.loc['Hardware', 'Top_Company'], est)
        self.assertNotIn(by_layer.loc['Cloud Providers', 'Top_Company'], est)

    def test_layer_exposure_subtotals(self):
        deps = ai_ecosystem_model.EmbeddedDataSource.get_industry_dependencies()
        df = self._tg()._generate_table_A_4_layer_exposure(deps)
        by_layer = df.set_index('Layer')
        self.assertAlmostEqual(by_layer.loc['committed-procurement', 'Value_$B'], 518.40, places=2)
        self.assertAlmostEqual(by_layer.loc['ownership-equity', 'Value_$B'], 52.00, places=2)
        self.assertAlmostEqual(by_layer.loc['credit-guarantee', 'Value_$B'], 105.00, places=2)
        self.assertAlmostEqual(by_layer.loc['TOTAL committed', 'Value_$B'], 675.40, places=2)
        self.assertAlmostEqual(by_layer.loc['TOTAL tracked', 'Value_$B'], 1188.33, places=2)
        self.assertIn('MEMO', by_layer.loc['prospective-capacity', 'Booked'])
        self.assertIsNone(self._tg()._generate_table_A_4_layer_exposure(
            pd.DataFrame({'a': [1]})))

    def test_bcr_break_even_identity(self):
        w = pd.DataFrame([{
            'Policy': 'Antitrust Enforcement', 'DWL_Reduction_%_Base': 25.0,
            'DWL_Reduction_$B_Base': 104.19194223421601, 'Cost_$B_Base': 2.0,
            'BCR_Base': 52.10, 'BCR_Conservative': 15.63}])
        b = pd.DataFrame([{
            'Intervention': 'GPU-collateral lending caps', 'Burst_Reduction_%_Base': 12.0,
            'Cost_$B_Base': 1.0, 'BCR_Base': 57.40}])
        df = self._tg()._generate_table_6_4_bcr_break_even(w, b, 478.32)
        wel = df.loc[df['Family'] == 'welfare (DWL)'].iloc[0]
        self.assertAlmostEqual(wel['Denominator_$B'], 416.77, places=2)
        self.assertAlmostEqual(wel['BreakEven_Reduction_Pct'], 0.48, places=2)
        self.assertAlmostEqual(wel['Headroom_pp'], 25.0 - 0.48, places=2)
        bst = df.loc[df['Family'] == 'burst (severe loss)'].iloc[0]
        self.assertAlmostEqual(bst['BreakEven_Reduction_Pct'], 0.21, places=2)
        self.assertTrue(np.isnan(bst['BCR_Conservative']))
        self.assertTrue((df['Pays_At_Base'] == 'Yes').all())
        self.assertIsNone(self._tg()._generate_table_6_4_bcr_break_even())

    def test_live_break_even_all_pay(self):
        tg = self._tg()
        pa = ai_ecosystem_model.PolicyInterventionAnalyzer(
            total_dwl=416.76776893686404, output_dir='/tmp')
        w = pa.calculate_policy_ranges()
        df = tg._generate_table_6_4_bcr_break_even(w, None, None)
        self.assertEqual(len(df), len(w))
        self.assertTrue((df['Pays_At_Base'] == 'Yes').all())
        self.assertTrue((df['BCR_Conservative'] >= 1.0).all())


class TestElasticity(unittest.TestCase):
    """TestElasticity."""
    def test_lerner_identity_and_ordering(self):
        with tempfile.TemporaryDirectory() as td:
            ea = ai_ecosystem_model.ElasticitySensitivityAnalyzer(output_dir=td)
            df = ea.calculate_lerner_ranges()
            self.assertGreater(len(df), 0)
            for _, row in df.iterrows():
                # Lerner rule L = -1/epsilon on the base elasticity.
                self.assertAlmostEqual(row['Base_Lerner'], -1.0 / row['Base_Elasticity'], places=6)
                self.assertLessEqual(row['Conservative_Lerner'], row['Base_Lerner'])
                self.assertLessEqual(row['Base_Lerner'], row['Aggressive_Lerner'])
                self.assertTrue(os.path.exists(os.path.join(td, 'elasticity_sensitivity_analysis.csv')))


class TestPolicy(unittest.TestCase):
    """TestPolicy."""
    def test_bcr_arithmetic(self):
        with tempfile.TemporaryDirectory() as td:
            pa = ai_ecosystem_model.PolicyInterventionAnalyzer(total_dwl=400.0, output_dir=td)
            df = pa.calculate_policy_ranges()
            self.assertGreater(len(df), 0)
            row = df.iloc[0]
            # Base BCR = base benefit / base cost; net = benefit - cost.
            self.assertAlmostEqual(row['BCR_Base'],
                                   row['DWL_Reduction_$B_Base'] / row['Cost_$B_Base'], places=6)
            self.assertAlmostEqual(row['Net_Benefit_$B_Base'],
                                   row['DWL_Reduction_$B_Base'] - row['Cost_$B_Base'], places=6)
            # Conservative pairs low benefit with high cost.
            self.assertLessEqual(row['Net_Benefit_$B_Conservative'], row['Net_Benefit_$B_Base'])
            self.assertTrue(os.path.exists(os.path.join(td, 'policy_interventions_enhanced.csv')))


class TestGameTheory(unittest.TestCase):
    """TestGameTheory."""
    @classmethod
    def setUpClass(cls):
        cls.data, cls.gf = solve_framework()

    def test_six_games_solved(self):
        self.assertEqual(len(self.gf.nash_equilibria), 6)

    def test_equilibria_valid(self):
        for name, res in self.gf.nash_equilibria.items():
            eq = res.get('nash_equilibria', [])
            if eq:
                for e in eq:
                    self.assertIn(e['position'][0], (0, 1))
                    self.assertIn(e['position'][1], (0, 1))
            else:
                self.assertIn('mixed_equilibrium', res)

    def test_welfare_metrics_sane(self):
        for name, res in self.gf.nash_equilibria.items():
            w = res['welfare_metrics']
            self.assertGreaterEqual(w['deadweight_loss'], 0)
            self.assertGreater(w['efficiency_ratio'], 0)
            self.assertLessEqual(w['efficiency_ratio'], 100)

    def test_game_types_classified(self):
        for name in self.gf.nash_equilibria:
            vd = self.gf._prepare_matrix_visualization_data(name)
            self.assertTrue(vd and isinstance(vd.get('game_type'), str) and vd['game_type'])

    def test_visualize_2x2_legend_branch(self):
        # Standalone-figure path adds the legend + tight layout.
        import matplotlib.pyplot as plt
        md = {'payoffs': [[(3.0, 3.0), (0.0, 5.0)],
                          [(5.0, 0.0), (1.0, 1.0)]],
              'nash_equilibrium': (1, 1), 'pareto_optimal': (0, 0),
              'players': ('Row', 'Col')}
        try:
            ax = self.gf.visualize_2x2_matrix('smoke', md)
            self.assertGreaterEqual(len(ax.figure.legends), 1)
        finally:
            plt.close('all')


class TestDataValidator(unittest.TestCase):
    """TestDataValidator."""
    @classmethod
    def setUpClass(cls):
        cls.data, cls.gf = solve_framework()

    def test_valid_data_passes(self):
        self.assertTrue(ai_ecosystem_model.DataValidator.validate_market_data(self.data['players']))

    def test_negative_revenue_fails(self):
        bad = self.data['players'].copy()
        bad.loc[bad.index[0], 'Current_Revenue_Billions'] = -5.0
        self.assertFalse(ai_ecosystem_model.DataValidator.validate_market_data(bad))

    def test_valid_matrix_passes(self):
        m = self.gf.payoff_matrices['Hardware-Cloud Providers']
        self.assertTrue(ai_ecosystem_model.DataValidator.validate_payoff_matrix(m, 'Hardware', 'Cloud Providers'))

    def test_nan_matrix_fails(self):
        bad = np.full((2, 2), np.nan, dtype=object)
        self.assertFalse(ai_ecosystem_model.DataValidator.validate_payoff_matrix(bad, 'P1', 'P2'))


class TestCoordination(unittest.TestCase):
    """TestCoordination."""
    @classmethod
    def setUpClass(cls):
        _, cls.gf = solve_framework()
        cls.ca = ai_ecosystem_model.CoordinationFailureAnalyzer(cls.gf)
        cls.ca.identify_coordination_failures()

    def test_core_game_flagged(self):
        coord = self.ca.coordination_failures['Hardware-Cloud Providers']
        self.assertTrue(coord.get('is_coordination_failure'))
        self.assertIn(coord.get('severity'), ('None', 'Low', 'Moderate', 'High', 'Critical'))

    def test_severity_map_matches_live_dwl_pct(self):
        # Sep 2026 unit fix: dwl_percent arrived as a fraction while the
        # severity bands are percent units, so every game graded 'Low'.
        # Pin the true per-game map (G1 Moderate, G2/G3/G5 Critical,
        # G4/G6 High) so a unit regression fails loudly.
        expected = {'Hardware-Cloud Providers': 'Moderate',
                    'Hardware-Foundation Models': 'Critical',
                    'Hardware-LLM Wrappers': 'Critical',
                    'Cloud Providers-Foundation Models': 'High',
                    'Cloud Providers-LLM Wrappers': 'Critical',
                    'Foundation Models-LLM Wrappers': 'High'}
        for name, sev in expected.items():
            self.assertEqual(self.ca.coordination_failures[name].get('severity'),
                             sev, name)

    def test_keys_subset_of_games(self):
        self.assertTrue(set(self.ca.coordination_failures) <= set(self.gf.nash_equilibria))

    def test_total_dwl_nonnegative(self):
        self.assertGreaterEqual(self.ca.total_coordination_dwl, 0)


class TestMonteCarlo(unittest.TestCase):
    """TestMonteCarlo."""
    @classmethod
    def setUpClass(cls):
        _, cls.gf = solve_framework()

    def test_small_run_shape(self):
        mc = ai_ecosystem_model.MonteCarloSimulator(self.gf, n_simulations=300, seed=11)
        out = mc.run_monte_carlo_analysis()
        self.assertEqual(len(out), 6)
        for name, res in out.items():
            self.assertIn('nash_stability_pct', res)
            self.assertGreaterEqual(res['nash_stability_pct'], 0)
            self.assertLessEqual(res['nash_stability_pct'], 100)

    def test_seed_determinism(self):
        a = ai_ecosystem_model.MonteCarloSimulator(self.gf, n_simulations=300, seed=11).run_monte_carlo_analysis()
        b = ai_ecosystem_model.MonteCarloSimulator(self.gf, n_simulations=300, seed=11).run_monte_carlo_analysis()
        for name in a:
            self.assertAlmostEqual(a[name]['nash_stability_pct'], b[name]['nash_stability_pct'], places=6)


class TestWelfare(unittest.TestCase):
    """TestWelfare."""
    @classmethod
    def setUpClass(cls):
        cls.data, cls.gf = solve_framework()
        cls.wa = ai_ecosystem_model.WelfareEconomicsAnalyzer(cls.gf, cls.data['players'],
                                             dependencies_df=cls.data['dependencies'])
        cls.res = cls.wa.calculate_aggregate_welfare_loss()

    def test_total_nonnegative(self):
        self.assertGreaterEqual(self.wa.welfare_results['total_dwl'], 0)

    def test_decomposition_sums(self):
        total = self.wa.welfare_results['total_dwl']
        parts = self.wa._decompose_dwl(total)
        s = sum(float(v) for v in parts.values() if isinstance(v, (int, float, np.floating)))
        self.assertAlmostEqual(s, total, delta=max(1.0, total * 0.05))


class TestSensitivity(unittest.TestCase):
    """TestSensitivity."""
    def test_small_sweep(self):
        data, gf = solve_framework()
        wa = ai_ecosystem_model.WelfareEconomicsAnalyzer(gf, data['players'], dependencies_df=data['dependencies'])
        wa.calculate_aggregate_welfare_loss()
        sa = ai_ecosystem_model.SensitivityAnalysis(wa, gf)
        df = sa.run_sensitivity_analysis(variation_pct=0.10, steps=3)
        self.assertEqual(len(df), 3)
        self.assertIn('recalculated_total_dwl_billions', df.columns)


class TestRobustnessBattery(unittest.TestCase):
    """TestRobustnessBattery."""
    def test_assumption_register_complete(self):
        reg = ai_ecosystem_model.RobustnessBattery.assumption_register()
        self.assertEqual(list(reg['#']), [f'A{i}' for i in range(1, 11)])

    def test_stationarity_synthetic(self):
        _, gf = solve_framework()
        rng = np.random.default_rng(3)
        mc_syn = {'G1': {'dwl_distribution': rng.normal(10, 1, 2000)},
                  'G2': {'dwl_distribution': np.concatenate([rng.normal(10, 1, 1000),
                                                             rng.normal(20, 1, 1000)])}}
        rb = ai_ecosystem_model.RobustnessBattery(gf, mc_results=mc_syn)
        df = rb.run_stationarity()
        v = dict(zip(df['Game'], df['Verdict']))
        self.assertEqual(v['G1'], 'stationary')
        self.assertNotEqual(v['G2'], 'stationary')


class TestRevenueProjection(unittest.TestCase):
    """TestRevenueProjection."""
    def test_projection_positive(self):
        data = ai_ecosystem_model.load_all_data()
        rp = ai_ecosystem_model.RevenueProjection(data)
        df = rp.project_revenues()
        self.assertGreater(len(df), 0)
        for col in df.columns:
            if 'Revenue' in str(col) or '2030' in str(col):
                self.assertTrue((pd.to_numeric(df[col], errors='coerce').dropna() > 0).all())


class TestPortfolioRisk(unittest.TestCase):
    """TestPortfolioRisk."""
    def test_small_simulation(self):
        data = ai_ecosystem_model.load_all_data()
        pr = ai_ecosystem_model.PortfolioRiskAnalysis(data)
        out = pr.run_simulation(n_simulations=500, seed=5)
        self.assertIsInstance(out, dict)
        keys = ' '.join(out.keys()).lower()
        self.assertTrue('var' in keys or 'cvar' in keys or 'sharpe' in keys,
                        f'unexpected keys: {list(out.keys())}')

    def test_seed_determinism(self):
        data = ai_ecosystem_model.load_all_data()
        a = ai_ecosystem_model.PortfolioRiskAnalysis(data).run_simulation(n_simulations=500, seed=5)
        b = ai_ecosystem_model.PortfolioRiskAnalysis(data).run_simulation(n_simulations=500, seed=5)
        self.assertEqual(json.dumps(a, sort_keys=True, default=str),
                         json.dumps(b, sort_keys=True, default=str))


class TestStabilityScores(unittest.TestCase):
    """TestStabilityScores."""
    def test_scores_bounded(self):
        data = ai_ecosystem_model.load_all_data()
        sm = ai_ecosystem_model.SuccessProbabilityModel(data, network_centrality={})
        df = sm.calculate_scores()
        self.assertGreater(len(df), 0)
        score_col = [c for c in df.columns if 'score' in c.lower()][0]
        vals = pd.to_numeric(df[score_col], errors='coerce').dropna()
        self.assertTrue(((vals >= 0) & (vals <= 1)).all())


class TestCooperativeGame(unittest.TestCase):
    """TestCooperativeGame."""
    def test_shapley_efficiency(self):
        data = ai_ecosystem_model.load_all_data()
        cg = ai_ecosystem_model.CooperativeGame(data)
        cg.calculate_shapley_values()
        # Efficiency axiom: Shapley values allocate the grand-coalition value.
        grand = cg.characteristic_function(cg.players)
        self.assertAlmostEqual(sum(cg.shapley_values.values()), grand, delta=abs(grand) * 0.01 + 1e-6)
        self.assertTrue(all(v >= 0 for v in cg.shapley_values.values()))

    def test_all_players_allocated(self):
        data = ai_ecosystem_model.load_all_data()
        cg = ai_ecosystem_model.CooperativeGame(data)
        cg.calculate_shapley_values()
        self.assertEqual(set(cg.shapley_values), set(cg.players))


class TestTableHelpers(unittest.TestCase):
    """TestTableHelpers."""
    def test_latex_structure(self):
        df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
        tex = ai_ecosystem_model.TableGenerator._table_latex(df, 'Test caption', 'test_stem')
        self.assertIn(r'\begin{tabular}', tex)
        self.assertIn('Test caption', tex)

    def test_sanitize_latex(self):
        out = ai_ecosystem_model.TableGenerator._sanitize_latex('100% A&B_C')
        self.assertNotIn('%', out.replace(r'\%', ''))
        self.assertIn(r'\%', out)

    def test_excel_roundtrip(self):
        df = pd.DataFrame({'A': [1.5], 'B': ['z']})
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 't.xlsx')
            self.assertTrue(ai_ecosystem_model.TableGenerator._table_excel(df, p))
            self.assertTrue(os.path.exists(p))

    def test_table_4_1_aggregates_without_core_cell(self):
        # No Hardware-Cloud cell: the mean/std/CI/stability aggregate
        # across the interactions that do report (a missing key is filtered
        # out, not defaulted).
        import types
        fake = types.SimpleNamespace(
            n_simulations=300,
            simulation_results={
                'A': {'mean_dwl': 100.0, 'std_dwl': 10.0,
                      'nash_stability_pct': 90.0},
                'B': {'mean_dwl': 200.0}})
        df = ai_ecosystem_model.TableGenerator()._generate_table_4_1(fake)
        row = df.set_index('Parameter')['Value']
        self.assertEqual(row['Mean Simulated DWL ($B)'], '150.00')
        self.assertEqual(row['Standard Deviation of DWL ($B)'], '10.00')
        self.assertEqual(row['98% CI Lower Bound ($B)'], '126.70')
        self.assertEqual(row['98% CI Upper Bound ($B)'], '173.30')
        self.assertEqual(row['Nash Equilibrium Stability (%)'], '90.0')
        self.assertIsNone(ai_ecosystem_model.TableGenerator()._generate_table_4_1(None))


class TestValuation(unittest.TestCase):
    """TestValuation."""
    def test_company_financials_and_metrics(self):
        data = ai_ecosystem_model.load_all_data()
        eva = ai_ecosystem_model.EnhancedValuationAnalyzer(data['players'], circular_analyzer=None,
                                           plots_dir='/tmp', tables_dir='/tmp')
        row = data['players'].iloc[0]
        fin = eva.create_company_financials(row)
        self.assertIsInstance(fin, ai_ecosystem_model.CompanyFinancials)
        self.assertGreater(fin.revenue_ttm, 0)
        calc = ai_ecosystem_model.ValuationMetricsCalculator()
        m = calc.calculate_all_metrics(fin, tam_2030=1000.0)
        self.assertIsInstance(m, dict)
        self.assertAlmostEqual(m['market_cap'], fin.market_cap)

    def test_valuation_figures_ship_pdf_twins(self):
        # figure_17/18 save vector twins like every other exhibit.
        data = ai_ecosystem_model.load_all_data()
        with tempfile.TemporaryDirectory() as td:
            eva = ai_ecosystem_model.EnhancedValuationAnalyzer(
                data['players'], circular_analyzer=None,
                plots_dir=td, tables_dir=td)
            eva.generate_visualizations()
            eva.generate_comparison_table()
            for stem in ('figure_17', 'figure_18'):
                for ext in ('.png', '.pdf'):
                    p = os.path.join(td, stem + ext)
                    self.assertTrue(os.path.exists(p), p)
                    self.assertGreater(os.path.getsize(p), 0, p)

    def test_forward_pe_growth_clamp(self):
        import dataclasses
        data = ai_ecosystem_model.load_all_data()
        eva = ai_ecosystem_model.EnhancedValuationAnalyzer(data['players'], circular_analyzer=None,
                                           plots_dir='/tmp', tables_dir='/tmp')
        fin = eva.create_company_financials(data['players'].iloc[0])
        calc = ai_ecosystem_model.ValuationMetricsCalculator()
        base = calc.calculate_all_metrics(fin, tam_2030=1000.0)['forward_pe']
        # Hyper-growth must stay positive (clamped haircut), never NaN-flip sign.
        hyper = dataclasses.replace(fin, revenue_growth_yoy=1.5)
        fwd = calc.calculate_all_metrics(hyper, tam_2030=1000.0)['forward_pe']
        if base == base:  # defined only when trailing P/E exists
            self.assertGreater(fwd, 0)
        # Current-vintage growth inputs sit below the clamp: no-op expected.
        self.assertLessEqual(fin.revenue_growth_yoy, 0.99)

    def test_tex_nvidia_multiple_matches_live_csv(self):
        # Sep 2026 audit: the prose cited NVIDIA at 30.0x while the live
        # screen says 31.7x. Pin the prose to the CSV.
        vdf = pd.read_csv(os.path.join(TABLES, 'enhanced_valuation_metrics.csv'))
        pe = float(vdf.loc[vdf['company_name'] == 'NVIDIA', 'pe_ratio'].iloc[0])
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        self.assertIn(f'NVIDIA at {pe:.1f}x trailing', tex)
        self.assertNotIn('NVIDIA at 30.0x', tex)


class TestCircularDeals(unittest.TestCase):
    """TestCircularDeals."""
    def test_metrics_bounded(self):
        ca = ai_ecosystem_model.CircularDealsAnalyzer()
        m = ca.calc_metrics()
        for k in ('circ_conc', 'systemic', 'density', 'bubble_score'):
            self.assertGreaterEqual(m[k], 0)
            self.assertLessEqual(m[k], 1)
        self.assertGreater(m['total_value'], 0)

    def test_dcf_haircut_direction(self):
        ca = ai_ecosystem_model.CircularDealsAnalyzer()
        adj = ca.dcf_adjust(1000.0, 'Foundation Models')
        self.assertGreaterEqual(adj['exposure'], 0)
        self.assertLessEqual(adj['scenarios']['base'], 1000.0)
        self.assertLessEqual(adj['scenarios']['base'], adj['scenarios']['opt'])

    def test_factor_nonnegative(self):
        ca = ai_ecosystem_model.CircularDealsAnalyzer()
        self.assertGreaterEqual(ca._get_factor('Hardware', 'Cloud Providers'), 0.0)


class TestBubbleBurst(unittest.TestCase):
    """TestBubbleBurst."""
    @classmethod
    def setUpClass(cls):
        cls.data = ai_ecosystem_model.load_all_data()
        cls.ca = ai_ecosystem_model.CircularDealsAnalyzer()
        cls.bb = ai_ecosystem_model.BubbleBurstAnalyzer(cls.data['players'], cls.ca,
                                        dependencies_df=cls.data['dependencies'],
                                        output_dir=TABLES)

    def test_formation_bounded(self):
        df = self.bb.formation_probability()
        for col in ('Formation_Prob_Pct', 'Prob_Low_Pct', 'Prob_High_Pct'):
            self.assertTrue(((df[col] >= 0) & (df[col] <= 100)).all())

    def test_tex_wrappers_band_matches_live(self):
        # Sep 2026 audit: the prose cited the Wrappers high band as 56.2
        # while the live formation table says 56.1. Pin the prose.
        # NOTE: must pass revenue-share hhi_shares like the pipeline does;
        # the {} default silently substitutes a uniform fallback and moves
        # every band (that is what the first version of this test caught).
        data = ai_ecosystem_model.load_all_data()
        _rev = data['players'].set_index('Player_Category')['Current_Revenue_Billions']
        _shares = (_rev / _rev.sum()).to_dict()
        bb = ai_ecosystem_model.BubbleBurstAnalyzer(
            data['players'], ai_ecosystem_model.CircularDealsAnalyzer(),
            dependencies_df=data.get('dependencies', pd.DataFrame()),
            hhi_shares=_shares, output_dir='/tmp')
        df = bb.formation_probability()
        w = df.loc[df['Archetype'] == 'LLM Wrappers'].iloc[0]
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        self.assertIn(f"({w['Prob_Low_Pct']:.1f}--{w['Prob_High_Pct']:.1f})", tex)
        self.assertNotIn('36.5--56.2', tex)
        self.assertTrue(((df['Prob_Low_Pct'] <= df['Formation_Prob_Pct']) &
                         (df['Formation_Prob_Pct'] <= df['Prob_High_Pct'])).all())

    def test_ranking_permutation_descending(self):
        df = self.bb.burst_impact()
        self.assertEqual(sorted(df['Burst_Rank']), [1, 2, 3, 4])
        totals = df['Total_Loss_Severe_$B'].tolist()
        self.assertEqual(totals, sorted(totals, reverse=True))

    def test_equity_capped(self):
        df = self.bb.burst_impact()
        self.assertTrue((df['Equity_Impact_Severe_Pct'] <= 95).all())

    def test_gdp_combined_sums_channels(self):
        df = self.bb.gdp_impact(100.0)
        for sev in ('mild', 'severe'):
            sub = df[df['Severity'] == sev]
            ch = sub[sub['Channel'] != 'Combined']['GDP_Loss_$B'].sum()
            co = sub[sub['Channel'] == 'Combined']['GDP_Loss_$B'].iloc[0]
            self.assertAlmostEqual(ch, co, places=6)

    def test_policy_bcr_math(self):
        df = self.bb.bubble_policy(200.0)
        row = df.iloc[0]
        self.assertAlmostEqual(row['BCR_Base'],
                               row['Loss_Reduction_$B_Base'] / row['Cost_$B_Base'], places=6)

    def test_save_all_writes_five_tables(self):
        with tempfile.TemporaryDirectory() as td:
            bb = ai_ecosystem_model.BubbleBurstAnalyzer(self.data['players'], self.ca,
                                        dependencies_df=self.data['dependencies'],
                                        output_dir=td)
            out = bb.save_all()
            self.assertEqual(len(out), 5)
            self.assertIn('table_7.19_geography_overlay', out)
            for stem in out:
                for ext in ('.csv', '.tex', '.xlsx'):
                    self.assertTrue(os.path.exists(os.path.join(td, stem + ext)))

    def test_invariants_raise(self):
        bb = ai_ecosystem_model.BubbleBurstAnalyzer(self.data['players'], self.ca,
                                    dependencies_df=self.data['dependencies'],
                                    output_dir='/tmp')
        real = bb.formation_probability()
        bad = real.copy()
        bad.loc[bad.index[0], 'Formation_Prob_Pct'] = 150.0
        bb.formation_probability = lambda **kw: bad
        with self.assertRaises(ValueError):
            bb.save_all()

    def test_burst_timing_window_ordered(self):
        # pred_G: five rows (four archetypes + first-tip system); every row
        # reads earliest <= median <= latest and fast <= median <= slow.
        tmg = pd.read_csv(os.path.join(TABLES, 'pred_G_burst_timing.csv'))
        self.assertEqual(len(tmg), 5)
        self.assertIn('System (first-tip)', tmg['Archetype'].tolist())
        for _, r in tmg.iterrows():
            self.assertLessEqual(r['Earliest_q'], r['Median_q'], r['Archetype'])
            self.assertLessEqual(r['Median_q'], r['Latest_q'], r['Archetype'])
            self.assertLessEqual(r['Fast_q'], r['Median_q'], r['Archetype'])
            self.assertLessEqual(r['Median_q'], r['Slow_q'], r['Archetype'])
            self.assertRegex(r['Median_quarter'], r"^Q[1-4]'\d{2}$")
            self.assertIn('-', r['Window'])
        # System row inherits the max-formation archetype's clock.
        form = pd.read_csv(os.path.join(TABLES, 'table_7.10_bubble_formation.csv'))
        top = form.loc[form['Formation_Prob_Pct'].idxmax(), 'Archetype']
        sys_med = tmg.loc[tmg['Archetype'] == 'System (first-tip)', 'Median_q'].iloc[0]
        top_med = tmg.loc[tmg['Archetype'] == top, 'Median_q'].iloc[0]
        self.assertEqual(sys_med, top_med)
        self.assertIn('tips first', tmg.loc[tmg['Archetype'] == 'System (first-tip)',
                                            'Reading'].iloc[0])

    def test_company_losses_reconcile_and_cover(self):
        # pred_H: one row per valuation-screen company; each basis sums back
        # to its archetype's Table 7.11 total; weights sum to 100.
        val = pd.read_csv(os.path.join(TABLES, 'enhanced_valuation_metrics.csv'))
        losses = pd.read_csv(os.path.join(TABLES, 'pred_H_company_losses.csv'))
        burst = pd.read_csv(os.path.join(TABLES, 'table_7.11_burst_ranking.csv')).set_index('Archetype')
        self.assertEqual(len(losses), len(val))
        self.assertTrue(set(('NVIDIA', 'OpenAI', 'Anthropic')) <= set(losses['Company']))
        for a in ('Hardware', 'Cloud Providers', 'Foundation Models', 'LLM Wrappers'):
            sub = losses[losses['Archetype'] == a]
            self.assertAlmostEqual(sub['Severe_mcap_$B'].sum(),
                                   burst.loc[a, 'Total_Loss_Severe_$B'], places=1)
            self.assertAlmostEqual(sub['Severe_expo_$B'].sum(),
                                   burst.loc[a, 'Total_Loss_Severe_$B'], places=1)
            self.assertAlmostEqual(sub['Mild_mcap_$B'].sum(),
                                   burst.loc[a, 'Total_Loss_Mild_$B'], places=1)
            self.assertAlmostEqual(sub['Mild_expo_$B'].sum(),
                                   burst.loc[a, 'Total_Loss_Mild_$B'], places=1)
            self.assertAlmostEqual(sub['Mcap_wt_pct'].sum(), 100.0, delta=0.05)
            self.assertAlmostEqual(sub['Expo_wt_pct'].sum(), 100.0, delta=0.05)
        # Mild never exceeds severe on the same basis (across bases a mild
        # deal-basis loss can top a severe mcap-basis one, e.g. Lambda Labs).
        self.assertTrue((losses['Mild_mcap_$B'] <= losses['Severe_mcap_$B']).all(),
                        'mild must not exceed severe (mcap basis)')
        self.assertTrue((losses['Mild_expo_$B'] <= losses['Severe_expo_$B']).all(),
                        'mild must not exceed severe (deal basis)')
        self.assertTrue(((losses['Haircut_severe_pct'] >= 0)
                         & (losses['Haircut_severe_pct'] <= 100)).all())
        # Headline ordering: NVIDIA absorbs the most on the midpoint basis.
        self.assertEqual(losses.iloc[0]['Company'], 'NVIDIA')

    def test_tex_timing_median_matches_live(self):
        # The burst-timing prose pins the first-tip system median to the
        # live pred_G ledger (median quarters + calendar quarter).
        tmg = pd.read_csv(os.path.join(TABLES, 'pred_G_burst_timing.csv'))
        sys_row = tmg[tmg['Archetype'] == 'System (first-tip)'].iloc[0]
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        self.assertIn(f"median of {sys_row['Median_q']:.1f} quarters", tex)
        self.assertIn(f"({sys_row['Median_quarter']})", tex)

    def test_tex_company_losses_match_live(self):
        # The firm-level prose pins NVIDIA/OpenAI midpoints to pred_H.
        losses = pd.read_csv(os.path.join(TABLES, 'pred_H_company_losses.csv')).set_index('Company')
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        for co in ('NVIDIA', 'OpenAI', 'Anthropic'):
            mid = losses.loc[co, 'Severe_mid_$B']
            self.assertIn(f'{co} midpoint \\${mid:.1f}B', tex)

    def test_tex_capture_matches_live(self):
        # The gauge prose pins the crash-capture share and lift to pred_J.
        cap = pd.read_csv(os.path.join(TABLES, 'pred_J_gauge_backtest.csv')).set_index('Metric')
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        share = cap.loc['Crash-week capture share', 'Value']
        lift = cap.loc['Lift over baseline', 'Value']
        self.assertIn(f'{share * 100:.1f}\\%', tex)
        self.assertIn(f'{lift:.2f}x lift', tex)

    def test_convex_tstat_unit(self):
        # Accelerating log-price fires hard; linear reads small; decelerating
        # reads negative. Fixed-seed noise keeps the pins deterministic.
        rng = np.random.default_rng(0)
        t = np.arange(52.0)
        e = rng.normal(0, 0.005, 52)
        acc = 3.0 + 0.02 * t + 0.0008 * t ** 2 + e
        lin = 3.0 + 0.02 * t + e
        dec = 3.0 + 0.02 * t - 0.0008 * t ** 2 + e
        self.assertGreater(enh.convex_tstat(acc), 5.0)
        self.assertLess(abs(enh.convex_tstat(lin)), 2.0)
        self.assertLess(enh.convex_tstat(dec), -5.0)
        self.assertEqual(enh.convex_tstat(acc), enh.convex_tstat(acc))
        self.assertEqual(enh.convex_tstat(np.ones(52)), 0.0)

    def test_price_gauges_backtest_ordering(self):
        # pred_I: both historical tops fire on convexity; the no-crash
        # Feb'25 control stays quiet; hazard ranks the dot-com top first.
        g = pd.read_csv(os.path.join(TABLES, 'pred_I_price_gauges.csv'))
        self.assertEqual(len(g), 6)
        row = {f"{r['Name']}@{r['Asof'][:7]}": r for _, r in g.iterrows()}
        self.assertGreater(row['QQQ@2000-03']['Convex_max'], 5.0)
        self.assertGreater(row['SPY@2007-10']['Convex_max'], 2.0)
        self.assertLess(row['SPY@2025-02']['Convex_max'], 0.0)
        self.assertEqual(row['SPY@2025-02']['Convex_verdict'], 'quiet')
        self.assertGreater(row['QQQ@2000-03']['Hazard'],
                           row['SPY@2007-10']['Hazard'])
        self.assertGreater(row['SPY@2007-10']['Hazard'],
                           row['SPY@2025-02']['Hazard'])
        self.assertTrue(((g['Hazard'] >= 0) & (g['Hazard'] <= 100)).all())
        self.assertTrue(set(g['Regime']) <= {'High', 'Mid', 'Low'})
        live = g[g['Check'] == 'current']
        self.assertTrue((live['Convex_max'] > 2.0).all())

    def test_gauge_backtest_capture(self):
        # pred_J: recompute crash-capture from the vendored closes through
        # the builder's pure functions, require the committed table to match
        # the code, and require lift over baseline with margin.
        cap = pd.read_csv(os.path.join(TABLES, 'pred_J_gauge_backtest.csv')).set_index('Metric')
        px = pd.read_csv(os.path.join(TABLES, 'price_history_weekly.csv'),
                         parse_dates=['Date'])
        spy = px[px['Ticker'] == 'SPY'].sort_values('Date').reset_index(drop=True)
        feats = enh.price_features(spy['AdjClose'].to_numpy())
        scores, _, _, _ = enh.hazard_frame(feats)
        got = enh.crash_capture(spy['AdjClose'].to_numpy(), scores)
        self.assertAlmostEqual(got['capture'], cap.loc['Crash-week capture share', 'Value'],
                               places=3)
        self.assertGreaterEqual(got['capture'], 0.35)
        self.assertGreaterEqual(got['lift'], 1.5)
        self.assertEqual(int(got['crash_weeks']),
                         int(cap.loc['Crash weeks (>=20% drawdown, SPY 2000-2026)', 'Value']))
        # Degenerate input (no drawdown weeks) returns the zero branch.
        flat = enh.crash_capture(np.ones(50), pd.Series(np.zeros(50)))
        self.assertEqual(flat['capture'], 0.0)
        self.assertEqual(flat['crash_weeks'], 0)

    def test_ml_horserace_recomputes_and_orders(self):
        # pred_M: rebuild the horse-race from the vendored closes through
        # the builder's pure helpers with train-only scaling; the committed
        # table must match, and must encode the negative result: dot-com-fit
        # weights lose the GFC test while GFC-fit weights win the dot-com
        # test -- no stable winner, so no upgrade ships.
        px = pd.read_csv(os.path.join(TABLES, 'price_history_weekly.csv'),
                         parse_dates=['Date'])
        spy = px[px['Ticker'] == 'SPY'].sort_values('Date').reset_index(drop=True)
        feats = enh.price_features(spy['AdjClose'].to_numpy())
        X = feats[enh.ML_FEATS].to_numpy(dtype=float)
        base = enh.hazard_frame(feats)[0].to_numpy(dtype=float)
        peak = np.maximum.accumulate(spy['AdjClose'].to_numpy())
        dd = np.where(peak > 0, (peak - spy['AdjClose'].to_numpy()) / peak, 0.0)
        y = (dd >= 0.20).astype(int)
        dates = spy['Date'].to_numpy(dtype='datetime64[D]')
        WF = dates < np.datetime64('2007-01-01')
        MID = ((dates >= np.datetime64('2007-01-01'))
               & (dates < np.datetime64('2016-01-01')))
        LATE = dates >= np.datetime64('2016-01-01')
        tab = pd.read_csv(os.path.join(TABLES, 'pred_M_ml_horserace.csv'))
        self.assertEqual(len(tab), 7)
        # dist carries no independent signal (collinear with dd).
        c = np.corrcoef(X[WF | MID][:, 0], feats['dist'].to_numpy()[WF | MID])
        self.assertGreater(c[0, 1], 0.999)
        want = {}
        for fname, ftr in (('FULL train 2000-15', WF | MID),
                           ('WF train 2000-06', WF),
                           ('REV train 2007+', MID | LATE)):
            mu, sd = enh.standardize_fit(X[ftr])
            self.assertTrue(np.allclose(mu, X[ftr].mean(axis=0)))
            clf = enh.fit_crash_logit((X[ftr] - mu) / sd, y[ftr])
            Z = (X - mu) / sd
            p = clf.predict_proba(Z)[:, 1]
            eras = {'train': ftr}
            if fname.startswith('FULL'):
                eras['late-test'] = LATE
            elif fname.startswith('WF'):
                eras.update({'wf-test': MID, 'late-test': LATE})
            else:
                eras['rev-test'] = WF
            for ename, era in eras.items():
                want[(fname, ename)] = (enh.era_scores(y, p, era),
                                        enh.era_scores(y, base, era), clf.coef_[0])
        for _, r in tab.iterrows():
            gm, gb, coef = want[(r['Fit'], r['Context'])]
            self.assertEqual(r['N_crash'], gm['n_crash'])
            for col, got in (('Model_cap', gm['capture']),
                             ('Base_cap', gb['capture']),
                             ('Model_auc', gm['auc']),
                             ('Base_auc', gb['auc'])):
                self.assertAlmostEqual(r[col], got, places=2)
            for col, k in (('Coef_dd', 0), ('Coef_dur', 1), ('Coef_vol', 2)):
                self.assertAlmostEqual(r[col], coef[k], places=2)
            self.assertEqual(r['Winner'],
                             'model' if gm['capture'] > gb['capture']
                             else 'base' if gb['capture'] > gm['capture'] else 'tie')
        # The asymmetry that kills the upgrade.
        wftest = tab[(tab['Fit'] == 'WF train 2000-06')
                     & (tab['Context'] == 'wf-test')].iloc[0]
        revtest = tab[(tab['Fit'] == 'REV train 2007+')
                      & (tab['Context'] == 'rev-test')].iloc[0]
        self.assertLess(wftest['Model_cap'], wftest['Base_cap'])
        self.assertGreater(revtest['Model_cap'], revtest['Base_cap'])

    def test_tex_ml_horserace_matches_live(self):
        # The ML-evaluation prose pins the walk-forward scoreboard to pred_M.
        race = pd.read_csv(os.path.join(TABLES, 'pred_M_ml_horserace.csv'))
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        wf = race[(race['Fit'] == 'WF train 2000-06')
                  & (race['Context'] == 'wf-test')].iloc[0]
        rev = race[(race['Fit'] == 'REV train 2007+')
                   & (race['Context'] == 'rev-test')].iloc[0]
        self.assertIn(f"{wf['Model_cap']:.2f} vs {wf['Base_cap']:.2f}", tex)
        self.assertIn(f"{rev['Model_cap']:.2f} vs {rev['Base_cap']:.2f}", tex)
        self.assertIn('no ML upgrade ships', tex)

    def test_expected_losses_blend_and_cover(self):
        # pred_K: one row per screen company; E_loss is the severity blend
        # P_burst * Severe_mid + (1 - P_burst) * Mild_mid with P_burst from
        # pred_G formation; E sits inside [mild-mid, severe-mid].
        val = pd.read_csv(os.path.join(TABLES, 'enhanced_valuation_metrics.csv'))
        tmg = pd.read_csv(os.path.join(TABLES, 'pred_G_burst_timing.csv')).set_index('Archetype')
        losses = pd.read_csv(os.path.join(TABLES, 'pred_H_company_losses.csv')).set_index('Company')
        exp = pd.read_csv(os.path.join(TABLES, 'pred_K_expected_losses.csv'))
        self.assertEqual(len(exp), len(val))
        self.assertTrue(set(exp['Driver']) <= {'deal-flow', 'size'})
        self.assertTrue(((exp['E_haircut_pct'] >= 0) & (exp['E_haircut_pct'] <= 100)).all())
        for _, r in exp.iterrows():
            h = losses.loc[r['Company']]
            p = float(tmg.loc[r['Archetype'], 'Formation_pct']) / 100.0
            mild = (float(h['Mild_mcap_$B']) + float(h['Mild_expo_$B'])) / 2.0
            sev = float(h['Severe_mid_$B'])
            self.assertAlmostEqual(r['P_burst_pct'], p * 100, places=1)
            self.assertAlmostEqual(r['Mild_mid_$B'], mild, places=1)
            self.assertAlmostEqual(r['E_loss_$B'], p * sev + (1 - p) * mild, delta=0.05)
            self.assertLessEqual(r['Mild_mid_$B'], r['E_loss_$B'])
            self.assertLessEqual(r['E_loss_$B'], r['Severe_mid_$B'])
        # Archetype roll-up: expected totals are the p-weighted blend.
        for a in ('Hardware', 'Cloud Providers', 'Foundation Models', 'LLM Wrappers'):
            sub = exp[exp['Archetype'] == a]
            p = float(tmg.loc[a, 'Formation_pct']) / 100.0
            want = p * sub['Severe_mid_$B'].sum() + (1 - p) * sub['Mild_mid_$B'].sum()
            self.assertAlmostEqual(sub['E_loss_$B'].sum(), want, delta=0.1)
        self.assertEqual(exp.iloc[0]['Company'], 'NVIDIA')

    def test_timing_rationale_adds_up(self):
        # pred_L: five why-rows; adjustments sum base + calm credit to the
        # conditional call, whose detail carries the calendar quarter.
        rat = pd.read_csv(os.path.join(TABLES, 'pred_L_timing_rationale.csv'))
        tmg = pd.read_csv(os.path.join(TABLES, 'pred_G_burst_timing.csv'))
        sys_med = float(tmg.loc[tmg['Archetype'] == 'System (first-tip)',
                                'Median_q'].iloc[0])
        self.assertEqual(len(rat), 7)
        self.assertTrue(set(rat['Reason']) >= {
            'Hazard base — system first-tip',
            'Trigger conditioning — calm credit',
            'Gauge triangulation — acceleration without stress',
            'Precedent topping lag', 'CONDITIONAL CALL (level)',
            'Sensitivity — no calm credit (level)',
            'Sensitivity — double calm credit (level)'})
        self.assertTrue(set(rat['Weight']) <= {'High', 'Medium', 'Low'})
        adj = rat.set_index('Reason')['Adj_q']
        self.assertAlmostEqual(adj['Hazard base — system first-tip'], 0.0)
        self.assertAlmostEqual(adj['Trigger conditioning — calm credit'], 1.0)
        self.assertAlmostEqual(adj['Gauge triangulation — acceleration without stress'], 0.0)
        self.assertAlmostEqual(adj['Precedent topping lag'], 0.0)
        cond = adj['CONDITIONAL CALL (level)']
        self.assertAlmostEqual(cond, sys_med + 1.0, places=1)
        call = rat.loc[rat['Reason'] == 'CONDITIONAL CALL (level)', 'Detail'].iloc[0]
        self.assertIn(f'median {cond:.1f}q', call)
        self.assertRegex(call, r"Q[1-4]'\d{2}")
        sens = rat.set_index('Reason')['Adj_q']
        self.assertAlmostEqual(sens['Sensitivity — no calm credit (level)'],
                               sys_med, places=1)
        self.assertAlmostEqual(sens['Sensitivity — double calm credit (level)'],
                               sys_med + 2.0, places=1)

    def test_tex_quant_memo_matches_live(self):
        # The quant-memo prose pins the conditional call and the top
        # expected losses to the live pred_K/pred_L ledgers.
        exp = pd.read_csv(os.path.join(TABLES, 'pred_K_expected_losses.csv')).set_index('Company')
        rat = pd.read_csv(os.path.join(TABLES, 'pred_L_timing_rationale.csv')).set_index('Reason')
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        call = rat.loc['CONDITIONAL CALL (level)', 'Detail']
        med = re.search(r'median (\d+\.\d+)q \((Q[1-4]\'\d{2})\)', call)
        self.assertIsNotNone(med)
        self.assertIn(f'conditional call at a median of {med.group(1)} quarters', tex)
        self.assertIn(f'({med.group(2)})', tex)
        tmg2 = pd.read_csv(os.path.join(TABLES, 'pred_G_burst_timing.csv'))
        sys_med = float(tmg2.loc[tmg2['Archetype'] == 'System (first-tip)',
                                 'Median_q'].iloc[0])
        self.assertIn(f'median {sys_med + 2.0:.1f}q', tex)
        for co in ('NVIDIA', 'Anthropic', 'OpenAI'):
            self.assertIn(f"{co} expected \\${exp.loc[co, 'E_loss_$B']:.1f}B", tex)


class TestOpenWeights(unittest.TestCase):
    """Open-weight margin-compression channel: formation covariate + DebtRank shock."""
    @classmethod
    def setUpClass(cls):
        cls.data = ai_ecosystem_model.load_all_data()
        cls.ca = ai_ecosystem_model.CircularDealsAnalyzer()
        cls.bb = ai_ecosystem_model.BubbleBurstAnalyzer(cls.data['players'], cls.ca,
                                        dependencies_df=cls.data['dependencies'],
                                        output_dir=TABLES)

    def test_flag_off_matches_published_schema(self):
        df = self.bb.formation_probability()
        self.assertNotIn('Open_Weight_Margin', list(df.columns))
        self.assertEqual(list(df.columns),
                         ['Archetype', 'Circular_Exposure', 'Capex_Intensity', 'HHI_Share',
                          'Exuberance', 'Z_Score', 'Formation_Prob_Pct',
                          'Prob_Low_Pct', 'Prob_High_Pct'])

    def test_open_weights_lifts_fm_and_preserves_order(self):
        base = self.bb.formation_probability().set_index('Archetype')
        scen = self.bb.formation_probability(open_weights=True)
        self.assertIn('Open_Weight_Margin', list(scen.columns))
        scen = scen.set_index('Archetype')
        for c in base.index:
            self.assertGreaterEqual(scen.loc[c, 'Formation_Prob_Pct'],
                                    base.loc[c, 'Formation_Prob_Pct'])
            self.assertTrue(0 <= scen.loc[c, 'Formation_Prob_Pct'] <= 100)
            self.assertTrue(scen.loc[c, 'Prob_Low_Pct']
                            <= scen.loc[c, 'Formation_Prob_Pct']
                            <= scen.loc[c, 'Prob_High_Pct'])
            self.assertAlmostEqual(scen.loc[c, 'Open_Weight_Margin'],
                                   self.bb.OPEN_WEIGHT_MARGIN[c])
        self.assertGreater(scen.loc['Foundation Models', 'Formation_Prob_Pct'],
                           base.loc['Foundation Models', 'Formation_Prob_Pct'])
        self.assertEqual(list(scen.sort_values('Formation_Prob_Pct', ascending=False).index),
                         list(base.sort_values('Formation_Prob_Pct', ascending=False).index))

    def test_readme_open_weights_delta_matches_live(self):
        # Sep 2026 audit: README cited FM 84.4% -> 90.2% while the live
        # committed vintage gives 82.9% -> 89.3%. Pin the README to live.
        # NOTE: revenue-share hhi_shares required (uniform fallback moves
        # every probability).
        data = ai_ecosystem_model.load_all_data()
        _rev = data['players'].set_index('Player_Category')['Current_Revenue_Billions']
        _shares = (_rev / _rev.sum()).to_dict()
        bb = ai_ecosystem_model.BubbleBurstAnalyzer(
            data['players'], ai_ecosystem_model.CircularDealsAnalyzer(),
            dependencies_df=data.get('dependencies', pd.DataFrame()),
            hhi_shares=_shares, output_dir='/tmp')
        base = bb.formation_probability().set_index('Archetype')
        scen = bb.formation_probability(open_weights=True).set_index('Archetype')
        b = float(base.loc['Foundation Models', 'Formation_Prob_Pct'])
        o = float(scen.loc['Foundation Models', 'Formation_Prob_Pct'])
        with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
            readme = f.read()
        self.assertIn(f'(FM {b:.1f}% \u2192 {o:.1f}%', readme)

    def test_open_weights_sigmoid_identity(self):
        bb = self.bb
        df = bb.formation_probability(open_weights=True)
        L = bb.LOGIT
        row = df[df['Archetype'] == 'Hardware'].iloc[0]
        z = (L['intercept'] + L['w_circ'] * row['Circular_Exposure']
             + L['w_capex'] * row['Capex_Intensity'] + L['w_hhi'] * row['HHI_Share']
             + L['w_exub'] * row['Exuberance']
             + L['w_open'] * row['Open_Weight_Margin'])
        self.assertAlmostEqual(row['Formation_Prob_Pct'], 1 / (1 + np.exp(-z)) * 100, delta=0.1)

    def test_burst_impact_threads_flag(self):
        plain = self.bb.burst_impact().set_index('Archetype')
        ow = self.bb.burst_impact(open_weights=True).set_index('Archetype')
        form_ow = self.bb.formation_probability(open_weights=True).set_index('Archetype')
        self.assertAlmostEqual(ow.loc['Foundation Models', 'Formation_Prob_Pct'],
                               form_ow.loc['Foundation Models', 'Formation_Prob_Pct'], places=1)
        self.assertGreaterEqual(ow.loc['Foundation Models', 'Formation_Prob_Pct'],
                                plain.loc['Foundation Models', 'Formation_Prob_Pct'])
        self.assertEqual(sorted(ow['Burst_Rank']), [1, 2, 3, 4])

    def test_open_weights_shock_is_fm_led(self):
        shock = self.bb.open_weights_shock()
        self.assertEqual(set(shock), set(self.bb.cats))
        self.assertTrue(all(v >= 0 for v in shock.values()))
        self.assertEqual(max(shock, key=shock.get), 'Foundation Models')
        self.assertGreater(shock['Foundation Models'], 0)

    def test_open_weights_debtrank_runs(self):
        out = self.bb.open_weights_debtrank()
        self.assertEqual(len(out), 4)
        self.assertTrue(((out['Final_Distress'] >= 0) & (out['Final_Distress'] <= 1)).all())
        self.assertTrue((out['Final_Distress'] >= out['Initial_Distress'] - 1e-9).all())
        self.assertGreaterEqual(float(out['Second_Round_Loss_$B'].sum()), 0)

    def test_open_weights_debtrank_needs_frame(self):
        with tempfile.TemporaryDirectory() as td:
            bb = ai_ecosystem_model.BubbleBurstAnalyzer(self.data['players'], self.ca,
                                        dependencies_df=pd.DataFrame(), output_dir=td)
            self.assertTrue(bb.dependencies_df.empty)
            with self.assertRaises(ValueError):
                bb.open_weights_debtrank()

    def test_register_covers_open_weights_channel(self):
        df = ai_ecosystem_model.build_validated_inputs_register()
        ids = set(df['ID'])
        for rid in ('X-CHN', 'S-LOGIT-w_open', 'S-OWM-HA', 'S-OWM-CL',
                    'S-OWM-FO', 'S-OWM-LL'):
            self.assertIn(rid, ids)


class TestDebtRankClearing(unittest.TestCase):
    """DebtRankClearingEngine: second-round contagion invariants."""
    @staticmethod
    def _toy():
        deps = pd.DataFrame([
            {'Dependent_Category': 'Hardware', 'Dependency_Category': 'Cloud Providers',
             'Dependency_Value_Billions': 100.0},
            {'Dependent_Category': 'Cloud Providers', 'Dependency_Category': 'Hardware',
             'Dependency_Value_Billions': 50.0},
        ])
        eq = {'Hardware': 1000.0, 'Cloud Providers': 2000.0,
              'Foundation Models': 500.0, 'LLM Wrappers': 300.0}
        losses = {'Hardware': 10.0, 'Cloud Providers': 20.0,
                  'Foundation Models': 0.0, 'LLM Wrappers': 0.0}
        rev = {'Hardware': 400.0, 'Cloud Providers': 400.0,
               'Foundation Models': 100.0, 'LLM Wrappers': 100.0}
        return ai_ecosystem_model.DebtRankClearingEngine(deps, eq, losses, rev, output_dir='/tmp')

    def test_zero_shock_zero_contagion(self):
        eng = self._toy()
        out = eng.run(shock={c: 0.0 for c in eng.cats})
        self.assertTrue((out['Second_Round_Loss_$B'] == 0).all())
        self.assertTrue((out['DebtRank'] == 0).all())

    def test_distress_bounded_and_monotone(self):
        eng = self._toy()
        out = eng.run()
        self.assertTrue(((out['Final_Distress'] >= 0) & (out['Final_Distress'] <= 1)).all())
        self.assertTrue((out['Final_Distress'] >= out['Initial_Distress'] - 1e-9).all())
        s = eng.summary(out)
        self.assertGreaterEqual(s['multiplier'], 1.0)

    def test_bigger_shock_bigger_second_round(self):
        eng = self._toy()
        small = eng.summary(eng.run(shock={'Hardware': 5.0, 'Cloud Providers': 5.0,
                                           'Foundation Models': 0.0, 'LLM Wrappers': 0.0}))
        big = eng.summary(eng.run(shock={'Hardware': 50.0, 'Cloud Providers': 50.0,
                                         'Foundation Models': 0.0, 'LLM Wrappers': 0.0}))
        self.assertGreaterEqual(big['second_total'], small['second_total'])

    def test_empty_exposure_raises(self):
        deps = pd.DataFrame([{'Dependent_Category': 'Hardware',
                              'Dependency_Category': 'Hardware',
                              'Dependency_Value_Billions': 10.0}])
        with self.assertRaises(ValueError):
            ai_ecosystem_model.DebtRankClearingEngine(
                deps, {'Hardware': 100.0, 'Cloud Providers': 100.0},
                {'Hardware': 1.0, 'Cloud Providers': 1.0},
                {'Hardware': 10.0, 'Cloud Providers': 10.0}, output_dir='/tmp')

    def test_save_all_writes_table(self):
        eng = self._toy()
        with tempfile.TemporaryDirectory() as td:
            eng.output_dir = td
            res = eng.save_all()
            self.assertIn('table_7.14_debtrank_contagion', res)
            self.assertTrue(os.path.exists(os.path.join(td, 'table_7.14_debtrank_contagion.csv')))

    def test_non_committed_rows_excluded_from_exposure(self):
        base = pd.DataFrame([
            {'Dependent_Category': 'Hardware', 'Dependency_Category': 'Cloud Providers',
             'Dependency_Value_Billions': 100.0, 'Booking_Status': 'committed'},
            {'Dependent_Category': 'Hardware', 'Dependency_Category': 'Foundation Models',
             'Dependency_Value_Billions': 900.0, 'Booking_Status': 'prospective'},
            {'Dependent_Category': 'Cloud Providers', 'Dependency_Category': 'Hardware',
             'Dependency_Value_Billions': 700.0, 'Booking_Status': 'vertical-integration'},
        ])
        eq = {'Hardware': 1000.0, 'Cloud Providers': 2000.0,
              'Foundation Models': 500.0, 'LLM Wrappers': 300.0}
        losses = {'Hardware': 10.0, 'Cloud Providers': 20.0,
                  'Foundation Models': 0.0, 'LLM Wrappers': 0.0}
        rev = {'Hardware': 400.0, 'Cloud Providers': 400.0,
               'Foundation Models': 100.0, 'LLM Wrappers': 100.0}
        eng = ai_ecosystem_model.DebtRankClearingEngine(base, eq, losses, rev, output_dir='/tmp')
        self.assertAlmostEqual(eng.exposure.loc['Hardware', 'Cloud Providers'], 100.0)
        self.assertAlmostEqual(eng.exposure.loc['Hardware', 'Foundation Models'], 0.0)
        self.assertAlmostEqual(eng.exposure.loc['Cloud Providers', 'Hardware'], 0.0)

    def test_live_frame_booking_taxonomy(self):
        deps = ai_ecosystem_model.EmbeddedDataSource.get_industry_dependencies()
        counts = deps['Booking_Status'].value_counts().to_dict()
        self.assertEqual(counts.get('committed'), 17)
        self.assertEqual(counts.get('prospective'), 1)
        self.assertEqual(counts.get('vertical-integration'), 1)
        committed = deps.loc[deps['Booking_Status'] == 'committed',
                             'Dependency_Value_Billions'].sum()
        self.assertAlmostEqual(committed, 675.40, places=2)
        self.assertAlmostEqual(deps['Dependency_Value_Billions'].sum(), 1188.33, places=2)
        self.assertAlmostEqual(deps.attrs['committed_circular_exposure_billions'], 675.40, places=2)
        meta = ai_ecosystem_model.EmbeddedDataSource.get_market_metadata()
        self.assertAlmostEqual(meta['circular_exposure_billions'], 675.40, places=2)
        fac = deps.loc[deps['Dependency_Value_Billions'] == 500.0].iloc[0]
        self.assertEqual(fac['Booking_Status'], 'prospective')
        hf = deps.loc[deps['Dependent_Player'] == 'Hugging Face'].iloc[0]
        self.assertEqual(hf['Booking_Status'], 'vertical-integration')


class TestCreditMonitor(unittest.TestCase):
    """CreditMonitor: sourced facts, hazard math, watchlist order."""
    @classmethod
    def setUpClass(cls):
        rank = pd.DataFrame([
            {'Archetype': 'Hardware', 'Burst_Rank': 1, 'Total_Loss_Severe_$B': 133.0,
             'Equity_Impact_Severe_Pct': 1.7},
            {'Archetype': 'Cloud Providers', 'Burst_Rank': 2, 'Total_Loss_Severe_$B': 132.7,
             'Equity_Impact_Severe_Pct': 1.1},
        ])
        cls.cm = ai_ecosystem_model.CreditMonitor(rank, output_dir='/tmp')

    def test_hazard_math(self):
        self.assertAlmostEqual(self.cm.hazard_approx(198.23), 3.3038, places=3)
        self.assertAlmostEqual(self.cm.hazard_approx(0.0), 0.0)
        with self.assertRaises(ValueError):
            self.cm.hazard_approx(100.0, recovery=1.0)

    def test_pipeline_ratio(self):
        self.assertAlmostEqual(self.cm.pipeline_to_capex(), 570.0 / 400.0)

    def test_watchlist_follows_live_rank(self):
        wl = self.cm.watchlist()
        self.assertEqual(wl['Watch_Rank'].tolist(), [1, 2])
        self.assertEqual(wl['Archetype'].tolist(), ['Hardware', 'Cloud Providers'])

    def test_haircut_schedule_ordered(self):
        for _, (base, stressed) in self.cm.HAIRCUTS.items():
            self.assertLessEqual(base, stressed)

    def test_save_all_writes_table(self):
        with tempfile.TemporaryDirectory() as td:
            self.cm.output_dir = td
            res = self.cm.save_all()
            self.assertIn('table_7.15_credit_monitor', res)
            df = pd.read_csv(os.path.join(td, 'table_7.15_credit_monitor.csv'))
            self.assertGreater(len(df), 6)


class TestStructuralEstimation(unittest.TestCase):
    """StructuralEstimationAnalyzer: criteria, bisection, intervals."""
    @classmethod
    def setUpClass(cls):
        cls.data, cls.gf = solve_framework()
        cls.bb = ai_ecosystem_model.BubbleBurstAnalyzer(
            cls.data['players'], ai_ecosystem_model.CircularDealsAnalyzer(),
            dependencies_df=cls.data.get('dependencies', pd.DataFrame()),
            hhi_shares={}, output_dir='/tmp')
        cls.se = ai_ecosystem_model.StructuralEstimationAnalyzer(cls.gf, cls.bb, output_dir='/tmp')

    def test_base_criteria_hold(self):
        self.assertEqual(self.se.nash_positions(rate_const=None), self.se.base_nash)
        self.assertEqual(self.se.formation_order(self.se.a8_base), self.se.base_order)

    def test_formation_order_matches_pipeline_rank(self):
        self.assertEqual(self.se.base_order[0], 'Foundation Models')

    def test_covariates_mirror_pipeline_lookups(self):
        cov = self.se._covariates()
        for a in self.bb.cats:
            self.assertAlmostEqual(
                cov[a][1], float(self.bb.CAPEX_INTENSITY.get(a, 0.5)))
            self.assertAlmostEqual(
                cov[a][0], float(self.bb.exposure.get(a, 0.0)))
        # The capex covariate must vary across archetypes: a constant
        # covariate would leave the formation order invariant in w_capex and
        # report a spuriously open interval.
        self.assertGreater(len({v[1] for v in cov.values()}), 1)

    def test_edge_bisect_toy(self):
        edge, opened = ai_ecosystem_model.StructuralEstimationAnalyzer._edge(
            1.0, 0.0, lambda v: v >= 0.4, -1, iters=10)
        self.assertFalse(opened)
        self.assertAlmostEqual(edge, 0.4, places=2)
        _, opened = ai_ecosystem_model.StructuralEstimationAnalyzer._edge(
            1.0, 2.0, lambda v: True, +1, iters=4)
        self.assertTrue(opened)

    def test_calibrate_structure(self):
        out = self.se.calibrate(iters=3)
        self.assertEqual(len(out), 9)
        self.assertTrue(((out['Ident_Low'] <= out['Base']) &
                         (out['Base'] <= out['Ident_High'])).all())

    def test_save_all_writes_table(self):
        with tempfile.TemporaryDirectory() as td:
            self.se.output_dir = td
            with mock.patch.object(ai_ecosystem_model.StructuralEstimationAnalyzer, 'calibrate',
                                   return_value=pd.DataFrame([
                                       {'Block': 'A3 rate maps', 'Parameter': 't_intercept',
                                        'Base': 0.1, 'Ident_Low': 0.0, 'Ident_High': 0.3,
                                        'Preserves': 'x', 'Status': 'open-low+high'}
                                       for _ in range(9)])):
                res = self.se.save_all()
            self.assertIn('table_8.7_structural_estimation', res)

    def test_joint_order_check_uses_extreme_levers(self):
        # The OAT battery must test severe misspecification: quartering and
        # quadrupling each lever (16 runs per family), not just halves/doubles.
        self.assertEqual(
            list(ai_ecosystem_model.StructuralEstimationAnalyzer.OAT_MULTS),
            [0.25, 0.5, 2.0, 4.0])
        jc = self.se.joint_order_check(n_draws=5, seed=7)
        self.assertEqual(jc['oat_formation_runs'], 16.0)
        self.assertEqual(jc['oat_burst_runs'], 16.0)
        self.assertLessEqual(jc['oat_formation_matches'], 16.0)
        self.assertLessEqual(jc['oat_burst_matches'], 16.0)
        self.assertGreaterEqual(jc['oat_formation_matches'], 0.0)
        self.assertGreaterEqual(jc['oat_burst_matches'], 0.0)

    def test_write_tex_macros_uses_live_status_edges(self):
        # The article fragment must render Status edges exactly: open highs
        # and lows get '$+$', interior bounds stay bare, intercept is skipped.
        df = pd.DataFrame([
            {'Block': 'A8 logit weights', 'Parameter': 'intercept',
             'Base': -1.2, 'Ident_Low': -3.0, 'Ident_High': 0.5,
             'Preserves': 'x', 'Status': 'open-low+high'},
            {'Block': 'A8 logit weights', 'Parameter': 'w_circ',
             'Base': 1.6, 'Ident_Low': 0.7362, 'Ident_High': 3.0,
             'Preserves': 'x', 'Status': 'open-high'},
            {'Block': 'A8 logit weights', 'Parameter': 'w_capex',
             'Base': 0.9, 'Ident_Low': 0.3454, 'Ident_High': 2.6275,
             'Preserves': 'x', 'Status': 'interior'},
            {'Block': 'A8 logit weights', 'Parameter': 'w_hhi',
             'Base': 0.7, 'Ident_Low': 0.0977, 'Ident_High': 1.6699,
             'Preserves': 'x', 'Status': 'interior'},
            {'Block': 'A8 logit weights', 'Parameter': 'w_exub',
             'Base': 0.8, 'Ident_Low': 0.0, 'Ident_High': 1.0447,
             'Preserves': 'x', 'Status': 'open-low'},
        ])
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'article', 'generated_a8_intervals.tex')
            self.assertEqual(
                self.se.write_tex_macros(out, df=df), out)
            with open(out, encoding='utf-8') as f:
                gen = f.read()
        self.assertIn('\\newcommand{\\AeightCapex}{[0.35, 2.63]}', gen)
        self.assertIn('\\newcommand{\\AeightHhi}{[0.10, 1.67]}', gen)
        self.assertIn('\\newcommand{\\AeightCirc}{[0.74, 3.00$+$]}', gen)
        self.assertIn('\\newcommand{\\AeightExub}{[0.00$+$, 1.04]}', gen)
        self.assertNotIn('intercept', gen)


class TestA8ProseMatchesLiveBattery(unittest.TestCase):
    """A8 prose (tex, report builder, dashboard builder) must restate the live
    seeded battery numbers, never hardcoded stale ones (Sep 2026: prose cited
    90%/56% joint exact-match rates and pre-re-spec frontier intervals after
    the battery was re-specified to real OAT runs + a 300-draw joint check)."""
    @classmethod
    def setUpClass(cls):
        cls.data, cls.gf = solve_framework()
        # Revenue-share hhi_shares mirror assumption_register (and equal the
        # pipeline's market shares); hhi_shares={} would silently substitute a
        # uniform 0.25 fallback and move the joint percentages.
        _rev = cls.data['players'].set_index('Player_Category')['Current_Revenue_Billions']
        _shares = (_rev / _rev.sum()).to_dict()
        cls.bb = ai_ecosystem_model.BubbleBurstAnalyzer(
            cls.data['players'], ai_ecosystem_model.CircularDealsAnalyzer(),
            dependencies_df=cls.data.get('dependencies', pd.DataFrame()),
            hhi_shares=_shares, output_dir='/tmp')
        cls.se = ai_ecosystem_model.StructuralEstimationAnalyzer(cls.gf, cls.bb, output_dir='/tmp')
        cls.jc = cls.se.joint_order_check()
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'), encoding='utf-8') as f:
            cls.tex = f.read()
        with open(os.path.join(HERE, 'build_technical_report.py'), encoding='utf-8') as f:
            cls.rpt = f.read()
        with open(os.path.join(HERE, 'build_interactive_dashboard.py'), encoding='utf-8') as f:
            cls.dash = f.read()

    def test_tex_matches_live_joint_check(self):
        jc = self.jc
        self.assertIn(f"exact-match {jc['formation_exact_pct']:.0f}\\%", self.tex)
        self.assertIn(f"exact match {jc['burst_exact_pct']:.0f}\\%", self.tex)
        self.assertIn(f"\\tau={jc['formation_tau']:.2f}", self.tex)
        self.assertIn(f"\\tau={jc['burst_tau']:.2f}", self.tex)
        self.assertIn(f"{jc['oat_formation_matches']:.0f}/{jc['oat_formation_runs']:.0f} formation", self.tex)
        self.assertIn(f"{jc['oat_burst_matches']:.0f}/{jc['oat_burst_runs']:.0f} burst", self.tex)

    def test_tex_matches_live_frontier_intervals(self):
        # The manuscript must feed Table 8.7 intervals from code output, never
        # restate them literally: the ai_ecosystem_model stage writes
        # article/generated_a8_intervals.tex, the manuscript \input's it, and
        # every A8 interval renders through a macro. This gate verifies the
        # wiring end to end against the live CSV (the oracle).
        p = os.path.join(TABLES, 'table_8.7_structural_estimation.csv')
        if not os.path.exists(p):
            self.skipTest('Table 8.7 not built')
        tab = pd.read_csv(p)
        a8 = tab[tab['Block'] == 'A8 logit weights']
        self.assertEqual(len(a8), 5)
        gen_path = os.path.join(HERE, 'article', 'generated_a8_intervals.tex')
        self.assertTrue(os.path.isfile(gen_path),
                        'article/generated_a8_intervals.tex missing -- '
                        'run the ai_ecosystem_model stage')
        with open(gen_path, encoding='utf-8') as f:
            gen = f.read()
        self.assertIn('\\input{generated_a8_intervals}', self.tex)
        suffix = {'w_capex': 'Capex', 'w_hhi': 'Hhi',
                  'w_circ': 'Circ', 'w_exub': 'Exub'}
        seen = set()
        for _, row in a8.iterrows():
            param = str(row['Parameter'])
            if param == 'intercept':
                continue
            status = str(row['Status'])
            opened = status.split('open-')[-1] if status.startswith('open-') else ''
            lo = f"{row['Ident_Low']:.2f}" + ('$+$' if 'low' in opened else '')
            hi = f"{row['Ident_High']:.2f}" + ('$+$' if 'high' in opened else '')
            macro = f"\\Aeight{suffix[param]}"
            self.assertIn(macro, self.tex, f'main tex must use {macro}')
            self.assertIn(f"\\newcommand{{{macro}}}{{[{lo}, {hi}]}}",
                          gen, param)
            # The interval must not also be restated literally in the
            # manuscript -- the macro is the single source in prose.
            self.assertNotIn(f"[{lo}, {hi}]", self.tex, param)
            seen.add(param)
        self.assertEqual(seen, set(suffix))

    def test_battery_uses_committed_valuation_vintage(self):
        # Sep 2026: the battery silently ran on fallback exub=0.5 (no vintage
        # file at output_dir) while Tables 7.10/8.7 used the real vintage.
        # The analyzer must resolve tables/ when output_dir has no vintage.
        vpath = os.path.join(TABLES, 'enhanced_valuation_metrics.csv')
        if not os.path.exists(vpath):
            self.skipTest('valuation vintage not built')
        vdf = pd.read_csv(vpath)
        self.assertGreater(len(set(self.bb.exub.values())), 1,
                           'uniform exub smells like the silent 0.5 fallback')
        for c in self.bb.cats:
            sub = vdf[vdf['category'].astype(str).str.lower() == c.lower()]
            ev = pd.to_numeric(sub['ev_revenue'], errors='coerce').median()
            expected = float(np.clip((ev / ai_ecosystem_model.BubbleBurstAnalyzer.EV_NORM - 1.0) / 2.0,
                                     0.0, 1.0)) if pd.notna(ev) else 0.5
            self.assertAlmostEqual(self.bb.exub[c], expected, places=6, msg=c)

    def test_no_stale_a8_tokens(self):
        stale = ('exact-match 90\\%', 'exact match 56\\%', 'tau=0.98', 'tau=0.93',
                 '15/16', '15 of 16', '16/16 formation', '[0.34, 2.65]',
                 '[0.10, 1.64]', 'near-tie', '56%/44%', '$133.0B',
                 'exact-match 76\\%', 'tau=0.92', '6/8 formation',
                 '[0.00$+$, 1.95]', '[0.00$+$, 1.29]', '[1.07, 3.00$+$]',
                 'formation-order 90', 'burst-order 56',
                 '1.42 elasticity', "sweep's 1.42")
        for blob, name in ((self.tex, 'tex'), (self.rpt, 'report builder'),
                           (self.dash, 'dashboard builder')):
            for tok in stale:
                self.assertNotIn(tok, blob, f'{name}: {tok}')

    def test_no_stale_gdp_debtrank_bcr_tokens(self):
        stale = ('$134.5B', '0.437%', '1.029', '362.7', 'BCR 43.5',
                 'exact in 90% of 300', 'exact in 56% of 300')
        for blob, name in ((self.rpt, 'report builder'), (self.dash, 'dashboard builder')):
            for tok in stale:
                self.assertNotIn(tok, blob, f'{name}: {tok}')
        for live in ('$139.1B', '0.452%', '1.032', '478.3', 'BCR 57.4',
                     '65% of 300 joint weight draws', 'tau 0.88', 'tau 1.00'):
            self.assertIn(live, self.dash, live)

    def test_report_section_75_uses_live_joint_check(self):
        self.assertIn('joint_order', self.rpt)


class TestGeographyOverlay(unittest.TestCase):
    """BubbleBurstAnalyzer.geography_overlay: ported GDP-at-risk (Table 7.19)."""
    @classmethod
    def setUpClass(cls):
        cls.data, cls.gf = solve_framework()
        _rev = cls.data['players'].set_index('Player_Category')['Current_Revenue_Billions']
        cls.bb = ai_ecosystem_model.BubbleBurstAnalyzer(
            cls.data['players'], ai_ecosystem_model.CircularDealsAnalyzer(),
            dependencies_df=cls.data.get('dependencies', pd.DataFrame()),
            hhi_shares=(_rev / _rev.sum()).to_dict(), output_dir='/tmp')
        rk = pd.read_csv(os.path.join(TABLES, 'table_7.11_burst_ranking.csv'))
        cls.severe_total = float(rk['Total_Loss_Severe_$B'].sum())
        cls.geo = cls.bb.geography_overlay(cls.severe_total)

    def test_us_row_reproduces_table_712(self):
        p = os.path.join(TABLES, 'table_7.12_gdp_impact.csv')
        if not os.path.exists(p):
            self.skipTest('Table 7.12 not built')
        ref = pd.read_csv(p)
        ref = ref[(ref['Severity'] == 'severe') & (ref['Channel'] == 'Combined')].iloc[0]
        us = self.geo[self.geo['Economy'] == 'United States'].iloc[0]
        self.assertAlmostEqual(us['Combined_Loss_$B'], ref['GDP_Loss_$B'], places=1)
        self.assertAlmostEqual(us['GDP_Share_Pct'], ref['GDP_Share_Pct'], places=3)
        self.assertEqual(us['Illustrative'], 'N')

    def test_channel_identity(self):
        for _, r in self.geo.iterrows():
            self.assertAlmostEqual(
                r['Combined_Loss_$B'],
                r['Capex_Channel_Loss_$B'] + r['Wealth_Channel_Loss_$B'],
                delta=0.15, msg=r['Economy'])
            self.assertGreaterEqual(r['Combined_Loss_$B'], 0)
            self.assertGreater(r['GDP_$B'], 0)

    def test_custom_economy_override(self):
        out = self.bb.geography_overlay(1000.0, economies=[
            {'economy': 'Testland', 'gdp_b': 10000.0, 'capex_b': 200.0,
             'impaired_b': 500.0, 'mpc': 0.05}])
        self.assertEqual(len(out), 2)
        t = out[out['Economy'] == 'Testland'].iloc[0]
        self.assertAlmostEqual(t['Capex_Channel_Loss_$B'], 200.0 * 0.3)
        self.assertAlmostEqual(t['Wealth_Channel_Loss_$B'], 500.0 * 0.05)
        self.assertAlmostEqual(t['Combined_Loss_$B'], 60.0 + 25.0)
        self.assertAlmostEqual(t['GDP_Share_Pct'], 85.0 / 10000.0 * 100)
        self.assertEqual(t['Illustrative'], 'Y')

    def test_illustrative_flags(self):
        flags = dict(zip(self.geo['Economy'], self.geo['Illustrative']))
        self.assertEqual(flags['United States'], 'N')
        for eco in ('European Union', 'China', 'Japan', 'South Korea'):
            self.assertEqual(flags[eco], 'Y', eco)

    def test_artifact_triple_exists(self):
        stem = 'table_7.19_geography_overlay'
        p = os.path.join(TABLES, stem + '.csv')
        if not os.path.exists(p):
            self.skipTest('Table 7.19 not built')
        for ext in ('.tex', '.xlsx'):
            self.assertTrue(os.path.exists(os.path.join(TABLES, stem + ext)), ext)

    def test_tex_matches_live_overlay(self):
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        for _, r in self.geo.iterrows():
            self.assertIn(f"{r['GDP_Share_Pct']:.3f}", tex, r['Economy'])
        self.assertNotIn('0.262', tex)


class TestManuscriptTitle(unittest.TestCase):
    """The canonical article title is pinned in tex and README so a stale
    title fails the build instead of shipping (Sep 2026 retitle)."""
    TITLE_HEAD = 'Circular Capital in the Artificial Intelligence Stack'
    TITLE_SUB = ('A Game-Theoretic Analysis of Market Power, Financial '
                 'Fragility, and Welfare')

    def test_title_pinned(self):
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
            readme = f.read()
        for blob, name in ((tex, 'tex'), (readme, 'README')):
            self.assertIn(self.TITLE_HEAD, blob, name)
            self.assertIn(self.TITLE_SUB, blob, name)
        self.assertNotIn('A Game-Theoretic Model of\nRound-Trip Deals', tex)
        self.assertNotIn('A Game-Theoretic Model of Round-Trip Deals', tex)

    def test_reference_keys_resolve(self):
        # Every in-text \cite key must have a \bibitem and vice versa; the
        # 2026 reference audit also pins the two entries that were ever
        # incomplete (Agrawal subtitle, Farrell chapter pages).
        import re
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        body, bib = tex.split('\\begin{thebibliography}', 1)
        cited = set(k.strip() for grp in re.findall(r'\\cite[tp]?\{([^}]+)\}', body)
                    for k in grp.split(','))
        defined = set(re.findall(r'\\bibitem\[[^\]]*\]\{([^}]+)\}', bib))
        self.assertFalse(cited - defined, f'cited but undefined: {cited - defined}')
        self.assertFalse(defined - cited, f'defined but never cited: {defined - cited}')
        self.assertIn('Prediction machines: The simple economics', tex)
        self.assertIn('pp.~1967--2072', tex)

    def test_tex_float_placement(self):
        # Sep 2026: all 11 figures drifted to the document end under
        # restrictive [ht] floats. Pin the fix: section barriers, [htbp]
        # on every figure, and a height guard on every includegraphics.
        import re
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        self.assertIn('\\usepackage[section]{placeins}', tex)
        specs = re.findall(r'\\begin\{figure\}\[([^\]]*)\]', tex)
        self.assertGreater(len(specs), 0)
        for spec in specs:
            self.assertEqual(spec, 'htbp', spec)
        widths = re.findall(r'\\includegraphics\[(width=[^\]]*)\]', tex)
        self.assertGreater(len(widths), 0)
        for w in widths:
            self.assertIn('keepaspectratio', w, w)

    def test_prose_class_count_matches_live(self):
        # Sep 2026 audit: prose said 36 classes while the model defines 35
        # top-level ones. Pin the count in tex and README to the live AST.
        import ast
        with open(os.path.join(HERE, 'ai_ecosystem_model.py'),
                  encoding='utf-8') as f:
            mod = ast.parse(f.read())
        n = sum(isinstance(node, ast.ClassDef) for node in mod.body)
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
            readme = f.read()
        for blob, name in ((tex, 'tex'), (readme, 'README')):
            self.assertIn(f'{n} classes', blob, name)

    def test_no_british_forms(self):
        # The paper uses American spelling throughout; British variants
        # (totalling, modelled, centred, flavoured, Data Centre) recurred
        # until this gate (Sep 2026 word-by-word audit).
        with open(os.path.join(HERE, 'article', 'ai_circularity_article.tex'),
                  encoding='utf-8') as f:
            tex = f.read()
        for tok in ('totalling', 'modelled', 'modelling', 'centred',
                    'flavoured', 'Data Centre', 'unmodelled'):
            self.assertNotIn(tok, tex, tok)
        for blob, name in ((tex, 'tex'),
                           (open(os.path.join(HERE, 'build_technical_report.py'),
                                 encoding='utf-8').read(), 'builder')):
            self.assertNotIn('Lerner markups', blob.replace('Lerner-implied markups', ''), name)


class TestLiteratureDiagnostics(unittest.TestCase):
    """TestLiteratureDiagnostics."""
    @classmethod
    def setUpClass(cls):
        cls.data, cls.gf = solve_framework()
        mc = ai_ecosystem_model.MonteCarloSimulator(cls.gf, n_simulations=300, seed=11)
        cls.mc_results = mc.run_monte_carlo_analysis()
        cls.ld = ai_ecosystem_model.LiteratureDiagnostics(
            players_df=cls.data['players'], dependencies_df=cls.data['dependencies'],
            game_framework=cls.gf, mc_results=cls.mc_results,
            aggregate_welfare={}, policy_df=pd.DataFrame(), circular_metrics={},
            output_dir='/tmp')

    def test_contestability_types_populated(self):
        df = self.ld.contestability_screen()
        self.assertGreater(len(df), 0)
        vals = df['Game_Type'].astype(str).str.lower()
        self.assertFalse((vals == 'n/a').all(), 'Game_Type column regressed to all-n/a')

    def test_fragility_bounded(self):
        df = self.ld.fragility_composite()
        comp = pd.to_numeric(df['Composite'], errors='coerce').dropna()
        if len(comp):
            self.assertTrue(((comp >= 0) & (comp <= 1)).all())

    def test_five_pillar_excludes_constant_p5(self):
        # P5 is ecosystem-wide (identical across rows): it must be reported as
        # context but excluded from the cross-sectional composite.
        eva = ai_ecosystem_model.EnhancedValuationAnalyzer(
            self.data['players'], circular_analyzer=None,
            plots_dir='/tmp', tables_dir='/tmp')
        eva.export_metrics('enhanced_valuation_metrics.csv')
        df = self.ld.five_pillar_screen()
        self.assertGreater(len(df), 0)
        self.assertEqual(df['P5_Capex_Intensity'].nunique(), 1)
        pcols = [c for c in ('P1_Narrative_Tilt', 'P2_Multiple_Exuberance',
                             'P3_Multiple_Expansion', 'P4_Circular_Hype_Proxy')
                 if c in df.columns and df[c].notna().any()]
        self.assertTrue((df['Informative_Pillars'] == len(pcols)).all())
        for _, r in df.iterrows():
            # places=2: CSV pillars are rounded to 3dp while the composite uses
            # full precision, so exact equality cannot hold.
            vals = [float(r[c]) for c in pcols if pd.notna(r[c])]
            self.assertAlmostEqual(float(r['Composite']), sum(vals) / len(vals), places=2)

    def test_equilibrium_audit_matches(self):
        df = self.ld.equilibrium_audit()
        self.assertGreater(len(df), 0)
        if 'Position_Match' in df.columns:
            self.assertTrue(df['Position_Match'].astype(bool).all())


class TestRptHelpers(unittest.TestCase):
    """TestRptHelpers."""
    def test_fmt(self):
        self.assertEqual(build_technical_report._fmt(None), '')
        self.assertEqual(build_technical_report._fmt(float('nan')), '')
        self.assertEqual(build_technical_report._fmt(5.0), '5')
        self.assertEqual(build_technical_report._fmt(5.256), '5.26')

    def test_fmt_infinite(self):
        # table_3.7's 'Nash reversion is not a credible punishment' rows
        # carry a genuine +inf Critical_Delta. _fmt() must render it, not
        # crash trying to round() an infinity.
        self.assertEqual(build_technical_report._fmt(float('inf')), '\u221e')
        self.assertEqual(build_technical_report._fmt(float('-inf')), '-\u221e')
        self.assertEqual(build_technical_report._fmt(np.inf), '\u221e')

    def test_load_audit_table_reads_from_tables_subdir(self):
        # load_audit_table() must look inside TABLES_DIR ('tables/'), not
        # gt_dir itself -- Tables 3.5-3.7 live at
        # tables/table_3.5_welfare_benchmark_audit.csv, never at the
        # project directory root.
        df = build_technical_report.load_audit_table(HERE, 'table_3.5_welfare_benchmark_audit.csv')
        self.assertIsNotNone(df, 'load_audit_table failed to find the file under tables/')
        self.assertFalse(df.empty)
        self.assertIn('classification', df.columns)

    def test_pct(self):
        self.assertAlmostEqual(build_technical_report._pct(0.5), 50.0)
        self.assertEqual(build_technical_report._pct('junk', default=-1.0), -1.0)

    def test_table_sort_key(self):
        ids = ['A.2', '1.10', '2.4', '1.3', '8_1']
        self.assertEqual(sorted(ids, key=build_technical_report._table_sort_key)[0], '1.3')

    def test_figure_number(self):
        self.assertEqual(build_technical_report.figure_number('figure_3.png'), '3')
        self.assertEqual(build_technical_report.figure_number('figure_17.png'), '17')
        self.assertEqual(build_technical_report.figure_number('figure_18.png'), '18')
        self.assertIsNone(build_technical_report.figure_number('nope.png'))

    def test_caption_for(self):
        desc, panels = build_technical_report.caption_for('missing.png', {})
        self.assertTrue(desc)  # falls back to registry or filename

    def test_code_profile(self):
        prof = build_technical_report.code_profile(os.path.join(HERE, 'ai_ecosystem_model.py'))
        self.assertGreater(prof['lines'], 5000)
        self.assertGreater(len(prof['classes']), 20)
        self.assertIn('GameTheoryFramework', prof['classes'])

    def test_load_table_meta(self):
        meta = build_technical_report.load_table_meta(HERE)
        self.assertIn('1.1', meta)
        self.assertTrue(os.path.exists(meta['1.1']['path']))

    def test_loaders_missing_dir(self):
        self.assertIsNone(build_technical_report.load_policy_table('/nonexistent_dir_xyz'))
        self.assertIsNone(build_technical_report.load_valuation_table('/nonexistent_dir_xyz'))
        self.assertIsNone(build_technical_report._exhibit_value('/nonexistent_dir_xyz', 'f.csv', 'x'))
        self.assertEqual(build_technical_report.load_table_meta('/nonexistent_dir_xyz'), {})

    def test_verify_placement_missing_docx(self):
        issues = build_technical_report.verify_placement('/nonexistent/report.docx', [], HERE)
        self.assertEqual(len(issues), 1)
        self.assertIn('cannot open', issues[0].lower())

    def test_gf_strategy_of(self):
        self.assertEqual(build_technical_report.gf_strategy_of(None, None, 'Hardware'), ('Open Access', 'Walled Garden'))
        self.assertEqual(build_technical_report.gf_strategy_of(None, None, 'Mystery Player'), ('Cooperate', 'Defect'))

    def test_core_ne_strategies_uses_live_equilibrium(self):
        # The DOCX core-game label must follow the solved equilibrium, never a
        # hardcoded pair: live (Open Access, Proprietary Stack) passes through,
        # a missing game degrades to n/a instead of failing the build.
        nash = {"Hardware-Cloud Providers":
                {"nash_equilibria": [{"strategies": ["Open Access", "Proprietary Stack"]}]}}
        self.assertEqual(build_technical_report._core_ne_strategies(nash),
                         ["Open Access", "Proprietary Stack"])
        self.assertEqual(build_technical_report._core_ne_strategies({}), ["n/a", "n/a"])

    def test_nash_dwl_missing(self):
        self.assertTrue(np.isnan(build_technical_report._nash_dwl({}, 'Nope-Nope')))


class TestDashboardHelpers(unittest.TestCase):
    """TestDashboardHelpers."""
    def test_num(self):
        self.assertAlmostEqual(dash.num('41.27%'), 41.27)
        self.assertAlmostEqual(dash.num('1,000'), 1000.0)
        self.assertAlmostEqual(dash.num('$5B'), 5.0)
        self.assertIsNone(dash.num(None))
        self.assertIsNone(dash.num('n/a'))

    def test_kv_table(self):
        rows = [{'Metric': 'HHI', 'Value': '3332'}, {'Metric': 'X', 'Value': 'y'}]
        self.assertEqual(dash.kv_table(rows)['HHI'], '3332')

    def test_parse_pair_cell(self):
        self.assertEqual(dash.parse_pair_cell('(1.5, 2.5)'), (1.5, 2.5))
        self.assertEqual(dash.parse_pair_cell('junk'), (None, None))

    def test_extract_walkthrough(self):
        w = dash.extract_walkthrough(os.path.join(HERE, 'ai_ecosystem_model.py'))
        names = [m['name'] for m in w['modules']]
        self.assertGreaterEqual(len(names), 27)
        for must in ('GameTheoryFramework', 'RobustnessBattery', 'BubbleBurstAnalyzer',
                     'TableGenerator', 'CircularDealsAnalyzer'):
            self.assertIn(must, names)
        self.assertTrue(all(m['doc'] for m in w['modules']))

    def test_extract_report_absent(self):
        r = dash.extract_report('/nonexistent/report.docx')
        self.assertIn('absent', r)

    def test_extract_report_real(self):
        p = os.path.join(HERE, 'AI_Ecosystem_Game_Theory_Technical_Report.docx')
        if not os.path.exists(p):
            self.skipTest('report docx not built')
        r = dash.extract_report(p)
        self.assertGreaterEqual(len(r['sections']), 15)


class TestIntegration(unittest.TestCase):
    """Cross-file continuity: artifacts on disk agree with each other."""

    def test_artifact_triples_match(self):
        # Results tables (table_*) ship as CSV + LaTeX + Excel triples...
        csvs = [f for f in os.listdir(TABLES) if f.startswith('table_') and f.endswith('.csv')]
        self.assertGreater(len(csvs), 30)
        for f in csvs:
            stem = f[:-4]
            self.assertTrue(os.path.exists(os.path.join(TABLES, stem + '.tex')), stem)
            self.assertTrue(os.path.exists(os.path.join(TABLES, stem + '.xlsx')), stem)
            df = pd.read_csv(os.path.join(TABLES, f))
            self.assertFalse(df.empty, f)
        # ...while run_pipeline working files are CSV-only by design.
        for f in ('circular_metrics.csv', 'category_level_dependencies.csv',
                  'elasticity_sensitivity_analysis.csv', 'policy_interventions_enhanced.csv'):
            self.assertTrue(os.path.exists(os.path.join(TABLES, f)), f)

    def test_index_all_saved(self):
        idx = pd.read_csv(os.path.join(TABLES, 'tables_index.csv'))
        bad = idx[~idx['Status'].isin(['saved', 'alias'])]
        self.assertTrue(bad.empty, bad.to_string())

    def test_diagnostics_sequence_complete(self):
        for i in ['7.1', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '7.8', '7.9',
                  '7.10', '7.11', '7.12', '7.13', '7.14', '7.15']:
            hits = [f for f in os.listdir(TABLES) if f.startswith(f'table_{i}') and f.endswith('.csv')]
            self.assertTrue(hits, f'missing table_{i}')
        for i in range(1, 7):
            self.assertTrue(os.path.exists(os.path.join(TABLES, f'table_8.{i}.csv')))
        self.assertTrue(os.path.exists(
            os.path.join(TABLES, 'table_8.7_structural_estimation.csv')))

    def test_burst_ranking_artifact(self):
        df = pd.read_csv(os.path.join(TABLES, 'table_7.11_burst_ranking.csv'))
        self.assertEqual(sorted(df['Burst_Rank']), [1, 2, 3, 4])
        self.assertEqual(df['Total_Loss_Severe_$B'].tolist(),
                         sorted(df['Total_Loss_Severe_$B'], reverse=True))

    def test_rpt_integrity_no_fail(self):
        # NOTE: build_technical_report.integrity_checks() emits UPPERCASE statuses ('PASS'/'WARN'/'FAIL');
        # matching lowercase 'fail' here was a vacuous pass -- it could never match.
        data = ai_ecosystem_model.load_all_data()
        prof = build_technical_report.code_profile(os.path.join(HERE, 'ai_ecosystem_model.py'))
        checks = build_technical_report.integrity_checks(ai_ecosystem_model, data, prof, TABLES, PLOTS)
        self.assertGreater(len(checks), 15)
        for c in checks:
            self.assertIn(c['status'], ('PASS', 'WARN', 'FAIL'), c)
            self.assertTrue(c['code'] and c['title'] and isinstance(c['detail'], str))
        fails = [c for c in checks if c['status'] == 'FAIL']
        self.assertEqual(fails, [], f"Integrity checks failed: {fails}")
        codes = {c['code'] for c in checks}
        for must in ('D1', 'D7', 'G1', 'G2', 'G3', 'G4', 'G5', 'T7', 'A1'):
            self.assertIn(must, codes, f"integrity gate {must} missing")

    def test_rpt_integrity_stale_tables_warn(self):
        # Empty tables dir: artifact-gated checks degrade to WARN
        # ("~table absent") instead of crashing or passing.
        data = ai_ecosystem_model.load_all_data()
        prof = build_technical_report.code_profile(os.path.join(HERE, 'ai_ecosystem_model.py'))
        with tempfile.TemporaryDirectory() as td:
            checks = build_technical_report.integrity_checks(
                ai_ecosystem_model, data, prof, td, td)
        by_code = {c['code']: c for c in checks}
        for must in ('G2', 'G3', 'G4', 'G5'):
            self.assertIn(must, by_code, must)
            self.assertEqual(by_code[must]['status'], 'WARN', must)

    def test_rpt_integrity_unreadable_tables_fail(self):
        # Unreadable artifact files: the same checks FAIL (no "~" downgrade).
        data = ai_ecosystem_model.load_all_data()
        prof = build_technical_report.code_profile(os.path.join(HERE, 'ai_ecosystem_model.py'))
        with tempfile.TemporaryDirectory() as td:
            for f in ('table_7.2_gatekeeper_index.csv', 'table_7.6_policy_synergy.csv',
                      'table_7.8_equilibrium_audit.csv', 'table_7.9_mc_convergence.csv'):
                os.mkdir(os.path.join(td, f))
            checks = build_technical_report.integrity_checks(
                ai_ecosystem_model, data, prof, td, td)
        by_code = {c['code']: c for c in checks}
        for must in ('G2', 'G3', 'G4', 'G5'):
            self.assertIn(must, by_code, must)
            self.assertEqual(by_code[must]['status'], 'FAIL', must)

    def test_dashboard_assemble_keys(self):
        data = dash.assemble(TABLES, PLOTS)
        for key in ('kpi', 'games', 'walkthrough', 'report', 'burst_ranking',
                    'burst_formation', 'burst_gdp', 'burst_policy',
                    'inputs_register'):
            self.assertIn(key, data)
        self.assertEqual(len(data['games']), 6)
        self.assertGreaterEqual(len(data['walkthrough']['modules']), 27)
        self.assertGreaterEqual(len(data['report']['sections']), 15)
        self.assertEqual(len(data['burst_ranking']), 4)
        reg = data['inputs_register']
        self.assertGreaterEqual(len(reg), 90)
        ids = {r.get('ID') for r in reg}
        for rid in ('M-GDP', 'M-ERP', 'M-CIRCEXP', 'S-WACC', 'P-1', 'D-17', 'X-GOOG'):
            self.assertIn(rid, ids)

    def test_dashboard_build_subprocess(self):
        out = os.path.join(tempfile.gettempdir(), 'dash_integration_test.html')
        if os.path.exists(out):
            os.remove(out)
        r = subprocess.run([sys.executable, os.path.join(HERE, 'build_interactive_dashboard.py'),
                            '--out', out], capture_output=True, text=True, cwd=HERE)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertGreater(os.path.getsize(out), 100 * 1024)

    def test_dashboard_type_scale(self):
        # Dashboard type scale: integer px steps only (no fractional sizes),
        # human figure captions for every linked PNG.
        import re as _re
        with open(os.path.join(HERE, 'build_interactive_dashboard.py'), encoding='utf-8') as f:
            src = f.read()
        frac = _re.findall(r'\d+\.\d+px', src)
        self.assertEqual(frac, [], frac)
        data = dash.assemble(TABLES, PLOTS)
        self.assertIn('figcaps', data)
        self.assertGreaterEqual(len(data['figcaps']), 16)
        for f in data['figures']:
            if f.startswith('figure_'):
                self.assertIn(f, data['figcaps'], f)

    def test_no_d_word(self):
        # The banned word is split so this test's own source stays clean.
        banned = 'disser' + 'tation'
        for fn in ('ai_ecosystem_model.py', 'build_technical_report.py', 'build_interactive_dashboard.py', 'test_ai_ecosystem_model.py'):
            with open(os.path.join(HERE, fn), encoding='utf-8') as f:
                src = f.read()
            self.assertNotIn(banned, src.lower(), fn)

    def test_rebuilt_figure_labels_and_files(self):
        with open(os.path.join(HERE, 'ai_ecosystem_model.py'), encoding='utf-8') as f:
            src = f.read()
        for marker in ('Equilibrium vs Efficient Welfare by Game',
                       'Core-Game Joint Payoffs by Cell',
                       'Nash Efficiency by Game',
                       'Sensitivity: DWL Response to DWL% Assumption',
                       '(Schematic Trajectories from Calibrated Endpoints)',
                       'Measured 2025 HHI Decomposition'):
            self.assertIn(marker, src, marker)
        # Consolidated registry (Sep 2026: 16 figures): composites 1, 2, 6, 7,
        # 11 plus key data figures must all ship as PNG + vector PDF.
        for n in (1, 2, 3, 6, 7, 9, 11, 14):
            self.assertTrue(os.path.exists(os.path.join(PLOTS, f'figure_{n}.png')), n)
            self.assertTrue(os.path.exists(os.path.join(PLOTS, f'figure_{n}.pdf')), n)

    def test_no_stale_figures_or_tables(self):
        # Publication set is figures 1-16 plus the numbered valuation
        # extras 17-18; anything else (or a gap) fails the build.
        nums = sorted(int(f.split('_')[1].split('.')[0])
                      for f in os.listdir(PLOTS)
                      if f.startswith('figure_') and f.endswith('.png'))
        self.assertEqual(nums, list(range(1, 19)), nums)
        for dead in ('table_3.3.3.5.csv', 'table_3.3.3.5.tex',
                     'table_3.3.3.5.xlsx', 'table_4.2.csv', 'table_4.2.tex',
                     'table_4.2.xlsx'):
            self.assertFalse(os.path.exists(os.path.join(TABLES, dead)), dead)

    def test_report_section_28_inputs_register(self):
        p = os.path.join(HERE, 'AI_Ecosystem_Game_Theory_Technical_Report.docx')
        if not os.path.exists(p):
            self.skipTest('report docx not built')
        r = dash.extract_report(p)
        heads = [b.get('h', '') for s in r['sections'] for b in s['blocks']]
        self.assertTrue(any('Validated inputs' in h for h in heads),
                        'docx §2.8 heading missing')
        mirrors = [t.get('mirror') for s in r['sections'] for b in s['blocks']
                   for t in b.get('tables', [])]
        self.assertIn('8.6', mirrors, 'Table 8.6 not embedded in docx')

    def test_assumption_register_gdp_current(self):
        reg = ai_ecosystem_model.RobustnessBattery.assumption_register()
        a10 = reg[reg['#'] == 'A10'].iloc[0]
        self.assertIn('$30.8T', a10['Assumption'])
        blob = ' '.join(reg['Assumption'].tolist())
        self.assertNotIn('$30.5T', blob)

    def test_report_mirrors_resolve(self):
        with open(os.path.join(HERE, 'dashboard.html'), encoding='utf-8') as f:
            html = f.read()
        m = re.search(r'const DATA = (\{.*?\});\nconst PALETTE', html, re.S)
        data = json.loads(m.group(1))
        files = set(os.listdir(TABLES))
        for s in data['report']['sections']:
            for b in s['blocks']:
                for t in b.get('tables', []):
                    if t.get('mirror'):
                        cands = [f for f in files if t['mirror'].replace('.', '_') in f.replace('.', '_')
                                 or t['mirror'] in f]
                        self.assertTrue(cands, f"mirror {t['mirror']} resolves to no CSV")

    def test_welfare_benchmark_audit_exhibits_present_in_report(self):
        # test_report_mirrors_resolve only checks mirrors that ARE present,
        # so it passes vacuously when Tables 3.5-3.7 are missing from the
        # DOCX entirely. Assert their mirrors actually show up.
        p = os.path.join(HERE, 'AI_Ecosystem_Game_Theory_Technical_Report.docx')
        if not os.path.exists(p):
            self.skipTest('report docx not built')
        r = dash.extract_report(p)
        mirrors = {t.get('mirror') for s in r['sections'] for b in s['blocks']
                   for t in b.get('tables', [])}
        for expected in ('3.5', '3.6', '3.7'):
            self.assertIn(expected, mirrors,
                          f'Table {expected} (welfare-benchmark / equilibrium-multiplicity / '
                          f'repeated-game audit) is missing from the DOCX report')
        # And it must not have silently fallen back to the placeholder text.
        paras = [pp['t'] for s in r['sections'] for b in s['blocks'] for pp in b.get('paras', [])]
        self.assertFalse(any('audit unavailable' in t.lower() for t in paras),
                         'welfare-benchmark audit fell back to its "unavailable" placeholder')

    def test_exec_summary_deal_count_matches_live_rows(self):
        # The exec-summary "N circular deals totaling $X" line once hardcoded
        # "Twelve" while the dependency table had grown to 19 rows. The count
        # is live now; pin it to the live row count so drift fails the build.
        p = os.path.join(HERE, 'AI_Ecosystem_Game_Theory_Technical_Report.docx')
        if not os.path.exists(p):
            self.skipTest('report docx not built')
        r = dash.extract_report(p)
        paras = [pp['t'] for s in r['sections'] for b in s['blocks'] for pp in b.get('paras', [])]
        hits = [t for t in paras if 'circular deals totaling' in t]
        self.assertTrue(hits, 'exec-summary circular-deals line missing from docx')
        n = len(ai_ecosystem_model.load_all_data()['dependencies'])
        self.assertIn(f'{n} circular deals', hits[0],
                      f'docx deal count is stale (live rows: {n})')

    def test_no_orphan_tables_in_dashboard_payload(self):
        # Every table_*.csv (plus the known run_pipeline working files) written
        # to tables/ must be reachable from the dashboard payload somewhere
        # -- via a literal R()/Ropt() read in the dashboard builder source,
        # the generic table_7.*/table_8.* diagnostics loop, or a Report-tab
        # mirror. A CSV that is silently dropped from all three is an orphan.
        data = dash.assemble(TABLES, PLOTS)
        diag_keys = set(data.get('diag', {}).keys())
        mirrors = {t.get('mirror') for s in data.get('report', {}).get('sections', [])
                   for b in s['blocks'] for t in b.get('tables', [])}
        with open(os.path.join(HERE, 'build_interactive_dashboard.py'), encoding='utf-8') as f:
            bd_source = f.read()
        named_reads = set(re.findall(r"R(?:opt)?\('([^']+\.csv)'\)", bd_source))

        def covered(fname):
            if fname in diag_keys or fname in named_reads:
                return True
            stem = fname[:-4]
            m = re.match(r'table_([A-Za-z0-9]+\.[A-Za-z0-9]+)', stem)
            return bool(m and m.group(1) in mirrors)

        orphans = [f for f in sorted(os.listdir(TABLES))
                  if f.endswith('.csv') and not covered(f)]
        self.assertEqual(orphans, [], f'tables/ CSVs never surfaced anywhere in the dashboard: {orphans}')

    def test_save_audit_table_survives_infinite_values(self):
        # save_audit_table() must survive real data whose columns
        # legitimately contain +inf ('Nash reversion is not a credible
        # punishment'): all three formats (.csv/.tex/.xlsx) must be written
        # with content.
        df = pd.DataFrame({
            'Game': ['A-B', 'C-D'],
            'Critical_Delta': [0.0, float('inf')],
            'Status': ['no deviation incentive', 'Nash reversion is not a credible punishment'],
        })
        with tempfile.TemporaryDirectory() as tmp:
            ai_ecosystem_model.save_audit_table(df, 'table_test_inf_audit', tmp)
            for ext in ('.csv', '.tex', '.xlsx'):
                path = os.path.join(tmp, 'table_test_inf_audit' + ext)
                self.assertTrue(os.path.exists(path), path)
                self.assertGreater(os.path.getsize(path), 0, f'{path} was written empty')

    def test_telecom_vintage_verdict_ledger(self):
        # The telecom backtest vintage must land its verdict ledger with the
        # registered U/A/G verdicts, honest grades only, plus its figures.
        import pandas as pd
        bt = os.path.join(HERE, 'tables', 'backtest')
        for stem in ('telecom_verdict', 'telecom_formation', 'telecom_burst',
                     'telecom_games', 'telecom_actuals'):
            p = os.path.join(bt, stem + '.csv')
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 0, p)
        v = pd.read_csv(os.path.join(bt, 'telecom_verdict.csv'))
        self.assertEqual(list(v.columns),
                         ['Test', 'Grade', 'Model says', 'Reading'])
        self.assertTrue(set(v['Grade']) <= {'PASS', 'PARTIAL', 'FAIL', 'INFO'},
                        set(v['Grade']))
        self.assertGreaterEqual(len(v), 11)
        for stem in ('payoff_matrices_2008', 'payoff_matrices_telecom',
                     'telecom_overview', 'telecom_games', 'telecom_rank'):
            p = os.path.join(HERE, 'figures', stem + '.png')
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 0, p)


class TestStaticRigor(unittest.TestCase):
    """Code-level rigor: everything compiles warning-free and documented."""

    FILES = ['ai_ecosystem_model.py', 'build_technical_report.py', 'build_interactive_dashboard.py', 'run_pipeline.py', 'test_ai_ecosystem_model.py']

    def test_all_modules_compile(self):
        import py_compile
        for f in self.FILES:
            self.assertTrue(py_compile.compile(os.path.join(HERE, f), doraise=True), f)

    def test_no_syntax_warnings(self):
        import warnings
        for f in self.FILES:
            with open(os.path.join(HERE, f), encoding='utf-8') as fh:
                src = fh.read()
            with warnings.catch_warnings():
                warnings.simplefilter('error', SyntaxWarning)
                compile(src, f, 'exec')  # catches invalid escapes like \s in strings

    def test_docstring_coverage(self):
        import ast as _ast
        for f in ['ai_ecosystem_model.py', 'build_technical_report.py', 'build_interactive_dashboard.py', 'run_pipeline.py']:
            with open(os.path.join(HERE, f), encoding='utf-8') as fh:
                tree = _ast.parse(fh.read())
            thin = []
            for node in tree.body:
                if isinstance(node, (_ast.ClassDef, _ast.FunctionDef)):
                    ds = _ast.get_docstring(node) or ''
                    if len(ds) < 50:
                        thin.append(f'{f}:{node.name}')
            self.assertEqual(thin, [], f'undocumented/short: {thin}')


class TestGameTheoryProofs(unittest.TestCase):
    """Re-derive equilibrium results from first principles; never trust the solver."""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.gf = solve_framework()

    @staticmethod
    def _cells(matrix):
        return {(r, c): (float(matrix[r, c][0]), float(matrix[r, c][1]))
                for r in range(2) for c in range(2)}

    def test_pure_ne_are_mutual_best_responses(self):
        for name, res in self.gf.nash_equilibria.items():
            cells = self._cells(res['matrix'])
            for e in res.get('nash_equilibria', []):
                r, c = e['position']
                self.assertEqual(cells[(r, c)][0], max(cells[(i, c)][0] for i in range(2)))
                self.assertEqual(cells[(r, c)][1], max(cells[(r, j)][1] for j in range(2)))

    def test_solver_finds_all_pure_ne(self):
        for name, res in self.gf.nash_equilibria.items():
            cells = self._cells(res['matrix'])
            expected = {(r, c) for (r, c) in cells
                        if cells[(r, c)][0] == max(cells[(i, c)][0] for i in range(2))
                        and cells[(r, c)][1] == max(cells[(r, j)][1] for j in range(2))}
            found = {tuple(e['position']) for e in res.get('nash_equilibria', [])}
            self.assertEqual(found, expected, name)

    def test_pareto_maximal_and_dwl_identity(self):
        for name, res in self.gf.nash_equilibria.items():
            cells = self._cells(res['matrix'])
            w = res['welfare_metrics']
            joint = {k: v[0] + v[1] for k, v in cells.items()}
            best = max(joint.values())
            self.assertAlmostEqual(w['pareto_optimal_welfare'], best, places=6)
            self.assertAlmostEqual(w['deadweight_loss'], best - w['nash_welfare'], places=6)
            self.assertAlmostEqual(w['efficiency_ratio'], w['nash_welfare'] / best * 100, places=4)

    def test_coordination_definition_agreement(self):
        ca = ai_ecosystem_model.CoordinationFailureAnalyzer(self.gf)
        ca.identify_coordination_failures()
        for name, res in self.gf.nash_equilibria.items():
            w = res['welfare_metrics']
            is_fail = w['nash_welfare'] < w['pareto_optimal_welfare'] - 1e-6
            self.assertEqual(bool(ca.coordination_failures[name].get('is_coordination_failure')),
                             is_fail, name)

    def test_mixed_matching_pennies(self):
        M = np.array([[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]], dtype=object)
        out = self.gf._find_mixed_strategy_nash(M)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out['p'], 0.5, places=6)
        self.assertAlmostEqual(out['q'], 0.5, places=6)
        self.assertAlmostEqual(out['u1'], out['u2'], places=6)
        # Indifference check: with q=0.5 both rows tie for P1.
        q = out['q']
        self.assertAlmostEqual((1 - q) * 1 + q * -1, (1 - q) * -1 + q * 1, places=6)

    def test_dominant_prisoners_dilemma(self):
        pd_pay = [[(3, 3), (0, 5)], [(5, 0), (1, 1)]]
        self.assertEqual(self.gf._find_dominant_strategies(pd_pay), (1, 1))

    def test_degenerate_uniform_focal(self):
        M = np.array([[(2, 2), (2, 2)], [(2, 2), (2, 2)]], dtype=object)
        out = self.gf._find_mixed_strategy_nash(M)
        self.assertEqual((out['p'], out['q']), (0.5, 0.5))


class TestEconomicsIdentities(unittest.TestCase):
    """TestEconomicsIdentities."""
    def test_markup_identity(self):
        with tempfile.TemporaryDirectory() as td:
            df = ai_ecosystem_model.ElasticitySensitivityAnalyzer(output_dir=td).calculate_lerner_ranges()
            for _, row in df.iterrows():
                for tag in ('Conservative', 'Base', 'Aggressive'):
                    L = row[f'{tag}_Lerner']
                    self.assertAlmostEqual(row[f'{tag}_Markup_%'], L / (1 - L) * 100, places=4)
                self.assertLess(row['Base_Elasticity'], 0)  # demand slopes down

    def test_cr4_identity(self):
        data = ai_ecosystem_model.load_all_data()
        ma = ai_ecosystem_model.MarketStructureAnalyzer(data['players'])
        conc = ma.calculate_market_concentration()
        shares = sorted(conc['market_shares'].values(), reverse=True)
        self.assertAlmostEqual(conc['CR4'], sum(shares[:4]), places=6)

    def test_gini_endpoints(self):
        data = ai_ecosystem_model.load_all_data()
        ma = ai_ecosystem_model.MarketStructureAnalyzer(data['players'])
        self.assertAlmostEqual(ma._calculate_gini(np.array([0.25] * 4)), 0.0, places=6)
        mono = ma._calculate_gini(np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertGreater(mono, 0.5)
        self.assertLess(mono, 1.0)

    def test_valuation_identities(self):
        data = ai_ecosystem_model.load_all_data()
        eva = ai_ecosystem_model.EnhancedValuationAnalyzer(data['players'], circular_analyzer=None,
                                           plots_dir='/tmp', tables_dir='/tmp')
        fin = eva.create_company_financials(data['players'].iloc[0])
        m = ai_ecosystem_model.ValuationMetricsCalculator().calculate_all_metrics(fin, tam_2030=1000.0)
        ev = fin.market_cap + max(fin.total_debt - fin.cash, 0.0)
        self.assertAlmostEqual(m['ev_revenue'], ev / max(fin.revenue_ttm, 1e-6))
        if fin.net_income > 0:
            self.assertAlmostEqual(m['pe_ratio'], fin.market_cap / fin.net_income)
            self.assertAlmostEqual(m['forward_pe'], m['pe_ratio'] * (1 - fin.revenue_growth_yoy))
        self.assertAlmostEqual(m['roic_wacc_spread'], m['roic'] - fin.wacc)
        self.assertTrue(0 <= m['sustainability_score'] <= 100)

    def test_valuation_company_level_differs(self):
        # The screen must use per-company TTM inputs, never the category
        # aggregate: every name would otherwise print identical revenue,
        # multiples and sustainability scores.
        data = ai_ecosystem_model.load_all_data()
        eva = ai_ecosystem_model.EnhancedValuationAnalyzer(data['players'], circular_analyzer=None,
                                           plots_dir='/tmp', tables_dir='/tmp')
        df = eva.calculate_all_enhanced_metrics()
        self.assertEqual(len(df), len(ai_ecosystem_model.COMPANY_VALUATION_INPUTS))
        self.assertGreater(df['market_cap'].nunique(), 4)
        self.assertGreater(df['pe_ratio'].nunique(), 4)
        self.assertGreater(df['sustainability_score'].nunique(), 1)
        self.assertGreater(len(set(df['screen_flag'])), 1)
        self.assertTrue(set(df['screen_flag']).issubset({'Positive', 'Neutral', 'Negative'}))
        self.assertEqual(
            sorted(df['screen_flag'].value_counts().to_dict().items()),
            sorted({'Positive': 4, 'Neutral': 17, 'Negative': 2}.items()))
        anth = df.loc[df['company_name'] == 'Anthropic'].iloc[0]
        self.assertTrue(pd.isna(anth['pe_ratio']))  # unattributed margins: fenced, not fabricated
        nvda = df.loc[df['company_name'] == 'NVIDIA'].iloc[0]
        self.assertAlmostEqual(nvda['market_cap'], 5280.0)
        self.assertLess(nvda['pe_ratio'], 60.0)
        intel = df.loc[df['company_name'] == 'Intel'].iloc[0]
        self.assertTrue(pd.isna(intel['pe_ratio']))  # GAAP loss-making: N/A, not 75x

    def test_dcf_scenario_identities(self):
        ca = ai_ecosystem_model.CircularDealsAnalyzer()
        adj = ca.dcf_adjust(1000.0, 'Hardware')
        e = adj['exposure']
        self.assertAlmostEqual(adj['scenarios']['base'], 1000.0 * (1 - e * 0.20))
        self.assertAlmostEqual(adj['scenarios']['opt'], 1000.0 * (1 + e * 0.10))
        self.assertAlmostEqual(adj['scenarios']['pess'], 1000.0 * (1 - e * 0.50))
        s = adj['scenarios']
        self.assertAlmostEqual(adj['expected'], s['opt'] * 0.3 + s['base'] * 0.4 + s['pess'] * 0.3)

    def test_welfare_dwl_override_monotonic(self):
        # NOTE: override runs return results WITHOUT touching .welfare_results
        # (by design: sensitivity sweeps must not clobber the base run) --
        # callers must use the return value.
        data, gf = solve_framework()
        wa = ai_ecosystem_model.WelfareEconomicsAnalyzer(gf, data['players'], dependencies_df=data['dependencies'])
        lo = wa.calculate_aggregate_welfare_loss(override_dwl_pct=0.20)['total_dwl']
        hi = wa.calculate_aggregate_welfare_loss(override_dwl_pct=0.40)['total_dwl']
        self.assertGreater(hi, lo)

    def test_sensitivity_sweep_brackets_base(self):
        data, gf = solve_framework()
        wa = ai_ecosystem_model.WelfareEconomicsAnalyzer(gf, data['players'], dependencies_df=data['dependencies'])
        wa.calculate_aggregate_welfare_loss()
        df = ai_ecosystem_model.SensitivityAnalysis(wa, gf).run_sensitivity_analysis(variation_pct=0.10, steps=5)
        base = float(ai_ecosystem_model.DWL_PCT_TOPDOWN)
        self.assertLess(df['test_dwl_pct'].min(), base)
        self.assertGreater(df['test_dwl_pct'].max(), base)

    def test_portfolio_risk_identities(self):
        data = ai_ecosystem_model.load_all_data()
        out = ai_ecosystem_model.PortfolioRiskAnalysis(data).run_simulation(n_simulations=2000, seed=5)
        self.assertAlmostEqual(out['sharpe_ratio'], out['mean_return'] / out['std_return'], places=9)
        self.assertLessEqual(out['cvar_95'], out['var_95'])
        self.assertGreaterEqual(out['std_return'], 0)

    def test_gdp_channel_identities(self):
        data = ai_ecosystem_model.load_all_data()
        bb = ai_ecosystem_model.BubbleBurstAnalyzer(data['players'], ai_ecosystem_model.CircularDealsAnalyzer(),
                                    dependencies_df=data['dependencies'], output_dir='/tmp')
        df = bb.gdp_impact(200.0)
        sev = df[df['Severity'] == 'severe']
        capex = sev[sev['Channel'] == 'AI capex cut']['GDP_Loss_$B'].iloc[0]
        self.assertAlmostEqual(capex, bb.US_AI_CAPEX_2025_B * 0.30, places=6)
        wealth = sev[sev['Channel'] == 'Equity wealth effect']['GDP_Loss_$B'].iloc[0]
        self.assertAlmostEqual(wealth, 200.0 * 1.0 * bb.WEALTH_MPC, places=6)

    def test_formation_sigmoid_identity(self):
        data = ai_ecosystem_model.load_all_data()
        bb = ai_ecosystem_model.BubbleBurstAnalyzer(data['players'], ai_ecosystem_model.CircularDealsAnalyzer(),
                                    dependencies_df=data['dependencies'], output_dir='/tmp')
        df = bb.formation_probability()
        L = bb.LOGIT
        row = df[df['Archetype'] == 'Hardware'].iloc[0]
        z = (L['intercept'] + L['w_circ'] * row['Circular_Exposure']
             + L['w_capex'] * row['Capex_Intensity'] + L['w_hhi'] * row['HHI_Share']
             + L['w_exub'] * row['Exuberance'])
        # delta covers the table's 1dp/3dp rounding (recompute is exact to ~0.03).
        self.assertAlmostEqual(row['Formation_Prob_Pct'], 1 / (1 + np.exp(-z)) * 100, delta=0.1)


class TestStatisticsIdentities(unittest.TestCase):
    """TestStatisticsIdentities."""
    def test_mcse_identity_and_verdicts(self):
        data, gf = solve_framework()
        rng = np.random.default_rng(11)
        good = rng.normal(10, 0.1, 2000)          # tight stationary draws
        bad = np.concatenate([np.zeros(500), np.full(500, 100.0)])  # regime shift
        ld = ai_ecosystem_model.LiteratureDiagnostics(
            players_df=data['players'], dependencies_df=data['dependencies'],
            game_framework=gf,
            mc_results={'Tight': {'dwl_distribution': good}, 'Shifted': {'dwl_distribution': bad}},
            aggregate_welfare={}, policy_df=pd.DataFrame(), circular_metrics={}, output_dir='/tmp')
        df = ld.mc_convergence().set_index('Game')
        expect = good.std(ddof=1) / np.sqrt(len(good))
        self.assertAlmostEqual(df.loc['Tight', 'MCSE_DWL_$B'], round(expect, 4), places=6)
        self.assertEqual(df.loc['Tight', 'Assessment'], 'adequate')
        self.assertEqual(df.loc['Shifted', 'Assessment'], 'increase draws')

    def test_geweke_p_bounds_and_holm_specificity(self):
        _, gf = solve_framework()
        rng = np.random.default_rng(4)
        mc_syn = {f'S{i}': {'dwl_distribution': rng.normal(5 + i, 1, 3000)} for i in range(5)}
        mc_syn['Drift'] = {'dwl_distribution': np.linspace(0, 50, 3000) + rng.normal(0, 0.5, 3000)}
        df = ai_ecosystem_model.RobustnessBattery(gf, mc_results=mc_syn).run_stationarity()
        self.assertTrue(((df['p_value'] >= 0) & (df['p_value'] <= 1)).all())
        # NOTE: Verdict casing is inconsistent in ai_ecosystem_model.py ('stationary' vs
        # 'NON-STATIONARY'); compare case-insensitively, don't enshrine it.
        v = {g: s.lower() for g, s in zip(df['Game'], df['Verdict'])}
        self.assertEqual(v['Drift'], 'non-stationary')
        for i in range(5):
            self.assertEqual(v[f'S{i}'], 'stationary')

    def test_saltelli_model_shape_finite_monotone(self):
        base = np.array([[300.0, 100.0, 0.3, 0.25, 0.5]])
        X = np.repeat(base, 7, axis=0)
        # Sweep shared circularity over the R4 operational band (dep ~= 0.29
        # +/- 20%). Global 0-1 monotonicity is NOT asserted: discrete Nash
        # switches can re-rank cells outside the sampled band, so DWL need not
        # rise everywhere; inside the band deeper lock-in must not lower DWL.
        X[:, 4] = np.linspace(0.15, 0.45, 7)  # sweep circ_shared only
        y = ai_ecosystem_model.RobustnessBattery._saltelli_model(X)
        self.assertEqual(y.shape, (7,))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(bool((y >= -1e-9).all()))
        self.assertTrue(bool((np.diff(y) >= -1e-9).all()), 'DWL must not fall as lock-in rises in-band')


class TestPayoffProportionality(unittest.TestCase):
    """Per-player proportional deviations (A3 improvement, Sep 6 2026)."""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.gf = solve_framework()
        cls.fund = (cls.data['players'].set_index('Player_Category')
                    [['Operating_Margin', 'Circular_Dependency_Index']])

    def test_deviation_rates_from_own_fundamentals(self):
        t1, s1, t2, s2 = self.gf._derive_strategic_deviations(
            'Hardware', 'LLM Wrappers', 0, 0, 0)
        m1, c1 = self.fund.loc['Hardware']
        m2, c2 = self.fund.loc['LLM Wrappers']
        self.assertAlmostEqual(t1, max(0.02, 0.10 + m1 * 0.40), places=9)
        self.assertAlmostEqual(s1, min(0.30, max(0.05, 0.05 + c1 * 0.25)), places=9)
        self.assertAlmostEqual(t2, max(0.02, 0.10 + m2 * 0.40), places=9)
        self.assertAlmostEqual(s2, min(0.30, max(0.05, 0.05 + c2 * 0.25)), places=9)

    def test_small_player_swings_scaled(self):
        m = self.gf.payoff_matrices['Hardware-LLM Wrappers']
        hw_gain = m[1, 0][0] - m[0, 0][0]
        wrap_gain = m[0, 1][1] - m[0, 0][1]
        self.assertGreater(hw_gain, 0)
        self.assertGreater(wrap_gain, 0)
        # A3: absolute swings scale with own stakes, not pair totals.
        self.assertLess(wrap_gain, hw_gain)

    def test_no_exact_mirror(self):
        for name, m in self.gf.payoff_matrices.items():
            j01 = m[0, 1][0] + m[0, 1][1]
            j10 = m[1, 0][0] + m[1, 0][1]
            self.assertGreater(abs(j01 - j10), 1e-9, name)


class TestInputsRegister(unittest.TestCase):
    """Table 8.6 lists every validated input live from canonical structures."""

    @classmethod
    def setUpClass(cls):
        cls.df = ai_ecosystem_model.build_validated_inputs_register()

    def test_register_shape(self):
        self.assertEqual(list(self.df.columns),
                         ['ID', 'Category', 'Parameter', 'Value', 'Unit', 'Source/Verification'])
        self.assertGreaterEqual(len(self.df), 90)
        self.assertTrue(self.df['ID'].is_unique)

    def test_register_spot_values(self):
        by_id = self.df.set_index('ID')
        self.assertEqual(by_id.loc['M-GDP', 'Value'], '30767.1')
        self.assertEqual(by_id.loc['M-ERP', 'Value'], '0.042')
        self.assertEqual(by_id.loc['M-CIRCEXP', 'Value'], '675.40')
        self.assertEqual(by_id.loc['M-MCAP', 'Value'], '23181.0')
        self.assertIn('Damodaran', by_id.loc['M-ERP', 'Source/Verification'])

    def test_register_matches_live_structures(self):
        deps = ai_ecosystem_model.EmbeddedDataSource.get_industry_dependencies()
        by_id = self.df.set_index('ID')
        committed = deps.loc[deps['Booking_Status'] == 'committed',
                             'Dependency_Value_Billions'].sum()
        self.assertEqual(by_id.loc['M-CIRCEXP', 'Value'], f"{committed:.2f}")
        d17 = by_id.loc['D-17']
        self.assertIn('Google', d17['Parameter'])
        players = ai_ecosystem_model.EmbeddedDataSource.get_industry_players()
        rev = by_id.loc['V-HA-REV', 'Value']
        self.assertEqual(rev, f"{players['Current_Revenue_Billions'].iloc[0]:.1f}")

    def test_register_artifact_and_dashboard_wiring(self):
        self.assertTrue(os.path.exists(os.path.join(TABLES, 'table_8.6.csv')))
        on_disk = pd.read_csv(os.path.join(TABLES, 'table_8.6.csv'))
        self.assertEqual(list(on_disk.columns), list(self.df.columns))
        self.assertEqual(len(on_disk), len(self.df))
        self.assertIn(('inputs', 'Inputs & Assumptions'), dash.TABS)

    def test_booking_reconciliation_entries(self):
        by_id = self.df.set_index('ID')
        for rid in ('X-FACILITY', 'X-HFVI'):
            self.assertIn(rid, by_id.index, rid)
            self.assertIn('not in committed total', by_id.loc[rid, 'Unit'])
        d07 = by_id.loc['D-07']
        self.assertIn('prospective', d07['Unit'])
        d11 = by_id.loc['D-11']
        self.assertIn('vertical-integration', d11['Unit'])
        self.assertIn('17 of 19', by_id.loc['M-CIRCEXP', 'Source/Verification'])
        # Deal-level reconciliation: every D-row carries its source material.
        for rid in ('D-01', 'D-04', 'D-07', 'D-08', 'D-11', 'D-19'):
            self.assertIn('source:', by_id.loc[rid, 'Source/Verification'], rid)
        self.assertIn('SEC filing', by_id.loc['D-04', 'Source/Verification'])
        self.assertIn('not a sourced deal', by_id.loc['D-09', 'Source/Verification'])
        self.assertIn('BEA National Income', by_id.loc['M-GDP', 'Source/Verification'])


class TestDashboardTolerantParsers(unittest.TestCase):
    """build_interactive_dashboard.py tolerant parsers: num/kv_table/parse_pair_cell/read_csv_rows_optional."""

    def test_num_parses_decorated_numbers(self):
        self.assertAlmostEqual(dash.num('41.27%'), 41.27)
        self.assertAlmostEqual(dash.num('$1,250.5B'), 1250.5)
        self.assertAlmostEqual(dash.num('  -99.4  '), -99.4)
        self.assertAlmostEqual(dash.num('1,000'), 1000.0)

    def test_num_passthrough_and_unparseable(self):
        self.assertAlmostEqual(dash.num(5), 5.0)
        self.assertAlmostEqual(dash.num(5.5), 5.5)
        # Unparseable inputs yield None (not NaN): callers use `is None` checks.
        self.assertIsNone(dash.num(None))
        self.assertIsNone(dash.num('n/a'))
        self.assertIsNone(dash.num('invalid_string'))
        self.assertIsNone(dash.num(''))

    def test_kv_table_first_two_columns(self):
        rows = [
            {'Metric': 'Total Revenue', 'Value': '1023.0', 'Extra': 'ignored'},
            {'Metric': '', 'Value': 'skipped-empty-metric'},
            {'Metric': 'HHI', 'Value': '3916'},
        ]
        kv = dash.kv_table(rows)
        self.assertEqual(kv['Total Revenue'], '1023.0')
        self.assertEqual(kv['HHI'], '3916')
        self.assertNotIn('', kv)
        self.assertEqual(len(kv), 2)

    def test_parse_pair_cell_signed_and_malformed(self):
        self.assertEqual(dash.parse_pair_cell('(487.9, 606.8)'), (487.9, 606.8))
        self.assertEqual(dash.parse_pair_cell('(1.5,2.5)'), (1.5, 2.5))
        self.assertEqual(dash.parse_pair_cell('(-10.5, 0.0)'), (-10.5, 0.0))
        self.assertEqual(dash.parse_pair_cell('malformed'), (None, None))
        self.assertEqual(dash.parse_pair_cell(''), (None, None))
        self.assertEqual(dash.parse_pair_cell(None), (None, None))

    def test_read_csv_rows_optional_missing_and_directory(self):
        self.assertEqual(dash.read_csv_rows_optional('/nonexistent_dir_xyz/nope.csv'), [])
        with tempfile.TemporaryDirectory() as td:
            # A directory path is unreadable as CSV -> [] rather than raising.
            self.assertEqual(dash.read_csv_rows_optional(td), [])

    def test_read_csv_rows_optional_roundtrip(self):
        import csv as _csv
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'sample.csv')
            with open(p, 'w', newline='', encoding='utf-8-sig') as f:
                w = _csv.DictWriter(f, fieldnames=['Metric', 'Value'])
                w.writeheader()
                w.writerow({'Metric': 'HHI', 'Value': '3916'})
            rows = dash.read_csv_rows_optional(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual(dash.kv_table(rows)['HHI'], '3916')


class TestGamesPayloadLive(unittest.TestCase):
    """build_interactive_dashboard.games_payload(): live 2x2 solutions behind the Games tab."""

    @classmethod
    def setUpClass(cls):
        cls.games, cls.order = dash.games_payload()
        cls.data = ai_ecosystem_model.load_all_data()

    def test_six_games_and_canonical_order(self):
        self.assertEqual(len(self.games), 6)
        self.assertEqual(self.order, ['Hardware', 'Cloud Providers',
                                      'Foundation Models', 'LLM Wrappers'])
        names = {g['name'] for g in self.games}
        self.assertEqual(names, {
            'Hardware-Cloud Providers', 'Hardware-Foundation Models', 'Hardware-LLM Wrappers',
            'Cloud Providers-Foundation Models', 'Cloud Providers-LLM Wrappers',
            'Foundation Models-LLM Wrappers'})

    def test_matrix_cells_are_finite_pairs(self):
        for g in self.games:
            self.assertEqual(len(g['matrix']), 2)
            for row in g['matrix']:
                self.assertEqual(len(row), 2)
                for a, b in row:
                    self.assertIsInstance(a, float)
                    self.assertIsInstance(b, float)
                    self.assertTrue(np.isfinite(a) and np.isfinite(b), g['name'])
            self.assertIn(g['p1'], self.order)
            self.assertIn(g['p2'], self.order)
            self.assertEqual(len(g['p1_strats']), 2)
            self.assertEqual(len(g['p2_strats']), 2)

    def test_defect_defect_matches_observed_revenue(self):
        revs = self.data['players'].set_index('Player_Category')['Current_Revenue_Billions'].to_dict()
        for g in self.games:
            dd = g['matrix'][1][1]
            self.assertAlmostEqual(dd[0], revs[g['p1']], places=5, msg=g['name'])
            self.assertAlmostEqual(dd[1], revs[g['p2']], places=5, msg=g['name'])

    def test_ne_positions_and_welfare_subset(self):
        allowed = {(0, 0), (0, 1), (1, 0), (1, 1)}
        welfare_keys = {'deadweight_loss', 'efficiency_ratio', 'pareto_optimal_welfare',
                        'nash_welfare', 'pareto_position', 'dwl_percent'}
        for g in self.games:
            for pos in g['ne']:
                self.assertIn(tuple(pos), allowed, g['name'])
            self.assertTrue(set(g['welfare']) <= welfare_keys, g['name'])
            if 'deadweight_loss' in g['welfare']:
                self.assertGreaterEqual(g['welfare']['deadweight_loss'], 0)
            if 'efficiency_ratio' in g['welfare']:
                self.assertGreater(g['welfare']['efficiency_ratio'], 0)
                self.assertLessEqual(g['welfare']['efficiency_ratio'], 100)


class TestExtractorEdges(unittest.TestCase):
    """extract_walkthrough / extract_report never raise: missing, broken, and absent inputs."""

    def test_walkthrough_missing_file_is_empty(self):
        w = dash.extract_walkthrough('/nonexistent_dir_xyz/ai_ecosystem_model.py')
        self.assertEqual(w, {'stages': [], 'modules': []})

    def test_walkthrough_unparseable_file_is_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'ai_ecosystem_model.py')
            with open(p, 'w', encoding='utf-8') as f:
                f.write('def broken(:\n  this is not python\n')
            self.assertEqual(dash.extract_walkthrough(p), {'stages': [], 'modules': []})

    def test_walkthrough_module_record_shape(self):
        w = dash.extract_walkthrough(os.path.join(HERE, 'ai_ecosystem_model.py'))
        self.assertGreaterEqual(len(w['stages']), 7)
        self.assertGreaterEqual(len(w['modules']), 25)
        for m in w['modules']:
            self.assertTrue(m['name'] and m['stage'] and m['doc'])
            self.assertIsInstance(m['methods'], list)
        names = {m['name'] for m in w['modules']}
        for must in ('GameTheoryFramework', 'MarketStructureAnalyzer', 'RobustnessBattery',
                     'BubbleBurstAnalyzer', 'DebtRankClearingEngine', 'CreditMonitor',
                     'TableGenerator', 'CircularDealsAnalyzer'):
            self.assertIn(must, names)

    def test_extract_report_absent_when_not_built(self):
        r = dash.extract_report('/nonexistent_dir_xyz/report.docx')
        self.assertIn('absent', r)
        self.assertIn('not built', r['absent'])

    def test_extract_report_without_docx_dependency(self):
        import sys as _sys
        with mock.patch.dict(_sys.modules, {'docx': None}):
            r = dash.extract_report(os.path.join(HERE, 'ai_ecosystem_model.py'))
        self.assertEqual(r, {'absent': 'python-docx not installed'})


class TestRptDocxPrimitives(unittest.TestCase):
    """build_technical_report.py DOCX styling: cell shading, header rows, word tables, insight callouts."""

    def test_fmt_grouping_and_infinities(self):
        self.assertEqual(build_technical_report._fmt(1234.0), '1,234')
        self.assertEqual(build_technical_report._fmt(1234567.89), '1,234,567.89')
        self.assertEqual(build_technical_report._fmt(12.3456), '12.35')
        self.assertEqual(build_technical_report._fmt(float('inf')), '\u221e')
        self.assertEqual(build_technical_report._fmt(float('-inf')), '-\u221e')

    def test_sort_key_full_order(self):
        ids = ['A.2', '1.10', '2.4', '1.3', '8_1', '7.15']
        # Numeric sections first (1.3 < 1.10 < 2.4 < 7.15), then A.x, then
        # underscored ids such as 8_1 which carry no dotted section.
        self.assertEqual(sorted(ids, key=build_technical_report._table_sort_key),
                         ['1.3', '1.10', '2.4', '7.15', 'A.2', '8_1'])

    def test_figure_number_sixteen(self):
        self.assertEqual(build_technical_report.figure_number('figure_16.png'), '16')
        self.assertIsNone(build_technical_report.figure_number('not_a_figure.jpg'))

    def test_cell_shading_and_header_row(self):
        from docx import Document as _Document
        doc = _Document()
        table = doc.add_table(rows=2, cols=2)
        cell = table.cell(0, 0)
        build_technical_report.set_cell_bg(cell, '2C3E50')
        self.assertIn('shd', cell._tc.xml)
        self.assertIn('2C3E50', cell._tc.xml)
        build_technical_report.mark_header_row(table)
        for c in table.rows[0].cells:
            self.assertIn('tblHeader', c._tc.xml)

    def test_add_word_table_empty_and_styled(self):
        from docx import Document as _Document
        doc = _Document()
        self.assertIsNone(build_technical_report.add_word_table(doc, None))
        self.assertIsNone(build_technical_report.add_word_table(doc, pd.DataFrame()))
        df = pd.DataFrame({'Metric': ['HHI', 'CR4'], 'Value': [3916.0, 0.85]})
        n_before = len(doc.tables)
        table = build_technical_report.add_word_table(doc, df, caption='Table 1.1 -- Market structure')
        self.assertIsNotNone(table)
        self.assertEqual(len(doc.tables), n_before + 1)
        self.assertEqual(len(table.rows), len(df) + 1)  # header + body
        self.assertEqual(table.cell(0, 0).text, 'Metric')
        self.assertEqual(table.cell(1, 0).text, 'HHI')
        self.assertEqual(table.cell(1, 1).text, '3,916')  # _fmt groups integral floats
        self.assertTrue(any('Table 1.1' in p.text for p in doc.paragraphs))

    def test_add_word_table_renders_infinity_cells(self):
        # Table 3.7 carries genuine +inf critical deltas; the DOCX build must
        # render them (via _fmt -> ∞), not raise.
        from docx import Document as _Document
        doc = _Document()
        df = pd.DataFrame({'Game': ['A-B'],
                           'Critical_Delta': [float('inf')],
                           'Status': ['Nash reversion is not a credible punishment']})
        table = build_technical_report.add_word_table(doc, df)
        self.assertIsNotNone(table)
        self.assertIn('\u221e', table.cell(1, 1).text)

    def test_add_insights_noop_and_callout(self):
        from docx import Document as _Document
        doc = _Document()
        n_before = len(doc.paragraphs)
        build_technical_report.add_insights(doc, [])
        self.assertEqual(len(doc.paragraphs), n_before)
        build_technical_report.add_insights(doc, [('Concentration binds', 'HHI implies pricing power')], section_label='Mkt')
        texts = [p.text for p in doc.paragraphs[n_before:]]
        self.assertTrue(any('Analytical insights' in t for t in texts))
        self.assertTrue(any('Concentration binds' in t for t in texts))


class TestReportAnalysisEndToEnd(unittest.TestCase):
    """build_technical_report.analyze() recomputation plus the payoff-exhibit DOCX renderer."""

    @classmethod
    def setUpClass(cls):
        cls.data = ai_ecosystem_model.load_all_data()
        cls.analysis = build_technical_report.analyze(ai_ecosystem_model, cls.data)

    def test_valuation_inference_renders_screen(self):
        # Regression: the valuation module once read enhanced_valuation_metrics.csv
        # from the workspace root instead of tables/, silently rendering n/a.
        mod = build_technical_report.build_module_analyses(self.analysis, HERE)
        inf = mod['valuation']['inference']
        self.assertIn('companies screen 4 Positive / 2 Negative / 17 Neutral', inf)

    def test_analysis_payload_keys(self):
        for key in ('concentration', 'market_power', 'nash', 'coordination',
                    'welfare', 'shapley', 'circular', 'dwl_pct',
                    'game_dwl_pct', 'mc_iterations', 'global_seed'):
            self.assertIn(key, self.analysis, f"analyze() missing '{key}'")
        self.assertEqual(len(self.analysis['nash']), 6)
        # Game-level DWL% is clamped to [0.05, 0.40] where it is derived.
        game_dwl_pct = float(self.analysis['game_dwl_pct'])
        self.assertTrue(np.isfinite(game_dwl_pct))
        self.assertGreaterEqual(game_dwl_pct, 0.05)
        self.assertLessEqual(game_dwl_pct, 0.40)

    def test_analysis_nash_welfare_present(self):
        for name, res in self.analysis['nash'].items():
            self.assertIn('welfare_metrics', res)
            self.assertGreaterEqual(res['welfare_metrics']['deadweight_loss'], 0)

    def test_payoff_exhibit_renders_ne_annotation(self):
        from docx import Document as _Document
        _, gf = solve_framework()
        name = 'Hardware-Cloud Providers'
        res = gf.nash_equilibria[name]
        s1 = build_technical_report.gf_strategy_of(None, None, 'Hardware')
        s2 = build_technical_report.gf_strategy_of(None, None, 'Cloud Providers')
        doc = _Document()
        n_tables = len(doc.tables)
        build_technical_report.add_payoff_exhibit(doc, 1, name, res, s1, s2)
        self.assertEqual(len(doc.tables), n_tables + 1)
        exhibit = doc.tables[-1]
        self.assertEqual(len(exhibit.rows), 3)
        self.assertEqual(len(exhibit.columns), 3)
        self.assertTrue(any('Exhibit 4.1' in p.text for p in doc.paragraphs))
        body_text = ' '.join(c.text for row in exhibit.rows[1:] for c in row.cells[1:])
        # Payoff pairs render as "(a, b)"; the Nash cell carries its tag.
        self.assertIn(',', body_text)
        if res.get('nash_equilibria'):
            self.assertIn('NE', body_text)


class TestSupplementalOutputs(unittest.TestCase):
    """Tables 7.16-7.18 generators and figures/supplemental/ builders in ai_ecosystem_model.py."""

    @staticmethod
    def _fnum(s):
        return float(str(s).strip())

    def test_table_7_16_shares_sum_and_sorted(self):
        import tempfile as _tf
        data = ai_ecosystem_model.load_all_data()
        rp = ai_ecosystem_model.RevenueProjection(data)
        with _tf.TemporaryDirectory() as td:
            tg = ai_ecosystem_model.TableGenerator(output_dir=td)
            df = tg._generate_table_7_16(rp)
        self.assertIsNotNone(df)
        self.assertEqual(list(df.columns), ['Archetype', 'Revenue 2025 ($B)', 'Share 2025 (%)',
                                            'Revenue 2030 ($B)', 'Share 2030 (%)', 'Share Change (pp)'])
        self.assertEqual(len(df), 4)
        shares = [self._fnum(v) for v in df['Share 2030 (%)']]
        self.assertAlmostEqual(sum(shares), 100.0, delta=0.1)
        self.assertEqual(shares, sorted(shares, reverse=True))
        changes = [self._fnum(v) for v in df['Share Change (pp)']]
        self.assertAlmostEqual(sum(changes), 0.0, delta=0.1)

    def test_table_7_17_bands_bracket_base(self):
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            edf = ai_ecosystem_model.ElasticitySensitivityAnalyzer(output_dir=td).calculate_lerner_ranges()
            tg = ai_ecosystem_model.TableGenerator(output_dir=td)
            df = tg._generate_table_7_17(edf)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 4)
        for _, r in df.iterrows():
            base = self._fnum(r['Base Lerner'])
            lo, hi = (self._fnum(v) for v in str(r['Lerner Band']).split('–'))
            self.assertLessEqual(lo, base + 1e-9)
            self.assertLessEqual(base, hi + 1e-9)

    def test_table_7_17_missing_columns_returns_none(self):
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            tg = ai_ecosystem_model.TableGenerator(output_dir=td)
            self.assertIsNone(tg._generate_table_7_17(pd.DataFrame({'a': [1]})))
            self.assertIsNone(tg._generate_table_7_17(None))

    def test_table_7_18_gaps_sum_to_zero(self):
        import tempfile as _tf
        data = ai_ecosystem_model.load_all_data()
        ma = ai_ecosystem_model.MarketStructureAnalyzer(data['players'])
        cg = ai_ecosystem_model.CooperativeGame(data)
        cg.calculate_shapley_values()
        with _tf.TemporaryDirectory() as td:
            tg = ai_ecosystem_model.TableGenerator(output_dir=td)
            df = tg._generate_table_7_18(cg, ma)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 4)
        gaps = [self._fnum(v) for v in df['Gap (pp, Shapley − Revenue)']]
        self.assertAlmostEqual(sum(gaps), 0.0, delta=0.5)
        shapely = [self._fnum(v) for v in df['Shapley Share (%)']]
        self.assertEqual(shapely, sorted(shapely, reverse=True))

    def test_supplemental_none_inputs_empty(self):
        self.assertEqual(ai_ecosystem_model.build_supplemental_figures(), [])

    def test_supplemental_full_and_partial(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data = ai_ecosystem_model.load_all_data()
        rp = ai_ecosystem_model.RevenueProjection(data)
        cg = ai_ecosystem_model.CooperativeGame(data)
        cg.calculate_shapley_values()
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            edf = ai_ecosystem_model.ElasticitySensitivityAnalyzer(output_dir=td).calculate_lerner_ranges()
        dd = pd.DataFrame({
            'Archetype': ['Hardware', 'Cloud Providers', 'Foundation Models', 'LLM Wrappers'],
            'Direct_Loss_Severe_$B': [133.0, 132.7, 67.0, 30.0],
            'Second_Round_Loss_$B': [4.0, 3.0, 2.0, 1.0],
            'Final_Distress': [0.03, 0.03, 0.07, 0.06],
        })
        _, gf = solve_framework()
        t35 = ai_ecosystem_model.ParetoClassifier(ai_ecosystem_model._audit_games(gf)).classify()
        self.assertIn('joint_gain', t35.columns)
        try:
            partial = ai_ecosystem_model.build_supplemental_figures(
                revenue_proj=rp, elasticity_df=edf, coop_analyzer=cg,
                players_df=data['players'], debtrank_df=None)
            self.assertEqual(len(partial), 3)
            full = ai_ecosystem_model.build_supplemental_figures(
                revenue_proj=rp, elasticity_df=edf, coop_analyzer=cg,
                players_df=data['players'], debtrank_df=dd, audit_df=t35)
            self.assertEqual(len(full), 5)
            stems = [s for s, _, _ in full]
            self.assertEqual(stems, [s for s, _ in ai_ecosystem_model.SUPPLEMENTAL_FIGURES])
            for _, _, fig in full:
                self.assertIsInstance(fig, plt.Figure)
            s5 = [f for s, _, f in full if s == 's5_joint_gains_transfers'][0]
            self.assertEqual(len(s5.axes[0].patches), len(t35))
        finally:
            plt.close('all')

    def test_run_log_file_mirrors_and_idempotent(self):
        import logging as _logging
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            root = _logging.getLogger()
            n_before = len(root.handlers)
            path = ai_ecosystem_model.setup_run_log_file(directory=td)
            self.assertIsNotNone(path)
            self.assertTrue(os.path.basename(path).startswith('ai_ecosystem_model_run_'))
            self.assertTrue(path.endswith('.log'))
            self.assertTrue(os.path.exists(path))
            # Second call for the same stamp reuses the handler: no duplicates.
            again = ai_ecosystem_model.setup_run_log_file(directory=td)
            self.assertEqual(again, path)
            try:
                ai_ecosystem_model.logger.info('run-log-probe-12345')
                for h in root.handlers:
                    h.flush()
                with open(path, encoding='utf-8') as f:
                    content = f.read()
                self.assertIn('run-log-probe-12345', content)
            finally:
                for h in [h for h in root.handlers
                          if isinstance(h, _logging.FileHandler)
                          and os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(path)]:
                    root.removeHandler(h)
                    h.close()
            self.assertEqual(len(root.handlers), n_before)

    def test_run_log_file_unwritable_returns_none(self):
        # A file path as the target directory cannot hold a log: must not raise.
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix='.log') as tf:
            self.assertIsNone(ai_ecosystem_model.setup_run_log_file(directory=tf.name))

    def test_polish_data_axes_never_raises(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            ai_ecosystem_model.VisualizationEngine._polish_data_axes(fig)
            self.assertIsNone(ai_ecosystem_model.VisualizationEngine._polish_data_axes(None))
        finally:
            plt.close('all')


class TestAuditTableRobustness(unittest.TestCase):
    """Table 3.7 robustness: non-finite critical deltas and audit edge content."""

    def test_fmt_cell_infinities(self):
        self.assertEqual(ai_ecosystem_model.TableGenerator._fmt_cell(float('inf')), '\u221e')
        self.assertEqual(ai_ecosystem_model.TableGenerator._fmt_cell(float('-inf')), '-\u221e')
        self.assertEqual(ai_ecosystem_model.TableGenerator._fmt_cell(np.inf), '\u221e')
        self.assertEqual(ai_ecosystem_model.TableGenerator._fmt_cell(float('nan')), 'N/A')
        self.assertEqual(ai_ecosystem_model.TableGenerator._fmt_cell(12.3456), '12.35')

    def test_excel_survives_nonfinite_cells(self):
        from openpyxl import load_workbook
        df = pd.DataFrame({'Game': ['A-B', 'C-D'],
                           'Critical_Delta': [0.5, float('inf')],
                           'Note': [float('nan'), 'ok']})
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 't.xlsx')
            self.assertTrue(ai_ecosystem_model.TableGenerator._table_excel(df, p))
            self.assertGreater(os.path.getsize(p), 0)
            wb = load_workbook(p)
            vals = [[c.value for c in row] for row in wb['Table'].iter_rows(min_row=2)]
            flat = [v for row in vals for v in row]
            self.assertIn('\u221e', flat)
            self.assertNotIn(float('inf'), flat)

    def test_save_audit_table_no_warning_and_complete_triple(self):
        import logging as _logging
        df = pd.DataFrame({
            'Game': ['A-B', 'C-D'],
            'Critical_Delta': [0.0, float('inf')],
            'Status': ['no deviation incentive', 'Nash reversion is not a credible punishment'],
        })
        records = []

        class _Cap(_logging.Handler):
            def emit(self, record):
                records.append(record)

        cap = _Cap(level=_logging.WARNING)
        glog = _logging.getLogger('ai_ecosystem_model')
        glog.addHandler(cap)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ai_ecosystem_model.save_audit_table(df, 'table_test_inf_audit', tmp)
                for ext in ('.csv', '.tex', '.xlsx'):
                    path = os.path.join(tmp, 'table_test_inf_audit' + ext)
                    self.assertTrue(os.path.exists(path), path)
                    self.assertGreater(os.path.getsize(path), 0, f'{path} was written empty')
                with open(os.path.join(tmp, 'table_test_inf_audit.tex'), encoding='utf-8') as f:
                    tex = f.read()
                self.assertIn('\u221e', tex)
        finally:
            glog.removeHandler(cap)
        warnings = [r for r in records if r.levelno >= _logging.WARNING]
        self.assertEqual(warnings, [], f"save_audit_table warned: {[r.getMessage() for r in warnings]}")

    def test_fonttools_subsetter_silenced(self):
        import logging as _logging
        # ai_ecosystem_model.py sets fontTools to WARNING at import: PDF subsetting must not
        # flood the run log with thousands of glyph-list INFO lines again.
        self.assertGreaterEqual(_logging.getLogger('fontTools').getEffectiveLevel(), _logging.WARNING)


class TestRunLogFiles(unittest.TestCase):
    """Per-tool run-log helpers: build_technical_report/dashboard session files plus this suite's own log."""

    @staticmethod
    def _detach(path):
        import logging as _logging
        root = _logging.getLogger()
        for h in [h for h in root.handlers
                  if isinstance(h, _logging.FileHandler)
                  and os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(path)]:
            root.removeHandler(h)
            h.close()

    def test_own_session_log_active(self):
        self.assertIsNotNone(TEST_LOG_PATH)
        self.assertTrue(os.path.basename(TEST_LOG_PATH).startswith('test_ai_ecosystem_model_run'))
        self.assertTrue(TEST_LOG_PATH.endswith('.log'))
        self.assertTrue(os.path.exists(TEST_LOG_PATH))

    def test_build_technical_report_run_log_mirrors_and_idempotent(self):
        import logging as _logging
        with tempfile.TemporaryDirectory() as td:
            root = _logging.getLogger()
            n_before = len(root.handlers)
            path = build_technical_report.setup_run_log_file(td)
            try:
                self.assertIsNotNone(path)
                self.assertTrue(os.path.basename(path).startswith('build_technical_report_run'))
                self.assertTrue(os.path.exists(path))
                self.assertEqual(build_technical_report.setup_run_log_file(td), path)
                build_technical_report.logger.info('build_technical_report-log-probe-12345')
                for h in root.handlers:
                    h.flush()
                with open(path, encoding='utf-8') as f:
                    self.assertIn('build_technical_report-log-probe-12345', f.read())
            finally:
                self._detach(path)
            self.assertEqual(len(root.handlers), n_before)
        # A file path as the target directory cannot hold a log: must not raise.
        with tempfile.NamedTemporaryFile(suffix='.log') as tf:
            self.assertIsNone(build_technical_report.setup_run_log_file(tf.name))

    def test_build_interactive_dashboard_run_log_mirrors_and_idempotent(self):
        import logging as _logging
        with tempfile.TemporaryDirectory() as td:
            root = _logging.getLogger()
            n_before = len(root.handlers)
            path = dash.setup_run_log_file(td)
            try:
                self.assertIsNotNone(path)
                self.assertTrue(os.path.basename(path).startswith('build_interactive_dashboard_run'))
                self.assertTrue(os.path.exists(path))
                self.assertEqual(dash.setup_run_log_file(td), path)
                dash.logger.info('dash-log-probe-12345')
                for h in root.handlers:
                    h.flush()
                with open(path, encoding='utf-8') as f:
                    self.assertIn('dash-log-probe-12345', f.read())
            finally:
                self._detach(path)
            self.assertEqual(len(root.handlers), n_before)
        with tempfile.NamedTemporaryFile(suffix='.log') as tf:
            self.assertIsNone(dash.setup_run_log_file(tf.name))

    def test_run_pipeline_run_log_mirrors_and_idempotent(self):
        import logging as _logging
        with tempfile.TemporaryDirectory() as td:
            root = _logging.getLogger()
            n_before = len(root.handlers)
            path = run_pipeline.setup_run_log_file(td)
            try:
                self.assertIsNotNone(path)
                self.assertTrue(os.path.basename(path).startswith('run_pipeline_run'))
                self.assertTrue(os.path.exists(path))
                self.assertEqual(run_pipeline.setup_run_log_file(td), path)
                run_pipeline.logger.info('run_pipeline-log-probe-12345')
                for h in root.handlers:
                    h.flush()
                with open(path, encoding='utf-8') as f:
                    self.assertIn('run_pipeline-log-probe-12345', f.read())
            finally:
                self._detach(path)
            self.assertEqual(len(root.handlers), n_before)
        # A file path as the target directory cannot hold a log: must not raise.
        with tempfile.NamedTemporaryFile(suffix='.log') as tf:
            self.assertIsNone(run_pipeline.setup_run_log_file(tf.name))


class TestTechnicalReportEnhancements(unittest.TestCase):
    """Report bundle test: one real build, then every enhancement asserted.

    Covers the six report fixes: supplemental figures/tables embedded in
    narrative sections, inline exhibits in List of Tables, figure alt text,
    new abbreviations, new references, and the §2.8 register excerpt.
    """

    def test_report_enhancements_bundle(self):
        from docx import Document as _Docx
        from docx.oxml.ns import qn as _qn
        data = ai_ecosystem_model.load_all_data()
        analysis = build_technical_report.analyze(ai_ecosystem_model, data)
        prof = build_technical_report.code_profile(os.path.join(HERE, 'ai_ecosystem_model.py'))
        checks = build_technical_report.integrity_checks(ai_ecosystem_model, data, prof, TABLES, PLOTS)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'bundle.docx')
            result = build_technical_report.build_report(ai_ecosystem_model, analysis, prof, checks, HERE, out)
            self.assertEqual(result.get('issues', []), [])
            d = _Docx(out)
            paras = [p.text for p in d.paragraphs]
            cells = [c.text for t in d.tables for r in t.rows for c in r.cells]
            # Check 1: supplemental figures + new tables embedded with captions.
            for cap in ('Supplemental S1', 'Supplemental S2', 'Supplemental S3',
                        'Supplemental S4', 'Supplemental S5',
                        'Table 7.16', 'Table 7.17', 'Table 7.18'):
                self.assertTrue(any(cap in p for p in paras), f'{cap} caption missing')
            # Check 2: List of Tables covers inline exhibits too. It is the table
            # whose header is exactly Table/Section (tables[0] is the cover
            # meta table, and lof/abbr tables have different headers).
            lots = [t for t in d.tables
                    if [c.text for c in t.rows[0].cells] == ['Table', 'Section']]
            self.assertEqual(len(lots), 1, 'List of Tables not found exactly once')
            lot_cells = [c.text for r in lots[0].rows for c in r.cells]
            for entry in ('Exhibit 3.2', 'Table 3.5', 'Table 7.16', 'Exhibit 7.1'):
                self.assertTrue(any(entry in c for c in lot_cells),
                                f'{entry} missing from List of Tables')
            # Check 3: every embedded figure carries alt text.
            self.assertGreater(len(d.inline_shapes), 0)
            for s in d.inline_shapes:
                docPr = s._inline.find(_qn('wp:docPr'))
                self.assertIsNotNone(docPr)
                self.assertTrue((docPr.get('descr') or '').strip(),
                                'figure without alt text')
            # Check 4: new abbreviations present.
            for abbr in ('K-H', 'MC', 'TTM'):
                self.assertTrue(any(abbr in c for c in cells), f'{abbr} missing')
            # Check 5: previously uncited sources now in References.
            for ref in ('Aldasoro', 'Federal Trade Commission. (2025)',
                        'Merger guidelines'):
                self.assertTrue(any(ref in p for p in paras), f'{ref} missing')
            # Check 6: §2.8 shows the excerpt, full register lives in Appendix A.
            self.assertTrue(any('Table 8.6 (excerpt)' in p for p in paras))
            self.assertTrue(any('Appendix A' in p for p in paras))


class TestReportAndDashboardEnhancements(unittest.TestCase):
    """Report and dashboard enhancement coverage."""

    def test_assemble_missing_tables_fails_loudly(self):
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td, _tf.TemporaryDirectory() as pd_:
            with self.assertRaises(FileNotFoundError) as ctx:
                dash.assemble(td, pd_)
            self.assertIn('run ai_ecosystem_model.py first', str(ctx.exception))

    def test_dashboard_accessibility_markers(self):
        # Tablist/tab/tabpanel roles with selected-state sync, keyboard-sortable
        # headers, labelled filter inputs, and the visually-hidden helper class.
        # NOTE: the tablist role is set via setAttribute (single-quoted JS), so
        # match that form -- 'role="tablist"' with double quotes never appears.
        self.assertIn("setAttribute('role','tablist')", dash.JS_CORE)
        self.assertIn('role="tab"', dash.JS_CORE)
        self.assertIn('aria-selected', dash.JS_CORE)
        self.assertIn('role="tabpanel"', dash.JS_CORE)
        self.assertIn('tabindex="0"', dash.JS_CORE)
        self.assertIn('onkeydown', dash.JS_CORE)
        self.assertIn('aria-label="Filter table rows"', dash.JS_CORE)
        self.assertIn('.vh{', dash.PAGE_HEAD)
        # Integer-px type scale must survive the new CSS (house gate).
        frac = re.findall(r'\d+\.\d+px', dash.PAGE_HEAD + dash.JS_CORE)
        self.assertEqual(frac, [], frac)

    def test_exhibit_3_2_shares_sorted_descending(self):
        # Exhibit 3.2 must sort by numeric share, never by its string display
        # column ("9.xx" would otherwise rank above "46.xx"). Build the real
        # report and read the exhibit back through the dashboard DOCX extractor.
        data = ai_ecosystem_model.load_all_data()
        analysis = build_technical_report.analyze(ai_ecosystem_model, data)
        prof = build_technical_report.code_profile(os.path.join(HERE, 'ai_ecosystem_model.py'))
        checks = build_technical_report.integrity_checks(ai_ecosystem_model, data, prof, TABLES, PLOTS)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'probe.docx')
            result = build_technical_report.build_report(ai_ecosystem_model, analysis, prof, checks, HERE, out)
            self.assertTrue(os.path.exists(out))
            rep = dash.extract_report(out)
            found = None
            for s in rep.get('sections', []):
                for b in s.get('blocks', []):
                    for t in b.get('tables', []):
                        if 'Exhibit 3.2' in (t.get('caption') or ''):
                            found = t
            self.assertIsNotNone(found, 'Exhibit 3.2 missing from built report')
            shares = [float(row[1]) for row in found['rows']]
            self.assertEqual(shares, sorted(shares, reverse=True))
            self.assertAlmostEqual(sum(shares), 100.0, delta=0.05)


class TestPipeline(unittest.TestCase):
    """Unit tests for run_pipeline.py: CLI parsing, helpers, and stage wiring.

    The heavy stages are stubbed here so these stay fast; only run()
    exercises real subprocesses (two one-line interpreter exits). Gates
    against real artifacts live in TestPipelineIntegration below.
    """

    def test_stage_chain_order_locked(self):
        self.assertEqual(run_pipeline.STAGES, ('ai_ecosystem_model', 'build_technical_report', 'dashboard', 'tests',
                                               'build_retirement_exposure', 'pdf',
                                               'build_prediction_enhancement', 'build_crisis_backtest',
                                               'build_compare_2008_ai', 'build_prediction_artifacts'))
        self.assertEqual(run_pipeline.PREDICTION_STAGES, ('build_prediction_enhancement', 'build_crisis_backtest',
                                                          'build_compare_2008_ai', 'build_prediction_artifacts'))
        self.assertEqual(run_pipeline.PREDICTION_STAGES, run_pipeline.STAGES[6:])

    def test_parse_args_defaults(self):
        args = run_pipeline.parse_args([])
        self.assertEqual(args.start, 'ai_ecosystem_model')
        self.assertFalse(args.skip_tests)
        self.assertFalse(args.skip_pdf)

    def test_parse_args_flags(self):
        args = run_pipeline.parse_args(['--from', 'build_technical_report', '--skip-tests', '--skip-pdf'])
        self.assertEqual(args.start, 'build_technical_report')
        self.assertTrue(args.skip_tests)
        self.assertTrue(args.skip_pdf)

    def test_parse_args_rejects_unknown_stage(self):
        with self.assertRaises(SystemExit):
            run_pipeline.parse_args(['--from', 'nope'])

    def test_missing_reports_absent_and_empty(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, 'ok.txt')
            with open(good, 'w', encoding='utf-8') as fh:
                fh.write('x')
            empty = os.path.join(td, 'empty.txt')
            open(empty, 'w').close()
            gone = os.path.join(td, 'gone.txt')
            self.assertEqual(run_pipeline.missing([good]), [])
            bad = run_pipeline.missing([good, empty, gone])
            self.assertEqual(len(bad), 2)
            self.assertTrue(any('empty.txt' in b for b in bad))
            self.assertTrue(any('gone.txt' in b for b in bad))

    def test_run_success_and_failure(self):
        run_pipeline.run([sys.executable, '-c', 'print("run_pipeline-probe-1")'], HERE, 'probe')
        with self.assertRaises(RuntimeError) as ctx:
            run_pipeline.run([sys.executable, '-c', 'import sys; sys.exit(3)'],
                         HERE, 'probe')
        self.assertIn('exit code 3', str(ctx.exception))
        self.assertIn('probe', str(ctx.exception))

    def _run_main_stubbed(self, argv, fail_at=None):
        calls = []
        def rec(name):
            def _f():
                calls.append(name)
                if name == fail_at:
                    raise RuntimeError(f'{name} boom')
            return _f
        with mock.patch.object(run_pipeline, 'setup_run_log_file', return_value=None), \
             mock.patch.object(run_pipeline, 'stage_gt', side_effect=rec('ai_ecosystem_model')), \
             mock.patch.object(run_pipeline, 'stage_rpt', side_effect=rec('build_technical_report')), \
             mock.patch.object(run_pipeline, 'stage_dashboard', side_effect=rec('dashboard')), \
             mock.patch.object(run_pipeline, 'stage_tests', side_effect=rec('tests')), \
             mock.patch.object(run_pipeline, 'stage_retirement', side_effect=rec('build_retirement_exposure')), \
             mock.patch.object(run_pipeline, 'stage_pdf', side_effect=rec('pdf')), \
             mock.patch.object(run_pipeline, 'stage_enhancement', side_effect=rec('build_prediction_enhancement')), \
             mock.patch.object(run_pipeline, 'stage_backtest', side_effect=rec('build_crisis_backtest')), \
             mock.patch.object(run_pipeline, 'stage_compare', side_effect=rec('build_compare_2008_ai')), \
             mock.patch.object(run_pipeline, 'stage_artifacts', side_effect=rec('build_prediction_artifacts')):
            rc = run_pipeline.main(argv)
        return rc, calls

    def test_main_runs_full_chain_in_order(self):
        rc, calls = self._run_main_stubbed([])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, list(run_pipeline.STAGES))

    def test_main_fails_fast_on_stage_error(self):
        rc, calls = self._run_main_stubbed([], fail_at='build_technical_report')
        self.assertEqual(rc, 1)
        self.assertEqual(calls, ['ai_ecosystem_model', 'build_technical_report'])

    def test_main_resume_starts_mid_chain(self):
        rc, calls = self._run_main_stubbed(['--from', 'dashboard'])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['dashboard', 'tests', 'build_retirement_exposure', 'pdf']
                         + list(run_pipeline.PREDICTION_STAGES))

    def test_main_skip_tests(self):
        rc, calls = self._run_main_stubbed(['--skip-tests'])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['ai_ecosystem_model', 'build_technical_report', 'dashboard',
                                 'build_retirement_exposure', 'pdf']
                         + list(run_pipeline.PREDICTION_STAGES))

    def test_main_skip_pdf(self):
        rc, calls = self._run_main_stubbed(['--skip-pdf'])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['ai_ecosystem_model', 'build_technical_report', 'dashboard', 'tests',
                                 'build_retirement_exposure']
                         + list(run_pipeline.PREDICTION_STAGES))

    def test_main_skip_prediction(self):
        rc, calls = self._run_main_stubbed(['--skip-prediction'])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['ai_ecosystem_model', 'build_technical_report', 'dashboard', 'tests',
                                 'build_retirement_exposure', 'pdf'])

    def test_main_resume_into_prediction_block(self):
        rc, calls = self._run_main_stubbed(['--from', 'build_crisis_backtest', '--skip-pdf', '--skip-tests'])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['build_crisis_backtest', 'build_compare_2008_ai', 'build_prediction_artifacts'])

    def test_main_resume_at_compare_verifies_backtest_inputs(self):
        # --from build_compare_2008_ai must pass check_backtest_outputs before running.
        with mock.patch.object(run_pipeline, 'check_artifact_prerequisites', return_value=None):
            rc, calls = self._run_main_stubbed(['--from', 'build_compare_2008_ai'])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ['build_compare_2008_ai', 'build_prediction_artifacts'])

    def test_parse_args_skip_prediction_default_off(self):
        self.assertFalse(run_pipeline.parse_args([]).skip_prediction)
        self.assertTrue(run_pipeline.parse_args(['--skip-prediction']).skip_prediction)
        args = run_pipeline.parse_args(['--from', 'build_compare_2008_ai'])
        self.assertEqual(args.start, 'build_compare_2008_ai')

    def test_main_resume_verifies_gt_outputs_first(self):
        # --from build_technical_report must refuse stale tables before running anything.
        with mock.patch.object(run_pipeline, 'setup_run_log_file', return_value=None), \
             mock.patch.object(run_pipeline, 'check_gt_outputs',
                               side_effect=RuntimeError('ai_ecosystem_model.py outputs incomplete')):
            rc = run_pipeline.main(['--from', 'build_technical_report', '--skip-pdf', '--skip-tests'])
        self.assertEqual(rc, 1)

    def test_check_gt_outputs_fails_loudly_on_empty_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(run_pipeline, 'TABLES', td), \
                 mock.patch.object(run_pipeline, 'PLOTS', td), \
                 mock.patch.object(run_pipeline, 'SUPP', td):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.check_gt_outputs()
        msg = str(ctx.exception)
        self.assertIn('table_3.5_welfare_benchmark_audit.csv', msg)
        self.assertIn('figure_1.png', msg)
        self.assertIn('s5_joint_gains_transfers.png', msg)

    def test_enhancement_verified_set_matches_produced_set(self):
        # The pipeline must verify every pred_* exhibit the stage writes:
        # a builder-side addition without a gate entry fails here.
        import glob as _glob
        on_disk = sorted(os.path.basename(p)
                         for p in _glob.glob(os.path.join(PLOTS, 'pred_*.png')))
        self.assertEqual(on_disk,
                         [f'pred_{s}.png' for s in run_pipeline.PRED_FIGURES])

    def test_check_enhancement_outputs_fails_loudly_on_empty_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(run_pipeline, 'PLOTS', td):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.check_enhancement_outputs()
        self.assertIn('pred_valuation_scatter.png', str(ctx.exception))

    def test_check_backtest_outputs_fails_loudly_on_empty_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            bt = os.path.join(td, 'backtest')
            os.makedirs(bt)
            with mock.patch.object(run_pipeline, 'BACKTEST_T', bt), \
                 mock.patch.object(run_pipeline, 'PLOTS', td):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.check_backtest_outputs()
        msg = str(ctx.exception)
        self.assertIn('backtest_verdict.csv', msg)
        self.assertIn('backtest_2008.png', msg)

    def test_check_retirement_outputs_fails_loudly_on_empty_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            bt = os.path.join(td, 'backtest')
            os.makedirs(bt)
            with mock.patch.object(run_pipeline, 'BACKTEST_T', bt), \
                 mock.patch.object(run_pipeline, 'PLOTS', td):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.check_retirement_outputs()
        msg = str(ctx.exception)
        self.assertIn('retirement_exposure.csv', msg)
        self.assertIn('retirement_exposure.png', msg)

    def test_check_retirement_outputs_passes_on_real_workspace(self):
        # The committed run_pipeline left both retirement artifacts on disk.
        run_pipeline.check_retirement_outputs()

    def test_check_compare_outputs_fails_loudly_on_empty_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(run_pipeline, 'TABLES', td), \
                 mock.patch.object(run_pipeline, 'PLOTS', td):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.check_compare_outputs()
        msg = str(ctx.exception)
        self.assertIn('compare_summary.csv', msg)
        self.assertIn('compare_rank_shape.png', msg)

    def test_check_artifact_prerequisites_names_missing_pandoc(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch('run_pipeline.shutil.which', return_value=None), \
                 mock.patch.object(run_pipeline, 'HERE', td), \
                 mock.patch.object(run_pipeline, 'ARTICLE', td), \
                 mock.patch.object(run_pipeline, 'PLOTS', td):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.check_artifact_prerequisites()
        self.assertIn('pandoc', str(ctx.exception))

    def test_check_artifact_prerequisites_names_missing_refdoc(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch('run_pipeline.shutil.which', return_value='/usr/bin/pandoc'), \
                 mock.patch.object(run_pipeline, 'HERE', td), \
                 mock.patch.object(run_pipeline, 'ARTICLE', td), \
                 mock.patch.object(run_pipeline, 'PLOTS', td):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.check_artifact_prerequisites()
        self.assertIn(run_pipeline.DOCX_NAME, str(ctx.exception))

    def test_check_pdf_prerequisites_names_missing_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(run_pipeline, 'ARTICLE', td), \
                 mock.patch.object(run_pipeline, 'PLOTS', td), \
                 mock.patch.object(run_pipeline, 'SUPP', td):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.check_pdf_prerequisites()
        self.assertIn('run the ai_ecosystem_model stage first', str(ctx.exception))

    def test_check_pdf_prerequisites_names_missing_compiler(self):
        with tempfile.TemporaryDirectory() as td:
            art = os.path.join(td, 'article')
            figs = os.path.join(td, 'figures')
            supp = os.path.join(figs, 'supplemental')
            os.makedirs(art)
            os.makedirs(supp)
            for d, f in ((art, run_pipeline.TEX_NAME),
                         (art, run_pipeline.GENERATED_TEX),
                         (figs, 'figure_3.png'),
                         (figs, 'figure_7.png'),
                         (figs, 'retirement_exposure.png'),
                         (supp, 's5_joint_gains_transfers.png')):
                with open(os.path.join(d, f), 'w', encoding='utf-8') as fh:
                    fh.write('x')
            with mock.patch.object(run_pipeline, 'ARTICLE', art), \
                 mock.patch.object(run_pipeline, 'PLOTS', figs), \
                 mock.patch.object(run_pipeline, 'SUPP', supp), \
                 mock.patch('run_pipeline.shutil.which', return_value=None):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.check_pdf_prerequisites()
        self.assertIn('pdflatex', str(ctx.exception))

    def test_run_clears_stale_bytecode_cache(self):
        with mock.patch.object(run_pipeline.shutil, 'rmtree') as rm:
            run_pipeline.run([sys.executable, '-c', 'pass'], HERE, 'probe')
        rm.assert_called_once_with(os.path.join(HERE, '__pycache__'),
                                   ignore_errors=True)

    def test_run_skips_cache_clear_for_non_python(self):
        with mock.patch.object(run_pipeline.shutil, 'rmtree') as rm:
            run_pipeline.run(['true'], HERE, 'probe')
        rm.assert_not_called()

    def test_ensure_console_adds_handler_when_none(self):
        import logging as _logging
        root = _logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        for h in saved_handlers:
            root.removeHandler(h)
        try:
            self.assertEqual(root.handlers, [])
            run_pipeline.ensure_console()
            self.assertEqual(len(root.handlers), 1)
            self.assertIsInstance(root.handlers[0], _logging.StreamHandler)
            self.assertEqual(root.level, _logging.INFO)
        finally:
            for h in root.handlers[:]:
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
            for h in saved_handlers:
                root.addHandler(h)
            root.setLevel(saved_level)

    def test_clear_bytecode_cache_removes_dir(self):
        with tempfile.TemporaryDirectory() as td:
            cache = os.path.join(td, '__pycache__')
            os.makedirs(cache)
            open(os.path.join(cache, 'stale.pyc'), 'w').close()
            run_pipeline.clear_bytecode_cache(td)
            self.assertFalse(os.path.exists(cache))
            run_pipeline.clear_bytecode_cache(td)  # missing dir: no-op, never raises

    def test_setup_run_log_file_idempotent_same_stamp(self):
        import logging as _logging
        from datetime import datetime as _dt
        frozen = _dt.now().replace(microsecond=0)

        class _FrozenDatetime:
            @staticmethod
            def now():
                return frozen

        with tempfile.TemporaryDirectory() as td:
            root = _logging.getLogger()
            n_before = len(root.handlers)
            with mock.patch.object(run_pipeline, 'datetime', _FrozenDatetime):
                first = run_pipeline.setup_run_log_file(td)
                n_mid = len(root.handlers)
                second = run_pipeline.setup_run_log_file(td)
            try:
                self.assertIsNotNone(first)
                self.assertEqual(first, second)
                self.assertEqual(len(root.handlers), n_mid)
            finally:
                for h in [h for h in root.handlers
                          if isinstance(h, _logging.FileHandler)
                          and os.path.abspath(getattr(h, 'baseFilename', ''))
                          == os.path.abspath(first)]:
                    root.removeHandler(h)
                    h.close()
            self.assertEqual(len(root.handlers), n_before)

    def test_setup_run_log_file_disambiguates_same_second(self):
        # Same wall-clock second as an unowned log must suffix, not
        # truncate: two in-process mains in one second shared one filename
        # and broke the new-log assertion downstream (Sep 2026 flake).
        import logging as _logging
        from datetime import datetime as _dt
        frozen = _dt.now().replace(microsecond=0)

        class _FrozenDatetime:
            @staticmethod
            def now():
                return frozen

        with tempfile.TemporaryDirectory() as td:
            root = _logging.getLogger()
            with mock.patch.object(run_pipeline, 'datetime', _FrozenDatetime):
                first = run_pipeline.setup_run_log_file(td)
                for h in [h for h in root.handlers
                          if isinstance(h, _logging.FileHandler)
                          and os.path.abspath(getattr(h, 'baseFilename', ''))
                          == os.path.abspath(first)]:
                    root.removeHandler(h)
                    h.close()
                second = run_pipeline.setup_run_log_file(td)
            try:
                self.assertIsNotNone(first)
                self.assertIsNotNone(second)
                self.assertNotEqual(first, second)
                self.assertTrue(os.path.exists(first))
                self.assertTrue(os.path.exists(second))
            finally:
                for h in [h for h in root.handlers
                          if isinstance(h, _logging.FileHandler)
                          and os.path.abspath(getattr(h, 'baseFilename', ''))
                          in (os.path.abspath(first), os.path.abspath(second))]:
                    root.removeHandler(h)
                    h.close()

    def test_stage_gt_invokes_gt_and_verifies(self):
        with mock.patch.object(run_pipeline, 'run') as run, \
             mock.patch.object(run_pipeline, 'check_gt_outputs') as chk:
            run_pipeline.stage_gt()
        run.assert_called_once_with(
            [sys.executable, os.path.join(run_pipeline.HERE, 'ai_ecosystem_model.py')],
            run_pipeline.HERE, 'ai_ecosystem_model')
        chk.assert_called_once_with()

    def test_stage_rpt_pass_and_missing(self):
        with mock.patch.object(run_pipeline, 'run'), \
             mock.patch.object(run_pipeline, 'missing', return_value=[]), \
             mock.patch('os.path.getsize', return_value=2048):
            run_pipeline.stage_rpt()
        with mock.patch.object(run_pipeline, 'run'), \
             mock.patch.object(run_pipeline, 'missing', return_value=['r.docx']):
            with self.assertRaises(RuntimeError) as ctx:
                run_pipeline.stage_rpt()
        self.assertIn('report is missing', str(ctx.exception))

    def test_stage_dashboard_pass_and_missing(self):
        with mock.patch.object(run_pipeline, 'run'), \
             mock.patch.object(run_pipeline, 'missing', return_value=[]), \
             mock.patch('os.path.getsize', return_value=1024):
            run_pipeline.stage_dashboard()
        with mock.patch.object(run_pipeline, 'run'), \
             mock.patch.object(run_pipeline, 'missing', return_value=['d.html']):
            with self.assertRaises(RuntimeError) as ctx:
                run_pipeline.stage_dashboard()
        self.assertIn('dashboard.html is missing', str(ctx.exception))

    def test_stage_tests_invokes_suite(self):
        with mock.patch.object(run_pipeline, 'run') as run:
            run_pipeline.stage_tests()
        run.assert_called_once_with(
            [sys.executable, '-m', 'unittest', 'test_ai_ecosystem_model', '-v'],
            run_pipeline.HERE, 'tests')

    def test_stage_glue_runs_checks_against_real_outputs(self):
        # Builder subprocesses mocked out: each stage still runs its
        # output check against the real workspace artifacts.
        with mock.patch.object(run_pipeline, 'run'):
            run_pipeline.stage_enhancement()
            run_pipeline.stage_backtest()
            run_pipeline.stage_compare()
            run_pipeline.stage_artifacts()

    def test_main_returns_one_on_stage_failure(self):
        # A raising stage surfaces as exit code 1, not a traceback.
        # Detaches main()'s session-log handler afterwards: leaking it
        # lets a later same-second setup call reuse this run's filename
        # instead of writing its own log (Sep 2026 flake).
        import logging as _logging
        root = _logging.getLogger()
        before = set(root.handlers)
        try:
            with mock.patch.object(run_pipeline, 'stage_dashboard',
                                   side_effect=RuntimeError('boom')):
                rc = run_pipeline.main(['--from', 'dashboard', '--skip-pdf',
                                        '--skip-tests', '--skip-prediction'])
        finally:
            for h in [h for h in root.handlers if h not in before]:
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
        self.assertEqual(rc, 1)

    @staticmethod
    def _pdf_env(td):
        art = os.path.join(td, 'article')
        figs = os.path.join(td, 'figures')
        supp = os.path.join(figs, 'supplemental')
        os.makedirs(supp)
        os.makedirs(art, exist_ok=True)
        for d, f in ((art, run_pipeline.TEX_NAME),
                     (art, run_pipeline.GENERATED_TEX),
                     (figs, 'figure_3.png'),
                     (figs, 'figure_7.png'),
                     (figs, 'retirement_exposure.png'),
                     (supp, 's5_joint_gains_transfers.png')):
            with open(os.path.join(d, f), 'w', encoding='utf-8') as fh:
                fh.write('x')
        return art, figs, supp

    def test_stage_pdf_no_pdf_produced(self):
        with tempfile.TemporaryDirectory() as td:
            art, figs, supp = self._pdf_env(td)
            with mock.patch.object(run_pipeline, 'ARTICLE', art), \
                 mock.patch.object(run_pipeline, 'PLOTS', figs), \
                 mock.patch.object(run_pipeline, 'SUPP', supp), \
                 mock.patch('run_pipeline.shutil.which', return_value='/usr/bin/pdflatex'), \
                 mock.patch.object(run_pipeline, 'run') as run:
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.stage_pdf()
        self.assertIn('no PDF was produced', str(ctx.exception))
        self.assertEqual(run.call_count, 2)
        for c in run.call_args_list:
            self.assertEqual(c.args[0][0], 'pdflatex')
            self.assertEqual(c.args[1], art)
            self.assertIn(c.args[2], ('pdflatex-1', 'pdflatex-2'))

    def test_stage_pdf_log_error_fails(self):
        with tempfile.TemporaryDirectory() as td:
            art, figs, supp = self._pdf_env(td)
            with open(os.path.join(art, run_pipeline.PDF_NAME), 'w', encoding='utf-8') as fh:
                fh.write('x')
            with open(os.path.join(art, 'ai_circularity_article.log'),
                      'w', encoding='utf-8') as fh:
                fh.write('! Fatal TeX error\n')
            with mock.patch.object(run_pipeline, 'ARTICLE', art), \
                 mock.patch.object(run_pipeline, 'PLOTS', figs), \
                 mock.patch.object(run_pipeline, 'SUPP', supp), \
                 mock.patch('run_pipeline.shutil.which', return_value='/usr/bin/pdflatex'), \
                 mock.patch.object(run_pipeline, 'run'):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.stage_pdf()
        self.assertIn('Fatal TeX error', str(ctx.exception))

    def test_stage_pdf_log_warnings_fail(self):
        with tempfile.TemporaryDirectory() as td:
            art, figs, supp = self._pdf_env(td)
            with open(os.path.join(art, run_pipeline.PDF_NAME), 'w', encoding='utf-8') as fh:
                fh.write('x')
            with open(os.path.join(art, 'ai_circularity_article.log'),
                      'w', encoding='utf-8') as fh:
                fh.write('LaTeX Warning: There were undefined references.\n'
                         'LaTeX Warning: Citation `foo` undefined on page 1.\n'
                         'This is a normal line.\n')
            with mock.patch.object(run_pipeline, 'ARTICLE', art), \
                 mock.patch.object(run_pipeline, 'PLOTS', figs), \
                 mock.patch.object(run_pipeline, 'SUPP', supp), \
                 mock.patch('run_pipeline.shutil.which', return_value='/usr/bin/pdflatex'), \
                 mock.patch.object(run_pipeline, 'run'):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.stage_pdf()
        self.assertIn('undefined references', str(ctx.exception))
        self.assertIn('Citation', str(ctx.exception))

    def test_stage_pdf_log_unreadable(self):
        with tempfile.TemporaryDirectory() as td:
            art, figs, supp = self._pdf_env(td)
            with open(os.path.join(art, run_pipeline.PDF_NAME), 'w', encoding='utf-8') as fh:
                fh.write('x')
            with mock.patch.object(run_pipeline, 'ARTICLE', art), \
                 mock.patch.object(run_pipeline, 'PLOTS', figs), \
                 mock.patch.object(run_pipeline, 'SUPP', supp), \
                 mock.patch('run_pipeline.shutil.which', return_value='/usr/bin/pdflatex'), \
                 mock.patch.object(run_pipeline, 'run'), \
                 mock.patch('builtins.open', side_effect=OSError('denied')):
                with self.assertRaises(RuntimeError) as ctx:
                    run_pipeline.stage_pdf()
        self.assertIn('cannot read TeX log', str(ctx.exception))

    def test_stage_pdf_clean_passes(self):
        with tempfile.TemporaryDirectory() as td:
            art, figs, supp = self._pdf_env(td)
            with open(os.path.join(art, run_pipeline.PDF_NAME), 'w', encoding='utf-8') as fh:
                fh.write('x')
            with open(os.path.join(art, 'ai_circularity_article.log'),
                      'w', encoding='utf-8') as fh:
                fh.write('This is a normal line.\n')
            with mock.patch.object(run_pipeline, 'ARTICLE', art), \
                 mock.patch.object(run_pipeline, 'PLOTS', figs), \
                 mock.patch.object(run_pipeline, 'SUPP', supp), \
                 mock.patch('run_pipeline.shutil.which', return_value='/usr/bin/pdflatex'), \
                 mock.patch.object(run_pipeline, 'run'):
                self.assertIsNone(run_pipeline.stage_pdf())


class TestPlacementOrder(unittest.TestCase):
    """Placement walk: labels match paragraph text exactly, never doubled.

    The walk must emit each caption verbatim -- even for pathological
    back-to-back duplicate caption+table pairs (explicit + registry
    double-embedding).
    """

    @staticmethod
    def _doc_with_duplicates():
        from docx import Document as _Docx
        doc = _Docx()
        doc.add_heading('1.  Probe chapter', level=1)
        doc.add_paragraph('Table 9.9: Probe table')
        doc.add_table(rows=1, cols=1)
        # Pathological: same caption+table twice in a row.
        doc.add_paragraph('Table 9.9: Probe table')
        doc.add_table(rows=1, cols=1)
        doc.add_paragraph('Exhibit 9.1  Lone exhibit')
        doc.add_table(rows=1, cols=1)
        return doc

    def test_walk_labels_are_single_and_ordered(self):
        doc = self._doc_with_duplicates()
        ordered = build_technical_report.complete_placement_order(doc, [])
        labels = [e['label'] for e in ordered]
        # One entry per caption+table pair; every label verbatim, never doubled.
        self.assertEqual(labels, ['Table 9.9: Probe table', 'Table 9.9: Probe table',
                                  'Exhibit 9.1  Lone exhibit'])
        for e in ordered:
            self.assertEqual(e['section'], 1)
            self.assertEqual(e['kind'], 'table')

    def test_walk_adopts_logged_entry_at_position(self):
        doc = self._doc_with_duplicates()
        logged = {'kind': 'table', 'ok': True, 'file': 'table_9.9.csv',
                  'label': 'Table 9.9: Probe table', 'section': 1}
        ordered = build_technical_report.complete_placement_order(doc, [logged])
        self.assertIs(ordered[0], logged)
        self.assertEqual([e['label'] for e in ordered],
                         ['Table 9.9: Probe table', 'Table 9.9: Probe table',
                          'Exhibit 9.1  Lone exhibit'])

    def test_walk_adopts_appendix_numbered_captions(self):
        # Table A.x captions lead with a letter, so the caption pattern must
        # accept letters as well as digits and dots.
        from docx import Document as _Docx
        doc = _Docx()
        doc.add_heading('7.  Probe chapter', level=1)
        doc.add_paragraph('Table A.0: Probe appendix table')
        doc.add_table(rows=1, cols=1)
        ordered = build_technical_report.complete_placement_order(doc, [])
        self.assertEqual([e['label'] for e in ordered],
                         ['Table A.0: Probe appendix table'])
        self.assertEqual(ordered[0]['section'], 7)

    def test_verify_reports_closest_paragraph(self):
        from docx import Document as _Docx
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'probe.docx')
            doc = _Docx()
            doc.add_paragraph('Table 9.9: Probe table')
            doc.save(out)
            issues = build_technical_report.verify_placement(
                out, [{'kind': 'table', 'ok': True, 'file': '',
                       'label': 'Table 9.9: Probe tableTable 9.9: Probe table',
                       'section': None}], td)
        self.assertEqual(len(issues), 1)
        self.assertIn('closest paragraph', issues[0])
        self.assertIn('Table 9.9: Probe table', issues[0])


class TestPipelineIntegration(unittest.TestCase):
    """Entire-run_pipeline gates against real artifacts (needs a ai_ecosystem_model.py run).

    Dispatch order and failure paths are stubbed in TestPipeline; here the
    gates run for real, plus one live stage (dashboard) driven through
    run_pipeline.main to prove the wiring holds end to end without the
    multi-minute ai_ecosystem_model/build_technical_report/suite stages.
    """

    def test_check_gt_outputs_passes_on_real_artifacts(self):
        self.assertIsNone(run_pipeline.check_gt_outputs())

    def test_check_pdf_prerequisites_passes_on_real_inputs(self):
        with mock.patch('run_pipeline.shutil.which', return_value='/usr/bin/pdflatex'):
            self.assertIsNone(run_pipeline.check_pdf_prerequisites())

    def test_all_stage_modules_importable(self):
        # Every pipeline stage module imports cleanly (catches syntax and
        # import-time errors in files the stubbed wiring tests never load).
        for mod in (retmod, btmod, cmpmod, artmod):
            self.assertTrue(hasattr(mod, 'main'), mod.__name__)

    def test_stage_enhancement_main_end_to_end(self):
        # Live run: regenerates all pred_* ledgers and exhibits
        # deterministically (fixed seeds, pure OLS) and must leave the
        # headline new-functionality artifacts on disk.
        enh.main()
        for f in ('pred_G_burst_timing.csv', 'pred_H_company_losses.csv',
                  'pred_I_price_gauges.csv', 'pred_J_gauge_backtest.csv',
                  'pred_tables_index.csv'):
            p = os.path.join(TABLES, f)
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 0, p)
        for f in ('pred_timing_window.png', 'pred_company_losses.png',
                  'pred_gauge_backtest.png', 'pred_valuation_scatter.png'):
            p = os.path.join(PLOTS, f)
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 0, p)

    def test_stage_retirement_main_end_to_end(self):
        retmod.main()
        for f in (os.path.join(TABLES, 'backtest', 'retirement_exposure.csv'),
                  os.path.join(PLOTS, 'retirement_exposure.png')):
            self.assertTrue(os.path.exists(f), f)
            self.assertGreater(os.path.getsize(f), 0, f)

    def test_stage_compare_main_end_to_end(self):
        cmpmod.main()
        for f in ('compare_summary.csv',):
            p = os.path.join(TABLES, f)
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 0, p)
        for f in ('compare_rank_shape.png', 'compare_remedies.png',
                  'compare_tail_risk.png'):
            p = os.path.join(PLOTS, f)
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 0, p)

    def test_stage_backtest_main_end_to_end(self):
        # Live run (~30 s, fully seeded): both verdict ledgers plus the
        # telecom figure set must land on disk.
        btmod.main()
        for f in ('telecom_verdict.csv', 'valuation_2008.csv'):
            p = os.path.join(TABLES, 'backtest', f)
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 0, p)
        for f in ('backtest_2008.png', 'telecom_rank.png',
                  'payoff_matrices_telecom.png'):
            p = os.path.join(PLOTS, f)
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 0, p)

    def test_stage_artifacts_main_end_to_end(self):
        # Live run through pandoc: the prediction DOCX + standalone HTML
        # must land on disk (needs the article PDF + pred_* figures first).
        artmod.main()
        for f in ('AI_Bubble_Burst_Prediction_Report.docx', 'prediction.html'):
            p = os.path.join(HERE, f)
            self.assertTrue(os.path.exists(p), p)
            self.assertGreater(os.path.getsize(p), 0, p)

    def test_artifacts_citation_helpers_unit(self):
        # apa_short spacing fix; rewrite_citations maps known keys and
        # passes unknown ones through on both cite commands.
        self.assertEqual(artmod.apa_short('Agrawal et al.(2019)'), 'Agrawal et al. (2019)')
        self.assertEqual(artmod.apa_short('NoYear'), 'NoYear')
        src = ('\\bibitem[Agrawal et al.(2019)]{agrawal2019}\n'
               '\\citet{agrawal2019} say boom; multi \\citep{agrawal2019, unknown2020} ok.')
        out, keys = artmod.rewrite_citations(src)
        self.assertEqual(keys, {'agrawal2019': 'Agrawal et al. (2019)'})
        self.assertIn('Agrawal et al. (2019) say boom', out)
        self.assertIn('(Agrawal et al. (2019); unknown2020)', out)

    def test_artifacts_run_failure_raises(self):
        # The pandoc wrapper surfaces failures as SystemExit, not silence.
        with self.assertRaises(SystemExit):
            artmod.run([sys.executable, '-c', 'import sys; sys.exit(1)'])

    def test_organize_docx_parts_branch_unit(self):
        # Synthetic docx: Heading 3 sections demote, parts keep H1 with a
        # page break each, and the footer gains page-number runs.
        import docx as _docx
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 't.docx')
            doc = _docx.Document()
            doc.add_paragraph('Part One', style='Heading 1')
            doc.add_paragraph('A section', style='Heading 3')
            doc.save(p)
            artmod.organize_docx(p, has_parts=True)
            back = _docx.Document(p)
            names = [q.style.name for q in back.paragraphs if q.text.strip()]
            self.assertNotIn('Heading 3', names)
            self.assertIn('Heading 1', names)
            self.assertIn('Heading 2', names)
            foot = back.sections[0].footer.paragraphs[0]
            self.assertIn('PAGE', foot._p.xml)

    def test_organize_html_branches_unit(self):
        # Flat layout demotes h1 sections (title-class kept) and wraps
        # appendix h2s; parts layout adds banners with grouped TOC links.
        from bs4 import BeautifulSoup as _Soup
        with tempfile.TemporaryDirectory() as td:
            flat = os.path.join(td, 'flat.html')
            open(flat, 'w', encoding='utf-8').write(
                '<html><body><nav id="TOC"></nav>'
                '<h1 class="title">T</h1>'
                '<h1 id="sec:howtoread">How</h1>'
                '<h2 id="sec:appendix-x">App X</h2><p>body</p>'
                '</body></html>')
            artmod.organize_html(flat)
            soup = _Soup(open(flat, encoding='utf-8'), 'html.parser')
            by_id = {h.get('id'): h.name for h in soup.find_all(['h1', 'h2', 'h3'])}
            self.assertEqual(by_id.get('sec:howtoread'), 'h3')
            self.assertIsNotNone(soup.find('details', {'class': 'appendix'}))
            parts = os.path.join(td, 'parts.html')
            open(parts, 'w', encoding='utf-8').write(
                '<html><body><nav id="TOC"></nav>'
                '<h1 id="part:one">One</h1>'
                '<h2 id="s1">S1</h2><p>x</p>'
                '<h1 id="part:two">Two</h1></body></html>')
            artmod.organize_html(parts)
            soup = _Soup(open(parts, encoding='utf-8'), 'html.parser')
            banners = soup.find_all('h1', {'class': 'part'})
            self.assertEqual(len(banners), 2)
            self.assertTrue(banners[0].find('span', {'class': 'part-eyebrow'}))
            links = soup.find('nav', id='TOC').find_all('a')
            self.assertGreaterEqual(len(links), 3)

    def test_pipeline_dashboard_stage_end_to_end(self):
        import glob as _glob
        import logging as _logging
        root = _logging.getLogger()
        before = set(root.handlers)
        logs_before = set(_glob.glob(os.path.join(HERE, 'run_pipeline_run_*.log')))
        try:
            rc = run_pipeline.main(['--from', 'dashboard', '--skip-pdf', '--skip-tests',
                                    '--skip-prediction'])
        finally:
            for h in [h for h in root.handlers if h not in before]:
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
        self.assertEqual(rc, 0)
        dash_out = os.path.join(HERE, 'dashboard.html')
        self.assertTrue(os.path.exists(dash_out))
        self.assertGreater(os.path.getsize(dash_out), 0)
        logs_after = set(_glob.glob(os.path.join(HERE, 'run_pipeline_run_*.log')))
        self.assertTrue(logs_after - logs_before, 'run_pipeline session log missing')


class TestRptMain(unittest.TestCase):
    """build_technical_report.main driver: CLI guards, mocked stages, placement-gate exits."""

    @staticmethod
    def _checks():
        return [{'status': 'PASS', 'code': 'D1', 'title': 't', 'detail': 'd'},
                {'status': 'FAIL', 'code': 'X1', 'title': 't', 'detail': 'd'},
                {'status': 'WARN', 'code': 'X2', 'title': 't', 'detail': 'd'}]

    def _run_main(self, argv, out, issues, real_log=True):
        """Runs build_technical_report.main with every heavy stage stubbed; returns its outcome."""
        gt_stub = mock.Mock()
        gt_stub.load_all_data.return_value = {}
        log_patch = (contextlib.nullcontext()
                     if real_log
                     else mock.patch.object(build_technical_report, 'setup_run_log_file',
                                            return_value=None))
        with mock.patch.object(sys, 'argv', ['build_technical_report.py', *argv]), \
             log_patch, \
             mock.patch.object(build_technical_report, 'load_gt', return_value=gt_stub), \
             mock.patch.object(build_technical_report, 'analyze',
                               return_value={'nash': {'g': 1},
                                             'shapley': {'a': 1, 'b': 2}}), \
             mock.patch.object(build_technical_report, 'code_profile', return_value={}), \
             mock.patch.object(build_technical_report, 'integrity_checks',
                               return_value=self._checks()), \
             mock.patch.object(build_technical_report, 'build_report',
                               return_value={'path': out, 'placement': [1],
                                             'issues': issues}):
            return build_technical_report.main()

    def test_main_missing_gt_dir_exits(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'r.docx')
            with mock.patch.object(sys, 'argv',
                                   ['build_technical_report.py', '--ai_ecosystem_model-dir', td, '--out', out]):
                with self.assertRaises(SystemExit) as ctx:
                    build_technical_report.main()
        self.assertIn('ai_ecosystem_model.py', str(ctx.exception))

    def test_load_gt_quiet_import(self):
        with tempfile.TemporaryDirectory() as td:
            mod = build_technical_report.load_gt(td)
        self.assertTrue(hasattr(mod, 'load_all_data'))
        self.assertIs(mod, ai_ecosystem_model)

    def test_main_success_warns_on_missing_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, 'ai_ecosystem_model.py'), 'w').close()
            out = os.path.join(td, 'r.docx')
            open(out, 'w').close()
            self.assertIsNone(
                self._run_main(['--ai_ecosystem_model-dir', td, '--out', out], out, []))

    def test_main_success_present_dirs_no_log(self):
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, 'ai_ecosystem_model.py'), 'w').close()
            os.makedirs(os.path.join(td, 'tables'))
            os.makedirs(os.path.join(td, 'figures'))
            out = os.path.join(td, 'r.docx')
            open(out, 'w').close()
            self.assertIsNone(
                self._run_main(['--ai_ecosystem_model-dir', td, '--out', out], out, [],
                               real_log=False))

    def test_main_issues_exit_one(self):
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, 'ai_ecosystem_model.py'), 'w').close()
            os.makedirs(os.path.join(td, 'tables'))
            os.makedirs(os.path.join(td, 'figures'))
            out = os.path.join(td, 'r.docx')
            open(out, 'w').close()
            with self.assertRaises(SystemExit) as ctx:
                self._run_main(['--ai_ecosystem_model-dir', td, '--out', out], out,
                               ['boom'], real_log=False)
        self.assertEqual(ctx.exception.code, 1)


class TestDashMain(unittest.TestCase):
    """build_interactive_dashboard.main: payload build, report present/absent, log on/off."""

    @staticmethod
    def _payload(sections=True):
        return {'meta': {'n_tables': 2}, 'games': [{'id': 'g'}],
                'figures': [{'f': 1}],
                'walkthrough': {'modules': [{'m': 1}]},
                'report': ({'sections': [{'s': 1}]} if sections
                           else {'absent': 'no docx'})}

    def test_main_writes_dashboard_with_report(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'd.html')
            with mock.patch.object(sys, 'argv',
                                   ['build_interactive_dashboard.py', '--out', out]), \
                 mock.patch.object(dash, 'assemble',
                                   return_value=self._payload(True)):
                self.assertIsNone(dash.main())
            with open(out, encoding='utf-8') as f:
                html = f.read()
            self.assertIn('Interactive Dashboard', html)
            self.assertIn('"n_tables": 2', html)

    def test_main_report_absent_and_no_log(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'd.html')
            with mock.patch.object(sys, 'argv',
                                   ['build_interactive_dashboard.py', '--out', out]), \
                 mock.patch.object(dash, 'assemble',
                                   return_value=self._payload(False)), \
                 mock.patch.object(dash, 'setup_run_log_file',
                                   return_value=None):
                self.assertIsNone(dash.main())
            self.assertGreater(os.path.getsize(out), 0)


class TestGtMain(unittest.TestCase):
    """ai_ecosystem_model entry points: self-test, full driver, gate failure, optional imports."""

    def test_run_data_self_test_passes(self):
        self.assertIsNone(ai_ecosystem_model.run_data_self_test())

    def test_gt_main_end_to_end(self):
        # Slow by design (~2 min): the only honest coverage of ai_ecosystem_model.main's
        # pass path is a real run writing real outputs, exactly as the
        # run_pipeline's ai_ecosystem_model stage does.
        result = ai_ecosystem_model.main()
        self.assertTrue(result['tests_passed'])
        # 29 tables: header + 29 rows in tables/tables_index.csv (26 legacy +
        # 6.4 BCR break-even, 7.1b within-layer HHI, A.4 layer exposure).
        self.assertEqual(result['tables_generated'], 29)
        self.assertGreater(result['figures_generated'], 10)

    def test_gt_main_gate_failure_raises(self):
        bad = {'gates': [{'code': 'G1', 'status': 'PASS', 'detail': 'ok'},
                          {'code': 'G2', 'status': 'FAIL', 'detail': 'boom'}],
               'pareto_summary': {'n_kaldor_hicks': 0, 'n_games': 6,
                                  'total_required_transfers_billions': 0.0}}
        with mock.patch.object(ai_ecosystem_model, 'run_welfare_benchmark_audits',
                               return_value=bad):
            with self.assertRaises(ValueError) as ctx:
                ai_ecosystem_model.main()
        self.assertIn('G2', str(ctx.exception))

    def test_optional_import_fallbacks(self):
        import importlib
        blocked = {'IPython': None, 'IPython.display': None,
                   'colorama': None}
        with mock.patch.dict(sys.modules, blocked):
            importlib.reload(ai_ecosystem_model)
        try:
            self.assertFalse(ai_ecosystem_model.HAS_IPYTHON)
            ai_ecosystem_model.display(object())
            self.assertEqual(ai_ecosystem_model.HTML('<b>x</b>').data, '<b>x</b>')
            self.assertEqual(ai_ecosystem_model.Fore.RED, '')
            self.assertEqual(ai_ecosystem_model.Style.RESET_ALL, '')
        finally:
            importlib.reload(ai_ecosystem_model)


class TestPrintSafeConversion(unittest.TestCase):
    """GT_PRINT_SAFE grayscale+hatch conversion."""
    @staticmethod
    def _fig():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar([0, 1, 2], [1.0, 2.0, 3.0], label='bars')
        ax.plot([0, 1, 2], [1.0, 2.0, 1.5], label='line')
        ax.fill_between([0, 1, 2], [0.5, 1.0, 0.5], [1.0, 2.0, 1.5])
        ax.legend()
        return fig, ax

    def test_apply_print_safe_hatches_deterministically(self):
        import matplotlib.pyplot as plt
        fig, ax = self._fig()
        try:
            ai_ecosystem_model.VisualizationEngine._apply_print_safe(fig)
            hatches = [p.get_hatch() for p in ax.patches]
            self.assertTrue(any(hatches))
            self.assertEqual(hatches[0],
                             ai_ecosystem_model.VisualizationEngine.PRINT_HATCHES[0])
            fig2, ax2 = self._fig()
            try:
                ai_ecosystem_model.VisualizationEngine._apply_print_safe(fig2)
                self.assertEqual([p.get_hatch() for p in ax2.patches], hatches)
            finally:
                plt.close(fig2)
        finally:
            plt.close(fig)

    def test_finalize_none_passthrough(self):
        self.assertIsNone(ai_ecosystem_model.VisualizationEngine._finalize_figure(None))

    def test_finalize_print_safe_branch(self):
        import matplotlib.pyplot as plt
        fig, _ = self._fig()
        try:
            with mock.patch.object(ai_ecosystem_model, 'PRINT_SAFE', True):
                out = ai_ecosystem_model.VisualizationEngine._finalize_figure(fig)
            self.assertIs(out, fig)
        finally:
            plt.close(fig)


class TestLegacyFigures(unittest.TestCase):
    """Superseded registry-external figure builders still render."""
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        data, gf = solve_framework()
        wa = ai_ecosystem_model.WelfareEconomicsAnalyzer(gf, data['players'],
                                                         dependencies_df=data['dependencies'])
        wa.calculate_aggregate_welfare_loss()
        mc = ai_ecosystem_model.MonteCarloSimulator(gf, n_simulations=300, seed=11)
        mc.run_monte_carlo_analysis()
        cls.eng = ai_ecosystem_model.VisualizationEngine(None, gf, wa, mc)

    def test_welfare_analysis_figure_renders(self):
        import matplotlib.pyplot as plt
        fig = self.eng._create_welfare_analysis_figure()
        try:
            self.assertIsInstance(fig, plt.Figure)
            self.assertGreater(len(fig.axes), 0)
        finally:
            plt.close(fig)

    def test_monte_carlo_figure_renders(self):
        import matplotlib.pyplot as plt
        fig = self.eng._create_monte_carlo_figure()
        try:
            self.assertIsInstance(fig, plt.Figure)
            self.assertGreater(len(fig.axes), 0)
        finally:
            plt.close(fig)

    def test_shapley_figure_shows_share_gaps(self):
        # figure_16 plots Shapley vs revenue shares with gap labels
        # (the free-riding finding), not absolute $B bars.
        import matplotlib.pyplot as plt
        import types
        coop = mock.Mock()
        coop.shapley_values = {'Hardware': 434.0, 'Cloud Providers': 387.0}
        coop.players = ['Hardware', 'Cloud Providers']
        coop.characteristic_function = mock.Mock(return_value=1000.0)
        ma = types.SimpleNamespace(players_df=pd.DataFrame(
            {'Player_Category': ['Hardware', 'Cloud Providers'],
             'Current_Revenue_Billions': [459.0, 403.0]}))
        eng = ai_ecosystem_model.VisualizationEngine(ma, None, None, None,
                                                     coop_analyzer=coop)
        try:
            fig = eng._create_shapley_value_figure()
            ax = fig.axes[0]
            self.assertEqual(len(ax.patches), 4)
            self.assertTrue(any('pp' in t.get_text() for t in ax.texts))
            self.assertIn('Share (%)', ax.get_ylabel())
        finally:
            plt.close('all')

    def test_monte_carlo_convergence_empty_results_fallback(self):
        # Simulator present but resultless: the illustrative demo figure
        # still renders instead of failing.
        import matplotlib.pyplot as plt
        eng = ai_ecosystem_model.VisualizationEngine(
            None, None, None, mock.Mock(simulation_results={}))
        try:
            fig = eng._create_monte_carlo_convergence_figure()
            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(len(fig.axes), 2)
            self.assertIn('Convergence', fig._suptitle.get_text())
        finally:
            plt.close('all')

    def test_dynamics_combined_market_fallback_renders(self):
        # No welfare analyzer: trajectory endpoints fall back to market
        # concentration revenue instead of failing.
        import matplotlib.pyplot as plt
        import types
        ma = types.SimpleNamespace(
            results={'concentration': {'total_revenue_billions': 1000.0}})
        eng = ai_ecosystem_model.VisualizationEngine(ma, None, None, None)
        try:
            fig = eng._create_dynamics_combined_figure()
            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(len(fig.axes), 2)
        finally:
            plt.close('all')


class TestMarketDataFallbacks(unittest.TestCase):
    """MarketDataSource graceful degradation (no network in tests)."""
    def test_players_invalid_json_falls_back(self):
        with mock.patch.dict(os.environ, {'AIGT_TICKERS_JSON': '{bad json'}):
            df = ai_ecosystem_model.MarketDataSource.get_industry_players()
        pd.testing.assert_frame_equal(
            df, ai_ecosystem_model.EmbeddedDataSource.get_industry_players())

    def test_players_empty_mapping_falls_back(self):
        with mock.patch.dict(os.environ, {'AIGT_TICKERS_JSON': '{}'}):
            df = ai_ecosystem_model.MarketDataSource.get_industry_players()
        pd.testing.assert_frame_equal(
            df, ai_ecosystem_model.EmbeddedDataSource.get_industry_players())

    def test_players_api_rows_built(self):
        payload = '{"Hardware": ["0000320193"], "Cloud Providers": ["0001652044"]}'
        with mock.patch.dict(os.environ, {'AIGT_TICKERS_JSON': payload}), \
             mock.patch.object(ai_ecosystem_model.MarketDataSource,
                               '_fetch_sec_revenue_ttm', return_value=50.0):
            df = ai_ecosystem_model.MarketDataSource.get_industry_players()
        self.assertEqual(len(df), 2)
        self.assertTrue((df['Current_Revenue_Billions'] == 50.0).all())

    def test_dwl_estimate_bounded(self):
        v = ai_ecosystem_model.estimate_market_dwl_pct()
        self.assertGreaterEqual(v, 0.05)
        self.assertLessEqual(v, 0.40)

    def test_dwl_estimate_last_resort(self):
        # Every data source down: the estimator still returns the
        # conservative mid-range constant instead of raising.
        with mock.patch.object(ai_ecosystem_model.EmbeddedDataSource,
                               'get_industry_players',
                               side_effect=RuntimeError('down')), \
             mock.patch.object(ai_ecosystem_model.EmbeddedDataSource,
                               'get_industry_dependencies',
                               side_effect=RuntimeError('down')):
            self.assertEqual(ai_ecosystem_model.estimate_market_dwl_pct(), 0.20)

    def test_fetch_sec_seams(self):
        import types
        fetch = ai_ecosystem_model.MarketDataSource._fetch_sec_revenue_ttm

        def fake_module(getter):
            mod = types.ModuleType('requests')
            mod.get = getter
            mod.exceptions = types.SimpleNamespace(
                RequestException=type('RequestException', (Exception,), {}))
            return mod

        def resp(status=200, payload=None, bad_json=False):
            class _Resp:
                status_code = status

                def json(self):
                    if bad_json:
                        raise ValueError('no json')
                    return payload
            return _Resp()

        good = {'facts': {'us-gaap': {'SalesRevenueNet': {'units': {'USD': [
            {'end': '2025-12-31', 'val': 50000000000.0},
            {'end': '2024-12-31', 'val': 40000000000.0}]}}}}}
        with mock.patch.dict('sys.modules', {'requests': fake_module(
                lambda url, headers=None, timeout=None: resp(payload=good))}):
            self.assertAlmostEqual(fetch('0000320193'), 50.0)
        usdm = {'facts': {'us-gaap': {'Revenues': {'units': {'USDm': [
            {'end': '2025-12-31', 'val': 12000.0}]}}}}}
        with mock.patch.dict('sys.modules', {'requests': fake_module(
                lambda url, headers=None, timeout=None: resp(payload=usdm))}):
            self.assertAlmostEqual(fetch('0000320193'), 12.0)
        with mock.patch.dict('sys.modules', {'requests': fake_module(
                lambda url, headers=None, timeout=None: resp(status=500))}):
            self.assertIsNone(fetch('0000320193'))
        with mock.patch.dict('sys.modules', {'requests': fake_module(
                lambda url, headers=None, timeout=None: resp(bad_json=True))}):
            self.assertIsNone(fetch('0000320193'))

        def raising(url, headers=None, timeout=None):
            raise fake.exceptions.RequestException('down')
        fake = fake_module(raising)
        with mock.patch.dict('sys.modules', {'requests': fake}):
            self.assertIsNone(fetch('0000320193'))
        with mock.patch.dict('sys.modules', {'requests': None}):
            self.assertIsNone(fetch('0000320193'))

    def test_dwl_market_source_failure_falls_back(self):
        with mock.patch.object(ai_ecosystem_model, 'USE_MARKET_DATA', True), \
             mock.patch.object(ai_ecosystem_model.MarketDataSource,
                               'get_industry_players', side_effect=RuntimeError('down')), \
             mock.patch.object(ai_ecosystem_model.MarketDataSource,
                               'get_industry_dependencies', side_effect=RuntimeError('down')):
            v = ai_ecosystem_model.estimate_market_dwl_pct()
        self.assertGreaterEqual(v, 0.05)
        self.assertLessEqual(v, 0.40)


class TestSmallHelpers(unittest.TestCase):
    """apply_axis_style, safe_centrality, validation report, _perturb_matrix."""
    def test_apply_axis_style_hides_spines(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            ax.plot([0, 1], [0, 1])
            ai_ecosystem_model.apply_axis_style(ax)
            self.assertFalse(ax.spines['top'].get_visible())
            self.assertFalse(ax.spines['right'].get_visible())
        finally:
            plt.close(fig)

    def test_safe_centrality_variants(self):
        import networkx as nx
        conn = nx.path_graph(4)
        out = ai_ecosystem_model.safe_centrality_calculation(conn)
        self.assertEqual(set(out), set(conn.nodes))
        self.assertTrue(all(v >= 0 for v in out.values()))
        disc = nx.DiGraph()
        disc.add_edges_from([(0, 1)])
        disc.add_node(9)
        out2 = ai_ecosystem_model.safe_centrality_calculation(disc)
        self.assertEqual(set(out2), {0, 1, 9})
        self.assertEqual(ai_ecosystem_model.safe_centrality_calculation(nx.Graph()), {})

    def test_print_data_validation_report(self):
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ai_ecosystem_model.print_data_validation_report()
        txt = buf.getvalue()
        self.assertIn('TOTAL CIRCULAR', txt)
        self.assertIn('MARKET PARAMETERS', txt)

    def test_perturb_matrix_shapes_and_graceful_none(self):
        _, gf = solve_framework()
        mc = ai_ecosystem_model.MonteCarloSimulator(gf, n_simulations=10, seed=3)
        m = np.empty((2, 2), dtype=object)
        m[0, 0] = (3.0, 3.0)
        m[0, 1] = (0.0, 5.0)
        m[1, 0] = (5.0, 0.0)
        m[1, 1] = (1.0, 1.0)
        out = mc._perturb_matrix(m, 0.05)
        self.assertEqual(out.shape, (2, 2))
        flat = [out[i, j] for i in range(2) for j in range(2)]
        self.assertTrue(all(isinstance(v, tuple) and len(v) == 2 for v in flat))
        self.assertTrue(all(float(a) >= 0 and float(b) >= 0 for a, b in flat))
        bad = np.empty((2, 2), dtype=object)
        bad.fill('junk')
        out2 = mc._perturb_matrix(bad, 0.05)
        self.assertEqual(out2.shape, (2, 2))


class TestReportFallbacks(unittest.TestCase):
    """Report builders degrade gracefully on missing inputs."""
    @classmethod
    def setUpClass(cls):
        cls.data = ai_ecosystem_model.load_all_data()
        cls.analysis = build_technical_report.analyze(ai_ecosystem_model, cls.data)

    def test_build_report_empty_inputs_stays_graceful(self):
        from docx import Document as _Doc
        with tempfile.TemporaryDirectory() as td:
            tables = os.path.join(td, 'tables')
            plots = os.path.join(td, 'figures')
            os.makedirs(tables)
            os.makedirs(plots)
            out = os.path.join(td, 'empty.docx')
            prof = build_technical_report.code_profile(
                os.path.join(HERE, 'ai_ecosystem_model.py'))
            res = build_technical_report.build_report(
                ai_ecosystem_model, self.analysis, prof, [], td, out)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(len(res['issues']), 0)
            text = '\n'.join(p.text for p in _Doc(out).paragraphs)
            self.assertIn('ai_ecosystem_model.py', text)

    def test_insights_hhi_bands(self):
        import copy
        for hhi, needle in ((2000.0, 'moderately concentrated'),
                            (1000.0, 'unconcentrated')):
            analysis = copy.deepcopy(self.analysis)
            analysis['concentration']['HHI'] = hhi
            ins = build_technical_report.build_insights(ai_ecosystem_model, analysis, HERE)
            body = ' '.join(b for _, b in ins['market'])
            self.assertIn(needle, body)

    def test_insights_fragility_bands(self):
        import copy
        for score, needle in ((0.8, 'elevated'), (0.6, 'material'), (0.4, 'moderate')):
            analysis = copy.deepcopy(self.analysis)
            analysis.setdefault('circular', {})['bubble_score'] = score
            ins = build_technical_report.build_insights(ai_ecosystem_model, analysis, HERE)
            body = ' '.join(b for _, b in ins['circular'])
            self.assertIn(needle, body)

    def test_add_word_table_transposed_note(self):
        from docx import Document as _Doc
        doc = _Doc()
        df = pd.DataFrame({f'C{i}': [1, 2, 3] for i in range(9)})
        tbl = build_technical_report.add_word_table(doc, df, caption='Wide', note='A note')
        self.assertIsNotNone(tbl)
        self.assertTrue(any('Transposed' in p.text for p in doc.paragraphs))


class TestDashboardFallbacks(unittest.TestCase):
    """Dashboard tolerant paths: viz fallback, corrupt report, table edges."""
    def test_games_payload_visualization_fallback(self):
        with mock.patch.object(ai_ecosystem_model.GameTheoryFramework,
                               '_prepare_matrix_visualization_data',
                               side_effect=RuntimeError('no viz')):
            games, _ = dash.games_payload()
        self.assertEqual(len(games), 6)
        self.assertTrue(all(g['game_type'] is None for g in games))

    def test_extract_report_corrupt_file(self):
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tf:
            tf.write(b'not a docx at all')
            path = tf.name
        try:
            res = dash.extract_report(path)
        finally:
            os.unlink(path)
        self.assertTrue(res.get('absent', '').startswith('cannot open'))

    def test_extract_report_table_cell_limits(self):
        from docx import Document as _Doc
        from docx.oxml.ns import qn
        doc = _Doc()
        doc.add_heading('Probe', level=1)
        doc.add_paragraph('Table 7.14: Probe caption')
        doc.add_table(rows=2, cols=2)
        tbl = doc.add_table(rows=2, cols=2)
        for tc in tbl.rows[0]._tr.findall(qn('w:tc')):
            tbl.rows[0]._tr.remove(tc)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'edge.docx')
            doc.save(path)
            self.assertIsInstance(dash.extract_report(path, max_cells=-1), dict)
            self.assertIsInstance(dash.extract_report(path), dict)

    def test_assemble_titles_from_dotted_stems(self):
        import shutil
        with tempfile.TemporaryDirectory() as td:
            tdir = os.path.join(td, 'tables')
            pdir = os.path.join(td, 'figures')
            os.makedirs(tdir)
            os.makedirs(pdir)
            for fn in os.listdir(TABLES):
                if fn.endswith('.csv') and fn != 'tables_index.csv':
                    shutil.copy(os.path.join(TABLES, fn), os.path.join(tdir, fn))
            payload = dash.assemble(tdir, pdir)
        titles = payload['meta']['titles']
        self.assertIn('table_7.16.market.share.evolution.csv', titles)
        self.assertTrue(titles['table_7.16.market.share.evolution.csv']
                        .startswith('Table 7.16'))


class TestFigureFallbacks(unittest.TestCase):
    """Convergence-figure fallback ladders and the sensitivity figure."""
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        cls.data, cls.gf = solve_framework()
        cls.wa = ai_ecosystem_model.WelfareEconomicsAnalyzer(
            cls.gf, cls.data['players'], dependencies_df=cls.data['dependencies'])
        cls.wa.calculate_aggregate_welfare_loss()
        mc = ai_ecosystem_model.MonteCarloSimulator(cls.gf, n_simulations=300, seed=11)
        mc.run_monte_carlo_analysis()
        cls.mc = mc

    def _close(self, fig):
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_convergence_falls_back_to_welfare(self):
        import matplotlib.pyplot as plt
        eng = ai_ecosystem_model.VisualizationEngine(None, self.gf, self.wa, None)
        fig = eng._create_monte_carlo_convergence_figure()
        try:
            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(len(fig.axes), 2)
        finally:
            self._close(fig)

    def test_convergence_falls_back_to_market(self):
        import matplotlib.pyplot as plt
        ma = ai_ecosystem_model.MarketStructureAnalyzer(self.data['players'])
        ma.results = {'concentration': {'total_revenue_billions': 1023.0}}
        eng = ai_ecosystem_model.VisualizationEngine(ma, self.gf, None, None)
        fig = eng._create_monte_carlo_convergence_figure()
        try:
            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(len(fig.axes), 2)
        finally:
            self._close(fig)

    def test_convergence_absolute_last_resort(self):
        import matplotlib.pyplot as plt
        eng = ai_ecosystem_model.VisualizationEngine(None, self.gf, None, None)
        fig = eng._create_monte_carlo_convergence_figure()
        try:
            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(len(fig.axes), 2)
        finally:
            self._close(fig)

    def test_sensitivity_figure_renders(self):
        import matplotlib.pyplot as plt
        sa = ai_ecosystem_model.SensitivityAnalysis(self.wa, self.gf)
        sa.run_sensitivity_analysis(variation_pct=0.10, steps=3)
        eng = ai_ecosystem_model.VisualizationEngine(None, self.gf, self.wa, self.mc)
        eng.sensitivity_analyzer = sa
        fig = eng._create_sensitivity_analysis_figure()
        try:
            self.assertIsInstance(fig, plt.Figure)
            self.assertGreater(len(fig.axes), 0)
        finally:
            self._close(fig)


class TestTaxRate(unittest.TestCase):
    """_calculate_effective_tax_rate computed, fallback, and default paths."""
    @staticmethod
    def _fin(**kw):
        import types
        return types.SimpleNamespace(**kw)

    def test_computed_rate_in_range(self):
        fin = self._fin(net_income=80.0, operating_income=100.0, category='Hardware')
        self.assertAlmostEqual(ai_ecosystem_model._calculate_effective_tax_rate(fin), 0.2)

    def test_out_of_range_falls_back_to_category(self):
        fin = self._fin(net_income=-100.0, operating_income=100.0, category='Hardware')
        self.assertAlmostEqual(ai_ecosystem_model._calculate_effective_tax_rate(fin), 0.17)

    def test_exception_falls_back_to_category(self):
        fin = self._fin(net_income='junk', operating_income=100.0, category='Cloud Providers')
        self.assertAlmostEqual(ai_ecosystem_model._calculate_effective_tax_rate(fin), 0.19)

    def test_unknown_category_default(self):
        fin = self._fin(net_income=None, operating_income=None, category='Nope')
        self.assertAlmostEqual(ai_ecosystem_model._calculate_effective_tax_rate(fin), 0.21)


class TestCrisisBacktestModules(unittest.TestCase):
    """2008 backtest compatibility: model classes run on non-AI archetypes.

    Guards build_crisis_backtest.py against engine refactors. Fast paths only
    (no Monte Carlo, no joint draws). Importing build_crisis_backtest rewrites
    tables/backtest/valuation_2008.csv with identical content (idempotent).
    """

    @classmethod
    def setUpClass(cls):
        import build_crisis_backtest as bcb
        cls.b = bcb
        cls.circ = bcb.CrisisDealsAnalyzer()
        cls.bb = ai_ecosystem_model.BubbleBurstAnalyzer(
            bcb.PLAYERS, cls.circ, dependencies_df=bcb.CATFLOWS,
            hhi_shares=bcb.HHI,
            valuation_path=os.path.join("tables", "backtest", "valuation_2008.csv"),
            output_dir=os.path.join("tables", "backtest"))
        cls.form = cls.bb.formation_probability()
        cls.burst = cls.bb.burst_impact()

    def test_exposures_bounded(self):
        for c in self.b.CATS:
            e = float(self.circ._exposure(c))
            self.assertGreaterEqual(e, 0.0, c)
            self.assertLessEqual(e, 1.0, c)

    def test_formation_top_two(self):
        top2 = set(self.form.sort_values("Formation_Prob_Pct", ascending=False)
                   ["Archetype"].iloc[:2])
        self.assertEqual(top2, {"Securitizers", "Investors"})

    def test_burst_head_is_investors(self):
        top = self.burst.sort_values("Total_Loss_Severe_$B", ascending=False
                                     )["Archetype"].iloc[0]
        self.assertEqual(top, "Investors")

    def test_stability_min_is_originators(self):
        nra = ai_ecosystem_model.NetworkRiskAnalysis(
            {"players": self.b.COS, "dependencies": self.b.DEPS})
        spm = ai_ecosystem_model.SuccessProbabilityModel(
            {"players": self.b.COS}, nra.analyze_centrality())
        scores = spm.calculate_scores()["stability_score"]
        cats = self.b.COS["Player_Category"].values
        means = {c: float(np.mean(scores[cats == c])) for c in self.b.CATS}
        self.assertEqual(min(means, key=means.get), "Originators")

    def test_sourced_category_revenues(self):
        sums = self.b.PLAYERS.set_index("Player_Category")[
            "Current_Revenue_Billions"].to_dict()
        self.assertAlmostEqual(sums["Securitizers"], 134.2, places=1)
        self.assertAlmostEqual(sums["Investors"], 186.8, places=1)
        self.assertAlmostEqual(sums["Guarantors"], 117.1, places=1)
        self.assertAlmostEqual(sums["Originators"], 15.5, places=1)

    def test_six_games_two_stuck_at_guarantor_nexus(self):
        gf = ai_ecosystem_model.GameTheoryFramework(self.b.PLAYERS)
        gf.construct_payoff_matrices()
        ne = gf.find_nash_equilibria()
        self.assertEqual(len(ne), 6)
        stuck = [g for g, r in ne.items()
                 if (r.get("welfare_metrics", {}) or {}).get("deadweight_loss", 0) > 0]
        self.assertEqual(set(stuck),
                         {"Securitizers-Guarantors", "Investors-Guarantors"})


if __name__ == '__main__':
    unittest.main(verbosity=2)
