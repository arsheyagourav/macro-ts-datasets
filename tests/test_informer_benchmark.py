from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarking.benchmark import run_benchmark
from benchmarking.config import BenchmarkConfig
from benchmarking.data import (
    build_matched_panels,
    load_collected_monthly,
    load_fred_md_monthly,
    train_standardize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InformerBenchmarkTests(unittest.TestCase):
    def make_config(self, **overrides: object) -> BenchmarkConfig:
        base = dict(
            project_root=PROJECT_ROOT,
            device="cpu",
            seed=7,
            context_length=24,
            prediction_length=6,
            eval_horizons=(1, 3, 6),
            rolling_splits=2,
            min_train_size=72,
            epochs=1,
            batch_size=16,
            d_model=32,
            n_heads=4,
            e_layers=1,
            d_ff=64,
            max_observations=96,
        )
        base.update(overrides)
        return BenchmarkConfig(**base)

    def test_collected_loader_orders_monthly_columns(self) -> None:
        config = self.make_config()
        panel = load_collected_monthly(config)
        self.assertEqual(panel.index.freqstr, "M")
        self.assertEqual(panel.columns.tolist(), list(config.matched_variables.keys()))
        self.assertTrue(panel.index.is_monotonic_increasing)

    def test_fred_loader_selects_mapped_columns(self) -> None:
        config = self.make_config()
        panel = load_fred_md_monthly(config)
        self.assertEqual(panel.index.freqstr, "M")
        self.assertEqual(panel.columns.tolist(), list(config.matched_variables.keys()))
        self.assertTrue(panel.index.is_monotonic_increasing)

    def test_matched_panels_share_same_targets_and_range(self) -> None:
        config = self.make_config()
        panels = build_matched_panels(config)
        self.assertEqual(panels.collected.columns.tolist(), panels.fred_md.columns.tolist())
        self.assertEqual(panels.collected.index[0], panels.fred_md.index[0])
        self.assertEqual(panels.collected.index[-1], panels.fred_md.index[-1])

    def test_train_standardize_uses_train_statistics_only(self) -> None:
        train = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        eval_values = np.array([[11.0, 14.0]])
        train_scaled, eval_scaled, mean, std = train_standardize(train, eval_values)
        np.testing.assert_allclose(mean, np.array([3.0, 6.0]))
        np.testing.assert_allclose(std, np.std(train, axis=0))
        np.testing.assert_allclose(eval_scaled, (eval_values - mean) / std)
        np.testing.assert_allclose(train_scaled.mean(axis=0), np.zeros(2), atol=1e-7)

    def test_smoke_run_writes_metrics_for_both_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(output_dir=Path(tmpdir))
            result = run_benchmark("both", config)
            self.assertTrue((result.output_dir / "per_target_metrics.csv").exists())
            self.assertTrue((result.output_dir / "summary_metrics.csv").exists())
            self.assertTrue((result.output_dir / "comparison_metrics.csv").exists())
            self.assertIn("collected", set(result.summary["dataset"]))
            self.assertIn("fred_md", set(result.summary["dataset"]))
            self.assertEqual(set(result.comparison["horizon"]), {1, 3, 6})


if __name__ == "__main__":
    unittest.main()
