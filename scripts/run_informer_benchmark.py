from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.benchmark import run_benchmark
from benchmarking.config import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Informer benchmark on collected monthly data and/or FRED-MD."
    )
    parser.add_argument(
        "--dataset",
        choices=["collected", "fred_md", "both"],
        default="both",
    )
    parser.add_argument("--fred-md-vintage", default="2026-03-md.csv")
    parser.add_argument("--fred-md-url", default=None)
    parser.add_argument("--context-length", type=int, default=36)
    parser.add_argument("--prediction-length", type=int, default=6)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--rolling-splits", type=int, default=6)
    parser.add_argument("--min-train-size", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-observations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fred_md_url = args.fred_md_url
    if fred_md_url is None:
        fred_md_url = (
            "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
            f"fred-md/monthly/{args.fred_md_vintage}"
        )

    config = BenchmarkConfig(
        project_root=PROJECT_ROOT,
        fred_md_vintage=args.fred_md_vintage,
        fred_md_url=fred_md_url,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        eval_horizons=tuple(args.horizons),
        rolling_splits=args.rolling_splits,
        min_train_size=args.min_train_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attention_factor=args.attention_factor,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        max_observations=args.max_observations,
    )
    result = run_benchmark(args.dataset, config)
    print(f"Wrote benchmark artifacts to {result.output_dir}")
    print(result.summary.to_string(index=False))
    if not result.comparison.empty:
        print()
        print(result.comparison.to_string(index=False))


if __name__ == "__main__":
    main()
