from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BenchmarkConfig:
    project_root: Path
    fred_md_vintage: str = "2026-03-md.csv"
    fred_md_url: str = (
        "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
        "fred-md/monthly/2026-03-md.csv"
    )
    context_length: int = 36
    prediction_length: int = 6
    eval_horizons: tuple[int, ...] = (1, 3, 6)
    rolling_splits: int = 6
    min_train_size: int = 120
    batch_size: int = 32
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    attention_factor: int = 4
    seed: int = 42
    device: str = "cpu"
    output_dir: Path | None = None
    max_observations: int | None = None
    matched_variables: dict[str, str] = field(
        default_factory=lambda: {
            "LFC": "CLF16OV",
            "EMPLOY": "CE16OV",
            "RUC": "UNRATE",
            "H": "AWHMAN",
            "IPT": "INDPRO",
            "IPM": "IPMANSICS",
            "CUM": "CUMFNS",
        }
    )

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def fred_md_path(self) -> Path:
        return self.raw_root / "fred_md" / self.fred_md_vintage

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        stem = Path(self.fred_md_vintage).stem.replace(".", "_")
        return self.project_root / "results" / "informer_benchmark" / stem

