from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "philly_fed" / "industrial_production_capacity_utilization"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_MAP = {
    "IPT": "iptMvMd.xlsx",
    "IPM": "ipmMvMd.xlsx",
    "CUT": "cutMvMd.xlsx",
    "CUM": "cumMvMd.xlsx",
}


def clean_one(var_name: str, filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_excel(path, index_col=0)

    series = df.iloc[:, -1]
    out = series.reset_index()
    out.columns = ["date", var_name]
    out = out.dropna(subset=["date"])
    return out


def main() -> None:
    missing_files = [
        filename
        for filename in FILE_MAP.values()
        if filename.endswith(".xlsx")
        and not filename.startswith("~$")
        and not (RAW_DIR / filename).exists()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing required raw files in {RAW_DIR}: {missing}")

    merged = None
    for var_name, filename in FILE_MAP.items():
        cleaned = clean_one(var_name, filename)
        if merged is None:
            merged = cleaned
        else:
            merged = merged.merge(cleaned, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    output_path = OUT_DIR / "monthly_industrial_production_capacity_utilization.csv"
    merged.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(merged.head())


if __name__ == "__main__":
    main()
