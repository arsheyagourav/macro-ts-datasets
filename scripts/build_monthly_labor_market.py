from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/philly_fed/labor_market")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

file_map = {
    "POP": "popMvMd.xlsx",
    "LFC": "lfcMvMd.xlsx",
    "LFPART": "lfpartMvMd.xlsx",
    "RUC": "rucQvMd.xlsx",
    "EMPLOY": "employMvMd.xlsx",
    "H": "hMvMd.xlsx",
    "HG": "hgMvMd.xlsx",
    "HS": "hsMvMd.xlsx",
}

def clean_one(var_name: str, filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename

    df = pd.read_excel(path, index_col=0)

    print(f"\n--- {var_name} raw columns ---")
    print(df.columns.tolist())

    series = df.iloc[:, -1]
    out = series.reset_index()
    out.columns = ["date", var_name]
    out = out.dropna(subset=["date"])
    return out

merged = None

for var_name, filename in file_map.items():
    cleaned = clean_one(var_name, filename)

    if merged is None:
        merged = cleaned
    else:
        merged = merged.merge(cleaned, on="date", how="outer")

merged.to_csv(OUT_DIR / "monthly_labor_market.csv", index=False)
print("\nSaved to data/processed/monthly_labor_market.csv")
print(merged.head())