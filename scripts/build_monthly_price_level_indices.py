from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/philly_fed/price_level_indices")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

file_map = {
    "PCONG": "pcongMvQd.xlsx",
    "PCONHH": "pconhhMvQd.xlsx",
    "PCONSHH": "pconshhMvQd.xlsx",
    "PCONSNP": "pconsnpMvQd.xlsx",
    "PCONX": "pconxMvQd.xlsx",
    "PCPI": "pcpiMvMd.xlsx",
    "PCPIX": "pcpixMvMd.xlsx",
    "P": "pMvQd.xlsx",
    "PPPI": "pppiMvMd.xlsx",
    "PPPIX": "pppixMvMd.xlsx",
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

merged.to_csv(OUT_DIR / "monthly_price_level_indices.csv", index=False)
print("\nSaved to data/processed/monthly_price_level_indices.csv")
print(merged.head())