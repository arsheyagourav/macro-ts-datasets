from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/philly_fed/income")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

file_map = {
    "DIV": "divQvQd.xlsx",
    "NDPI": "ndpiQvQd.xlsx",
    "NPI": "npiQvQd.xlsx",
    "OLI": "oliQvQd.xlsx",
    "PINTI": "pintiQvQd.xlsx",
    "PROPI": "propiQvQd.xlsx",
    "WSD": "wsdQvQd.xlsx",
    "NPSAV": "npsavQvQd.xlsx",
    "PTAX": "ptaxQvQd.xlsx",
    "RATESAV": "ratesavQvQd.xlsx",
    "TRANR": "tranrQvQd.xlsx",
}

def clean_one(var_name: str, filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_excel(path)

    print(f"\n--- {var_name} raw columns ---")
    print(df.columns.tolist())

    first_col = df.columns[0]
    last_col = df.columns[-1]

    df = df.rename(columns={first_col: "date"})
    df = df[["date", last_col]]
    df.columns = ["date", var_name]

    df = df.dropna(subset=["date"])
    return df

merged = None

for var_name, filename in file_map.items():
    cleaned = clean_one(var_name, filename)

    if merged is None:
        merged = cleaned
    else:
        merged = merged.merge(cleaned, on="date", how="outer")

merged.to_csv(OUT_DIR / "quarterly_income.csv", index=False)
print("\nSaved to data/processed/quarterly_income.csv")
print(merged.head())