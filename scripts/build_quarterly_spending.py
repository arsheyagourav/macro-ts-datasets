from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/philly_fed/spending")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

file_map = {
    "gdp": "ROUTPUTQvQd.xlsx",
    "consumption_total": "RCONQvQd.xlsx",
    "consumption_nondurables": "RCONNDQvQd.xlsx",
    "consumption_durables": "RCONDQvQd.xlsx",
    "consumption_services": "RCONSQvQd.xlsx",
    "business_investment": "rinvbfQvQd.xlsx",
    "residential_investment": "rinvresidQvQd.xlsx",
    "inventory_change": "rinvchiQvQd.xlsx",
    "government_total": "RGQvQd.xlsx",
    "government_federal": "RGFQvQd.xlsx",
    "government_state_local": "RGSLQvQd.xlsx",
    "net_exports": "RNXQvQd.xlsx",
    "exports": "REXQvQd.xlsx",
    "imports": "RIMPQvQd.xlsx",
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

merged.to_csv(OUT_DIR / "quarterly_spending.csv", index=False)
print("\nSaved to data/processed/quarterly_spending.csv")
print(merged.head())