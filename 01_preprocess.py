"""
STEP 1 - Data cleaning + feature engineering
--------------------------------------------
Run :  python 01_preprocess.py
Input :  data/india_housing_prices.csv   (put the downloaded dataset here)
Output:  data/cleaned_housing.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
RAW = BASE / "data" / "india_housing_prices.csv"
OUT = BASE / "data" / "cleaned_housing.csv"
CURRENT_YEAR = 2026

NUM_COLS = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
            "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools",
            "Nearby_Hospitals"]   # Parking_Space and Security are Yes/No text


def line(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ------------------------------------------------------------------ 1. LOAD
line("1. LOAD DATA")
df = pd.read_csv(RAW)
print("Shape:", df.shape)
print(df.dtypes)
print(df.head())

# tidy text columns (strip spaces, blank -> NaN) and force numeric columns
for c in df.select_dtypes(exclude="number").columns:
    df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
    df[c] = df[c].replace({"": np.nan, "nan": np.nan, "None": np.nan})
for c in NUM_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ------------------------------------------------------------ 2. DUPLICATES
line("2. DUPLICATE CHECK + REMOVAL")
no_id = [c for c in df.columns if c != "ID"]
print("Fully duplicated rows            :", df.duplicated().sum())
if "ID" in df.columns:
    print("Duplicated IDs                   :", df["ID"].duplicated().sum())
print("Same details but different ID    :", df.duplicated(subset=no_id).sum(),
      "(reported only, not removed)")

before = len(df)
df = df.drop_duplicates()
if "ID" in df.columns:
    df = df.drop_duplicates(subset="ID", keep="first")
print(f"Rows before: {before:,} | after: {len(df):,} | removed: {before - len(df):,}")

# ---------------------------------------------------------- 3. MISSING VALUES
line("3. MISSING VALUES - CHECK")
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
report = pd.DataFrame({"missing": miss, "percent": miss_pct})
print(report[report["missing"] > 0] if miss.sum() else "No missing values found.")

line("3b. MISSING VALUES - IMPUTE")
num_present = [c for c in NUM_COLS if c in df.columns]
cat_present = [c for c in df.columns if c not in num_present and c != "ID"]
for c in num_present:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())
        print(f"{c}: filled with median")
for c in cat_present:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode()[0])
        print(f"{c}: filled with mode")
print("Missing values left:", int(df.isnull().sum().sum()))

# ---------------------------------------------------- 4. LOGIC / SANITY CHECKS
line("4. DATA SANITY CHECKS")
bad = (df["Size_in_SqFt"] <= 0) | (df["Price_in_Lakhs"] <= 0)
print("Rows with size/price <= 0 removed:", int(bad.sum()))
df = df[~bad].copy()

df["Age_of_Property"] = (CURRENT_YEAR - df["Year_Built"]).clip(lower=0)
wrong_floor = df["Floor_No"] > df["Total_Floors"]
print(f"Floor_No > Total_Floors          : {int(wrong_floor.sum()):,} rows "
      f"({wrong_floor.mean() * 100:.1f}%) - kept as is, flagged only (too many to edit safely)")
df["Floor_Data_Inconsistent"] = wrong_floor.astype(int)

# ------------------------------------------------------------- 5. OUTLIERS
line("5. OUTLIER DETECTION (IQR) + CAPPING (1st-99th percentile)")
for c in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    if n_out == 0:
        print(f"{c:16s} IQR outliers: 0 -> no change")
        continue
    lo, hi = df[c].quantile([0.01, 0.99])
    df[c] = df[c].clip(lo, hi)
    print(f"{c:16s} IQR outliers: {n_out:,} -> capped to [{lo:.3f}, {hi:.3f}]")

# ------------------------------------------------------ 6. FEATURE ENGINEERING
line("6. FEATURE ENGINEERING")


def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


# number of amenities listed
df["Amenity_Count"] = df["Amenities"].fillna("").apply(
    lambda s: len([a for a in str(s).split(",") if a.strip()]))

# transport text (Low/Medium/High) -> number
tmap = {"low": 1, "medium": 2, "med": 2, "high": 3}
t_num = pd.to_numeric(df["Public_Transport_Accessibility"], errors="coerce")
t_txt = df["Public_Transport_Accessibility"].astype(str).str.lower().map(tmap)
df["Transport_Score"] = t_num.fillna(t_txt)
df["Transport_Score"] = df["Transport_Score"].fillna(df["Transport_Score"].median())

df["School_Density_Score"] = (minmax(df["Nearby_Schools"]) * 10).round(2)
df["Hospital_Density_Score"] = (minmax(df["Nearby_Hospitals"]) * 10).round(2)
df["Infrastructure_Score"] = ((df["School_Density_Score"] + df["Hospital_Density_Score"]
                               + minmax(df["Transport_Score"]) * 10) / 3).round(2)

city_median_ppsf = df.groupby("City")["Price_per_SqFt"].transform("median")
df["PPSF_vs_City_Median"] = (df["Price_per_SqFt"] / city_median_ppsf).round(3)

# ---------------------------------------------------- 7. TARGET: FUTURE PRICE
line("7. TARGET 1 (REGRESSION): PRICE AFTER 5 YEARS")
# Feature-based growth rate: better infrastructure -> faster appreciation (5% to 11% a year)
df["Growth_Rate"] = (0.05 + 0.006 * df["Infrastructure_Score"]).clip(0.05, 0.11).round(4)
df["Future_Price_5Y"] = (df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** 5).round(2)
df["Investment_Return_Pct"] = ((df["Future_Price_5Y"] / df["Price_in_Lakhs"] - 1) * 100).round(2)
print(df[["Price_in_Lakhs", "Growth_Rate", "Future_Price_5Y"]].describe().round(2))

# ------------------------------------------------- 8. TARGET: GOOD INVESTMENT
line("8. TARGET 2 (CLASSIFICATION): GOOD INVESTMENT")
# Multi-factor score (1 point each). Good investment if score >= 3 out of 5.
ready = df["Availability_Status"].astype(str).str.lower().str.contains("ready")
score = (
    (df["Price_per_SqFt"] <= city_median_ppsf).astype(int)           # cheaper than city median
    + (df["BHK"] >= 3).astype(int)                                   # 3+ BHK
    + ready.astype(int)                                              # ready to move
    + (df["Transport_Score"] >= df["Transport_Score"].median()).astype(int)
    + (df["Infrastructure_Score"] >= df["Infrastructure_Score"].median()).astype(int)
)
df["Investment_Score"] = score
df["Good_Investment"] = (score >= 3).astype(int)
print(df["Good_Investment"].value_counts(normalize=True).round(3))

# ----------------------------------------------------------------- 9. SAVE
line("9. SAVE")
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows x {df.shape[1]} columns -> {OUT}")
