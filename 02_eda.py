"""
STEP 2 - Exploratory Data Analysis (all 20 questions + 2 extras)
Run :  python 02_eda.py   (after 01_preprocess.py)
Output: charts saved in eda_outputs/  +  key findings printed in the terminal
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(__file__).parent
OUT = BASE / "eda_outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
BLUE = "#3b7dd8"

df = pd.read_csv(BASE / "data" / "cleaned_housing.csv")
SAMPLE = df.sample(min(len(df), 5000), random_state=42)   # for scatter plots


def done(n, title, fig_name):
    plt.tight_layout()
    plt.savefig(OUT / f"{fig_name}.png", dpi=120)
    plt.close()
    print(f"\n[Q{n}] {title}")


def hist(col, n, title, name):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=40, kde=True, color=BLUE)
    plt.title(title)
    done(n, title, name)
    print(df[col].describe().round(2).to_string())
    print("Skewness:", round(df[col].skew(), 2))


def bar(series, n, title, name, horizontal=False, figsize=(9, 5), ylabel=""):
    plt.figure(figsize=figsize)
    if horizontal:
        series.plot(kind="barh", color=BLUE)
    else:
        series.plot(kind="bar", color=BLUE)
        plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    done(n, title, name)
    print(series.round(3).to_string())


# ------------------------------------------------ 1-5  PRICE & SIZE ANALYSIS
hist("Price_in_Lakhs", 1, "Distribution of property prices (Lakhs)", "q01_price_dist")
hist("Size_in_SqFt", 2, "Distribution of property sizes (SqFt)", "q02_size_dist")

plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt", color=BLUE)
plt.title("Price per SqFt by property type")
done(3, "Price per sq ft by property type", "q03_ppsf_by_type")
print(df.groupby("Property_Type")["Price_per_SqFt"].mean().round(3).to_string())

plt.figure(figsize=(7, 5))
sns.scatterplot(data=SAMPLE, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.4, color=BLUE)
plt.title("Property size vs price")
done(4, "Relationship between size and price", "q04_size_vs_price")
print("Correlation (size, price):", round(df["Size_in_SqFt"].corr(df["Price_in_Lakhs"]), 3))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df["Price_per_SqFt"], ax=ax[0], color=BLUE)
ax[0].set_title("Price per SqFt")
sns.boxplot(y=df["Size_in_SqFt"], ax=ax[1], color=BLUE)
ax[1].set_title("Size in SqFt")
done(5, "Outliers in price per sq ft and size", "q05_outliers")
for c in ["Price_per_SqFt", "Size_in_SqFt"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    n_out = ((df[c] < q1 - 1.5 * (q3 - q1)) | (df[c] > q3 + 1.5 * (q3 - q1))).sum()
    print(f"{c}: {n_out:,} IQR outliers (after capping in step 1)")

# ------------------------------------------------ 6-10  LOCATION ANALYSIS
bar(df.groupby("State")["Price_per_SqFt"].mean().sort_values(), 6,
    "Average price per sq ft by state", "q06_ppsf_by_state", horizontal=True, figsize=(8, 7))

bar(df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(20), 7,
    "Average property price by city (top 20, Lakhs)", "q07_price_by_city")

loc_age = df.groupby("Locality")["Age_of_Property"].median().sort_values(ascending=False).head(15)
bar(loc_age, 8, "Median age of properties by locality (15 oldest)", "q08_age_by_locality")

top_cities = df["City"].value_counts().head(10).index
ct = pd.crosstab(df[df["City"].isin(top_cities)]["City"], df["BHK"])
ct.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="Blues")
plt.title("BHK distribution across top 10 cities")
plt.xticks(rotation=45, ha="right")
done(9, "BHK distribution across cities", "q09_bhk_by_city")
print(ct.to_string())

top5 = df.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(5).index
trend = (df[df["Locality"].isin(top5)].groupby(["Year_Built", "Locality"])["Price_in_Lakhs"]
         .mean().unstack())
trend.plot(figsize=(10, 5), marker="o", markersize=3)
plt.title("Price trend (by year built) - top 5 most expensive localities")
plt.ylabel("Avg price (Lakhs)")
done(10, "Price trends for the top 5 most expensive localities", "q10_top5_locality_trend")
print("Top 5 localities:", list(top5))

# ------------------------------------------------ 11-15  RELATIONSHIPS
num_cols = [c for c in df.select_dtypes("number").columns if c not in ("ID",)]
plt.figure(figsize=(12, 9))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 6})
plt.title("Correlation matrix of numeric features")
done(11, "Correlation between numeric features", "q11_correlation")
corr_target = df[num_cols].corr()["Price_per_SqFt"].drop("Price_per_SqFt").sort_values()
print("Correlation with Price_per_SqFt:\n", corr_target.round(3).to_string())

for n, col, name in [(12, "Nearby_Schools", "q12_schools_vs_ppsf"),
                     (13, "Nearby_Hospitals", "q13_hospitals_vs_ppsf")]:
    g = df.groupby(col)["Price_per_SqFt"].mean()
    plt.figure(figsize=(8, 4))
    g.plot(marker="o", color=BLUE)
    plt.ylabel("Avg price per SqFt")
    plt.title(f"{col} vs price per sq ft")
    done(n, f"{col} vs price per sq ft", name)
    print(f"Correlation: {df[col].corr(df['Price_per_SqFt']):.3f}")

plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", color=BLUE, errorbar=None)
plt.title("Average price by furnished status")
done(14, "Price by furnished status", "q14_price_by_furnished")
print(df.groupby("Furnished_Status")["Price_in_Lakhs"].mean().round(2).to_string())

plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="Facing", y="Price_per_SqFt", color=BLUE, errorbar=None)
plt.title("Average price per sq ft by facing")
done(15, "Price per sq ft by facing direction", "q15_ppsf_by_facing")
print(df.groupby("Facing")["Price_per_SqFt"].mean().round(3).to_string())

# ------------------------------------------------ 16-20  OWNERSHIP / AMENITIES
vc = df["Owner_Type"].value_counts()
bar(vc, 16, "Properties by owner type", "q16_owner_type", figsize=(6, 4))

vc = df["Availability_Status"].value_counts()
bar(vc, 17, "Properties by availability status", "q17_availability", figsize=(6, 4))

plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs", color=BLUE, errorbar=None)
plt.title("Average price by parking spaces")
done(18, "Does parking space affect price?", "q18_parking_vs_price")
print(df.groupby("Parking_Space")["Price_in_Lakhs"].mean().round(2).to_string())

am = df[["Amenities", "Price_per_SqFt"]].copy()
am["Amenity"] = am["Amenities"].str.split(",")
am = am.explode("Amenity")
am["Amenity"] = am["Amenity"].str.strip()
am = am[am["Amenity"].notna() & (am["Amenity"] != "")]
by_amenity = am.groupby("Amenity")["Price_per_SqFt"].mean().sort_values()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
by_amenity.plot(kind="barh", ax=ax[0], color=BLUE)
ax[0].set_title("Avg price/SqFt by amenity")
df.groupby("Amenity_Count")["Price_per_SqFt"].mean().plot(marker="o", ax=ax[1], color=BLUE)
ax[1].set_title("Avg price/SqFt by number of amenities")
done(19, "How do amenities affect price per sq ft?", "q19_amenities")
print(by_amenity.round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt",
            color=BLUE, errorbar=None, ax=ax[0])
ax[0].set_title("Avg price per SqFt")
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Good_Investment",
            color=BLUE, errorbar=None, ax=ax[1])
ax[1].set_title("Share of Good Investments")
done(20, "Public transport vs price per sq ft / investment potential", "q20_transport")
print(df.groupby("Public_Transport_Accessibility")[["Price_per_SqFt", "Good_Investment"]]
      .mean().round(3).to_string())

# ------------------------------------------------ EXTRA (from the project brief)
df["Size_Band"] = pd.qcut(df["Size_in_SqFt"], 5, duplicates="drop")
g = df.groupby("Size_Band", observed=True)["Investment_Return_Pct"].mean()
g.index = g.index.astype(str)
bar(g, 21, "Area vs 5-year investment return (%)", "q21_size_vs_return", figsize=(8, 4))

g = df.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(15)
bar(g, 22, "Share of good investments by city (top 15)", "q22_good_by_city")

print(f"\nDone. {len(list(OUT.glob('*.png')))} charts saved in {OUT}")
