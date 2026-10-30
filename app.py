"""
STEP 4 - Streamlit dashboard
Run :  streamlit run app.py      (after 01_preprocess.py and 03_train.py)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
CURRENT_YEAR = 2026

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


# ------------------------------------------------------------------ LOADERS
@st.cache_data
def load_data():
    return pd.read_csv(BASE / "data" / "cleaned_housing.csv")


@st.cache_resource
def load_models():
    return (joblib.load(BASE / "models" / "classifier.joblib"),
            joblib.load(BASE / "models" / "regressor.joblib"))


@st.cache_data
def load_json(name):
    with open(BASE / "models" / name) as f:
        return json.load(f)


try:
    df = load_data()
    clf, reg = load_models()
    cfg = load_json("feature_config.json")
    metrics = load_json("metrics.json")
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run 01_preprocess.py and 03_train.py first.")
    st.stop()

# dataset's own unit for price-per-sqft, so the form matches the training data
PPSF_RATIO = float((df["Price_per_SqFt"] / (df["Price_in_Lakhs"] / df["Size_in_SqFt"])).median())
AMENITIES = sorted({a.strip() for s in df["Amenities"].dropna().unique()
                    for a in str(s).split(",") if a.strip()})


def options(col):
    return sorted(df[col].dropna().unique().tolist())


st.title("🏠 Real Estate Investment Advisor")
st.caption("Predict whether a property is a good investment and what it may be worth in 5 years. "
           "For learning purposes - not financial advice.")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "🔎 Property Explorer",
                                   "📊 Market Insights", "🧪 Model Performance"])

# ================================================================ TAB 1: PREDICT
with tab1:
    st.subheader("Enter property details")
    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", options("State"))
        cities = sorted(df.loc[df["State"] == state, "City"].unique())
        city = st.selectbox("City", cities)
        ptype = st.selectbox("Property type", options("Property_Type"))
        bhk = st.number_input("BHK", 1, 10, 3)
        size = st.number_input("Size (SqFt)", 100, 20000, int(df["Size_in_SqFt"].median()), step=50)
        price = st.number_input("Current price (Lakhs)", 1.0, 100000.0,
                                float(round(df["Price_in_Lakhs"].median(), 1)), step=5.0)
    with c2:
        year_built = st.number_input("Year built", 1950, CURRENT_YEAR, 2015)
        total_floors = st.number_input("Total floors", 1, 100, 10)
        floor_no = st.number_input("Floor number", 0, 100, 3)
        furnished = st.selectbox("Furnished status", options("Furnished_Status"))
        facing = st.selectbox("Facing", options("Facing"))
        availability = st.selectbox("Availability", options("Availability_Status"))
    with c3:
        schools = st.number_input("Nearby schools", int(df["Nearby_Schools"].min()),
                                  int(df["Nearby_Schools"].max()), int(df["Nearby_Schools"].median()))
        hospitals = st.number_input("Nearby hospitals", int(df["Nearby_Hospitals"].min()),
                                    int(df["Nearby_Hospitals"].max()), int(df["Nearby_Hospitals"].median()))
        transport = st.selectbox("Public transport", options("Public_Transport_Accessibility"))
        parking = st.selectbox("Parking space", options("Parking_Space"))
        security = st.selectbox("Security", options("Security"))
        owner = st.selectbox("Owner type", options("Owner_Type"))
    amenities = st.multiselect("Amenities", AMENITIES, default=AMENITIES[:2])

    if st.button("Analyze investment", type="primary"):
        row = {"State": state, "City": city, "Property_Type": ptype, "BHK": bhk,
               "Size_in_SqFt": size, "Price_in_Lakhs": price,
               "Price_per_SqFt": price / size * PPSF_RATIO,
               "Age_of_Property": max(CURRENT_YEAR - year_built, 0),
               "Floor_No": min(floor_no, total_floors), "Total_Floors": total_floors,
               "Furnished_Status": furnished, "Facing": facing,
               "Availability_Status": availability, "Nearby_Schools": schools,
               "Nearby_Hospitals": hospitals, "Public_Transport_Accessibility": transport,
               "Parking_Space": parking, "Security": security, "Owner_Type": owner,
               "Amenity_Count": len(amenities)}
        X = pd.DataFrame([row])[cfg["num_cols"] + cfg["cat_cols"]]

        is_good = int(clf.predict(X)[0])
        proba = clf.predict_proba(X)[0]
        confidence = float(proba.max())
        future = float(reg.predict(X)[0])
        gain = future - price
        cagr = ((future / price) ** (1 / 5) - 1) * 100

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("### Is this a Good Investment?")
            if is_good:
                st.success("✅ Yes - Good Investment")
            else:
                st.warning("⚠️ No - Not a Good Investment")
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            st.progress(confidence)
        with r2:
            st.markdown("### Estimated price after 5 years")
            st.metric("Future price", f"₹ {future:,.1f} Lakhs",
                      delta=f"{gain:,.1f} Lakhs ({cagr:.1f}% per year)")

        years = list(range(0, 6))
        proj = pd.DataFrame({"Year": years,
                             "Value (Lakhs)": [price * (1 + cagr / 100) ** y for y in years]})
        st.plotly_chart(px.line(proj, x="Year", y="Value (Lakhs)", markers=True,
                                title="Projected value over 5 years"))

        st.markdown("#### What drives the predictions (feature importance)")
        f1, f2 = st.columns(2)
        for col, fname, title in [(f1, "feature_importance_classifier.csv", "Good Investment model"),
                                  (f2, "feature_importance_regressor.csv", "Future price model")]:
            imp = pd.read_csv(BASE / "models" / fname).head(10).iloc[::-1]
            col.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", title=title))

# ================================================================ TAB 2: EXPLORER
with tab2:
    st.subheader("Filter properties")
    f1, f2, f3 = st.columns(3)
    sel_cities = f1.multiselect("City", options("City"))
    sel_types = f2.multiselect("Property type", options("Property_Type"))
    sel_bhk = f3.multiselect("BHK", sorted(df["BHK"].unique().tolist()))
    pmin, pmax = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
    smin, smax = float(df["Size_in_SqFt"].min()), float(df["Size_in_SqFt"].max())
    price_rng = st.slider("Price (Lakhs)", pmin, pmax, (pmin, pmax))
    size_rng = st.slider("Area (SqFt)", smin, smax, (smin, smax))
    only_good = st.checkbox("Show only good investments")

    f = df[df["Price_in_Lakhs"].between(*price_rng) & df["Size_in_SqFt"].between(*size_rng)]
    if sel_cities:
        f = f[f["City"].isin(sel_cities)]
    if sel_types:
        f = f[f["Property_Type"].isin(sel_types)]
    if sel_bhk:
        f = f[f["BHK"].isin(sel_bhk)]
    if only_good:
        f = f[f["Good_Investment"] == 1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Matching properties", f"{len(f):,}")
    m2.metric("Average price (Lakhs)", f"{f['Price_in_Lakhs'].mean():,.1f}" if len(f) else "-")
    m3.metric("Good investments", f"{f['Good_Investment'].mean() * 100:.1f}%" if len(f) else "-")
    show = ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Future_Price_5Y", "Good_Investment"]
    st.dataframe(f[show].head(500))
    st.download_button("Download filtered data (CSV)", f[show].to_csv(index=False),
                       "filtered_properties.csv", "text/csv")

# ================================================================ TAB 3: INSIGHTS
with tab3:
    st.subheader("Market insights")
    top_c = df["City"].value_counts().head(15).index
    heat = (df[df["City"].isin(top_c)]
            .pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean"))
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                              title="Heatmap: avg price per SqFt (city x property type)"))

    pick = st.multiselect("Cities for price trend", options("City"), default=list(top_c[:3]))
    if pick:
        tr = (df[df["City"].isin(pick)].groupby(["Year_Built", "City"])["Price_in_Lakhs"]
              .mean().reset_index())
        st.plotly_chart(px.line(tr, x="Year_Built", y="Price_in_Lakhs", color="City",
                                title="Average price by year built"))

    good = df.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(15).reset_index()
    st.plotly_chart(px.bar(good, x="City", y="Good_Investment",
                           title="Share of good investments by city (top 15)"))
    st.plotly_chart(px.histogram(df, x="Price_in_Lakhs", nbins=50, title="Price distribution (Lakhs)"))

# ================================================================ TAB 4: MODELS
with tab4:
    st.subheader("Model performance (test set)")
    st.markdown(f"**Best classifier:** {metrics['best_classifier']}  |  "
                f"**Best regressor:** {metrics['best_regressor']}")
    st.markdown("#### Classification - Good Investment")
    st.dataframe(pd.DataFrame(metrics["classification"]).round(4))
    cm = BASE / "models" / "confusion_matrix.png"
    if cm.exists():
        st.image(str(cm), width=380)
    st.markdown("#### Regression - Price after 5 years")
    st.dataframe(pd.DataFrame(metrics["regression"]).round(4))
    st.info("The Good Investment label comes from a rule-based score, and the 5-year price from "
            "a feature-based growth rate. Accuracy is therefore high by design.")
