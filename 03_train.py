
"""
REAL ESTATE INVESTMENT ADVISOR - interactive Streamlit app (black theme)
Replaces 03_train.py. Needs data/cleaned_housing.csv (made by 01_preprocess.py).

Run : streamlit run 03_train.py
MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db

Pages: Business Problem | EDA | Train Models | Predict | Recommendations | Investor Quiz
Output: models/classifier.joblib, models/regressor.joblib, models/metrics.json,
        models/feature_importance_*.csv, models/confusion_matrix.png
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

BASE = Path(__file__).parent
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)
SEED = 42
GOLD = "#F5B700"

# ------------------------------------------------------------------ PAGE + THEME
st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

st.markdown(f"""
<style>
.stApp, [data-testid="stHeader"], [data-testid="stSidebar"] {{ background-color: #000000; }}
html, body, p, label, span, li, h1, h2, h3, h4, div {{ color: #EDEDED; }}
h1, h2, h3 {{ color: {GOLD} !important; }}
[data-testid="stMetric"] {{ background: #111; border: 1px solid #333; border-radius: 12px; padding: 12px; }}
[data-testid="stMetricValue"] {{ color: {GOLD}; }}
.stButton > button {{ background: {GOLD}; color: #000; font-weight: 700; border: 0; border-radius: 10px; }}
.stButton > button:hover {{ background: #ffd54a; color: #000; }}
.card {{ background: #111; border-left: 4px solid {GOLD}; padding: 14px 18px; border-radius: 8px; margin-bottom: 10px; }}
</style>
""", unsafe_allow_html=True)

LOGO = f"""
<svg width="70" height="70" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <polygon points="50,8 94,46 84,46 84,90 16,90 16,46 6,46" fill="none" stroke="{GOLD}" stroke-width="4"/>
  <rect x="30" y="58" width="9" height="24" fill="{GOLD}"/>
  <rect x="45" y="46" width="9" height="36" fill="{GOLD}"/>
  <rect x="60" y="34" width="9" height="48" fill="{GOLD}"/>
</svg>"""


def px_dark(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000")
    return fig


# ------------------------------------------------------------------ DATA
@st.cache_data
def load_data():
    return pd.read_csv(BASE / "data" / "cleaned_housing.csv")


try:
    df_full = load_data()
except FileNotFoundError:
    st.error("data/cleaned_housing.csv not found. Run 01_preprocess.py first.")
    st.stop()

NUM = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
       "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count"]
CAT = ["State", "City", "Property_Type", "Furnished_Status",
       "Public_Transport_Accessibility", "Parking_Space", "Security", "Facing", "Owner_Type",
       "Availability_Status"]
NUM = [c for c in NUM if c in df_full.columns]
CAT = [c for c in CAT if c in df_full.columns]
# Locality is left out on purpose: too many unique values for one-hot encoding.


# ------------------------------------------------------------------ MODEL HELPERS
def make_prep():
    # scaling + encoding live INSIDE the pipeline, so the app applies the same steps
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def importance(pipe, top=20):
    names = pd.Series(pipe.named_steps["prep"].get_feature_names_out())
    names = names.str.replace(r"^(num|cat)__", "", regex=True)
    est = pipe.named_steps["model"]
    vals = est.feature_importances_ if hasattr(est, "feature_importances_") \
        else np.abs(np.ravel(est.coef_))
    return (pd.DataFrame({"feature": names, "importance": vals})
            .sort_values("importance", ascending=False).head(top))


def clean_params(model):
    return {k: str(v)[:200] for k, v in model.get_params().items() if v is not None}


def train_all(max_rows, test_size, use_mlflow, progress):
    df = df_full
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=SEED)
    X = df[NUM + CAT]
    y_clf, y_reg = df["Good_Investment"], df["Future_Price_5Y"]
    json.dump({"num_cols": NUM, "cat_cols": CAT}, open(MODELS / "feature_config.json", "w"))
    Xtr, Xte, ytr_c, yte_c, ytr_r, yte_r = train_test_split(
        X, y_clf, y_reg, test_size=test_size, random_state=SEED, stratify=y_clf)

    if use_mlflow:
        mlflow.set_tracking_uri(f"sqlite:///{(BASE / 'mlflow.db').as_posix()}")

    # ---------- classification
    clf_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=-1,
                                               random_state=SEED)}
    if HAS_XGB:
        clf_models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                              eval_metric="logloss", n_jobs=-1, random_state=SEED)
    reg_models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=40, max_depth=12, max_features=0.3,
                                              min_samples_leaf=5, n_jobs=-1, random_state=SEED)}
    if HAS_XGB:
        reg_models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                             n_jobs=-1, random_state=SEED)
    total = len(clf_models) + len(reg_models)
    done = 0

    if use_mlflow:
        mlflow.set_experiment("RealEstate_Classification")
    clf_results = []
    for name, model in clf_models.items():
        progress.progress(done / total, text=f"Training classifier: {name}")
        pipe = Pipeline([("prep", make_prep()), ("model", model)])
        pipe.fit(Xtr, ytr_c)
        pred = pipe.predict(Xte)
        m = {"accuracy": accuracy_score(yte_c, pred), "precision": precision_score(yte_c, pred),
             "recall": recall_score(yte_c, pred), "f1": f1_score(yte_c, pred)}
        run_id = None
        if use_mlflow:
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_param("model_name", name)
                mlflow.log_params(clean_params(model))
                mlflow.log_metrics(m)
                mlflow.sklearn.log_model(pipe, artifact_path="model",
                                         serialization_format="cloudpickle")
                run_id = run.info.run_id
        clf_results.append({"model": name, **m, "run_id": run_id, "pipe": pipe})
        done += 1

    best_c = max(clf_results, key=lambda r: r["f1"])
    joblib.dump(best_c["pipe"], MODELS / "classifier.joblib", compress=3)
    importance(best_c["pipe"]).to_csv(MODELS / "feature_importance_classifier.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(yte_c, best_c["pipe"].predict(Xte), ax=ax,
                                            display_labels=["Not good", "Good"], cmap="Blues")
    ax.set_title(f"Confusion matrix - {best_c['model']}")
    plt.tight_layout()
    plt.savefig(MODELS / "confusion_matrix.png", dpi=120)
    plt.close()
    if use_mlflow and best_c["run_id"]:
        try:
            mlflow.register_model(f"runs:/{best_c['run_id']}/model",
                                  "RealEstate_Good_Investment_Classifier")
        except Exception as e:
            st.warning(f"Model registry step skipped: {e}")

    # ---------- regression
    if use_mlflow:
        mlflow.set_experiment("RealEstate_Regression")
    reg_results = []
    for name, model in reg_models.items():
        progress.progress(done / total, text=f"Training regressor: {name}")
        pipe = Pipeline([("prep", make_prep()), ("model", model)])
        pipe.fit(Xtr, ytr_r)
        pred = pipe.predict(Xte)
        m = {"rmse": float(np.sqrt(mean_squared_error(yte_r, pred))),
             "mae": mean_absolute_error(yte_r, pred), "r2": r2_score(yte_r, pred)}
        run_id = None
        if use_mlflow:
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_param("model_name", name)
                mlflow.log_params(clean_params(model))
                mlflow.log_metrics(m)
                mlflow.sklearn.log_model(pipe, artifact_path="model",
                                         serialization_format="cloudpickle")
                run_id = run.info.run_id
        reg_results.append({"model": name, **m, "run_id": run_id, "pipe": pipe})
        done += 1

    best_r = min(reg_results, key=lambda r: r["rmse"])
    joblib.dump(best_r["pipe"], MODELS / "regressor.joblib", compress=3)
    importance(best_r["pipe"]).to_csv(MODELS / "feature_importance_regressor.csv", index=False)
    if use_mlflow and best_r["run_id"]:
        try:
            mlflow.register_model(f"runs:/{best_r['run_id']}/model",
                                  "RealEstate_Future_Price_Regressor")
        except Exception as e:
            st.warning(f"Model registry step skipped: {e}")

    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("pipe", "run_id")} for r in rows]
    out = {"classification": strip(clf_results), "regression": strip(reg_results),
           "best_classifier": best_c["model"], "best_regressor": best_r["model"]}
    json.dump(out, open(MODELS / "metrics.json", "w"), indent=2)
    progress.progress(1.0, text="Done")
    return out


# ------------------------------------------------------------------ HEADER + NAV
c1, c2 = st.columns([1, 9])
c1.markdown(LOGO, unsafe_allow_html=True)
c2.markdown("# Real Estate Investment Advisor")
c2.caption("Find good investments and forecast property prices after 5 years")

page = st.sidebar.radio("Go to", ["🏢 Business Problem", "📊 EDA", "🤖 Train Models",
                                  "🔮 Predict", "💡 Recommendations", "🎯 Investor Quiz"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Rows: {len(df_full):,} | XGBoost: {'yes' if HAS_XGB else 'no'} | "
                   f"MLflow: {'yes' if HAS_MLFLOW else 'no'}")

# ================================================================== BUSINESS PROBLEM
if page.startswith("🏢"):
    st.header("Business Problem")
    st.markdown("""
<div class="card">Buying a property is a big decision. Buyers and investors find it hard to know
<b>which property is a good investment</b> and <b>what its price will be after 5 years</b>.</div>
""", unsafe_allow_html=True)
    st.markdown("""
- **Goal 1 (classification):** Is the property a good investment? Yes / No
- **Goal 2 (regression):** What will the price be after 5 years (in Lakhs)?
- **Users:** buyers, investors, agents and builders
""")
    m1, m2, m3 = st.columns(3)
    m1.metric("Properties", f"{len(df_full):,}")
    m2.metric("Features used", len(NUM) + len(CAT))
    if "City" in df_full.columns:
        m3.metric("Cities", df_full["City"].nunique())

# ================================================================== EDA
elif page.startswith("📊"):
    st.header("Exploratory Data Analysis")

    # filters
    st.sidebar.subheader("EDA filters")
    d = df_full
    if "State" in d.columns:
        states = st.sidebar.multiselect("State", sorted(d["State"].unique()))
        if states:
            d = d[d["State"].isin(states)]
    if "City" in d.columns:
        cities = st.sidebar.multiselect("City", sorted(d["City"].unique()))
        if cities:
            d = d[d["City"].isin(cities)]
    st.caption(f"Showing {len(d):,} rows after filters")

    t1, t2, t3, t4, t5 = st.tabs(["Overview", "Distributions", "Investment Split",
                                  "Price Analysis", "Correlation"])

    with t1:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(d):,}")
        b.metric("Columns", d.shape[1])
        c.metric("Missing values", int(d.isna().sum().sum()))
        st.subheader("Sample data")
        st.dataframe(d.head(50), use_container_width=True)
        st.subheader("Summary statistics")
        st.dataframe(d.describe().T, use_container_width=True)
        st.subheader("Missing values per column")
        miss = d.isna().sum()
        st.dataframe(miss[miss > 0] if miss.sum() else pd.DataFrame({"result": ["No missing values"]}))

    with t2:
        col = st.selectbox("Numeric column", NUM)
        st.plotly_chart(px_dark(px.histogram(d, x=col, nbins=40, color_discrete_sequence=[GOLD])),
                        use_container_width=True)
        st.plotly_chart(px_dark(px.box(d, y=col, color_discrete_sequence=[GOLD])),
                        use_container_width=True)
        ccol = st.selectbox("Categorical column", CAT)
        vc = d[ccol].value_counts().head(15).reset_index()
        vc.columns = [ccol, "count"]
        st.plotly_chart(px_dark(px.bar(vc, x=ccol, y="count", color_discrete_sequence=[GOLD])),
                        use_container_width=True)

    with t3:
        if "Good_Investment" in d.columns:
            vc = d["Good_Investment"].value_counts().reset_index()
            vc.columns = ["Good_Investment", "count"]
            st.plotly_chart(px_dark(px.pie(vc, names="Good_Investment", values="count", hole=0.5,
                                           color_discrete_sequence=[GOLD, "#777"])),
                            use_container_width=True)
            by = st.selectbox("Good investment rate by", CAT, key="by_inv")
            rate = (d.groupby(by)["Good_Investment"].mean().sort_values(ascending=False)
                    .head(15).reset_index())
            st.plotly_chart(px_dark(px.bar(rate, x=by, y="Good_Investment",
                                           color_discrete_sequence=[GOLD])),
                            use_container_width=True)

    with t4:
        if {"Size_in_SqFt", "Price_in_Lakhs"} <= set(d.columns):
            sample = d.sample(min(len(d), 5000), random_state=SEED)
            st.plotly_chart(px_dark(px.scatter(sample, x="Size_in_SqFt", y="Price_in_Lakhs",
                                               color="Good_Investment" if "Good_Investment" in d.columns else None,
                                               opacity=0.6)),
                            use_container_width=True)
        if "City" in d.columns and "Price_per_SqFt" in d.columns:
            cp = (d.groupby("City")["Price_per_SqFt"].mean().sort_values(ascending=False)
                  .head(15).reset_index())
            st.subheader("Average price per sqft by city (top 15)")
            st.plotly_chart(px_dark(px.bar(cp, x="City", y="Price_per_SqFt",
                                           color_discrete_sequence=[GOLD])),
                            use_container_width=True)
        if "Price_in_Lakhs" in d.columns and "Future_Price_5Y" in d.columns:
            st.subheader("Current price vs price after 5 years")
            sample = d.sample(min(len(d), 5000), random_state=SEED)
            st.plotly_chart(px_dark(px.scatter(sample, x="Price_in_Lakhs", y="Future_Price_5Y",
                                               opacity=0.5, color_discrete_sequence=[GOLD])),
                            use_container_width=True)

    with t5:
        cols = NUM + [c for c in ["Future_Price_5Y", "Good_Investment"] if c in d.columns]
        corr = d[cols].corr()
        st.plotly_chart(px_dark(px.imshow(corr, text_auto=".2f", color_continuous_scale="YlOrBr",
                                          aspect="auto")),
                        use_container_width=True)

# ================================================================== TRAIN
elif page.startswith("🤖"):
    st.header("Train Models")
    st.write("Pick the settings and press the button. Scaling and encoding live inside the pipeline.")
    a, b, c = st.columns(3)
    max_rows = a.slider("Training rows (sample)", 5_000, min(len(df_full), 200_000),
                        min(50_000, len(df_full)), step=5_000)
    test_size = b.slider("Test size", 0.1, 0.4, 0.2, step=0.05)
    use_mlflow = c.checkbox("Track with MLflow", value=HAS_MLFLOW, disabled=not HAS_MLFLOW)
    if not HAS_XGB:
        st.info("xgboost not installed - XGBoost models will be skipped (pip install xgboost)")
    if not HAS_MLFLOW:
        st.info("mlflow not installed - training without tracking (pip install mlflow)")

    if st.button("🚀 Train now"):
        bar = st.progress(0.0, text="Starting...")
        st.session_state["results"] = train_all(max_rows, test_size, use_mlflow, bar)

    results = st.session_state.get("results")
    if results is None and (MODELS / "metrics.json").exists():
        results = json.load(open(MODELS / "metrics.json"))
        st.caption("Showing results from the last saved run")

    if results:
        st.success(f"Best classifier: {results['best_classifier']} | "
                   f"Best regressor: {results['best_regressor']}")
        x, y = st.columns(2)
        x.subheader("Classification")
        x.dataframe(pd.DataFrame(results["classification"]).round(4), use_container_width=True)
        y.subheader("Regression")
        y.dataframe(pd.DataFrame(results["regression"]).round(4), use_container_width=True)

        p, q = st.columns(2)
        cm = MODELS / "confusion_matrix.png"
        if cm.exists():
            p.image(str(cm))
        fi = MODELS / "feature_importance_classifier.csv"
        if fi.exists():
            imp = pd.read_csv(fi).head(10).iloc[::-1]
            q.plotly_chart(px_dark(px.bar(imp, x="importance", y="feature", orientation="h",
                                          title="Top features (classifier)",
                                          color_discrete_sequence=[GOLD])),
                           use_container_width=True)
        if HAS_MLFLOW:
            st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db")

# ================================================================== PREDICT
elif page.startswith("🔮"):
    st.header("Predict for a Property")
    if not (MODELS / "classifier.joblib").exists() or not (MODELS / "regressor.joblib").exists():
        st.warning("Train the models first (🤖 Train Models page).")
        st.stop()
    clf = joblib.load(MODELS / "classifier.joblib")
    reg = joblib.load(MODELS / "regressor.joblib")

    vals = {}
    cols = st.columns(3)
    for i, c in enumerate(NUM):
        vals[c] = cols[i % 3].number_input(c, value=float(df_full[c].median()))
    for i, c in enumerate(CAT):
        opts = sorted(df_full[c].dropna().unique().tolist())
        vals[c] = cols[i % 3].selectbox(c, opts, index=opts.index(df_full[c].mode()[0]))

    if st.button("Predict"):
        row = pd.DataFrame([vals])[NUM + CAT]
        proba = clf.predict_proba(row)[0][1]
        future = float(reg.predict(row)[0])
        now = vals.get("Price_in_Lakhs", np.nan)
        r1, r2, r3 = st.columns(3)
        r1.metric("Good investment?", "YES ✅" if proba >= 0.5 else "NO ❌")
        r2.metric("Confidence (good)", f"{proba:.0%}")
        r3.metric("Price after 5 years", f"{future:,.1f} Lakhs",
                  delta=f"{(future - now) / now:.0%}" if now else None)

# ================================================================== RECOMMENDATIONS
elif page.startswith("💡"):
    st.header("Business Recommendations")
    d = df_full
    tips = []
    if {"City", "Good_Investment"} <= set(d.columns):
        top = d.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(3)
        tips.append(f"Focus on cities with the most good investments: {', '.join(top.index)}.")
    if {"Property_Type", "Good_Investment"} <= set(d.columns):
        pt = d.groupby("Property_Type")["Good_Investment"].mean().sort_values(ascending=False)
        tips.append(f"{pt.index[0]} properties have the highest good-investment rate ({pt.iloc[0]:.0%}).")
    if {"Price_in_Lakhs", "Future_Price_5Y"} <= set(d.columns):
        g = ((d["Future_Price_5Y"] - d["Price_in_Lakhs"]) / d["Price_in_Lakhs"]).median()
        tips.append(f"A typical property grows about {g:.0%} in 5 years - compare deals with this.")
    fi = MODELS / "feature_importance_classifier.csv"
    if fi.exists():
        tips.append("Check these first before buying: " + ", ".join(pd.read_csv(fi).head(3)["feature"]) + ".")
    tips.append("Retrain the model every few months with fresh data.")
    for t in tips:
        st.markdown(f'<div class="card">{t}</div>', unsafe_allow_html=True)
    st.caption("Predictions are based on past data and are not a guarantee of returns.")

# ================================================================== INVESTOR QUIZ
else:
    st.header("Investor Quiz")
    st.write("Answer 4 quick questions and get the top 5 properties for you.")
    if not (MODELS / "classifier.joblib").exists() or not (MODELS / "regressor.joblib").exists():
        st.warning("Train the models first (🤖 Train Models page).")
        st.stop()
    clf = joblib.load(MODELS / "classifier.joblib")
    reg = joblib.load(MODELS / "regressor.joblib")

    lo, hi = int(df_full["Price_in_Lakhs"].min()), int(df_full["Price_in_Lakhs"].max())
    mid = int(df_full["Price_in_Lakhs"].median())
    q1, q2 = st.columns(2)
    budget = q1.slider("1. Your budget (Lakhs)", lo, hi, (lo, mid))
    city = q2.selectbox("2. Preferred city", ["Any city"] + sorted(df_full["City"].unique().tolist()))
    bhk = q1.selectbox("3. BHK", ["Any"] + sorted(df_full["BHK"].unique().tolist()))
    goal = q2.radio("4. Your goal", ["Invest (best growth)", "Live in it (best facilities)"])

    if st.button("🎯 Find my properties"):
        d = df_full[(df_full["Price_in_Lakhs"] >= budget[0]) & (df_full["Price_in_Lakhs"] <= budget[1])]
        if city != "Any city":
            d = d[d["City"] == city]
        if bhk != "Any":
            d = d[d["BHK"] == bhk]
        if len(d) == 0:
            st.session_state["quiz"] = None
            st.warning("No properties match. Try a bigger budget or Any city.")
        else:
            d = d.sample(min(len(d), 3000), random_state=SEED).copy() # sample for speed
            d["Good_Probability"] = clf.predict_proba(d[NUM + CAT])[:, 1]
            d["Predicted_Price_5Y"] = reg.predict(d[NUM + CAT])
            d["Growth_%"] = (d["Predicted_Price_5Y"] - d["Price_in_Lakhs"]) / d["Price_in_Lakhs"] * 100
            if goal.startswith("Invest"):
                d = d.sort_values(["Good_Probability", "Growth_%"], ascending=False)
            else:
                fac = [c for c in ["Amenity_Count", "Nearby_Schools", "Nearby_Hospitals"] if c in d.columns]
                d["Facility_Score"] = d[fac].sum(axis=1)
                d = d.sort_values(["Facility_Score", "Good_Probability"], ascending=False)
            st.session_state["quiz"] = (d.head(5).reset_index(drop=True), len(d))

    quiz = st.session_state.get("quiz")
    if quiz:
        top, found = quiz
        st.success(f"Checked {found:,} matching properties. Here are your top 5:")
        show = [c for c in ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                            "Predicted_Price_5Y", "Growth_%", "Good_Probability"] if c in top.columns]
        st.dataframe(top[show].round(2), use_container_width=True)

        chart = top[["Price_in_Lakhs", "Predicted_Price_5Y"]].copy()
        chart["Property"] = [f"{i + 1}. {c}" for i, c in enumerate(top["City"])]
        chart = chart.melt(id_vars="Property", var_name="When", value_name="Lakhs")
        st.plotly_chart(px_dark(px.bar(chart, x="Property", y="Lakhs", color="When", barmode="group",
                                       color_discrete_sequence=[GOLD, "#777"])),
                        use_container_width=True)
        st.download_button("⬇️ Download my top 5 (CSV)", top[show].to_csv(index=False),
                           file_name="my_top_properties.csv", mime="text/csv")