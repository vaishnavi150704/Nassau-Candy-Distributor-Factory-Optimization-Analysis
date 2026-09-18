import os
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Nassau Candy Factory Optimizer", layout="wide", page_icon="🍫")

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(HERE, "processed.csv")
MODEL_PATH = os.path.join(HERE, "best_model.pkl")
RECOMMENDATIONS_PATH = os.path.join(HERE, "recommendations.csv")

FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

REGION_COORDS = {
    "Interior": (39.8283, -98.5795),
    "Atlantic": (38.9072, -77.0369),
    "Gulf": (29.7604, -95.3698),
    "Pacific": (37.7749, -122.4194),
}

categorical_features = ['Current Factory', 'Region', 'Ship Mode', 'Division', 'Product Name']
numeric_features = ['Distance_Miles', 'Sales', 'Units', 'Cost']


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def simulate_product(product, df_hist, pipeline):
    prod_hist = df_hist[df_hist['Product Name'] == product]
    if prod_hist.empty:
        prod_hist = df_hist

    avg_sales = prod_hist['Sales'].mean()
    avg_units = prod_hist['Units'].mean()
    avg_cost = prod_hist['Cost'].mean()
    division = prod_hist['Division'].mode().iloc[0]
    ship_mode = prod_hist['Ship Mode'].mode().iloc[0]

    rows = []
    for factory, (flat, flon) in FACTORY_COORDS.items():
        for region, (rlat, rlon) in REGION_COORDS.items():
            dist = haversine_miles(rlat, rlon, flat, flon)
            rows.append({
                'Current Factory': factory, 'Region': region, 'Ship Mode': ship_mode,
                'Division': division, 'Product Name': product, 'Distance_Miles': dist,
                'Sales': avg_sales, 'Units': avg_units, 'Cost': avg_cost
            })
    candidates = pd.DataFrame(rows)
    candidates['Predicted_Lead_Time'] = pipeline.predict(candidates[categorical_features + numeric_features])

    weights = prod_hist['Region'].value_counts(normalize=True).to_dict()
    default_weight = 1.0 / len(REGION_COORDS)
    candidates['_weight'] = candidates['Region'].map(lambda r: weights.get(r, default_weight))

    factory_scores = candidates.groupby('Current Factory').apply(
        lambda g: np.average(g['Predicted_Lead_Time'], weights=g['_weight'])
    ).rename('Expected_Lead_Time').reset_index()

    current_factory = PRODUCT_FACTORY_MAP.get(product)
    current_lt = factory_scores.loc[factory_scores['Current Factory'] == current_factory, 'Expected_Lead_Time']
    current_lt = float(current_lt.iloc[0]) if not current_lt.empty else np.nan

    factory_scores['Current_Assignment'] = factory_scores['Current Factory'] == current_factory
    factory_scores['Lead_Time_Reduction_Pct'] = ((current_lt - factory_scores['Expected_Lead_Time']) / current_lt) * 100
    factory_scores = factory_scores.sort_values('Expected_Lead_Time').reset_index(drop=True)
    return factory_scores


# ---- Load saved files (with clear error messages instead of crashing) ----
@st.cache_data
def load_processed_data():
    if not os.path.exists(PROCESSED_PATH):
        st.error("processed.csv not found. Run the notebook first (make sure Cell 45 ran).")
        st.stop()
    return pd.read_csv(PROCESSED_PATH)


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        st.error("best_model.pkl not found. Run the notebook's model training cells first.")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load best_model.pkl ({e}). Re-run the notebook in this same environment.")
        st.stop()


@st.cache_data
def load_recommendations():
    return pd.read_csv(RECOMMENDATIONS_PATH) if os.path.exists(RECOMMENDATIONS_PATH) else None


df = load_processed_data()
model_bundle = load_model_bundle()
pipeline = model_bundle['pipeline']

st.title("🍫 Nassau Candy — Factory Optimization Simulator")
st.caption(f"Model: **{model_bundle['name']}** | RMSE={model_bundle['metrics']['RMSE']:.2f} days | R²={model_bundle['metrics']['R2']:.2f}")

# ---- Sidebar ----
st.sidebar.header("Controls")
selected_product = st.sidebar.selectbox("Product", sorted(PRODUCT_FACTORY_MAP.keys()))

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["🏭 Simulator", "🔍 What-If Comparison", "📊 All Recommendations"])

sim_df = simulate_product(selected_product, df, pipeline)

with tab1:
    st.subheader(f"Expected lead time by factory — {selected_product}")
    fig = px.bar(sim_df, x="Current Factory", y="Expected_Lead_Time",
                 color="Current_Assignment",
                 color_discrete_map={True: "#2E7D32", False: "#90A4AE"},
                 text_auto=".2f")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sim_df, hide_index=True, use_container_width=True)

with tab2:
    current_factory = PRODUCT_FACTORY_MAP[selected_product]
    best_row = sim_df.iloc[0]
    current_row = sim_df[sim_df["Current Factory"] == current_factory].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Factory", current_factory, f"{current_row['Expected_Lead_Time']:.2f} days")
    c2.metric("Best Factory", best_row["Current Factory"], f"{best_row['Expected_Lead_Time']:.2f} days")
    c3.metric("Improvement", f"{best_row['Lead_Time_Reduction_Pct']:.1f}%")

with tab3:
    all_recs = load_recommendations()
    if all_recs is None:
        st.warning("recommendations.csv not found. Run the notebook's simulation cells first.")
    else:
        st.dataframe(all_recs, hide_index=True, use_container_width=True)