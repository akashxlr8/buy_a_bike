# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="India 350-650cc Motorcycles — Evaluation Studio",
                   layout="wide",
                   page_icon="🏍️")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("advanced_motorcycle_evaluation_complete.csv")
    # Coerce types where needed
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df

# Two-line explanations for each column
EXPLAIN = {
    "Bike_Name": [
        "Concatenation of Manufacturer and Model for unique identification.",
        "Used across charts and comparisons as the display label."
    ],
    "Manufacturer": [
        "Brand building and service presence affect long-term ownership.",
        "Also used for group insights and average scoring by brand."
    ],
    "Model": [
        "Exact commercial model identifier from the dataset.",
        "Useful for precise comparisons within a brand."
    ],
    "Ex_Showroom_Price_Lakh": [
        "Base ex-showroom price in lakh rupees for pan-India comparison.",
        "Major driver of affordability and value-for-money perception."
    ],
    "On_Road_Price_Est_Lakh": [
        "Estimated on-road price at ~20% above ex-showroom for planning.",
        "Useful for budget allocation before purchase."
    ],
    "Insurance_Annual_Est": [
        "Estimated annual insurance based on engine displacement.",
        "Recurring cost to include in total cost of ownership."
    ],
    "EMI_2000_Down_36m": [
        "Illustrative EMI with minimal down payment over 36 months.",
        "Rough guide to monthly outflow for financing decisions."
    ],
    "Price_Score": [
        "Normalized affordability score relative to the segment.",
        "Higher is more affordable or better priced for features."
    ],
    "Insurance_Score": [
        "Lower recurring insurance implies a higher score.",
        "Helps compare running cost differences across bikes."
    ],
    "Displacement_CC": [
        "Engine size in cubic centimeters indicating potential output.",
        "Often correlates with performance and insurance cost."
    ],
    "Power_PS": [
        "Peak power output measured in PS for acceleration/top speed.",
        "Higher values generally aid performance riding."
    ],
    "Torque_Nm": [
        "Peak torque (Nm) indicating tractability and low-end pull.",
        "Important for city riding and overtakes in higher gears."
    ],
    "Mileage_KMPL": [
        "Claimed or indicative fuel efficiency in KMPL.",
        "Key factor for commuting and touring range."
    ],
    "Power_to_Weight_Ratio": [
        "Power scaled per 1000 kg for dynamic performance feel.",
        "Useful single-number performance comparator."
    ],
    "Displacement_Score": [
        "Score derived from CC bands aligned with use-cases.",
        "Higher favors versatile performance envelopes."
    ],
    "Power_Score": [
        "Power normalized to a 10-point scale.",
        "Higher equals stronger acceleration/top-end potential."
    ],
    "Torque_Score": [
        "Torque normalized to a 10-point scale.",
        "Higher suits low-speed drivability and hill climbs."
    ],
    "Mileage_Score": [
        "Efficiency normalized to a 10-point scale.",
        "Higher equals lower running costs and longer range."
    ],
    "Power_to_Weight_Score": [
        "PTW normalized to a 10-point scale.",
        "Higher equals more responsive and lively feel."
    ],
    "Engine_Configuration": [
        "Single/Twin/Triple configuration impacting refinement and feel.",
        "Affects vibrations, sound, and complexity."
    ],
    "Cooling_System": [
        "Air / Air-oil / Liquid cooling affecting heat management.",
        "Critical for hot climate and spirited riding."
    ],
    "Transmission": [
        "Gearbox type and count e.g., 5-speed/6-speed manual.",
        "Impacts cruising RPM and acceleration spread."
    ],
    "Number_of_Gears": [
        "Parsed from transmission for quick filtering.",
        "Higher gear count helps efficiency and flexibility."
    ],
    "Engine_Config_Score": [
        "Refinement-oriented scoring (Triple> Twin> Single).",
        "Proxy for smoothness and premium feel."
    ],
    "Cooling_Score": [
        "Liquid>Air-oil>Air for hot weather performance.",
        "Better cooling improves consistency over long rides."
    ],
    "Transmission_Score": [
        "More gears generally score better for versatility.",
        "Impacts highway comfort and city tractability."
    ],
    "Braking_System": [
        "ABS configuration string from spec sheet.",
        "Dual-channel preferred for safety and stability."
    ],
    "ABS_Score": [
        "Dual-channel > Single-channel in safety scoring.",
        "Higher equals better braking stability."
    ],
    "Seat_Height_MM": [
        "Seat height in mm for rider reach and confidence.",
        "Optimal range 750–800 mm for wider accessibility."
    ],
    "Kerb_Weight_KG": [
        "Ready-to-ride weight affecting agility and stability.",
        "Lighter is more nimble; heavier is more planted."
    ],
    "Seat_Height_Score": [
        "Best access in 750–800mm band for most riders.",
        "Higher score equals easier ground reach."
    ],
    "Weight_Score": [
        "Lighter bikes score higher for urban maneuverability.",
        "Balances stability and ease of handling."
    ],
    "Fuel_Tank_Capacity_L": [
        "Tank capacity in liters for touring range planning.",
        "Bigger tanks often preferred for highway touring."
    ],
    "Theoretical_Range_KM": [
        "Fuel tank × mileage gives indicative full-tank range.",
        "Useful for route planning and fuel stops."
    ],
    "Fuel_Tank_Score": [
        "Normalized capacity scoring with diminishing returns.",
        "Balance capacity vs added bulk/weight."
    ],
    "Range_Score": [
        "Normalized range (km) scoring on a 10-point scale.",
        "Higher equals fewer refueling breaks."
    ],
    "Brand_Reputation_Score": [
        "Brand trust proxy from market presence and reliability.",
        "Impacts resale, community, and peace of mind."
    ],
    "Service_Network_Score": [
        "Coverage proxy for ease of service across regions.",
        "Higher equals better support availability."
    ],
    "Spare_Parts_Score": [
        "Proxy for parts availability and lead times.",
        "Higher equals easier maintenance."
    ],
    "Overall_Weighted_Score": [
        "Weighted blend of key scores for balanced ranking.",
        "Top-line indicator for quick shortlisting."
    ],
    "Value_for_Money": [
        "Composite of performance, safety, and price.",
        "Higher equals more performance/features per rupee."
    ],
}

def two_line(col):
    lines = EXPLAIN.get(col, ["No description available.", ""] )
    return f"{lines}\n{lines[1]}"


def higher_better_for():
    """Return a mapping of column -> whether higher values are better.

    True means greener when delta > 0. False means greener when delta < 0.
    """
    return {
        # positive is better
        "Mileage_KMPL": True,
        "Displacement_CC": True,
        "Power_PS": True,
        "Torque_Nm": True,
        "Power_to_Weight_Ratio": True,
        "Price_Score": True,
        "Power_Score": True,
        "Torque_Score": True,
        "Mileage_Score": True,
        # negative is better (lower price, lower insurance, lower EMI)
        "Ex_Showroom_Price_Lakh": False,
        "On_Road_Price_Est_Lakh": False,
        "Insurance_Annual_Est": False,
        "EMI_2000_Down_36m": False,
    }


def style_deltas(df, col_preferences=None, cmap_pos="Greens", cmap_neg="Reds"):
    """Return a pandas Styler for the deltas DataFrame where colors reflect meaningful direction.

    - col_preferences: dict col->bool where True means higher is better.
    - Positive deltas shown using Greens (more positive -> greener when higher is better,
      or redder when lower is better). Negative deltas shown using Reds.
    """
    if col_preferences is None:
        col_preferences = higher_better_for()

    # normalize deltas per column to map to 0..1
    styled = df.copy()

    def color_for_value(col, val):
        # pick preference (True => higher better)
        pref = col_preferences.get(col, True)
        # use symmetric scaling based on max absolute value
        col_vals = df[col].astype(float)
        max_abs = max(col_vals.abs().max(), 1e-6)
        norm = abs(val) / max_abs
        norm = min(norm, 1.0)
        if val == 0:
            return "background-color: transparent"
        if (val > 0 and pref) or (val < 0 and not pref):
            # positive is 'good' -> use green palette
            cmap = matplotlib.colormaps.get_cmap(cmap_pos)
            rgba = cmap(norm)
        else:
            # positive is 'bad' -> use red palette (i.e., negative is good)
            cmap = matplotlib.colormaps.get_cmap(cmap_neg)
            rgba = cmap(norm)
        hexc = mcolors.to_hex(rgba)
        
        return f"background-color: {hexc}; color: white"

    def apply_row(row):
        return [color_for_value(col, row[col]) for col in df.columns]

    styler = df.style.apply(lambda r: apply_row(r), axis=1)
    return styler

# UI Header
st.title("🏍️ India 350–650cc Motorcycles — Interactive Evaluation Studio")
st.caption("Explore, filter, and compare every motorcycle with rich charts, tables, and a full evaluation matrix.")

# Data loading
with st.sidebar:
    st.header("Data Source")
    file = st.file_uploader("Upload advanced_motorcycle_evaluation_complete.csv", type=["csv"])
    df = load_data(file)
    st.success(f"Loaded {len(df)} motorcycles with {len(df.columns)} columns")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    st.info(f"DataFrame columns: {list(df.columns)}")
    manu = st.multiselect("Manufacturer", sorted(df["Manufacturer"].unique().tolist()), default=None)
    cfg = st.multiselect("Engine Configuration", sorted(df["Engine_Configuration"].unique().tolist()), default=None)
    cool = st.multiselect("Cooling System", sorted(df["Cooling_System"].unique().tolist()), default=None)
    abs_type = st.multiselect("Braking System (ABS)", sorted(df["Braking_System"].unique().tolist()), default=None)

    def rng(col):
        vmin, vmax = float(df[col].min()), float(df[col].max())
        return st.slider(col.replace("_"," "), vmin, vmax, (vmin, vmax))

    price_rng = rng("Ex_Showroom_Price_Lakh")
    cc_rng = rng("Displacement_CC")
    pwr_rng = rng("Power_PS")
    tq_rng = rng("Torque_Nm")
    mpg_rng = rng("Mileage_KMPL")
    ht_rng = rng("Seat_Height_MM")
    wt_rng = rng("Kerb_Weight_KG")
    tank_rng = rng("Fuel_Tank_Capacity_L")
    range_rng = rng("Theoretical_Range_KM")

def apply_filters(df):
    out = df.copy()
    if manu:
        out = out[out["Manufacturer"].isin(manu)]
    if cfg:
        out = out[out["Engine_Configuration"].isin(cfg)]
    if cool:
        out = out[out["Cooling_System"].isin(cool)]
    if abs_type:
        out = out[out["Braking_System"].isin(abs_type)]

    def between(col, rng):
        return (out[col] >= rng[0]) & (out[col] <= rng[1])

    out = out[between("Ex_Showroom_Price_Lakh", price_rng)]
    out = out[between("Displacement_CC", cc_rng)]
    out = out[between("Power_PS", pwr_rng)]
    out = out[between("Torque_Nm", tq_rng)]
    out = out[between("Mileage_KMPL", mpg_rng)]
    out = out[between("Seat_Height_MM", ht_rng)]
    out = out[between("Kerb_Weight_KG", wt_rng)]
    out = out[between("Fuel_Tank_Capacity_L", tank_rng)]
    out = out[between("Theoretical_Range_KM", range_rng)]
    return out

fil = apply_filters(df)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Bike Explorer", "Compare Bikes", "Chart Studio", "Manufacturers", "Export"
])

with tab1:
    st.subheader("Overview Dashboard")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Bikes", len(fil))
    k2.metric("Avg Ex-Showroom (L)", f"{fil['Ex_Showroom_Price_Lakh'].mean():.2f}")
    k3.metric("Avg Power (PS)", f"{fil['Power_PS'].mean():.1f}")
    k4.metric("Avg Mileage (KMPL)", f"{fil['Mileage_KMPL'].mean():.1f}")
    k5.metric("Avg Overall Score", f"{fil['Overall_Weighted_Score'].mean():.2f}")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("#### Ranking by Overall Weighted Score")
        top = fil.sort_values("Overall_Weighted_Score", ascending=False)
        fig = px.bar(top.head(20), x="Bike_Name", y="Overall_Weighted_Score",
                     color="Manufacturer", text="Overall_Weighted_Score",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(xaxis_tickangle=-35, height=500, margin=dict(l=10,r=10,b=10,t=40))
        st.plotly_chart(fig, width='stretch')
        st.caption("Ranks motorcycles by the blended score to quickly shortlist options.")
        st.caption("Use the sidebar to refine by brand, engine, or budget, and see ranks update.")

    with c2:
        st.markdown("#### Price vs Power (Bubble = PTW)")
        fig2 = px.scatter(fil, x="Ex_Showroom_Price_Lakh", y="Power_PS",
                          color="Engine_Configuration",
                          size="Power_to_Weight_Ratio",
                          hover_data=["Bike_Name","Manufacturer","Torque_Nm","Mileage_KMPL"],
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=500, margin=dict(l=10,r=10,b=10,t=40))
        st.plotly_chart(fig2, width='stretch')
        st.caption("Bubbles farther right and higher indicate more expensive and powerful bikes.")
        st.caption("Larger bubbles have better power-to-weight ratios for a lively feel.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Mileage Distribution")
        fig3 = px.histogram(fil, x="Mileage_KMPL", nbins=20, color="Manufacturer",
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(height=420, margin=dict(l=10,r=10,b=10,t=40))
        st.plotly_chart(fig3, width='stretch')
        st.caption("Visualizes fuel efficiency spread across all filtered bikes.")
        st.caption("Right-skewed histograms indicate more efficient sets in view.")

    with c4:
        st.markdown("#### Score Radar (Select One)")
        target = st.selectbox("Pick a bike", fil["Bike_Name"].tolist())
        one_row = fil[fil["Bike_Name"]==target].iloc[0]
        radar_fields = ["Price_Score","Power_Score","Mileage_Score","ABS_Score","Engine_Config_Score",
                "Cooling_Score","Transmission_Score","Seat_Height_Score","Weight_Score",
                "Range_Score","Service_Network_Score","Brand_Reputation_Score"]
        theta = radar_fields + [radar_fields[0]]
        r = [float(one_row[x]) for x in radar_fields] + [float(one_row[radar_fields[0]])]
        fig4 = go.Figure(data=go.Scatterpolar(r=r, theta=theta, fill='toself', name=one_row["Bike_Name"]))
        fig4.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=False, height=420)
        st.plotly_chart(fig4, width='stretch')
        st.caption("Spider chart highlights where the selected bike excels or lags.")
        st.caption("Use to match strengths with personal priorities and routes.")

with tab2:
    st.subheader("Bike Explorer")
    st.caption("Filter further by searching the table and click a row to pin details below.")
    st.dataframe(fil, width='stretch', height=420)
    st.markdown("---")
    st.markdown("#### Column Guide — Two-line explanations")
    for col in fil.columns:
        with st.expander(col):
            st.write(two_line(col))

with tab3:
    st.subheader("Bike-to-Bike Comparison Matrix")
    opts = fil["Bike_Name"].tolist()
    picks = st.multiselect("Choose 2–5 bikes to compare", opts, default=opts[:2])
    if len(picks) >= 2:
        comp = fil[fil["Bike_Name"].isin(picks)].copy()
        st.markdown("##### Full Specification Table")
        st.dataframe(comp.set_index("Bike_Name"), width='stretch', height=400)

        # Differences vs baseline
        base = comp.iloc[0]
        st.markdown(f"##### Delta vs baseline: {base['Bike_Name']}")
        num_cols = [c for c in comp.columns if pd.api.types.is_numeric_dtype(comp[c])]
        deltas = comp.copy()
        for c in num_cols:
            deltas[c] = comp[c] - float(base[c])
        df_for_style = deltas.set_index("Bike_Name")[num_cols]
        styler = style_deltas(df_for_style)
        st.dataframe(styler, width='stretch')
        st.caption("Positive deltas mean higher than baseline; negative implies lower than baseline.")
        st.caption("Use to quickly spot where each bike gains or loses versus the selected baseline.")
    else:
        st.info("Pick at least two bikes to see the full matrix and deltas.")

with tab4:
    st.subheader("Chart Studio — Build Custom Visuals")
    numeric_cols = [c for c in fil.columns if pd.api.types.is_numeric_dtype(fil[c])]
    cat_cols = [c for c in fil.columns if c not in numeric_cols]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        chart_type = st.selectbox("Chart Type", ["Scatter","Bar","Histogram","Box","Violin"])
    with col2:
        x_col = st.selectbox("X Axis", numeric_cols + cat_cols)
    with col3:
        y_col = st.selectbox("Y Axis (if applicable)", [None] + numeric_cols + cat_cols)
    with col4:
        color_col = st.selectbox("Color", [None] + cat_cols + numeric_cols)

    size_col = st.selectbox("Bubble Size (scatter only)", [None] + numeric_cols)
    agg = st.selectbox("Aggregation (Bar/Histogram)", ["None","count","mean","median","sum","min","max"])

    if chart_type == "Scatter":
        fig = px.scatter(fil, x=x_col, y=y_col, color=color_col, size=size_col,
                         hover_data=["Bike_Name","Manufacturer","Power_PS","Mileage_KMPL","Ex_Showroom_Price_Lakh"])
    elif chart_type == "Bar":
        if agg == "None":
            fig = px.bar(fil, x=x_col, y=y_col, color=color_col, barmode="group")
        else:
            grouped = fil.groupby(x_col, as_index=False).agg({y_col:agg})
            fig = px.bar(grouped, x=x_col, y=y_col, color=color_col)
    elif chart_type == "Histogram":
        fig = px.histogram(fil, x=x_col, color=color_col, nbins=25)
    elif chart_type == "Box":
        fig = px.box(fil, x=color_col, y=x_col, points="all")
    elif chart_type == "Violin":
        fig = px.violin(fil, x=color_col, y=x_col, box=True, points="all")
    else:
        fig = go.Figure()

    fig.update_layout(height=520, margin=dict(l=10,r=10,b=10,t=40))
    st.plotly_chart(fig, width='stretch')
    st.caption("Use the studio to build the exact visualization needed for a specific question.")
    st.caption("Try Price vs Power with bubble size as PTW and color by Engine Configuration for performance insights.")

with tab5:
    st.subheader("Manufacturer Insights")
    grp = fil.groupby("Manufacturer").agg(
        Models=("Bike_Name","count"),
        Avg_Price=("Ex_Showroom_Price_Lakh","mean"),
        Avg_Power=("Power_PS","mean"),
        Avg_Mileage=("Mileage_KMPL","mean"),
        Avg_Overall=("Overall_Weighted_Score","mean"),
        Avg_VFM=("Value_for_Money","mean")
    ).reset_index().sort_values("Avg_Overall", ascending=False)

    st.dataframe(grp, width='stretch')
    fig = px.bar(grp, x="Manufacturer", y="Avg_Overall", text="Avg_Overall",
                 color="Avg_Price", color_continuous_scale="Blues")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=480, margin=dict(l=10,r=10,b=10,t=40))
    st.plotly_chart(fig, width='stretch')
    st.caption("Ranks brands by average overall score across filtered models.")
    st.caption("Hover for power, mileage, and price context per manufacturer.")

with tab6:
    st.subheader("Export Filtered Dataset")
    st.caption("Download the current filtered dataset, preserving all columns and computed fields.")
    csv = fil.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="filtered_motorcycles_evaluation.csv", mime="text/csv")
    st.caption("Use exported data for offline analysis or sharing.")
    st.caption("Filters in the sidebar control which rows are included in the export.")
