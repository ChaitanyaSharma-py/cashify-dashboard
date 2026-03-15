"""
Cashify MIS Live Brand Study — Interactive Decision Support Dashboard
Run: streamlit run cashify_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cashify Brand Study — DSS Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
  .section-header {
      background: linear-gradient(90deg, #0B2A3B, #1a4a6b);
      color: white; padding: 10px 18px; border-radius: 6px;
      font-size: 1.05rem; font-weight: 700; margin-bottom: 10px;
  }
  .kpi-card {
      background: #f8fafc; border: 1px solid #e2e8f0;
      border-radius: 8px; padding: 14px 18px; text-align: center;
  }
  .cashify-badge {
      background: #008891; color: white; padding: 3px 10px;
      border-radius: 12px; font-size: 0.78rem; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
CASHIFY_COLOR = "#008891"
COLORS = {
    "Cashify": "#008891",
    "OLX": "#1B3A5C",
    "Amazon Exchange": "#F59E0B",
    "Amazon Renewed": "#F59E0B",
    "Flipkart Exchange": "#6366F1",
    "Flipkart Reset": "#6366F1",
    "Quickr": "#94A3B8",
    "Local shop": "#10B981",
    "FB Marketplace": "#3B82F6",
    "GetInstaCash": "#EC4899",
    "PhoneCash": "#8B5CF6",
    "Refit Global": "#64748B",
    "Sahivalue": "#D97706",
    "ControlZ": "#9CA3AF",
    "XtraCover": "#6B7280",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    from pathlib import Path
    base = Path(__file__).parent

    # Auto-detect buyback file — works regardless of exact filename
    bb_file = next((f for f in base.glob("*.xlsx") if "uyback" in f.name), None)
    rf_file = next((f for f in base.glob("*.xlsx") if "efurb" in f.name), None)

    if bb_file is None or rf_file is None:
        found = [f.name for f in base.glob("*.xlsx")]
        raise FileNotFoundError(
            f"Excel files not found in {base}. Files present: {found}"
        )

    bb_raw = pd.read_excel(bb_file)
    rf_raw = pd.read_excel(rf_file)
    bb = bb_raw.iloc[1:].copy().reset_index(drop=True)
    rf = rf_raw.iloc[1:].copy().reset_index(drop=True)

    # ── Standardise demographic columns ─────────────────────────────────────
    for df in [bb, rf]:
        df["Q1"]  = df["Q1"].astype(str).str.strip()
        df["Q2"]  = df["Q2"].astype(str).str.strip()
        df["Q3"]  = df["Q3"].astype(str).str.strip()
        df["Q7"]  = df["Q7"].astype(str).str.strip()
        df["Q8"]  = df["Q8"].astype(str).str.strip()

        # City tier grouping
        df["City_Tier"] = df["Q1"].apply(lambda x:
            "Metro / Tier-1" if x in ["Delhi NCR","Mumbai","Bangalore","Kolkata",
                                       "Hyderabad","Chennai","Ahmedabad"]
            else ("Tier-2/3" if "Tier" in x else "Other"))

        # Clean gender
        df["Gender"] = df["Q2"].apply(lambda x:
            "Male" if x=="Male" else ("Female" if x=="Female" else "Other/Not Specified"))

        # Age bucket
        df["Age"]    = df["Q3"].apply(lambda x:
            x if x in ["<18","18–25","26–35","36–45","46+"] else "Unknown")

        # Income tier
        df["Income"] = df["Q8"].apply(lambda x:
            "High (>₹1.5L)" if x in [">₹2L","₹1.5–2L"]
            else ("Mid (₹50k–1.5L)" if x in ["₹1–1.5L","₹75k–1L","₹50–75k"]
            else ("Low (<₹50k)" if x in ["₹25–50k","<₹25k"] else "Other")))

        # Working status
        df["Occupation"] = df["Q7"].apply(lambda x:
            x if x in ["Working (full-time)","Student","Self-employed",
                        "Homemaker","Unemployed","Retired"] else "Other")

    return bb, rf

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: MULTI-SELECT MENTION COUNTER
# ─────────────────────────────────────────────────────────────────────────────
def count_multiselect(series, term):
    """Count rows where term appears in a comma-separated multi-select column."""
    return series.dropna().apply(lambda x: term.lower() in str(x).lower()).sum()

def parse_multiselect_counts(series, terms):
    """Returns dict of {term: count} for a list of terms."""
    return {t: count_multiselect(series, t) for t in terms}

def compute_nps(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return None, None, None, None
    p  = (vals >= 9).sum() / len(vals) * 100
    pa = ((vals >= 7) & (vals <= 8)).sum() / len(vals) * 100
    d  = (vals <= 6).sum() / len(vals) * 100
    nps = round(p - d, 1)
    return nps, round(p,1), round(pa,1), round(d,1)

def weighted_rank_score(df, cols, labels, n_ranks=5):
    """Compute weighted rank score: rank 1 = n_ranks pts, rank 5 = 1 pt."""
    scores = {}
    for col, label in zip(cols, labels):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        score = sum((n_ranks + 1 - r) * (vals == r).sum() for r in range(1, n_ranks+1))
        scores[label] = int(score)
    return pd.DataFrame({"Driver": list(scores.keys()), "Score": list(scores.values())}).sort_values("Score", ascending=False)

# ─────────────────────────────────────────────────────────────────────────────
# BRAND / QUESTION MAPS
# ─────────────────────────────────────────────────────────────────────────────
BB_BRANDS_Q14 = {
    "Q14_1":"Cashify","Q14_2":"OLX","Q14_3":"Amazon Exchange",
    "Q14_4":"Flipkart Exchange","Q14_5":"Quickr","Q14_6":"Flipkart Reset",
    "Q14_7":"GetInstaCash","Q14_8":"PhoneCash","Q14_9":"Refit Global"
}
BB_BRANDS_Q15 = {
    "Q15_1":"Cashify","Q15_2":"OLX","Q15_3":"Amazon Exchange",
    "Q15_4":"Flipkart Exchange","Q15_5":"Quickr","Q15_6":"Flipkart Reset",
    "Q15_7":"GetInstaCash","Q15_8":"PhoneCash","Q15_9":"Refit Global"
}
BB_BRANDS_Q16 = {
    "Q16_1":"Cashify","Q16_2":"OLX","Q16_3":"Amazon Exchange",
    "Q16_4":"Flipkart Exchange","Q16_5":"Quickr","Q16_6":"Flipkart Reset",
    "Q16_7":"GetInstaCash","Q16_8":"PhoneCash","Q16_9":"Refit Global"
}
BB_BRANDS_Q13 = {
    "Q13_1":"Cashify","Q13_2":"OLX","Q13_3":"Quickr",
    "Q13_4":"Flipkart Reset","Q13_5":"Refit Global","Q13_6":"GetInstaCash",
    "Q13_7":"PhoneCash","Q13_8":"Amazon Exchange","Q13_9":"Flipkart Exchange"
}

RF_BRANDS_Q14 = {
    "Q14_1":"Cashify","Q14_2":"OLX","Q14_3":"Amazon Renewed",
    "Q14_4":"Local shop","Q14_5":"Flipkart Reset","Q14_6":"FB Marketplace",
    "Q14_7":"Sahivalue","Q14_8":"XtraCover","Q14_9":"Refit Global","Q14_10":"ControlZ"
}
RF_BRANDS_Q15 = {
    "Q15_1":"Cashify","Q15_2":"OLX","Q15_3":"Amazon Renewed",
    "Q15_4":"Local shop","Q15_5":"Flipkart Reset","Q15_6":"FB Marketplace",
    "Q15_7":"Sahivalue","Q15_8":"XtraCover","Q15_9":"Refit Global","Q15_10":"ControlZ"
}
RF_BRANDS_Q16 = {
    "Q16_1":"Cashify","Q16_2":"OLX","Q16_3":"Amazon Renewed",
    "Q16_4":"Local shop","Q16_5":"Flipkart Reset","Q16_6":"FB Marketplace",
    "Q16_7":"Sahivalue","Q16_8":"XtraCover","Q16_9":"Refit Global","Q16_10":"ControlZ"
}
RF_BRANDS_Q13 = {
    "Q13_1":"Cashify","Q13_2":"Amazon Renewed","Q13_3":"Flipkart Reset",
    "Q13_4":"OLX","Q13_5":"Local shop","Q13_6":"FB Marketplace",
    "Q13_7":"Sahivalue","Q13_8":"XtraCover","Q13_9":"Refit Global","Q13_10":"ControlZ"
}

AWARENESS_SOURCES = [
    "Google search",
    "Social media ad (Instagram/Facebook)",
    "TV ad",
    "YouTube review/Unboxing",
    "Saw as brand integration on a Youtube show",
    "Influencer video recommending it",
    "Price–comparison site/Deal–blog",
    "Offline store",
]
AWARENESS_SOURCE_SHORT = {
    "Google search": "Google Search",
    "Social media ad (Instagram/Facebook)": "Social Media",
    "TV ad": "TV Ad",
    "YouTube review/Unboxing": "YT Review",
    "Saw as brand integration on a Youtube show": "YT Show",
    "Influencer video recommending it": "Influencer",
    "Price–comparison site/Deal–blog": "Deal Blog",
    "Offline store": "Offline Store",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    bb_full, rf_full = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"⚠️  Could not load data files. Make sure both Excel files are uploaded to the ROOT of your GitHub repository (same level as cashify_dashboard.py).\n\nError: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — GLOBAL FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📱 Cashify")
    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")

    journey = st.radio("Journey", ["Buyback (Sell)", "Refurbished (Buy)", "Both"], index=2)

    st.markdown("**Demographics**")
    city_opts   = ["All"] + sorted([c for c in bb_full["City_Tier"].unique() if c != "Other"])
    gender_opts = ["All"] + sorted([g for g in bb_full["Gender"].unique() if g != "Other/Not Specified"])
    age_opts    = ["All"] + [a for a in ["<18","18–25","26–35","36–45","46+"] if a in bb_full["Age"].unique()]
    income_opts = ["All"] + ["Low (<₹50k)","Mid (₹50k–1.5L)","High (>₹1.5L)"]
    occ_opts    = ["All"] + sorted([o for o in bb_full["Occupation"].unique() if o != "Other"])

    f_city   = st.selectbox("City Tier",    city_opts)
    f_gender = st.selectbox("Gender",       gender_opts)
    f_age    = st.selectbox("Age Bucket",   age_opts)
    f_income = st.selectbox("Income",       income_opts)
    f_occ    = st.selectbox("Occupation",   occ_opts)

    st.markdown("---")
    st.caption("MIS Live Brand Study | Cashify DSS")

# ─────────────────────────────────────────────────────────────────────────────
# FILTER HELPER
# ─────────────────────────────────────────────────────────────────────────────
def apply_filters(df):
    d = df.copy()
    if f_city   != "All": d = d[d["City_Tier"]   == f_city]
    if f_gender != "All": d = d[d["Gender"]       == f_gender]
    if f_age    != "All": d = d[d["Age"]          == f_age]
    if f_income != "All": d = d[d["Income"]       == f_income]
    if f_occ    != "All": d = d[d["Occupation"]   == f_occ]
    return d

bb = apply_filters(bb_full)
rf = apply_filters(rf_full)

n_bb = len(bb)
n_rf = len(rf)

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION TABS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📱 Cashify Consumer Intelligence — Decision Support System")

tabs = st.tabs([
    "🏠 Overview",
    "📣 Brand Awareness",
    "🔽 Health Funnel",
    "⭐ NPS",
    "📡 Source of Awareness",
    "🎯 Consideration Set",
    "💡 Choice Drivers",
    "🚧 Barriers",
    "📊 Category Insights",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Study Overview & KPI Summary</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Respondents", f"{n_bb+n_rf:,}")
    c2.metric("Buyback (Sell-side)", f"{n_bb:,}")
    c3.metric("Refurbished (Buy-side)", f"{n_rf:,}")
    c4.metric("Active Filters", sum([f_city!="All", f_gender!="All", f_age!="All", f_income!="All", f_occ!="All"]))

    st.markdown("---")

    # KPIs
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📊 Buyback Key Metrics (Cashify)")
        nps_bb,_,_,_ = compute_nps(bb["Q16_1"])
        tom_bb = bb["Q10"].dropna().apply(lambda x: "cashify" in str(x).lower()).sum()
        aided_bb = bb["Q12"].dropna().apply(lambda x: "cashify" in str(x).lower()).sum()
        consid_bb = bb["Q15_1"].dropna().apply(lambda x: "first choice" in str(x).lower() or "seriously consider" in str(x).lower()).sum()

        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("TOM", f"{tom_bb/n_bb*100:.1f}%" if n_bb>0 else "—")
        mc2.metric("Aided Aware.", f"{aided_bb/n_bb*100:.1f}%" if n_bb>0 else "—")
        mc3.metric("Consideration", f"{consid_bb/n_bb*100:.1f}%" if n_bb>0 else "—")
        mc4.metric("NPS", f"{nps_bb}" if nps_bb is not None else "—",
                   delta="Best in category" if nps_bb is not None else None)

    with col2:
        st.markdown("##### 📊 Refurbished Key Metrics (Cashify)")
        nps_rf,_,_,_ = compute_nps(rf["Q16_1"])
        tom_rf = rf["Q10"].dropna().apply(lambda x: "cashify" in str(x).lower()).sum()
        aided_rf = rf["Q12"].dropna().apply(lambda x: "cashify" in str(x).lower()).sum()
        consid_rf = rf["Q15_1"].dropna().apply(lambda x: "first choice" in str(x).lower() or "seriously consider" in str(x).lower()).sum()

        mr1,mr2,mr3,mr4 = st.columns(4)
        mr1.metric("TOM", f"{tom_rf/n_rf*100:.1f}%" if n_rf>0 else "—")
        mr2.metric("Aided Aware.", f"{aided_rf/n_rf*100:.1f}%" if n_rf>0 else "—")
        mr3.metric("Consideration", f"{consid_rf/n_rf*100:.1f}%" if n_rf>0 else "—")
        mr4.metric("NPS", f"{nps_rf}" if nps_rf is not None else "—",
                   delta="Best in category" if nps_rf is not None else None)

    st.markdown("---")

    # Demographic breakdown
    st.markdown("##### 🧑‍🤝‍🧑 Respondent Profile")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("###### Buyback — Demographics")
        tabs_demo = st.tabs(["City","Gender","Age","Income","Occupation"])
        for i, col_name in enumerate(["City_Tier","Gender","Age","Income","Occupation"]):
            with tabs_demo[i]:
                d = bb[col_name].value_counts(normalize=True).mul(100).round(1).reset_index()
                d.columns = [col_name, "%"]
                d = d[~d[col_name].isin(["Other","Other/Not Specified","Unknown"])]
                fig = px.bar(d, x=col_name, y="%", color_discrete_sequence=[CASHIFY_COLOR],
                             text=d["%"].astype(str)+"%")
                fig.update_layout(height=200, margin=dict(t=10,b=10), showlegend=False,
                                  yaxis_title=None, xaxis_title=None)
                st.plotly_chart(fig, width='stretch')
    with dc2:
        st.markdown("###### Refurbished — Demographics")
        tabs_demo2 = st.tabs(["City","Gender","Age","Income","Occupation"])
        for i, col_name in enumerate(["City_Tier","Gender","Age","Income","Occupation"]):
            with tabs_demo2[i]:
                d = rf[col_name].value_counts(normalize=True).mul(100).round(1).reset_index()
                d.columns = [col_name, "%"]
                d = d[~d[col_name].isin(["Other","Other/Not Specified","Unknown"])]
                fig = px.bar(d, x=col_name, y="%", color_discrete_sequence=["#1B3A5C"],
                             text=d["%"].astype(str)+"%")
                fig.update_layout(height=200, margin=dict(t=10,b=10), showlegend=False,
                                  yaxis_title=None, xaxis_title=None)
                st.plotly_chart(fig, width='stretch')

    # Pivot table
    st.markdown("---")
    st.markdown("##### 📋 Summary Pivot — KPIs by Demographic")
    pivot_dim = st.selectbox("Slice by:", ["City_Tier","Gender","Age","Income","Occupation"], key="overview_pivot")
    pivot_journey = st.radio("Journey for pivot:", ["Buyback","Refurbished"], horizontal=True, key="ov_j")
    df_piv = bb if pivot_journey == "Buyback" else rf
    q16_col = "Q16_1"
    q12_col = "Q12"
    q15_col = "Q15_1"
    q10_col = "Q10"

    rows = []
    for grp, grp_df in df_piv.groupby(pivot_dim):
        if grp in ["Other","Other/Not Specified","Unknown"]: continue
        n = len(grp_df)
        nps_v,_,_,_ = compute_nps(grp_df[q16_col])
        tom_v = grp_df[q10_col].dropna().apply(lambda x: "cashify" in str(x).lower()).sum()/n*100
        aided_v = grp_df[q12_col].dropna().apply(lambda x: "cashify" in str(x).lower()).sum()/n*100
        consid_v = grp_df[q15_col].dropna().apply(lambda x: "first choice" in str(x).lower() or "seriously consider" in str(x).lower()).sum()/n*100
        rows.append({pivot_dim: grp, "n": n,
                     "TOM %": round(tom_v,1), "Aided Aware %": round(aided_v,1),
                     "Consideration %": round(consid_v,1), "NPS": nps_v})
    if rows:
        piv_df = pd.DataFrame(rows)
        st.dataframe(piv_df.style.format({"TOM %":"{:.1f}","Aided Aware %":"{:.1f}",
                                           "Consideration %":"{:.1f}","NPS":"{:.1f}"}),
                     width='stretch')

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — BRAND AWARENESS FUNNEL
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Brand Awareness: Top-of-Mind → Spontaneous → Aided</div>', unsafe_allow_html=True)

    aw_journey = st.radio("Journey:", ["Buyback","Refurbished"], horizontal=True, key="aw_j")
    df_aw = bb if aw_journey == "Buyback" else rf
    n_aw  = len(df_aw)

    if aw_journey == "Buyback":
        tom_brands  = {"Cashify":0,"OLX":0,"Amazon Exchange":0,"Flipkart Exchange":0,"Quickr":0,"Local shop":0}
        aided_brands = {b: df_aw["Q12"].dropna().apply(lambda x: b.lower().split()[0] in str(x).lower()).sum()
                        for b in ["Cashify","OLX","Amazon Exchange","Flipkart Exchange","Quickr","Flipkart Reset"]}
        spont_brands = {b: df_aw["Q11"].dropna().apply(lambda x: b.lower().split()[0] in str(x).lower()).sum()
                        for b in ["Cashify","OLX","Amazon Exchange","Flipkart Exchange","Quickr"]}
    else:
        tom_brands  = {"Cashify":0,"Amazon Renewed":0,"OLX":0,"Apple":0,"Samsung":0,"Flipkart":0}
        aided_brands = {b: df_aw["Q12"].dropna().apply(lambda x: b.lower().split()[0] in str(x).lower()).sum()
                        for b in ["Cashify","OLX","Amazon Renewed","Local shop","Flipkart Reset"]}
        spont_brands = {b: df_aw["Q11"].dropna().apply(lambda x: b.lower().split()[0] in str(x).lower()).sum()
                        for b in ["Cashify","OLX","Amazon Renewed","Local shop","Flipkart Reset"]}

    # TOM from Q10
    for brand in list(tom_brands.keys()):
        tom_brands[brand] = df_aw["Q10"].dropna().apply(
            lambda x: brand.lower().split()[0] in str(x).lower() and
                      str(x).lower().strip() not in ["none","no","na","n/a","not aware","nan"]
        ).sum()

    # Build awareness comparison dataframe
    all_aw_brands = sorted(set(list(tom_brands.keys())+list(spont_brands.keys())+list(aided_brands.keys())))
    aw_rows = []
    for b in all_aw_brands:
        aw_rows.append({
            "Platform": b,
            "TOM (%)": round(tom_brands.get(b,0)/n_aw*100, 1),
            "Spontaneous (%)": round(spont_brands.get(b,0)/n_aw*100, 1),
            "Aided (%)": round(aided_brands.get(b,0)/n_aw*100, 1),
        })
    aw_df = pd.DataFrame(aw_rows).sort_values("Aided (%)", ascending=False)

    c1, c2 = st.columns([2,1])
    with c1:
        # Grouped bar: TOM vs Spontaneous vs Aided
        fig = go.Figure()
        metrics = ["TOM (%)","Spontaneous (%)","Aided (%)"]
        colours_trio = [CASHIFY_COLOR,"#1B3A5C","#94A3B8"]
        for metric, col in zip(metrics, colours_trio):
            fig.add_trace(go.Bar(
                name=metric, x=aw_df["Platform"], y=aw_df[metric],
                marker_color=[CASHIFY_COLOR if p=="Cashify" else col for p in aw_df["Platform"]],
                text=aw_df[metric].astype(str)+"%", textposition="outside",
            ))
        fig.update_layout(
            barmode="group", title=f"Awareness Funnel — {aw_journey}",
            yaxis_title="%", height=400, legend_title="Metric",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown("##### Awareness Table")
        st.dataframe(aw_df.set_index("Platform").style.format("{:.1f}"),
                     width='stretch')

    st.markdown("---")
    st.markdown("##### 📋 Demographic Cut — Aided Awareness")

    demo_col = st.selectbox("Slice aided awareness by:", ["City_Tier","Gender","Age","Income"], key="aw_demo")
    brand_sel = st.multiselect("Platforms:", all_aw_brands, default=["Cashify","OLX"],
                               key="aw_brand_sel")

    demo_rows = []
    for grp, grp_df in df_aw.groupby(demo_col):
        if grp in ["Other","Other/Not Specified","Unknown"]: continue
        n_g = len(grp_df)
        row = {demo_col: grp, "n": n_g}
        for b in brand_sel:
            row[b] = round(grp_df["Q12"].dropna().apply(
                lambda x: b.lower().split()[0] in str(x).lower()).sum() / n_g * 100, 1)
        demo_rows.append(row)

    if demo_rows and brand_sel:
        demo_pivot = pd.DataFrame(demo_rows)
        fig2 = px.bar(demo_pivot.melt(id_vars=[demo_col,"n"], var_name="Platform", value_name="Aided %"),
                      x=demo_col, y="Aided %", color="Platform",
                      color_discrete_map=COLORS, barmode="group",
                      title=f"Aided Awareness by {demo_col}")
        fig2.update_layout(height=350, plot_bgcolor="white")
        st.plotly_chart(fig2, width='stretch')
        st.dataframe(demo_pivot, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — BRAND HEALTH FUNNEL
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Brand Health Funnel: Awareness → Familiarity → Ever Used → Recent Usage</div>', unsafe_allow_html=True)

    fj = st.radio("Journey:", ["Buyback","Refurbished"], horizontal=True, key="fj")
    df_f = bb if fj=="Buyback" else rf
    n_f  = len(df_f)

    FAMIL_MAP = {
        "Never heard of it": "Never Heard",
        "Heard a little but never considered using it": "Heard — Not Considered",
        "Heard a lot but never considered using it": "Heard — Not Considered",
        "Have considered using it but never used": "Considered",
        "Have used it to sell an old device but it was sometime back.": "Ever Used",
        "Have used it to buy a refurbished device but it was sometime back.": "Ever Used",
        "I used it recently to sell my device within last 6 months": "Recent User",
        "I used it recently to buy a device within last 6 months": "Recent User",
    }

    brand_q14 = BB_BRANDS_Q14 if fj=="Buyback" else RF_BRANDS_Q14
    brand_q15 = BB_BRANDS_Q15 if fj=="Buyback" else RF_BRANDS_Q15
    brand_q12 = "Q12"

    funnel_rows = []
    for col, brand in brand_q14.items():
        if col not in df_f.columns: continue
        aware = df_f[brand_q12].dropna().apply(lambda x: brand.lower().split()[0] in str(x).lower()).sum()
        famil_vals = df_f[col].dropna().apply(lambda x: FAMIL_MAP.get(str(x).strip(),"Unknown"))
        considered = (famil_vals.isin(["Considered","Ever Used","Recent User"])).sum()
        ever_used  = (famil_vals.isin(["Ever Used","Recent User"])).sum()
        recent     = (famil_vals == "Recent User").sum()
        q15_col    = [c for c,b in brand_q15.items() if b==brand]
        strong_consider = 0
        if q15_col:
            strong_consider = df_f[q15_col[0]].dropna().apply(
                lambda x: "first choice" in str(x).lower() or "seriously consider" in str(x).lower()
            ).sum()

        funnel_rows.append({
            "Platform": brand,
            "Awareness": round(aware/n_f*100,1),
            "Familiarity": round((n_f-famil_vals.eq("Never Heard").sum())/n_f*100,1),
            "Ever Used": round(ever_used/n_f*100,1),
            "Intent to Use": round(strong_consider/n_f*100,1),
        })

    funnel_df = pd.DataFrame(funnel_rows).sort_values("Awareness", ascending=False)

    # Funnel chart — all platforms
    c1, c2 = st.columns([3,2])
    with c1:
        fig = go.Figure()
        stages = ["Awareness","Familiarity","Ever Used","Intent to Use"]
        for _, row in funnel_df.iterrows():
            col = CASHIFY_COLOR if row["Platform"]=="Cashify" else COLORS.get(row["Platform"],"#94A3B8")
            fig.add_trace(go.Bar(
                name=row["Platform"],
                x=stages, y=[row[s] for s in stages],
                marker_color=col,
            ))
        fig.update_layout(barmode="group", title=f"Brand Health Funnel — {fj}",
                          yaxis_title="%", height=400, plot_bgcolor="white",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown("##### Funnel Table (with Conversion Rates)")
        fdf = funnel_df.copy().set_index("Platform")
        fdf["Aware→Famil %"] = (fdf["Familiarity"]/fdf["Awareness"]*100).round(1)
        fdf["Famil→Used %"]  = (fdf["Ever Used"]/fdf["Familiarity"]*100).round(1)
        fdf["Used→Intent %"] = (fdf["Intent to Use"]/fdf["Ever Used"]*100).round(1)
        st.dataframe(fdf.style.format("{:.1f}"), width='stretch')

    # Cashify funnel waterfall
    st.markdown("---")
    cashify_row = funnel_df[funnel_df["Platform"]=="Cashify"]
    if not cashify_row.empty:
        st.markdown("##### Cashify — Detailed Funnel Waterfall")
        cr = cashify_row.iloc[0]
        stages_w = ["Awareness","Familiarity","Ever Used","Intent to Use"]
        vals_w   = [cr[s] for s in stages_w]
        fig_w = go.Figure()
        bar_colors = [CASHIFY_COLOR, "rgba(0,168,145,0.7)", "rgba(0,168,145,0.45)", "rgba(0,168,145,0.25)"]
        fig_w.add_trace(go.Bar(
            x=vals_w, y=stages_w, orientation="h",
            marker_color=bar_colors,
            text=[f"{v}%" for v in vals_w],
            textposition="outside",
        ))
        fig_w.update_layout(
            title="Cashify Funnel (% of total sample)",
            height=350, plot_bgcolor="white",
            xaxis=dict(range=[0, 100], title="%"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_w, width='stretch')

    # Demographic funnel slice
    st.markdown("---")
    st.markdown("##### Funnel by Demographic")
    fd_col = st.selectbox("Slice by:", ["City_Tier","Gender","Age","Income"], key="fd_col")
    fd_stage = st.selectbox("Stage:", stages, key="fd_stage")
    fd_rows = []
    for grp, grp_df in df_f.groupby(fd_col):
        if grp in ["Other","Other/Not Specified","Unknown"]: continue
        ng = len(grp_df)
        for q15c, brand in brand_q15.items():
            if q15c not in grp_df.columns: continue
            if fd_stage == "Awareness":
                v = grp_df["Q12"].dropna().apply(lambda x: brand.lower().split()[0] in str(x).lower()).sum()/ng*100
            elif fd_stage == "Intent to Use":
                v = grp_df[q15c].dropna().apply(lambda x: "first choice" in str(x).lower() or "seriously consider" in str(x).lower()).sum()/ng*100
            elif fd_stage in ["Familiarity","Ever Used"]:
                q14c = [c for c,b in brand_q14.items() if b==brand]
                if not q14c: continue
                fv = grp_df[q14c[0]].dropna().apply(lambda x: FAMIL_MAP.get(str(x).strip(),"Unknown"))
                if fd_stage == "Familiarity":
                    v = (1 - fv.eq("Never Heard").sum()/len(fv))*100
                else:
                    v = fv.isin(["Ever Used","Recent User"]).sum()/ng*100
            else:
                continue
            fd_rows.append({fd_col: grp, "Platform": brand, fd_stage+" %": round(v,1)})

    if fd_rows:
        fd_df = pd.DataFrame(fd_rows)
        fig_fd = px.bar(fd_df, x=fd_col, y=fd_stage+" %", color="Platform",
                        color_discrete_map=COLORS, barmode="group",
                        title=f"{fd_stage} by {fd_col} — {fj}")
        fig_fd.update_layout(height=350, plot_bgcolor="white")
        st.plotly_chart(fig_fd, width='stretch')

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — NPS DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">Net Promoter Score — Cashify vs All Competitors</div>', unsafe_allow_html=True)

    nj = st.radio("Journey:", ["Buyback","Refurbished","Both"], horizontal=True, key="nj")

    def build_nps_df(df, brand_q16):
        rows = []
        for col, brand in brand_q16.items():
            if col not in df.columns: continue
            nps, p, pa, d = compute_nps(df[col])
            if nps is not None:
                rows.append({"Platform":brand, "NPS":nps,
                             "Promoters %":p, "Passives %":pa, "Detractors %":d,
                             "n": pd.to_numeric(df[col], errors="coerce").dropna().shape[0]})
        return pd.DataFrame(rows).sort_values("NPS", ascending=False)

    nps_bb_df = build_nps_df(bb, BB_BRANDS_Q16)
    nps_rf_df = build_nps_df(rf, RF_BRANDS_Q16)

    if nj == "Buyback":
        nps_frames = [("Buyback", nps_bb_df)]
    elif nj == "Refurbished":
        nps_frames = [("Refurbished", nps_rf_df)]
    else:
        nps_frames = [("Buyback", nps_bb_df), ("Refurbished", nps_rf_df)]

    for label, ndf in nps_frames:
        st.markdown(f"#### {label} Journey")
        if ndf.empty:
            st.warning("No data for this filter."); continue

        # KPI row
        cashify_nps = ndf[ndf["Platform"]=="Cashify"].iloc[0] if "Cashify" in ndf["Platform"].values else None
        if cashify_nps is not None:
            ck1,ck2,ck3,ck4 = st.columns(4)
            ck1.metric("Cashify NPS", cashify_nps["NPS"])
            ck2.metric("Promoters",   f"{cashify_nps['Promoters %']:.1f}%")
            ck3.metric("Passives",    f"{cashify_nps['Passives %']:.1f}%")
            ck4.metric("Detractors",  f"{cashify_nps['Detractors %']:.1f}%")

        cc1, cc2 = st.columns([3,2])
        with cc1:
            # NPS bar chart
            fig_nps = go.Figure(go.Bar(
                x=ndf["Platform"], y=ndf["NPS"],
                marker_color=[CASHIFY_COLOR if p=="Cashify" else ("#EF4444" if v<-50 else "#94A3B8")
                              for p,v in zip(ndf["Platform"], ndf["NPS"])],
                text=ndf["NPS"], textposition="outside",
            ))
            fig_nps.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            fig_nps.update_layout(title=f"NPS Scores — {label}", height=380,
                                   yaxis_title="NPS", plot_bgcolor="white")
            st.plotly_chart(fig_nps, width='stretch')

        with cc2:
            # Stacked bar: P/Pa/D breakdown
            fig_ppd = go.Figure()
            fig_ppd.add_trace(go.Bar(name="Promoters", x=ndf["Platform"], y=ndf["Promoters %"],
                                     marker_color="#10B981"))
            fig_ppd.add_trace(go.Bar(name="Passives",  x=ndf["Platform"], y=ndf["Passives %"],
                                     marker_color="#FCD34D"))
            fig_ppd.add_trace(go.Bar(name="Detractors",x=ndf["Platform"], y=ndf["Detractors %"],
                                     marker_color="#EF4444"))
            fig_ppd.update_layout(barmode="stack", title="P / Pa / D Breakdown",
                                   height=380, plot_bgcolor="white",
                                   legend=dict(orientation="h",y=-0.3))
            st.plotly_chart(fig_ppd, width='stretch')

        # NPS pivot by demographic
        st.markdown(f"##### {label} — NPS by Demographic")
        nps_demo = st.selectbox("Slice by:", ["City_Tier","Gender","Age","Income"], key=f"nps_demo_{label}")
        df_nj = bb if label=="Buyback" else rf
        nps_pivot_rows = []
        for grp, grp_df in df_nj.groupby(nps_demo):
            if grp in ["Other","Other/Not Specified","Unknown"]: continue
            nps_v, p_v, pa_v, d_v = compute_nps(grp_df["Q16_1"])
            nps_pivot_rows.append({nps_demo: grp, "n": len(grp_df),
                                    "Cashify NPS": nps_v,
                                    "Promoters %": p_v, "Passives %": pa_v, "Detractors %": d_v})
        if nps_pivot_rows:
            nps_piv = pd.DataFrame(nps_pivot_rows)
            fig_n2 = px.bar(nps_piv, x=nps_demo, y="Cashify NPS",
                            text="Cashify NPS", color_discrete_sequence=[CASHIFY_COLOR],
                            title=f"Cashify NPS by {nps_demo} — {label}")
            fig_n2.add_hline(y=0, line_dash="dash", line_color="red")
            fig_n2.update_layout(height=320, plot_bgcolor="white")
            st.plotly_chart(fig_n2, width='stretch')
            st.dataframe(nps_piv.style.format({"Cashify NPS":"{:.1f}",
                                                "Promoters %":"{:.1f}",
                                                "Passives %":"{:.1f}",
                                                "Detractors %":"{:.1f}"}),
                         width='stretch')
        st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — SOURCE OF AWARENESS (HEATMAP)
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">Source of Awareness — Platform × Channel Heatmap</div>', unsafe_allow_html=True)

    sj = st.radio("Journey:", ["Buyback","Refurbished"], horizontal=True, key="sj")
    df_s = bb if sj=="Buyback" else rf
    brand_q13 = BB_BRANDS_Q13 if sj=="Buyback" else RF_BRANDS_Q13
    n_s = len(df_s)

    # Build heatmap matrix
    heat_data = {}
    for col, brand in brand_q13.items():
        if col not in df_s.columns: continue
        heat_data[brand] = {}
        for src in AWARENESS_SOURCES:
            heat_data[brand][AWARENESS_SOURCE_SHORT[src]] = count_multiselect(df_s[col], src)

    heat_df = pd.DataFrame(heat_data).T  # rows=brands, cols=sources
    heat_pct = (heat_df.div(n_s) * 100).round(1)

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("##### Heatmap — Mention Counts")
        fig_heat = px.imshow(heat_df, text_auto=True, aspect="auto",
                             color_continuous_scale="Teal",
                             title=f"Awareness Source × Platform — {sj} (Counts)")
        fig_heat.update_layout(height=400, coloraxis_showscale=True)
        st.plotly_chart(fig_heat, width='stretch')

    with c2:
        st.markdown("##### Heatmap — % of Sample")
        fig_heat2 = px.imshow(heat_pct, text_auto=True, aspect="auto",
                              color_continuous_scale="Blues",
                              title=f"Awareness Source × Platform — {sj} (%)")
        fig_heat2.update_layout(height=400)
        st.plotly_chart(fig_heat2, width='stretch')

    st.markdown("---")
    st.markdown("##### Overall Top Sources — Combined Across All Platforms")
    all_srcs = {AWARENESS_SOURCE_SHORT[s]: 0 for s in AWARENESS_SOURCES}
    for col, brand in brand_q13.items():
        if col not in df_s.columns: continue
        for src in AWARENESS_SOURCES:
            all_srcs[AWARENESS_SOURCE_SHORT[src]] += count_multiselect(df_s[col], src)

    src_df = pd.DataFrame(list(all_srcs.items()), columns=["Source","Total Mentions"])\
               .sort_values("Total Mentions", ascending=True)
    fig_src = px.bar(src_df, x="Total Mentions", y="Source", orientation="h",
                     text="Total Mentions", color="Total Mentions",
                     color_continuous_scale="teal",
                     title=f"Total Awareness Source Mentions — {sj}")
    fig_src.update_layout(height=350, plot_bgcolor="white", coloraxis_showscale=False)
    st.plotly_chart(fig_src, width='stretch')

    st.markdown("---")
    st.markdown("##### Detailed Source Table")
    st.dataframe(heat_df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — CONSIDERATION SET
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Consideration Set — Platform Shortlist Analysis</div>', unsafe_allow_html=True)

    cj = st.radio("Journey:", ["Buyback","Refurbished"], horizontal=True, key="cj")
    df_c = bb if cj=="Buyback" else rf
    n_c  = len(df_c)
    brand_q15c = BB_BRANDS_Q15 if cj=="Buyback" else RF_BRANDS_Q15

    # Consideration levels
    CONSIDER_LEVELS = {
        "It would be my first choice": 4,
        "I would seriously consider it": 3,
        "I might consider it": 2,
        "I would not consider it": 1,
    }

    consid_rows = []
    for col, brand in brand_q15c.items():
        if col not in df_c.columns: continue
        vals = df_c[col].dropna()
        n_resp = len(vals)
        if n_resp == 0: continue
        first   = vals.apply(lambda x: "first choice" in str(x).lower()).sum()
        serious = vals.apply(lambda x: "seriously consider" in str(x).lower()).sum()
        might   = vals.apply(lambda x: "might consider" in str(x).lower()).sum()
        would_not = vals.apply(lambda x: "would not consider" in str(x).lower()).sum()
        strong = first + serious
        consid_rows.append({
            "Platform": brand,
            "First Choice %": round(first/n_c*100,1),
            "Seriously Consider %": round(serious/n_c*100,1),
            "Might Consider %": round(might/n_c*100,1),
            "Would Not Consider %": round(would_not/n_c*100,1),
            "Strong Consideration %": round(strong/n_c*100,1),
        })

    consid_df = pd.DataFrame(consid_rows).sort_values("Strong Consideration %", ascending=False)

    cc1, cc2 = st.columns([3,2])
    with cc1:
        # Stacked consideration bar
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(name="First Choice", x=consid_df["Platform"],
                               y=consid_df["First Choice %"], marker_color=CASHIFY_COLOR))
        fig_c.add_trace(go.Bar(name="Seriously Consider", x=consid_df["Platform"],
                               y=consid_df["Seriously Consider %"], marker_color="rgba(0,168,145,0.44)"))
        fig_c.add_trace(go.Bar(name="Might Consider", x=consid_df["Platform"],
                               y=consid_df["Might Consider %"], marker_color="#94A3B8"))
        fig_c.add_trace(go.Bar(name="Would NOT", x=consid_df["Platform"],
                               y=consid_df["Would Not Consider %"], marker_color="#EF4444"))
        fig_c.update_layout(barmode="stack", title=f"Consideration Set — {cj}",
                            height=400, plot_bgcolor="white",
                            legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_c, width='stretch')

    with cc2:
        # Strong consideration ranking
        fig_c2 = px.bar(consid_df.sort_values("Strong Consideration %"),
                        x="Strong Consideration %", y="Platform", orientation="h",
                        text="Strong Consideration %",
                        color="Platform", color_discrete_map=COLORS,
                        title="Strong Consideration Ranking")
        fig_c2.update_layout(height=400, plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig_c2, width='stretch')

    # Full consideration table
    st.markdown("---")
    st.markdown("##### Consideration Table")
    st.dataframe(consid_df.set_index("Platform").style.format("{:.1f}"),
                 width='stretch')

    # Demographic consideration pivot
    st.markdown("---")
    st.markdown("##### Consideration by Demographic")
    cdemo = st.selectbox("Slice by:", ["City_Tier","Gender","Age","Income"], key="cdemo")
    rows_cd = []
    for grp, grp_df in df_c.groupby(cdemo):
        if grp in ["Other","Other/Not Specified","Unknown"]: continue
        ng = len(grp_df)
        for col, brand in brand_q15c.items():
            if col not in grp_df.columns: continue
            v = grp_df[col].dropna().apply(lambda x: "first choice" in str(x).lower() or "seriously consider" in str(x).lower()).sum()/ng*100
            rows_cd.append({cdemo: grp, "Platform": brand, "Strong Consideration %": round(v,1)})
    if rows_cd:
        cd_df = pd.DataFrame(rows_cd)
        fig_cd = px.line(cd_df, x=cdemo, y="Strong Consideration %", color="Platform",
                         color_discrete_map=COLORS, markers=True,
                         title=f"Strong Consideration by {cdemo} — {cj}")
        fig_cd.update_layout(height=380, plot_bgcolor="white")
        st.plotly_chart(fig_cd, width='stretch')

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — CHOICE DRIVERS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">Choice Drivers — Cashify vs Competitor Weighted Rank Scores</div>', unsafe_allow_html=True)

    drj = st.radio("Journey:", ["Buyback","Refurbished"], horizontal=True, key="drj")
    df_dr = bb if drj=="Buyback" else rf

    if drj == "Buyback":
        cashify_q20 = {
            "Q20_1":"Instant payment","Q20_2":"Best price","Q20_3":"Technician check",
            "Q20_4":"Trust data wiping","Q20_5":"Verified & secure","Q20_6":"Convenient",
            "Q20_7":"Trusted brand","Q20_8":"Recommended","Q20_9":"Wide store network",
            "Q20_10":"No negotiation","Q20_11":"Customer support",
        }
        comp_q21a = {
            "Q21A_11":"Instant payment","Q21A_12":"Best price","Q21A_13":"Trusted brand",
            "Q21A_14":"Trust data wiping","Q21A_15":"Verified & secure","Q21A_16":"Convenient",
            "Q21A_17":"Trusted brand (2)","Q21A_18":"Recommended","Q21A_19":"Wide store network",
            "Q21A_20":"No negotiation","Q21A_21":"Customer support",
        }
    else:
        cashify_q20 = {
            "Q20_1":"EMI available","Q20_2":"Multiple payment options","Q20_3":"Verified & secure",
            "Q20_4":"Trusted brand","Q20_5":"Recommended by friends","Q20_6":"Wide store network",
            "Q20_7":"32-pt quality check","Q20_8":"Warranty included","Q20_9":"Safer than local",
            "Q20_10":"Better value","Q20_11":"Hassle-free returns","Q20_12":"Good past experience",
            "Q20_13":"Recommended (2)","Q20_14":"Environmental reasons",
        }
        comp_q21a = {
            "Q21A_1":"EMI available","Q21A_2":"Multiple payment options","Q21A_3":"Verified & secure",
            "Q21A_4":"Trusted brand","Q21A_5":"Recommended","Q21A_6":"Wide store network",
            "Q21A_7":"32-pt quality check","Q21A_8":"Warranty","Q21A_9":"Safer",
            "Q21A_10":"Better value","Q21A_11":"Hassle-free returns","Q21A_12":"Good experience",
            "Q21A_13":"Recommended (2)","Q21A_14":"Environmental",
        }

    cashify_valid_cols = {c:l for c,l in cashify_q20.items() if c in df_dr.columns}
    comp_valid_cols    = {c:l for c,l in comp_q21a.items()   if c in df_dr.columns}

    cashify_scores = weighted_rank_score(df_dr, list(cashify_valid_cols.keys()), list(cashify_valid_cols.values()))
    comp_scores    = weighted_rank_score(df_dr, list(comp_valid_cols.keys()),    list(comp_valid_cols.values()))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Why Consumers Choose Cashify (Weighted Score)")
        fig_d1 = px.bar(cashify_scores.head(10), x="Score", y="Driver", orientation="h",
                        text="Score", color="Score", color_continuous_scale="teal",
                        title="Cashify Choice Drivers")
        fig_d1.update_layout(height=380, plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig_d1, width='stretch')

    with c2:
        st.markdown("##### Why Consumers Choose a Competitor (Weighted Score)")
        fig_d2 = px.bar(comp_scores.head(10), x="Score", y="Driver", orientation="h",
                        text="Score", color="Score", color_continuous_scale="blues",
                        title="Competitor Choice Drivers")
        fig_d2.update_layout(height=380, plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig_d2, width='stretch')

    # Side-by-side comparison on shared drivers
    st.markdown("---")
    st.markdown("##### Driver Comparison — Cashify vs Competitor (Common Drivers)")
    shared = set(cashify_scores["Driver"]) & set(comp_scores["Driver"])
    if shared:
        cf_dict = dict(zip(cashify_scores["Driver"], cashify_scores["Score"]))
        cp_dict = dict(zip(comp_scores["Driver"],    comp_scores["Score"]))
        cmp_rows = [{"Driver":d, "Cashify":cf_dict[d], "Competitor":cp_dict[d]} for d in sorted(shared)]
        cmp_df = pd.DataFrame(cmp_rows).sort_values("Cashify", ascending=False)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="Cashify",    x=cmp_df["Driver"], y=cmp_df["Cashify"],    marker_color=CASHIFY_COLOR))
        fig_cmp.add_trace(go.Bar(name="Competitor", x=cmp_df["Driver"], y=cmp_df["Competitor"], marker_color="#1B3A5C"))
        fig_cmp.update_layout(barmode="group", title="Driver Comparison — Cashify vs Competitor",
                               height=350, plot_bgcolor="white")
        st.plotly_chart(fig_cmp, width='stretch')

    # Raw tables
    with st.expander("📋 Raw Rank Score Tables"):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Cashify Drivers**")
            st.dataframe(cashify_scores, use_container_width=True)
        with t2:
            st.markdown("**Competitor Drivers**")
            st.dataframe(comp_scores, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — BARRIERS TO CASHIFY
# ═════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">Barriers to Choosing Cashify</div>', unsafe_allow_html=True)

    bj = st.radio("Journey:", ["Buyback","Refurbished"], horizontal=True, key="bj")
    df_b = bb if bj=="Buyback" else rf

    barrier_vals = df_b["Q21B"].dropna().astype(str).str.strip()
    barrier_counts = barrier_vals.value_counts().reset_index()
    barrier_counts.columns = ["Barrier","Count"]
    barrier_counts = barrier_counts[~barrier_counts["Barrier"].isin(["Other","nan",""])]
    barrier_counts["% of respondents"] = (barrier_counts["Count"]/len(df_b)*100).round(1)

    # Also parse multi-select barriers
    if bj == "Refurbished":
        rf_barrier_terms = [
            "Was not aware that Cashify sells refurbished",
            "Found a better price",
            "Prefer buying brand-new",
            "Limited availability",
            "Concerns about refurbished quality",
            "Unsure about battery health",
            "Warranty or return policy was unclear",
            "Trust in refurbished devices was low",
            "Prefer buying from offline",
            "Negative reviews",
            "Delivery time",
        ]
        parsed_barriers = {t: count_multiselect(df_b["Q21B"], t) for t in rf_barrier_terms}
        parsed_df = pd.DataFrame(list(parsed_barriers.items()), columns=["Barrier","Count"])\
                      .sort_values("Count", ascending=False)
        parsed_df["% of respondents"] = (parsed_df["Count"]/len(df_b)*100).round(2)
    else:
        bb_barrier_terms = [
            "Prefer selling to known local shop",
            "Price offered is low",
            "Didn't know about it",
            "Prefer exchange on Amazon/Flipkart",
            "Prefer OLX negotiation",
            "Concern about data privacy",
            "Don't trust process",
            "Stores not nearby",
            "Payment doubt",
        ]
        parsed_barriers = {t: count_multiselect(df_b["Q21B"], t) for t in bb_barrier_terms}
        parsed_df = pd.DataFrame(list(parsed_barriers.items()), columns=["Barrier","Count"])\
                      .sort_values("Count", ascending=False)
        parsed_df["% of respondents"] = (parsed_df["Count"]/len(df_b)*100).round(2)

    c1, c2 = st.columns([3,2])
    with c1:
        fig_b = px.bar(parsed_df.head(8).sort_values("Count"),
                       x="Count", y="Barrier", orientation="h",
                       text="Count", color="Count",
                       color_continuous_scale="Reds",
                       title=f"Barriers to Choosing Cashify — {bj}")
        fig_b.update_layout(height=400, plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig_b, width='stretch')

    with c2:
        # Treemap of barriers
        tree_df = parsed_df[parsed_df["Count"]>0]
        if not tree_df.empty:
            fig_tree = px.treemap(tree_df, path=["Barrier"], values="Count",
                                  color="Count", color_continuous_scale="Reds",
                                  title="Barrier Treemap")
            fig_tree.update_layout(height=400)
            st.plotly_chart(fig_tree, width='stretch')

    st.markdown("---")
    st.markdown("##### Barriers Detail Table")
    st.dataframe(parsed_df.style.format({"Count":"{:,}","% of respondents":"{:.2f}"}),
                 width='stretch')

    # Barrier by demographic
    st.markdown("---")
    st.markdown("##### Barrier Frequency by Demographic")
    bdemo = st.selectbox("Slice by:", ["City_Tier","Gender","Age","Income"], key="bdemo")
    top_barrier = parsed_df.iloc[0]["Barrier"] if not parsed_df.empty else ""
    if top_barrier:
        b_rows = []
        for grp, grp_df in df_b.groupby(bdemo):
            if grp in ["Other","Other/Not Specified","Unknown"]: continue
            ng = len(grp_df)
            cnt = count_multiselect(grp_df["Q21B"], top_barrier)
            b_rows.append({bdemo:grp, "n":ng, "Count":cnt, "%":round(cnt/ng*100,1)})
        if b_rows:
            b_demo_df = pd.DataFrame(b_rows)
            fig_bd = px.bar(b_demo_df, x=bdemo, y="%",
                            text="%", color_discrete_sequence=["#EF4444"],
                            title=f'Top Barrier "{top_barrier[:40]}…" by {bdemo}')
            fig_bd.update_layout(height=300, plot_bgcolor="white")
            st.plotly_chart(fig_bd, width='stretch')

# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — CATEGORY DRIVERS & FEARS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="section-header">Category-Level Drivers & Hesitations (Ecosystem Insights)</div>', unsafe_allow_html=True)

    cat_j = st.radio("Journey:", ["Buyback","Refurbished"], horizontal=True, key="catj")
    df_cat = bb if cat_j=="Buyback" else rf
    n_cat  = len(df_cat)

    if cat_j == "Buyback":
        driver_terms = [
            "Verified platform / official brand",
            "Instant payment",
            "Price quote matches final value",
            "Customer support",
            "App rating/reviews",
            "Technician checks phone in front of me",
            "Formal Data Wiping Certificate",
            "Safe location / Meet-at-store",
            "Buyback guarantee",
            "No negotiation",
        ]
        fear_terms = [
            "Fraudulent buyer",
            "Data theft / privacy",
            "Payment delay",
            "Low price",
            "Negotiation stress",
            "Stranger coming home",
        ]
        driver_label = "What matters most when choosing a SELLING platform?"
        fear_label   = "Biggest fears when SELLING a phone"
    else:
        driver_terms = [
            "Warranty",
            "Battery health score",
            "Certified refurbisher",
            "Replacement guarantee",
            "Lower price vs new",
            "Looks / feels like new",
            "Reviews / testimonials",
            "32/64 point quality check",
            "Trial window",
            "EMI available",
        ]
        fear_terms = [
            "Fear of bad quality",
            "Fear of fake/duplicate parts",
            "Warranty doubts",
            "No trust in seller",
            "Data privacy concern",
            "Prefer only brand",
            "Don't know where to buy safely",
            "Social stigma",
        ]
        driver_label = "What matters most when choosing a BUYING platform?"
        fear_label   = "Biggest fears/hesitations when BUYING refurbished"

    # Compute from Q22 and Q23
    driver_counts = {d: count_multiselect(df_cat["Q22"], d) for d in driver_terms}
    fear_counts   = {f: count_multiselect(df_cat["Q23"], f) for f in fear_terms}

    driver_df = pd.DataFrame(list(driver_counts.items()), columns=["Driver","Count"])\
                  .assign(Pct=lambda x: (x["Count"]/n_cat*100).round(1))\
                  .sort_values("Count", ascending=False)
    fear_df   = pd.DataFrame(list(fear_counts.items()), columns=["Fear","Count"])\
                  .assign(Pct=lambda x: (x["Count"]/n_cat*100).round(1))\
                  .sort_values("Count", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### ✅ {driver_label}")
        fig_drv = go.Figure(go.Bar(
            x=driver_df["Pct"], y=driver_df["Driver"], orientation="h",
            text=driver_df["Pct"].astype(str)+"%", textposition="outside",
            marker=dict(color=driver_df["Pct"], colorscale="Teal"),
        ))
        fig_drv.update_layout(height=420, plot_bgcolor="white",
                               xaxis_title="% of respondents", yaxis_title=None,
                               title=driver_label)
        st.plotly_chart(fig_drv, width='stretch')

    with c2:
        st.markdown(f"##### ⚠️ {fear_label}")
        fig_fear = go.Figure(go.Bar(
            x=fear_df["Pct"], y=fear_df["Fear"], orientation="h",
            text=fear_df["Pct"].astype(str)+"%", textposition="outside",
            marker=dict(color=fear_df["Pct"], colorscale="Reds"),
        ))
        fig_fear.update_layout(height=420, plot_bgcolor="white",
                                xaxis_title="% of respondents", yaxis_title=None,
                                title=fear_label)
        st.plotly_chart(fig_fear, width='stretch')

    # Q24 Buyback only — Top 3 drivers
    if cat_j == "Buyback":
        st.markdown("---")
        st.markdown("##### Top-3 Platform Choice Drivers (Q24 — Check-3)")
        q24_map = {
            "Q24_1":"Best price","Q24_2":"Instant payment","Q24_3":"Safe platform",
            "Q24_4":"Data wiping certificate","Q24_5":"Technician behaviour",
            "Q24_6":"Verified buyer network","Q24_7":"Wide availability",
            "Q24_8":"App convenience","Q24_9":"Trustworthiness",
            "Q24_10":"Customer support","Q24_11":"Recommended by friends",
            "Q24_12":"Transparent price estimate","Q24_13":"All payment methods",
        }
        q24_counts = {}
        for col, label in q24_map.items():
            if col in df_cat.columns:
                q24_counts[label] = pd.to_numeric(df_cat[col], errors="coerce").notna().sum()
        q24_df = pd.DataFrame(list(q24_counts.items()), columns=["Driver","Count"])\
                   .assign(Pct=lambda x: (x["Count"]/n_cat*100).round(1))\
                   .sort_values("Count", ascending=False)
        fig_q24 = px.bar(q24_df, x="Driver", y="Pct", text="Pct",
                         color="Pct", color_continuous_scale="teal",
                         title="Top-3 Selling Platform Drivers (Q24 — multiple check)")
        fig_q24.update_layout(height=350, plot_bgcolor="white",
                               xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig_q24, width='stretch')

    # Demographic cut for fears
    st.markdown("---")
    st.markdown("##### Fear Frequency by Demographic")
    fdemo = st.selectbox("Slice by:", ["City_Tier","Gender","Age","Income"], key="fdemo")
    top_fear = fear_df.iloc[0]["Fear"] if not fear_df.empty else ""
    if top_fear:
        f_rows = []
        for grp, grp_df in df_cat.groupby(fdemo):
            if grp in ["Other","Other/Not Specified","Unknown"]: continue
            ng = len(grp_df)
            cnt = count_multiselect(grp_df["Q23"], top_fear)
            f_rows.append({fdemo:grp, "%":round(cnt/ng*100,1)})
        if f_rows:
            f_demo_df = pd.DataFrame(f_rows)
            fig_fd2 = px.bar(f_demo_df, x=fdemo, y="%", text="%",
                             color_discrete_sequence=["#EF4444"],
                             title=f'Top Fear "{top_fear}" by {fdemo}')
            fig_fd2.update_layout(height=300, plot_bgcolor="white")
            st.plotly_chart(fig_fd2, width='stretch')

    # Summary stats
    st.markdown("---")
    st.markdown("##### Summary Statistics Tables")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"**Category Drivers — {cat_j}**")
        st.dataframe(driver_df.style.format({"Count":"{:,}","Pct":"{:.1f}"}),
                     width='stretch')
    with t2:
        st.markdown(f"**Category Fears — {cat_j}**")
        st.dataframe(fear_df.style.format({"Count":"{:,}","Pct":"{:.1f}"}),
                     width='stretch')
