# streamlit_app.py
# --- AI-Powered Analytics & Insights Dashboard (MVP) ---
# Author: Sense Data Lab (James) — August 2025
# Notes:
# - Modular, brandable Streamlit app blending GA4 funnels/pathing + sentiment + GPT summaries
# - Upload CSVs or use Demo Mode with generated sample data
# - Replace OpenAI call with your key in environment or st.secrets["OPENAI_API_KEY"]
# - Charts built with Plotly (Sankey, bars, lines). Altair optional placeholder included.

import os
import io
import json
import textwrap
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# ---------- THEME -----------
# -----------------------------
st.set_page_config(
    page_title="AI Insights Dashboard — Sense Data Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY = "#0e2a47"   # navy
ACCENT_1 = "#008080"   # teal
ACCENT_2 = "#ff6f61"   # coral

# Inject minimal CSS to align brand accents
st.markdown(
    f"""
    <style>
        :root {{
          --primary: {PRIMARY};
          --teal: {ACCENT_1};
          --coral: {ACCENT_2};
        }}
        .stButton>button {{
            background: var(--primary); color: white; border-radius: 8px; padding: 0.5rem 1rem;
        }}
        .metric-card {{ border-left: 4px solid var(--teal); padding: 0.75rem 1rem; background: #f8fafc; border-radius: 10px; }}
        .muted {{ color: #6b7280; font-size: 0.9rem; }}
        .section-title {{ margin-top: 8px; }}
        header .st-emotion-cache-1avcm0n {{ display: none; }} /* hide streamlit deploy header in some builds */
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# ------- UTILITIES -----------
# -----------------------------

def _dt(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file: Optional[io.BytesIO], parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    df = pd.read_csv(uploaded_file)
    for c in (parse_dates or []):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

@st.cache_data(show_spinner=False)
def generate_demo_data(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=120)
    pages = ["/", "/pages/plans", "/pages/deals", "/collections/all", "/product/12m-sim"]
    devices = ["mobile", "desktop"]

    # Page daily metrics
    recs = []
    for d in dates:
        for p in pages:
            for dev in devices:
                sessions = rng.integers(50, 800)
                bounces = int(sessions * rng.uniform(0.25, 0.65))
                view_item = int(sessions * rng.uniform(0.20, 0.75))
                add_to_cart = int(view_item * rng.uniform(0.25, 0.65))
                purchases = int(add_to_cart * rng.uniform(0.35, 0.7))
                recs.append({
                    "date": d, "page": p, "device": dev,
                    "sessions": sessions, "bounces": bounces,
                    "view_item": view_item, "add_to_cart": add_to_cart, "purchase": purchases
                })
    page_df = pd.DataFrame.from_records(recs)

    # Pathing (from_page -> to_page counts)
    transitions = []
    for d in dates:
        for _ in range(300):
            f,t = rng.choice(pages, 2, replace=True)
            if f!=t:
                transitions.append({"date": d, "from": f, "to": t, "count": int(rng.integers(1, 30))})
    path_df = pd.DataFrame(transitions)

    # Reviews (sentiment + themes)
    themes = ["navigation", "price", "activation", "delivery", "support"]
    sentiments = ["positive", "negative", "neutral"]
    revs = []
    for d in dates:
        for _ in range(rng.integers(2, 10)):
            s = sentiments[rng.choice([0,1,2], p=[0.5,0.35,0.15])]
            t = themes[rng.integers(0, len(themes))]
            revs.append({
                "date": d,
                "rating": int(rng.integers(1,6)),
                "sentiment": s,
                "theme": t,
                "text": f"Synthetic {s} review about {t} on {d.date()}"
            })
    review_df = pd.DataFrame(revs)

    return page_df, path_df, review_df


def kpi(value, label, help_text=None, cols=None):
    (c,) = st.columns(1) if cols is None else cols
    with c:
        st.markdown(f"<div class='metric-card'><h3>{value:,}</h3><div class='muted'>{label}</div>" + (f"<div class='muted'>{help_text}</div>" if help_text else "") + "</div>", unsafe_allow_html=True)


def fmt_pct(num, den):
    if den == 0 or pd.isna(den) or pd.isna(num):
        return 0.0
    return float(num)/float(den)

# -----------------------------
# ------- SIDEBAR -------------
# -----------------------------

st.sidebar.title("⚙️ Data Inputs")
demo = st.sidebar.toggle("Use Demo Data", value=True, help="Turn off to upload your own CSVs")

if not demo:
    ga4_file = st.sidebar.file_uploader("GA4 page metrics CSV (date,page,device,sessions,bounces,view_item,add_to_cart,purchase)", type=["csv"])
    path_file = st.sidebar.file_uploader("Pathing CSV (date,from,to,count)", type=["csv"])
    rev_file = st.sidebar.file_uploader("Reviews CSV (date,rating,sentiment,theme,text)", type=["csv"])
    page_df = load_csv(ga4_file, parse_dates=["date"])
    path_df = load_csv(path_file, parse_dates=["date"])
    review_df = load_csv(rev_file, parse_dates=["date"])
else:
    page_df, path_df, review_df = generate_demo_data()

st.sidebar.markdown("---")
st.sidebar.subheader("Branding")
client_name = st.sidebar.text_input("Client Name", value="Boost / Equifax / DeckTube")
primary_page = st.sidebar.selectbox("Primary page focus", sorted(page_df["page"].unique()) if not page_df.empty else ["/pages/plans","/pages/deals"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
min_date = page_df["date"].min() if not page_df.empty else None
max_date = page_df["date"].max() if not page_df.empty else None

if min_date is not None:
    date_range = st.sidebar.date_input("Date range", (min_date.date(), max_date.date()))
else:
    date_range = None

devices = ["mobile", "desktop"]
sel_devices = st.sidebar.multiselect("Devices", devices, default=devices)
sel_pages = st.sidebar.multiselect("Pages", sorted(page_df["page"].unique()) if not page_df.empty else [], default=list(sorted(page_df["page"].unique())) if not page_df.empty else [])

# Apply filters
if not page_df.empty:
    mask = (
        (page_df["device"].isin(sel_devices)) &
        (page_df["page"].isin(sel_pages if sel_pages else page_df["page"]))
    )
    if date_range:
        d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        mask &= (page_df["date"].between(d0, d1))
    page_df_f = page_df.loc[mask].copy()
else:
    page_df_f = pd.DataFrame()

if not path_df.empty:
    maskp = True
    if date_range:
        d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        maskp = path_df["date"].between(d0, d1)
    path_df_f = path_df.loc[maskp].copy()
else:
    path_df_f = pd.DataFrame()

if not review_df.empty:
    maskr = True
    if date_range:
        d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        maskr = review_df["date"].between(d0, d1)
    review_df_f = review_df.loc[maskr].copy()
else:
    review_df_f = pd.DataFrame()

# -----------------------------
# ------- HEADER --------------
# -----------------------------

st.title("📊 AI-Powered Analytics & Insights Dashboard")
st.caption("Quant + Qual + GPT summaries — reusable, brandable, client-ready")
st.markdown(f"**Client:** {client_name}  ·  **Primary Focus:** `{primary_page}`")

# -----------------------------
# ------- KPIs ----------------
# -----------------------------

col1, col2, col3, col4 = st.columns(4)
if not page_df_f.empty:
    total_sessions = int(page_df_f["sessions"].sum())
    total_purchases = int(page_df_f["purchase"].sum())
    conv = fmt_pct(total_purchases, total_sessions)
    bounce = fmt_pct(page_df_f["bounces"].sum(), total_sessions)
    kpi(total_sessions, "Total Sessions", cols=(col1,))
    kpi(total_purchases, "Total Purchases", cols=(col2,))
    kpi(f"{conv:.1%}", "Overall Conversion", cols=(col3,))
    kpi(f"{bounce:.1%}", "Overall Bounce Rate", cols=(col4,))

st.markdown("---")

# -----------------------------------
# SECTION: Page Performance View
# -----------------------------------
st.header("Page Performance View")

if page_df_f.empty:
    st.info("Upload GA4 page metrics CSV to populate charts.")
else:
    # Aggregate by page and device
    grp = page_df_f.groupby(["page", "device"], as_index=False).agg(
        sessions=("sessions", "sum"),
        bounces=("bounces", "sum"),
        purchases=("purchase", "sum"),
        view_item=("view_item", "sum"),
        add_to_cart=("add_to_cart", "sum"),
    )
    grp["conv_rate"] = grp.apply(lambda r: fmt_pct(r["purchases"], r["sessions"]), axis=1)
    grp["bounce_rate"] = grp.apply(lambda r: fmt_pct(r["bounces"], r["sessions"]), axis=1)

    colA, colB = st.columns([2,1])
    with colA:
        fig = px.bar(
            grp,
            x="page", y="sessions", color="device",
            barmode="group",
            title="Sessions by Page & Device",
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig2 = px.scatter(
            grp,
            x="bounce_rate", y="conv_rate",
            size="sessions", color="device",
            hover_name="page", title="Bounce vs Conversion (size=sessions)"
        )
        fig2.update_layout(yaxis_tickformat=".1%", xaxis_tickformat=".1%")
        st.plotly_chart(fig2, use_container_width=True)

    # Time series for a selected page
    st.subheader("Trend — Sessions & Conversion by Page")
    sel_trend_page = st.selectbox("Select page for trend", sorted(grp["page"].unique()))
    tdf = page_df_f[page_df_f["page"]==sel_trend_page].groupby(["date","device"], as_index=False).agg(
        sessions=("sessions","sum"), purchases=("purchase","sum")
    )
    tdf["conv"] = tdf.apply(lambda r: fmt_pct(r["purchases"], r["sessions"]), axis=1)

    fig3 = px.line(tdf, x="date", y="sessions", color="device", title=f"Sessions — {sel_trend_page}")
    st.plotly_chart(fig3, use_container_width=True)
    fig4 = px.line(tdf, x="date", y="conv", color="device", title=f"Conversion Rate — {sel_trend_page}")
    fig4.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# -----------------------------------
# SECTION: Funnel Breakdown
# -----------------------------------

st.header("Funnel Breakdown")

if page_df_f.empty:
    st.info("Upload GA4 page metrics CSV to see funnel drop-offs.")
else:
    fcols = ["view_item","add_to_cart","purchase"]
    fdf = page_df_f.groupby(["page","device"], as_index=False)[fcols].sum()
    fdf["step1_drop"] = (fdf["view_item"] - fdf["add_to_cart"]).clip(lower=0)
    fdf["step2_drop"] = (fdf["add_to_cart"] - fdf["purchase"]).clip(lower=0)
    fdf["view_to_cart"] = fdf.apply(lambda r: fmt_pct(r["add_to_cart"], r["view_item"]), axis=1)
    fdf["cart_to_purchase"] = fdf.apply(lambda r: fmt_pct(r["purchase"], r["add_to_cart"]), axis=1)

    sel_pages_f = st.multiselect("Pages to compare", sorted(fdf["page"].unique()), default=list(sorted(fdf["page"].unique()))[:3])
    fshow = fdf[fdf["page"].isin(sel_pages_f)]

    fig5 = px.bar(
        fshow, x="page", y=["view_item","add_to_cart","purchase"],
        barmode="group", facet_col="device", title="Funnel Counts by Page & Device"
    )
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.bar(
        fshow, x="page", y=["view_to_cart","cart_to_purchase"],
        barmode="group", facet_col="device", title="Step Conversion Rates by Page & Device"
    )
    fig6.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# -----------------------------------
# SECTION: Pathing Visualisation
# -----------------------------------

st.header("Pathing Visualisation")
if path_df_f.empty:
    st.info("Upload Pathing CSV to render Sankey.")
else:
    # Aggregate transitions
    agg = path_df_f.groupby(["from","to"], as_index=False)["count"].sum().sort_values("count", ascending=False).head(80)
    nodes = pd.Index(pd.unique(agg[["from","to"]].values.ravel())).tolist()
    node_map = {n:i for i,n in enumerate(nodes)}

    sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=12, thickness=16,
            label=nodes,
            color=[ACCENT_1 if n==primary_page else "#a3b8cc" for n in nodes]
        ),
        link=dict(
            source=[node_map[s] for s in agg["from"]],
            target=[node_map[t] for t in agg["to"]],
            value=agg["count"].tolist(),
        )
    )])
    sankey.update_layout(title_text="Top Navigation Flows (Sankey)")
    st.plotly_chart(sankey, use_container_width=True)

st.markdown("---")

# -----------------------------------
# SECTION: Sentiment Trends
# -----------------------------------

st.header("Sentiment Trends & Themes")

if review_df_f.empty:
    st.info("Upload Reviews CSV to see sentiment timelines.")
else:
    # Timeline by sentiment
    rts = review_df_f.assign(date=pd.to_datetime(review_df_f["date"]).dt.date)
    rts = rts.groupby(["date","sentiment"], as_index=False).size().rename(columns={"size":"count"})
    fig7 = px.line(rts, x="date", y="count", color="sentiment", title="Review Counts by Sentiment over Time")
    st.plotly_chart(fig7, use_container_width=True)

    # Theme filter + polarity share
    theme_options = sorted(review_df_f["theme"].dropna().unique().tolist())
    sel_theme = st.selectbox("Filter by theme", ["(All)"] + theme_options)
    rtheme = review_df_f if sel_theme == "(All)" else review_df_f[review_df_f["theme"]==sel_theme]
    share = rtheme.groupby("sentiment", as_index=False).size().rename(columns={"size":"count"})
    fig8 = px.pie(share, names="sentiment", values="count", title=f"Sentiment Share — Theme: {sel_theme}")
    st.plotly_chart(fig8, use_container_width=True)

    with st.expander("🔎 View sample theme reviews"):
        st.dataframe(rtheme[["date","rating","sentiment","theme","text"]].sort_values("date", ascending=False).head(25), use_container_width=True)

st.markdown("---")

# -----------------------------------
# SECTION: GPT Summary & Recs
# -----------------------------------

st.header("GPT Summary & Recommendations")

SYSTEM_PROMPT = """
You are a senior digital analytics & CRO consultant.
Blend GA4 funnel/pathing data and customer sentiment to produce:
- A concise summary of what's happening
- Top 3–5 issues (quant + qual evidence)
- Recommended actions with estimated impact and ease (H/M/L), separating mobile vs desktop if relevant
Be cautious with causality; call out uncertainty. Use bullets.
"""

INSIGHT_PROMPT_TEMPLATE = """
Client: {client}
Date Range: {date_range}
Primary Pages Analyzed: {pages}

Quant Highlights (aggregates):
{quant}

Sentiment Highlights (timeline + themes):
{qual}

Pathing Notes:
{path}

Write a plain-English summary with 3–5 recommendations.
"""

# Prepare prompt ingredients from current filtered data
if not page_df_f.empty:
    quant_bits = []
    by_page = page_df_f.groupby("page", as_index=False).agg(
        sessions=("sessions","sum"), purchases=("purchase","sum"), bounces=("bounces","sum"),
        views=("view_item","sum"), carts=("add_to_cart","sum")
    )
    by_page["conv"] = by_page.apply(lambda r: fmt_pct(r["purchases"], r["sessions"]), axis=1)
    by_page["bounce"] = by_page.apply(lambda r: fmt_pct(r["bounces"], r["sessions"]), axis=1)
    top_pages = by_page.sort_values("sessions", ascending=False).head(5)
    for _,r in top_pages.iterrows():
        quant_bits.append(f"{r['page']}: sessions={int(r['sessions'])}, conv={r['conv']:.1%}, bounce={r['bounce']:.1%}, view→cart={fmt_pct(r['carts'], r['views']):.1%}, cart→purchase={fmt_pct(r['purchases'], r['carts']):.1%}")
    quant_text = "\n".join(quant_bits)
else:
    quant_text = "No GA4 aggregates available."

if not review_df_f.empty:
    qual_bits = []
    trend = review_df_f.groupby("sentiment", as_index=False).size().rename(columns={"size":"count"}).sort_values("count", ascending=False)
    qual_bits.append("Sentiment distribution: " + ", ".join([f"{s}:{c}" for s,c in zip(trend['sentiment'], trend['count'])]))
    top_themes = review_df_f.groupby("theme", as_index=False).size().sort_values("size", ascending=False).head(3)
    if not top_themes.empty:
        qual_bits.append("Top themes: " + ", ".join([f"{t}:{n}" for t,n in zip(top_themes['theme'], top_themes['size'])]))
    qual_text = " | ".join(qual_bits)
else:
    qual_text = "No review data."

if not path_df_f.empty:
    top_links = path_df_f.groupby(["from","to"], as_index=False)["count"].sum().sort_values("count", ascending=False).head(5)
    path_text = "; ".join([f"{f}→{t} ({c})" for f,t,c in zip(top_links['from'], top_links['to'], top_links['count'])])
else:
    path_text = "No pathing data."

prompt = INSIGHT_PROMPT_TEMPLATE.format(
    client=client_name,
    date_range=f"{date_range[0] if date_range else 'N/A'} to {date_range[1] if date_range else 'N/A'}",
    pages=", ".join(sorted(page_df_f["page"].unique())) if not page_df_f.empty else "N/A",
    quant=quant_text,
    qual=qual_text,
    path=path_text,
)

st.text_area("Generated analysis prompt (editable)", value=prompt, height=260, key="analysis_prompt")

colg1, colg2 = st.columns([1,1])
with colg1:
    run_llm = st.button("✨ Generate GPT Summary")
with colg2:
    st.download_button("⬇️ Download Prompt (txt)", data=prompt.encode("utf-8"), file_name="insight_prompt.txt")

if run_llm:
    # Lazy import to keep app light without key
    try:
        import openai
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.warning("Add OPENAI_API_KEY to st.secrets or env to run the model. Showing a mocked response instead.")
            raise RuntimeError("No API key")
        openai.api_key = api_key

        # Minimal call (you can swap to responses.create with newer SDKs)
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        sys = SYSTEM_PROMPT
        user = st.session_state.get("analysis_prompt", prompt)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2,
        )
        out = resp.choices[0].message.content
    except Exception as e:
        # Fallback mocked output to keep the demo usable
        out = (
            "**Summary (mocked):**\n"
            "• Mobile funnel is weaker on product-heavy pages; consider simplifying PDP and sticky CTAs.\n"
            "• Pathing shows frequent loops between Plans and Deals; add comparison module and clarify entry points.\n"
            "• Reviews indicate persistent navigation/activation pain; ship clearer plan naming and guided activation.\n\n"
            "**Top Recommendations:**\n"
            "1) Add on-page comparison + canonicalise Deals vs Plans (Impact: High, Effort: Medium)\n"
            "2) Mobile-first PDP tidy + reduce duplicate CTAs (High, Medium)\n"
            "3) Guided activation wizard + FAQ placement (Medium, Low)\n"
        )
    st.markdown(out)

# -----------------------------
# SECTION: Data Model Hints & Templates
# -----------------------------

with st.expander("📥 CSV Schema & Templates"):
    st.markdown(
        """
        **GA4 Page Metrics CSV** — required columns:
        `date, page, device, sessions, bounces, view_item, add_to_cart, purchase`

        **Pathing CSV** — required columns:
        `date, from, to, count`

        **Reviews CSV** — required columns:
        `date, rating, sentiment, theme, text`

        *Tip:* You can export GA4 to BigQuery and assemble these via SQL; or export Looker Studio tables as CSV.
        """
    )
    # Provide small inline CSV samples
    sample_ga4 = (
        "date,page,device,sessions,bounces,view_item,add_to_cart,purchase\n"
        "2025-06-01,/pages/plans,mobile,1000,500,700,300,180\n"
        "2025-06-01,/pages/deals,desktop,800,280,500,260,140\n"
    )
    sample_path = (
        "date,from,to,count\n"
        "2025-06-01,/pages/plans,/pages/deals,120\n"
        "2025-06-01,/pages/deals,/collections/all,80\n"
    )
    sample_reviews = (
        "date,rating,sentiment,theme,text\n"
        "2025-06-03,3,negative,navigation,Kept bouncing between pages to compare plans.\n"
        "2025-06-04,5,positive,price,Great value and smooth checkout.\n"
    )
    st.download_button("Download GA4 sample CSV", data=sample_ga4, file_name="sample_ga4.csv")
    st.download_button("Download Pathing sample CSV", data=sample_path, file_name="sample_path.csv")
    st.download_button("Download Reviews sample CSV", data=sample_reviews, file_name="sample_reviews.csv")

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.caption("MVP • Modular • Swap chart libs or LLMs easily. Built by Sense Data Lab.")
