"""
IPO GMP Tracker & Predictor
A full-stack Streamlit dashboard for Indian IPO grey market analysis.

Tabs:
  1. 📊 Dashboard   — live KPIs, GMP vs listing scatter, sector heatmap
  2. 🔮 Predictor   — enter any IPO's details, get ML prediction
  3. 📈 Backtest    — monthly signal vs all-IPO return comparison
  4. 🗃️ IPO Table   — full searchable / filterable table with export
  5. 🧠 Model Info  — feature importances, CV metrics, how it works

Run:
    streamlit run src/app.py
"""

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import predict_one, SECTOR_MAP, MODEL_DIR, DB_PATH  # noqa: E402  # import must follow sys.path.insert above


@st.cache_resource
def _bootstrap() -> None:
    """Build the demo's data and models on first boot if they are absent.

    The dataset (*.db) and trained models (*.pkl) are gitignored, so a fresh
    clone or a hosted deployment starts empty. Seeding is deterministic (RNG
    seeded to 42) and training the XGBoost models takes ~1.5s on 800 synthetic
    IPOs, so rebuilding on boot is cheaper than committing binaries that would
    also risk a pickle/library-version mismatch. Everything the app shows is
    therefore computed by the current code, not replayed from a stale artifact.
    """
    if not DB_PATH.exists():
        import importlib.util  # noqa: PLC0415

        seed_path = Path(__file__).resolve().parent.parent / "scripts" / "seed_data.py"
        spec = importlib.util.spec_from_file_location("seed_data", seed_path)
        seeder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(seeder)
        seeder.seed()
    if not (MODEL_DIR / "regressor.pkl").exists():
        import model as _model  # noqa: PLC0415

        _model.train()


_bootstrap()

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="IPO GMP Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Topbar title */
  .ipo-title {
    font-size: 2rem; font-weight: 700; letter-spacing: -0.03em;
    color: #0f172a;
  }
  .ipo-subtitle {
    font-size: 0.95rem; color: #64748b; margin-top: -4px;
  }

  /* KPI cards */
  .kpi-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .kpi-val {
    font-size: 2rem; font-weight: 700; color: #0f172a;
    font-variant-numeric: tabular-nums;
  }
  .kpi-label {
    font-size: 0.78rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-top: 4px;
  }
  .kpi-green  { color: #16a34a !important; }
  .kpi-red    { color: #dc2626 !important; }
  .kpi-amber  { color: #d97706 !important; }

  /* Signal badge */
  .badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-size: 0.82rem; font-weight: 600; letter-spacing: 0.04em;
  }
  .badge-green  { background:#dcfce7; color:#15803d; }
  .badge-yellow { background:#fef9c3; color:#a16207; }
  .badge-red    { background:#fee2e2; color:#b91c1c; }
  .badge-blue   { background:#dbeafe; color:#1d4ed8; }

  /* Prediction output card */
  .pred-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px; padding: 32px 36px; color: white;
    margin: 16px 0;
  }
  .pred-gain { font-size: 3.5rem; font-weight: 800; letter-spacing: -0.04em; }
  .pred-meta { font-size: 0.9rem; color: #94a3b8; margin-top: 4px; }

  /* Mono font for code/numbers */
  .mono { font-family: 'JetBrains Mono', monospace; }

  /* Table tweaks */
  .dataframe { font-size: 0.85rem !important; }

  /* Tabs */
  [data-baseweb="tab"] { font-weight: 500; }

  /* Hide default streamlit branding */
  #MainMenu, footer { visibility: hidden; }

  div[data-testid="stMetricValue"] {
    font-size: 1.7rem;
    font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM ipos ORDER BY listing_date DESC", conn)
    conn.close()
    df["listing_date"] = pd.to_datetime(df["listing_date"])
    df["open_date"]    = pd.to_datetime(df["open_date"])
    df["year"]         = df["listing_date"].dt.year
    df["month"]        = df["listing_date"].dt.to_period("M").astype(str)
    df["gmp_accurate"] = (df["listing_gain_pct"] - df["gmp_pct"]).abs() < 10
    return df


@st.cache_resource
def load_models():
    try:
        with open(MODEL_DIR / "regressor.pkl", "rb") as f:
            reg = pickle.load(f)
        with open(MODEL_DIR / "classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        with open(MODEL_DIR / "meta.json") as f:
            meta = json.load(f)
        return reg, clf, meta
    except FileNotFoundError:
        return None, None, None


df = load_df()
regressor, classifier, meta = load_models()
SECTORS = list(SECTOR_MAP.keys())

# ── Header ────────────────────────────────────────────────────────
col_logo, col_spacer = st.columns([3, 1])
with col_logo:
    st.markdown('<div class="ipo-title">📈 IPO GMP Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="ipo-subtitle">Grey Market Premium tracker & listing return predictor · Indian Markets</div>',
                unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🔮 Predictor", "📈 Backtest", "🗃️ IPO Table", "🧠 Model Info"
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════
with tab1:
    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        years = sorted(df["year"].dropna().unique().astype(int), reverse=True)
        sel_years = st.multiselect("Year", years, default=years[:3], key="dash_years")
    with fc2:
        sel_cat = st.multiselect("Category", ["Mainboard", "SME"], default=["Mainboard", "SME"])
    with fc3:
        sel_sectors = st.multiselect("Sector", SECTORS, default=SECTORS[:6])

    fdf = df.copy()
    if sel_years:
        fdf = fdf[fdf["year"].isin(sel_years)]
    if sel_cat:
        fdf = fdf[fdf["category"].isin(sel_cat)]
    if sel_sectors:
        fdf = fdf[fdf["sector"].isin(sel_sectors)]

    # ── KPIs ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    total = len(fdf)
    winners = (fdf["listing_gain_pct"] > 0).sum()
    avg_gain = fdf["listing_gain_pct"].mean()
    avg_gmp = fdf["gmp_pct"].mean()
    gmp_hit_rate = fdf["gmp_accurate"].mean() * 100

    k1.metric("Total IPOs", f"{total:,}")
    k2.metric("Winners", f"{winners:,}", f"{winners/total*100:.0f}% hit rate" if total else "")
    k3.metric("Avg Listing Gain", f"{avg_gain:+.1f}%",
              delta_color="normal" if avg_gain >= 0 else "inverse")
    k4.metric("Avg GMP", f"{avg_gmp:+.1f}%")
    k5.metric("GMP Accuracy (±10%)", f"{gmp_hit_rate:.0f}%")

    st.divider()

    # ── Main chart: GMP vs Listing Gain scatter ───────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**GMP vs Actual Listing Gain**")
        scatter_df = fdf.dropna(subset=["gmp_pct", "listing_gain_pct"])
        fig = px.scatter(
            scatter_df,
            x="gmp_pct",
            y="listing_gain_pct",
            color="category",
            size="total_subscription",
            size_max=20,
            hover_data={"company": True, "sector": True, "issue_price": True,
                        "total_subscription": ":.1fx", "gmp_pct": ":.1f%",
                        "listing_gain_pct": ":.1f%", "category": False},
            color_discrete_map={"Mainboard": "#0f172a", "SME": "#f59e0b"},
            labels={"gmp_pct": "GMP (%)", "listing_gain_pct": "Listing Gain (%)"},
            title="",
        )
        # Identity line: if GMP perfectly predicted listing
        x_range = [scatter_df["gmp_pct"].min(), scatter_df["gmp_pct"].max()]
        fig.add_trace(go.Scatter(
            x=x_range, y=x_range, mode="lines",
            line=dict(color="#cbd5e1", dash="dash", width=1.5),
            name="Perfect prediction",
            hoverinfo="skip",
        ))
        # Zero lines
        fig.add_hline(y=0, line_color="#e2e8f0", line_width=1)
        fig.add_vline(x=0, line_color="#e2e8f0", line_width=1)
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=380, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(gridcolor="#f1f5f9"), yaxis=dict(gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bubble size = total subscription. Dashed line = perfect GMP accuracy.")

    with c2:
        st.markdown("**Listing Gain Distribution**")
        hist_df = fdf["listing_gain_pct"].dropna()
        fig_hist = px.histogram(
            hist_df, nbins=40,
            color_discrete_sequence=["#0f172a"],
            labels={"value": "Listing Gain (%)", "count": "IPOs"},
        )
        fig_hist.add_vline(x=0, line_color="#ef4444", line_width=1.5)
        fig_hist.add_vline(x=hist_df.mean(), line_color="#16a34a",
                           line_width=1.5, line_dash="dash",
                           annotation_text=f"Avg {hist_df.mean():.1f}%",
                           annotation_position="top right")
        fig_hist.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=180, margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("**Category Comparison**")
        cat_df = fdf.groupby("category").agg(
            ipos=("id", "count"),
            avg_gain=("listing_gain_pct", "mean"),
            win_rate=("listing_gain_pct", lambda x: (x > 0).mean() * 100),
        ).reset_index().round(1)
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    # ── Sector heatmap ────────────────────────────────────────────
    st.markdown("**Average Listing Gain by Sector × Year**")
    heat_df = (fdf.groupby(["sector", "year"])["listing_gain_pct"]
                  .mean().reset_index())
    heat_pivot = heat_df.pivot(index="sector", columns="year", values="listing_gain_pct").fillna(0)

    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale=[[0, "#fee2e2"], [0.5, "#fef9c3"], [1, "#dcfce7"]],
        color_continuous_midpoint=0,
        text_auto=".0f",
        labels=dict(x="Year", y="Sector", color="Avg Gain %"),
        aspect="auto",
    )
    fig_heat.update_layout(
        height=420, margin=dict(t=10, b=10),
        coloraxis_colorbar=dict(title="Avg %"),
        xaxis=dict(tickmode="linear"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Subscription vs GMP trend ──────────────────────────────────
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Subscription Rate vs Listing Gain**")
        sub_bins = pd.cut(fdf["total_subscription"], bins=[0, 5, 20, 50, 100, 500, 10000],
                          labels=["<5x", "5–20x", "20–50x", "50–100x", "100–500x", ">500x"])
        sub_df = fdf.groupby(sub_bins, observed=True)["listing_gain_pct"].mean().reset_index()
        sub_df.columns = ["Subscription Band", "Avg Gain (%)"]
        fig_sub = px.bar(sub_df, x="Subscription Band", y="Avg Gain (%)",
                         color="Avg Gain (%)",
                         color_continuous_scale=["#fee2e2", "#fef9c3", "#dcfce7"],
                         color_continuous_midpoint=0)
        fig_sub.update_layout(plot_bgcolor="white", height=280, margin=dict(t=10),
                               showlegend=False)
        st.plotly_chart(fig_sub, use_container_width=True)

    with sc2:
        st.markdown("**Monthly IPO Count & Avg GMP**")
        monthly = fdf.groupby("month").agg(
            count=("id", "count"),
            avg_gmp=("gmp_pct", "mean"),
        ).reset_index().sort_values("month")
        fig_mo = go.Figure()
        fig_mo.add_trace(go.Bar(x=monthly["month"], y=monthly["count"],
                                name="IPO Count", marker_color="#e2e8f0",
                                yaxis="y"))
        fig_mo.add_trace(go.Scatter(x=monthly["month"], y=monthly["avg_gmp"],
                                    name="Avg GMP (%)", line=dict(color="#0f172a", width=2),
                                    yaxis="y2"))
        fig_mo.update_layout(
            plot_bgcolor="white", height=280, margin=dict(t=10),
            yaxis=dict(title="IPO Count", gridcolor="#f1f5f9"),
            yaxis2=dict(title="Avg GMP (%)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_mo, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2: PREDICTOR
# ═══════════════════════════════════════════════════════════════════
with tab2:
    if regressor is None:
        st.error("Model not trained. Run `python src/model.py` first.")
        st.stop()

    st.markdown("### Predict Listing Return for Any IPO")
    st.markdown("Enter pre-listing data to get an ML-powered return estimate.")
    st.divider()

    col_inp, col_out = st.columns([1, 1], gap="large")

    with col_inp:
        st.markdown("**IPO Details**")
        company_name = st.text_input("Company name (for display)", placeholder="e.g. Bajaj Housing Finance")

        p1, p2 = st.columns(2)
        issue_price  = p1.number_input("Issue Price (₹)", min_value=1, value=200, step=5)
        lot_size     = p2.number_input("Lot Size (shares)", min_value=1, value=50)

        p3, p4 = st.columns(2)
        gmp_rs       = p3.number_input("Current GMP (₹)", value=40, step=5,
                                        help="Grey market premium in ₹ above issue price")
        gmp_day1     = p4.number_input("GMP on Day 1 open (₹)", value=20, step=5)

        st.markdown("**Subscription Data**")
        s1, s2, s3 = st.columns(3)
        qib_sub  = s1.number_input("QIB", min_value=0.0, value=15.0, step=1.0, format="%.1f",
                                    help="Times subscribed by Qualified Institutional Buyers")
        nii_sub  = s2.number_input("NII (HNI)", min_value=0.0, value=25.0, step=1.0, format="%.1f")
        ret_sub  = s3.number_input("Retail", min_value=0.0, value=8.0, step=1.0, format="%.1f")
        total_sub = (qib_sub * 0.5 + nii_sub * 0.15 + ret_sub * 0.35) / 0.35  # weighted

        st.markdown("**Other Details**")
        o1, o2 = st.columns(2)
        issue_size_cr = o1.number_input("Issue Size (₹ Cr)", min_value=1.0, value=500.0, step=50.0)
        category      = o2.selectbox("Category", ["Mainboard", "SME"])

        o3, o4 = st.columns(2)
        sector         = o3.selectbox("Sector", SECTORS)
        days_to_listing = o4.slider("Days close → listing", 7, 20, 12)

    with col_out:
        st.markdown("**Prediction**")

        input_data = {
            "gmp_rs": gmp_rs,
            "gmp_pct": gmp_rs / issue_price * 100 if issue_price > 0 else 0,
            "gmp_day1": gmp_day1,
            "gmp_day2": (gmp_day1 + gmp_rs) / 2,
            "issue_price": issue_price,
            "lot_size": lot_size,
            "issue_size_cr": issue_size_cr,
            "qib_subscription": qib_sub,
            "nii_subscription": nii_sub,
            "retail_subscription": ret_sub,
            "total_subscription": total_sub,
            "category": category,
            "sector": sector,
            "close_date": "2025-01-01",
            "listing_date": f"2025-01-{12 + days_to_listing:02d}",
        }

        result = predict_one(input_data, regressor, classifier, meta)
        gain   = result["predicted_gain_pct"]
        sig    = result["signal"]
        prob   = result["positive_prob"]

        gain_color = "#16a34a" if gain > 0 else "#dc2626"
        badge_cls  = "badge-blue" if "STRONG" in sig else "badge-green" if sig == "BUY" else "badge-yellow" if sig == "NEUTRAL" else "badge-red"

        st.markdown(f"""
        <div class="pred-card">
          <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:8px;">
            {'<strong style="color:white">' + company_name + '</strong> · ' if company_name else ''}
            {category} · {sector}
          </div>
          <div class="pred-gain" style="color:{gain_color}">{gain:+.1f}%</div>
          <div class="pred-meta">Expected listing gain · confidence band: {result['lower_band']:+.0f}% to {result['upper_band']:+.0f}%</div>
          <br>
          <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:4px;">
            <span class="badge {badge_cls}">{sig}</span>
            <span style="color:#94a3b8;font-size:0.88rem">
              {prob:.0f}% probability of positive listing
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Key input summary
        exp_listing = round(issue_price + gmp_rs, 2)
        ml_price = round(issue_price * (1 + gain / 100), 2)

        st.markdown("**Input summary**")
        summ_data = {
            "Metric": ["Issue Price", "GMP", "Expected (GMP-based)", "ML Predicted Price", "QIB Sub", "NII Sub", "Retail Sub"],
            "Value":  [f"₹{issue_price}", f"₹{gmp_rs} ({gmp_rs/issue_price*100:.1f}%)",
                       f"₹{exp_listing}", f"₹{ml_price}",
                       f"{qib_sub:.1f}x", f"{nii_sub:.1f}x", f"{ret_sub:.1f}x"],
        }
        st.dataframe(pd.DataFrame(summ_data), use_container_width=True, hide_index=True)

        st.markdown("**How this compares to historical IPOs**")
        similar = df[
            (df["gmp_pct"].between(gmp_rs / issue_price * 100 - 10, gmp_rs / issue_price * 100 + 10)) &
            (df["category"] == category)
        ]["listing_gain_pct"].dropna()

        if len(similar) > 0:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Histogram(x=similar, nbinsx=20, name="Similar IPOs",
                                            marker_color="#e2e8f0"))
            fig_comp.add_vline(x=gain, line_color="#0f172a", line_width=2,
                               annotation_text="Our prediction",
                               annotation_position="top right")
            fig_comp.update_layout(
                plot_bgcolor="white", height=200, margin=dict(t=20, b=10),
                showlegend=False,
                xaxis_title="Listing Gain (%)",
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.caption(f"Based on {len(similar)} historical IPOs with similar GMP% ({category})")


# ═══════════════════════════════════════════════════════════════════
# TAB 3: BACKTEST
# ═══════════════════════════════════════════════════════════════════
with tab3:
    if meta is None:
        st.error("Model not trained. Run `python src/model.py` first.")
    else:
        bt = meta["backtest"]
        st.markdown("### Signal Backtest — Monthly Performance")
        st.caption(
            "Strategy: apply for every IPO where predicted gain > 10% AND direction probability > 60%. "
            "Compare average return vs applying for all IPOs."
        )

        b1, b2, b3 = st.columns(3)
        b1.metric("Months tested", bt["months_tested"])
        b2.metric("Avg return (all IPOs)", f"{bt['avg_return_all_ipos']:+.1f}%")
        b3.metric("Avg return (signal only)", f"{bt['avg_return_signal_ipos']:+.1f}%",
                  delta=f"{bt['avg_return_signal_ipos'] - bt['avg_return_all_ipos']:+.1f}% vs all",
                  delta_color="normal")

        st.divider()

        bt_df = pd.DataFrame(bt["monthly"])
        bt_df = bt_df[bt_df["total_ipos"] > 0].copy()

        # Chart: signal vs all
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Bar(
            x=bt_df["month"], y=bt_df["avg_return_all"],
            name="All IPOs avg", marker_color="#e2e8f0",
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt_df["month"], y=bt_df["avg_return_signal"],
            mode="lines+markers", name="Signal IPOs avg",
            line=dict(color="#0f172a", width=2.5),
            marker=dict(size=8, color="#0f172a"),
        ))
        fig_bt.add_hline(y=0, line_color="#94a3b8", line_width=0.8)
        fig_bt.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=350, margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.05),
            yaxis=dict(title="Avg Listing Gain (%)", gridcolor="#f1f5f9"),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # Cumulative return comparison
        st.markdown("**Cumulative Return (₹10,000 invested per IPO)**")
        val_all    = 10000.0
        val_signal = 10000.0
        rows_cum = []
        for _, row in bt_df.iterrows():
            val_all    *= (1 + row["avg_return_all"] / 100)
            if row["signal_ipos"] > 0:
                val_signal *= (1 + row["avg_return_signal"] / 100)
            rows_cum.append({"month": row["month"], "All IPOs": val_all, "Signal Only": val_signal})

        cum_df = pd.DataFrame(rows_cum)
        fig_cum = px.line(cum_df, x="month", y=["All IPOs", "Signal Only"],
                          color_discrete_map={"All IPOs": "#94a3b8", "Signal Only": "#0f172a"},
                          labels={"value": "Portfolio Value (₹)", "month": ""},)
        fig_cum.add_hline(y=10000, line_color="#e2e8f0", line_dash="dot")
        fig_cum.update_layout(
            plot_bgcolor="white", height=300, margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.05),
            yaxis=dict(gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        st.caption("⚠️ Backtest uses the same dataset the model was trained on — treat as in-sample illustration, not true out-of-sample performance.")

        # Monthly breakdown table
        with st.expander("Monthly breakdown table"):
            display_bt = bt_df.rename(columns={
                "month": "Month", "total_ipos": "All IPOs",
                "signal_ipos": "Signal IPOs",
                "avg_return_all": "Avg All (%)", "avg_return_signal": "Avg Signal (%)"
            })
            st.dataframe(display_bt, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4: IPO TABLE
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Full IPO Database")

    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    t_search  = t_c1.text_input("Search company", placeholder="e.g. Bajaj")
    t_cat     = t_c2.multiselect("Category", ["Mainboard", "SME"], default=["Mainboard", "SME"])
    t_sector  = t_c3.multiselect("Sector", SECTORS, default=SECTORS)
    t_min_sub = t_c4.slider("Min total subscription (x)", 0.0, 100.0, 0.0)

    tdf = df.copy()
    if t_search:
        tdf = tdf[tdf["company"].str.contains(t_search, case=False, na=False)]
    if t_cat:
        tdf = tdf[tdf["category"].isin(t_cat)]
    if t_sector:
        tdf = tdf[tdf["sector"].isin(t_sector)]
    tdf = tdf[tdf["total_subscription"] >= t_min_sub]

    PAGE = 50
    total_pages = max(1, (len(tdf) - 1) // PAGE + 1)
    page = st.number_input("Page", 1, total_pages, 1) - 1
    tdf_page = tdf.iloc[page * PAGE: (page + 1) * PAGE]

    display_df = tdf_page[[
        "listing_date", "company", "category", "sector",
        "issue_price", "gmp_rs", "gmp_pct",
        "total_subscription", "qib_subscription", "nii_subscription", "retail_subscription",
        "listing_price", "listing_gain_pct",
    ]].copy()
    display_df["listing_date"] = display_df["listing_date"].dt.strftime("%d %b %Y")
    display_df.columns = [
        "Listing Date", "Company", "Cat", "Sector",
        "Issue ₹", "GMP ₹", "GMP %",
        "Total Sub", "QIB", "NII", "Retail",
        "Listing ₹", "Gain %",
    ]
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Gain %": st.column_config.NumberColumn(format="%.1f%%"),
            "GMP %": st.column_config.NumberColumn(format="%.1f%%"),
            "Total Sub": st.column_config.NumberColumn(format="%.1fx"),
        }
    )
    st.caption(f"{len(tdf):,} IPOs · page {page+1}/{total_pages}")

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇️ Download CSV",
        tdf[[c for c in tdf.columns if c not in ["year", "month", "gmp_accurate"]]].to_csv(index=False).encode(),
        file_name="ipo_gmp_data.csv", mime="text/csv",
    )
    try:
        import io

        import openpyxl  # noqa: F401  # unused directly; to_excel needs it, so its absence must raise ImportError

        buf = io.BytesIO()
        tdf[[c for c in tdf.columns if c not in ["year","month","gmp_accurate"]]].to_excel(buf, index=False)
        dl2.download_button(
            "⬇️ Download Excel", buf.getvalue(),
            file_name="ipo_gmp_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════
# TAB 5: MODEL INFO
# ═══════════════════════════════════════════════════════════════════
with tab5:
    if meta is None:
        st.error("Model not trained.")
    else:
        st.markdown("### Model Performance & Explainability")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CV MAE", f"{meta['cv_mae']:.1f}%", help="Mean absolute error on held-out folds")
        m2.metric("CV R²", f"{meta['cv_r2']:.3f}", help="Variance explained by model")
        m3.metric("Directional Accuracy", f"{meta['cv_dir_acc']*100:.1f}%", help="% of IPOs where gain direction (up/down) was correctly predicted")
        m4.metric("Training IPOs", f"{meta['n_train']:,}")

        st.divider()

        # Feature importances
        st.markdown("**Feature Importances (XGBoost gain)**")
        imp_df = pd.DataFrame(
            list(meta["importances"].items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)

        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=[[0, "#e2e8f0"], [1, "#0f172a"]],
                         labels={"Importance": "XGBoost Feature Importance (gain)", "Feature": ""})
        fig_imp.update_layout(
            plot_bgcolor="white", height=430,
            margin=dict(t=10, b=10), showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.divider()
        st.markdown("**How It Works**")
        st.markdown("""
        The model predicts **listing day return** (%) using two XGBoost models trained on pre-listing signals:

        **Regressor** → predicted gain %
        **Classifier** → probability that listing gain > 0%

        **Features used:**
        | Feature | What it captures |
        |---|---|
        | `gmp_pct` | GMP as % of issue price — strongest signal |
        | `gmp_rs` | Raw GMP in ₹ |
        | `gmp_momentum` | How much GMP moved from Day 1 to close |
        | `total_subscription` | Overall demand |
        | `qib_subscription` | Institutional demand (smart money) |
        | `nii_subscription` | HNI demand |
        | `retail_subscription` | Retail demand |
        | `sub_qib_nii_ratio` | QIB vs NII — tracks smart vs leveraged money |
        | `issue_price` | Price point (affects retail access) |
        | `issue_size_cr` | Deal size (affects float and volatility) |
        | `lot_size` | Affects minimum investment and retail participation |
        | `days_to_listing` | Longer gap = more time for GMP to drift |
        | `is_sme` | SME IPOs have different risk/return profile |
        | `sector_encoded` | Sector-specific premium/discount |

        **Validation:** Time-series 5-fold CV (no data leakage — future folds never seen in past training).
        """)
