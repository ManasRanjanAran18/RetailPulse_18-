"""RetailPulse - interactive analytics dashboard (Streamlit).

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # allows running without `pip install -e .`

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from retailpulse.config import load_config
from retailpulse.io import read_report, read_table

st.set_page_config(page_title="RetailPulse", page_icon="📈", layout="wide")
CFG = load_config()
RISK_COLORS = {"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"}


@st.cache_data(ttl=300, show_spinner=False)
def table(name: str) -> pd.DataFrame:
    return read_table(name, CFG)


@st.cache_data(ttl=300, show_spinner=False)
def report(name: str) -> dict:
    return read_report(name, CFG)


def money(x: float) -> str:
    return f"${x:,.0f}"


# ------------------------------------------------------------------------------------------ guard
try:
    tx = table("transactions")
    demand = table("demand_daily")
except FileNotFoundError:
    st.title("📈 RetailPulse")
    st.error("No processed data found. Run the pipeline first:  `retailpulse all --with-data`  (or `make pipeline`).")
    st.stop()

sales = tx[tx["quantity"] > 0]

st.title("📈 RetailPulse")
st.caption("AI-powered customer analytics & demand forecasting for retail")

with st.sidebar:
    st.header("Filters")
    dmin, dmax = sales["order_date"].min().date(), sales["order_date"].max().date()
    date_range = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    cats = sorted(sales["category"].unique())
    sel_cats = st.multiselect("Categories", cats, default=cats)
    stores = sorted(sales["store_id"].unique())
    sel_stores = st.multiselect("Stores", stores, default=stores)
    st.divider()
    mon = report("monitoring_report")
    if mon:
        (st.warning if mon.get("retrain_recommended") else st.success)(
            "Retraining recommended" if mon.get("retrain_recommended") else "Models healthy (no drift alerts)")
    if st.button("Reload data"):
        st.cache_data.clear()
        st.rerun()

if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = map(pd.Timestamp, date_range)
else:
    d0, d1 = pd.Timestamp(dmin), pd.Timestamp(dmax)
view = sales[(sales["order_date"].between(d0, d1)) & sales["category"].isin(sel_cats) & sales["store_id"].isin(sel_stores)]

tabs = st.tabs(["Overview", "Demand forecast", "Customers", "Churn risk", "Inventory", "Live feed", "Model monitoring"])

# ------------------------------------------------------------------------------------------ overview
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    rev = view["revenue"].sum()
    c1.metric("Revenue", money(rev))
    c2.metric("Gross margin", f"{view['margin'].sum() / rev:.1%}" if rev else "-")
    c3.metric("Orders", f"{view['order_id'].nunique():,}")
    c4.metric("Avg order value", money(rev / max(view["order_id"].nunique(), 1)))
    c5.metric("Active customers", f"{view['customer_id'].nunique():,}")

    ts = view.groupby(pd.Grouper(key="order_date", freq="W"))["revenue"].sum().reset_index()
    st.plotly_chart(px.area(ts, x="order_date", y="revenue", title="Weekly revenue"), width="stretch")
    a, b = st.columns(2)
    cat = view.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    a.plotly_chart(px.bar(cat, x="category", y="revenue", title="Revenue by category", color="category"), width="stretch")
    st_rev = view.groupby("store_id")["revenue"].sum().reset_index()
    b.plotly_chart(px.pie(st_rev, names="store_id", values="revenue", title="Revenue by store", hole=0.45), width="stretch")
    top = view.groupby(["product_id"]).agg(revenue=("revenue", "sum"), units=("quantity", "sum")).nlargest(10, "revenue").reset_index()
    st.subheader("Top 10 products")
    st.dataframe(top, width="stretch", hide_index=True)

# ------------------------------------------------------------------------------------------ forecast
with tabs[1]:
    try:
        fc, met, bt = table("forecast"), table("forecast_metrics"), table("forecast_backtest")
    except FileNotFoundError:
        st.info("Run the forecasting stage first.")
    else:
        series = sorted(fc["series_id"].unique())
        sid = st.selectbox("Series", series, key="fc_series")
        hist = demand[demand["series_id"] == sid].tail(120)
        f = fc[fc["series_id"] == sid]
        fig = go.Figure()
        fig.add_scatter(x=hist["date"], y=hist["units"], name="Actual", line=dict(color="#1f77b4"))
        fig.add_scatter(x=f["date"], y=f["yhat_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=f["date"], y=f["yhat_lower"], fill="tonexty", fillcolor="rgba(255,127,14,0.2)", line=dict(width=0), name="80% interval")
        fig.add_scatter(x=f["date"], y=f["yhat"], name=f"Forecast ({f['model'].iloc[0]})", line=dict(color="#ff7f0e", dash="dash"))
        fig.update_layout(title=f"{sid}: daily units - last 120 days + {len(f)}-day forecast", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        k1, k2, k3 = st.columns(3)
        k1.metric("Forecast units (horizon)", f"{f['yhat'].sum():,.0f}")
        k2.metric("Peak day", f"{f['yhat'].max():,.0f}")
        sel = met[(met["series_id"] == sid) & met["selected"]].iloc[0]
        k3.metric("Back-test WAPE (selected)", f"{sel['wape']:.1%}")
        st.subheader("Model comparison (hold-out back-test)")
        st.dataframe(met[met["series_id"] == sid].drop(columns=["series_id"]).style.format(
            {"mae": "{:.2f}", "rmse": "{:.2f}", "wape": "{:.1%}", "smape": "{:.1%}", "bias": "{:.2f}"}), width="stretch", hide_index=True)
        b = bt[(bt["series_id"] == sid)]
        figb = px.line(b, x="date", y="predicted", color="model", title="Back-test: predictions vs actual")
        figb.add_scatter(x=b[b["model"] == b["model"].iloc[0]]["date"], y=b[b["model"] == b["model"].iloc[0]]["actual"], name="actual", line=dict(color="black", width=3))
        st.plotly_chart(figb, width="stretch")

# ------------------------------------------------------------------------------------------ customers
with tabs[2]:
    try:
        seg, prof = table("segments"), table("segment_profile")
    except FileNotFoundError:
        st.info("Run the segmentation stage first.")
    else:
        st.subheader("RFM segments")
        a, b = st.columns([1, 1])
        a.plotly_chart(px.bar(prof, x="segment", y="revenue_share", color="segment", title="Revenue share by segment").update_yaxes(tickformat=".0%"), width="stretch")
        b.plotly_chart(px.pie(prof, names="segment", values="customers", hole=0.45, title="Customers by segment"), width="stretch")
        st.dataframe(prof.style.format({"avg_recency_days": "{:.0f}", "avg_orders": "{:.1f}", "avg_spend": "${:,.0f}", "total_revenue": "${:,.0f}", "revenue_share": "{:.1%}"}),
                     width="stretch", hide_index=True)
        sample = seg.sample(min(len(seg), 2500), random_state=1)
        st.plotly_chart(px.scatter(sample, x="frequency", y="monetary", color="segment", size_max=6, log_x=True, log_y=True,
                                   hover_data=["customer_id", "recency"], title="Frequency vs monetary (sample)"), width="stretch")
        pick = st.selectbox("Export customers from segment", ["-"] + sorted(seg["segment"].unique()))
        if pick != "-":
            st.download_button(f"Download '{pick}' (CSV)", seg[seg["segment"] == pick].to_csv(index=False), f"segment_{pick.replace(' ', '_')}.csv")

# ------------------------------------------------------------------------------------------ churn
with tabs[3]:
    try:
        sc, imp, cm = table("churn_scores"), table("churn_importance"), report("churn_metrics")
    except FileNotFoundError:
        st.info("Run the churn stage first.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", cm.get("model", "-").replace("_", " "))
        m2.metric("Test ROC-AUC", f"{cm.get('test_roc_auc', 0):.3f}")
        m3.metric("Precision / Recall", f"{cm.get('test_precision', 0):.2f} / {cm.get('test_recall', 0):.2f}")
        m4.metric("High-risk customers", f"{(sc['risk_band'] == 'High').sum():,}")
        a, b = st.columns(2)
        dist = sc["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).reset_index()
        dist.columns = ["risk_band", "customers"]
        a.plotly_chart(px.bar(dist, x="risk_band", y="customers", color="risk_band", color_discrete_map=RISK_COLORS, title="Customers by churn-risk band"), width="stretch")
        b.plotly_chart(px.bar(imp.head(10).iloc[::-1], x="importance", y="feature", orientation="h", title="Top drivers (permutation importance, ΔAUC)"), width="stretch")
        st.subheader("Win-back list")
        thr = st.slider("Minimum churn probability", 0.3, 0.95, 0.6, 0.05)
        min_val = st.slider("Minimum lifetime spend ($)", 0, int(sc["monetary_all"].max()), 300, 50)
        lst = sc[(sc["churn_probability"] >= thr) & (sc["monetary_all"] >= min_val)]
        st.write(f"{len(lst):,} customers - lifetime value at risk: **{money(lst['monetary_all'].sum())}**")
        st.dataframe(lst.head(500), width="stretch", hide_index=True)
        st.download_button("Download win-back list (CSV)", lst.to_csv(index=False), "winback_list.csv")

# ------------------------------------------------------------------------------------------ inventory
with tabs[4]:
    try:
        plan = table("inventory_plan")
    except FileNotFoundError:
        st.info("Run the inventory stage first.")
    else:
        s = report("inventory_summary")
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("High stock-out risk SKUs", s.get("high_risk_skus", 0))
        i2.metric("Overstocked SKUs", s.get("overstocked_skus", 0))
        i3.metric("Units to reorder", f"{s.get('units_to_order', 0):,}")
        i4.metric("Excess capital tied up", money(s.get("excess_capital", 0)))
        risk = st.multiselect("Stock-out risk", ["High", "Medium", "Low"], default=["High", "Medium"])
        cols = ["product_id", "product_name", "category", "on_hand", "forecast_daily_demand", "days_of_cover", "lead_time_days",
                "safety_stock", "reorder_point", "eoq", "recommended_order_qty", "stockout_risk", "units_at_risk"]
        shown = plan[plan["stockout_risk"].isin(risk)][cols]
        st.dataframe(shown.style.format({"forecast_daily_demand": "{:.1f}", "days_of_cover": "{:.1f}"}), width="stretch", hide_index=True)
        st.download_button("Download purchase recommendations (CSV)", plan[plan["recommended_order_qty"] > 0][cols].to_csv(index=False), "reorder_plan.csv")
        fig = px.scatter(plan, x="days_of_cover", y="lead_time_days", color="stockout_risk", color_discrete_map=RISK_COLORS,
                         hover_data=["product_id", "on_hand"], title="Days of cover vs supplier lead time")
        fig.add_shape(type="line", x0=0, y0=0, x1=plan["lead_time_days"].max(), y1=plan["lead_time_days"].max(), line=dict(dash="dot"))
        st.plotly_chart(fig, width="stretch")
        st.caption("SKUs below the dotted line have more cover than their lead time; SKUs above it will stock out before a new order can arrive.")

# ------------------------------------------------------------------------------------------ live
with tabs[5]:
    st.caption("Simulated real-time stream: replays the most recent orders in sequence. In production, point this at your "
               "event source (Kafka / Kinesis / database CDC).")
    orders = (sales.groupby("order_id").agg(ts=("order_date", "first"), store=("store_id", "first"), revenue=("revenue", "sum"),
                                            lines=("product_id", "count")).reset_index().tail(3000).reset_index(drop=True))
    if "live_pos" not in st.session_state:
        st.session_state.live_pos = 200
    run = st.toggle("Stream", value=True)

    @st.fragment(run_every=2 if run else None)
    def live_panel():
        pos = st.session_state.live_pos
        if run:
            st.session_state.live_pos = min(len(orders), pos + 25)
            pos = st.session_state.live_pos
        window = orders.iloc[max(0, pos - 200):pos]
        k1, k2, k3 = st.columns(3)
        k1.metric("Orders in window", len(window))
        k2.metric("Revenue in window", money(window["revenue"].sum()))
        k3.metric("Avg basket (lines)", f"{window['lines'].mean():.2f}")
        by_store = window.groupby("store")["revenue"].sum().reset_index()
        st.plotly_chart(px.bar(by_store, x="store", y="revenue", title="Rolling revenue by store"), width="stretch")
        st.dataframe(window.tail(10).iloc[::-1], width="stretch", hide_index=True)
        st.progress(pos / len(orders), text=f"{pos:,}/{len(orders):,} orders replayed")

    live_panel()

# ------------------------------------------------------------------------------------------ monitoring
with tabs[6]:
    mon = report("monitoring_report")
    if not mon:
        st.info("Run the monitoring stage first.")
    else:
        (st.warning if mon["retrain_recommended"] else st.success)(
            "Retraining recommended: " + "; ".join(mon["reasons"]) if mon["retrain_recommended"] else "No retraining needed.")
        st.caption(f"Report generated {mon['generated_at']}")
        a, b = st.columns(2)
        a.subheader("Churn feature drift (PSI / KS)")
        fd = table("feature_drift")
        a.dataframe(fd.style.format({"psi": "{:.3f}", "ks_stat": "{:.3f}", "ks_pvalue": "{:.4f}"}), width="stretch", hide_index=True)
        b.subheader("Demand drift (vs same window last year)")
        dd = table("demand_drift")
        b.dataframe(dd.style.format({"ref_mean": "{:.1f}", "recent_mean": "{:.1f}", "level_change_pct": "{:+.1%}",
                                     "change_vs_prior_window_pct": "{:+.1%}", "psi": "{:.3f}", "ks_pvalue": "{:.3f}"}), width="stretch", hide_index=True)
        st.subheader("Data quality (last ETL run)")
        dq = pd.DataFrame(report("data_quality").get("checks", []))
        if not dq.empty:
            st.dataframe(dq, width="stretch", hide_index=True)
