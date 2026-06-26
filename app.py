"""
Netexlir — AI-Assisted Ecommerce Forecasting
Streamlit prototype: ingest → forecast → budget sim → AI insights
"""

import os
import sys
import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.loader import (
    load_daily_aggregate, load_daily_by_channel,
    load_daily_by_campaign_type, campaign_summary,
)
from src.forecaster import run_aggregate_forecast, run_channel_forecasts, trailing_actuals
from src.anomaly import detect_anomalies
from src.budget_sim import simulate_budget

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Netexlir · Forecast",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-label { color:#9ca3af; font-size:0.78rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
  .pill-high   { background:#7f1d1d; color:#fca5a5; border-radius:12px; padding:2px 8px; font-size:.75rem; font-weight:600; }
  .pill-medium { background:#78350f; color:#fcd34d; border-radius:12px; padding:2px 8px; font-size:.75rem; font-weight:600; }
  .pill-low    { background:#1e3a5f; color:#93c5fd; border-radius:12px; padding:2px 8px; font-size:.75rem; font-weight:600; }
  .info-box    { background:#1e293b; border-left:3px solid #6366f1; border-radius:6px; padding:12px 16px; margin:8px 0; }
  code.file    { background:#0f172a; color:#a5b4fc; padding:2px 6px; border-radius:4px; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached helpers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_data():
    daily        = load_daily_aggregate()
    channel_data = load_daily_by_channel()
    camp_types   = load_daily_by_campaign_type()
    summary      = campaign_summary()
    trailing     = trailing_actuals(daily, days=30)
    return daily, channel_data, camp_types, summary, trailing


@st.cache_data(show_spinner=False)
def _run_forecast(daily_json):
    daily = pd.read_json(daily_json, orient="split", convert_dates=["ds"])
    return run_aggregate_forecast(daily, uncertainty_samples=500)


@st.cache_data(show_spinner=False)
def _run_channel_forecasts(channel_json):
    channel_data = {ch: pd.read_json(v, orient="split", convert_dates=["ds"])
                    for ch, v in channel_json.items()}
    return run_channel_forecasts(channel_data, uncertainty_samples=300)


@st.cache_data(show_spinner=False)
def _run_camptype_forecasts(camptype_json):
    from src.forecaster import run_campaign_type_forecasts
    camp_types = {k: pd.read_json(v, orient="split", convert_dates=["ds"])
                  for k, v in camptype_json.items()}
    return run_campaign_type_forecasts(camp_types, uncertainty_samples=200)


@st.cache_data(show_spinner=False)
def _detect_anomalies(daily_json, channel_json):
    daily        = pd.read_json(daily_json, orient="split", convert_dates=["ds"])
    channel_data = {ch: pd.read_json(v, orient="split", convert_dates=["ds"])
                    for ch, v in channel_json.items()}
    return detect_anomalies(daily, channel_data=channel_data)


@st.cache_data(show_spinner=False)
def _backtest(daily_json):
    """Train on data before Oct 2025, test Oct–Dec 2025 for MAPE validation."""
    import numpy as np
    from prophet import Prophet
    from src.forecaster import _PROMO_HOLIDAYS

    daily = pd.read_json(daily_json, orient="split", convert_dates=["ds"])
    cutoff = pd.Timestamp("2025-10-01")
    train = daily[daily["ds"] < cutoff].copy()
    test  = daily[(daily["ds"] >= cutoff) & (daily["ds"] < "2026-01-01")]
    actual_90d = test["revenue"].sum()

    df = train[["ds", "revenue"]].rename(columns={"revenue": "y"}).copy()
    df["y"] = np.log1p(df["y"].clip(lower=0))
    df["log_spend"] = np.log1p(train["spend"].values)

    m = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        holidays=_PROMO_HOLIDAYS, interval_width=0.80, uncertainty_samples=300,
        changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
    )
    m.add_regressor("log_spend", standardize=True)
    m.fit(df)

    future = m.make_future_dataframe(periods=90, freq="D")
    future["log_spend"] = np.log1p(train["spend"].mean())
    fc = m.predict(future)
    last = train["ds"].max()
    window = fc[fc["ds"] > last].head(90)
    pred_90d = np.expm1(window["yhat"].clip(lower=0)).sum()
    pred_lo  = np.expm1(window["yhat_lower"].clip(lower=0)).sum()
    pred_hi  = np.expm1(window["yhat_upper"].clip(lower=0)).sum()
    mape = abs(pred_90d - actual_90d) / actual_90d * 100
    return {
        "train_cutoff": str(cutoff.date()),
        "test_period": "Oct–Dec 2025",
        "actual_90d": actual_90d,
        "predicted_90d": pred_90d,
        "ci_lower": pred_lo,
        "ci_upper": pred_hi,
        "mape": mape,
        "ci_covers": pred_lo <= actual_90d <= pred_hi,
    }


def _fmt_usd(val):
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if val >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.0f}"


def _range_chart(forecasts, title, y_label, is_roas=False):
    days   = [f["days"] for f in forecasts]
    points = [f["point"] for f in forecasts]
    lowers = [f["lower"] for f in forecasts]
    uppers = [f["upper"] for f in forecasts]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{d}d" for d in days],
        y=points,
        error_y=dict(type="data", symmetric=False,
                     array=[u - p for p, u in zip(points, uppers)],
                     arrayminus=[p - l for p, l in zip(points, lowers)],
                     color="#6366f1", thickness=2.5, width=8),
        marker_color=["#4f46e5", "#7c3aed", "#a855f7"],
        text=[f"{p:,.2f}×" if is_roas else _fmt_usd(p) for p in points],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=y_label,
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font=dict(color="#d1d5db"), margin=dict(t=50, b=30, l=40, r=20),
        showlegend=False, height=280,
    )
    fig.update_xaxes(gridcolor="#1f2937")
    fig.update_yaxes(gridcolor="#1f2937")
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Netexlir")
    st.caption("AI-Assisted Ecommerce Forecasting · AIgnition 3.0")
    st.divider()

    with st.spinner("Loading data…"):
        daily, channel_data, camp_types, summary, trailing = _load_data()

    st.success(f"Data loaded · {daily['ds'].min().date()} → {daily['ds'].max().date()}")
    st.caption(f"{len(daily):,} daily rows · 3 channels · {len(summary)} campaigns")
    st.divider()

    st.subheader("AI Insights")
    gemini_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        placeholder="AIza…",
    )
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
    st.divider()

    st.subheader("Budget Simulation")
    st.caption("Daily spend per channel (USD)")
    budget_google = st.number_input("Google Ads", min_value=0.0, value=3000.0, step=100.0)
    budget_meta   = st.number_input("Meta Ads",   min_value=0.0, value=500.0,  step=50.0)
    budget_bing   = st.number_input("Bing Ads",   min_value=0.0, value=200.0,  step=50.0)
    run_sim = st.button("Run Budget Simulation", use_container_width=True, type="primary")
    st.divider()
    st.caption("Prediction intervals: 80% CI via Prophet Monte Carlo (500 samples)")


# ── Serialise for cache ────────────────────────────────────────────────────────

daily_json      = daily.to_json(orient="split", date_format="iso")
channel_json    = {ch: df.to_json(orient="split", date_format="iso")
                   for ch, df in channel_data.items()}
camptype_json   = {k: df.to_json(orient="split", date_format="iso")
                   for k, df in camp_types.items()}


# ── Run forecasts ──────────────────────────────────────────────────────────────

with st.spinner("Training Prophet models — first run ~60 s, then cached…"):
    forecast     = _run_forecast(daily_json)
    ch_forecasts = _run_channel_forecasts(channel_json)
    ct_forecasts = _run_camptype_forecasts(camptype_json)
    anomalies    = _detect_anomalies(daily_json, channel_json)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_fcst, tab_ch, tab_budget, tab_anom, tab_ai, tab_val, tab_sub, tab_demo = st.tabs([
    "Forecast",
    "Channels & Types",
    "Budget Simulation",
    "Anomalies",
    "AI Insights",
    "Validation",
    "Submission",
    "Demo Guide",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Aggregate Forecast
# ════════════════════════════════════════════════════════════════════════════

with tab_fcst:
    st.header("Aggregate Revenue & ROAS Forecast")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trailing 30d Revenue", _fmt_usd(trailing["revenue"]))
    with col2:
        st.metric("Trailing 30d Spend", _fmt_usd(trailing["spend"]))
    with col3:
        st.metric("Trailing 30d ROAS", f"{trailing['roas']:.2f}×")
    with col4:
        st.metric("Data through", str(trailing["end"].date()))

    rev30 = forecast["revenue_forecasts"][0]
    delta_pct = (rev30["point"] / trailing["revenue"] - 1) * 100

    if delta_pct < -10:
        st.info(
            f"**Why does the 30d forecast ({_fmt_usd(rev30['point'])}) look lower than trailing ({_fmt_usd(trailing['revenue'])})?**  "
            f"The trailing 30d covers late April – early June, which historically peaks in May. "
            f"June is a lower-revenue month based on 2025 data (May 2025 ≈ $418K vs June 2025 ≈ $127K). "
            f"Prophet correctly predicts the seasonal dip, not a business decline."
        )

    st.divider()

    col_rev, col_roas = st.columns(2)
    with col_rev:
        st.plotly_chart(_range_chart(forecast["revenue_forecasts"],
                                     "Revenue Forecast (80% CI)", "USD"),
                        use_container_width=True)
        rows = [{"Window": f"{r['days']}d",
                 "Lower": _fmt_usd(r["lower"]),
                 "Point": _fmt_usd(r["point"]),
                 "Upper": _fmt_usd(r["upper"])}
                for r in forecast["revenue_forecasts"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col_roas:
        st.plotly_chart(_range_chart(forecast["roas_forecasts"],
                                     "ROAS Forecast (80% CI)", "× ROAS", is_roas=True),
                        use_container_width=True)
        rows = [{"Window": f"{r['days']}d",
                 "Lower": f"{r['lower']:.2f}×",
                 "Point": f"{r['point']:.2f}×",
                 "Upper": f"{r['upper']:.2f}×"}
                for r in forecast["roas_forecasts"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Historical Daily Revenue")

    # Monthly revenue bar
    monthly = daily.copy()
    monthly["month"] = monthly["ds"].dt.to_period("M").astype(str)
    monthly_rev = monthly.groupby("month")["revenue"].sum().reset_index()
    fig_m = go.Figure(go.Bar(
        x=monthly_rev["month"], y=monthly_rev["revenue"],
        marker_color=["#a855f7" if "11" in m or "12" in m else "#4f46e5"
                      for m in monthly_rev["month"]],
        text=[_fmt_usd(v) for v in monthly_rev["revenue"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:$,.0f}<extra></extra>",
    ))
    fig_m.update_layout(
        title="Monthly Revenue (purple = Nov/Dec holiday spike)",
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font=dict(color="#d1d5db"), height=300,
        margin=dict(t=50, b=30, l=40, r=20),
    )
    fig_m.update_xaxes(gridcolor="#1f2937", tickangle=-45)
    fig_m.update_yaxes(gridcolor="#1f2937")
    st.plotly_chart(fig_m, use_container_width=True)
    st.caption("Key insight: Nov–Dec account for ~53% of annual revenue. "
               "The model correctly predicts a June seasonal dip following the May peak.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Channels & Campaign Types
# ════════════════════════════════════════════════════════════════════════════

with tab_ch:
    st.header("Channel & Campaign-Type Forecasts")

    subtab_ch, subtab_type, subtab_camp = st.tabs(
        ["Channel (30/60/90d)", "Campaign Type (30d)", "Top Campaigns"]
    )

    with subtab_ch:
        if ch_forecasts:
            # Only show channels with meaningful forecasts
            valid_channels = {ch: r for ch, r in ch_forecasts.items()
                              if r["revenue_forecasts"][0]["point"] > 100}
            skipped = [ch for ch in ch_forecasts if ch not in valid_channels]

            if skipped:
                st.warning(
                    f"**{', '.join(c.title() for c in skipped)}** excluded from channel charts "
                    f"— insufficient historical data for a reliable Prophet fit "
                    f"(< 90 days with non-zero revenue)."
                )

            if valid_channels:
                cols = st.columns(len(valid_channels))
                for col, (ch, result) in zip(cols, valid_channels.items()):
                    rev30  = result["revenue_forecasts"][0]
                    roas30 = result["roas_forecasts"][0]
                    with col:
                        st.subheader(ch.title())
                        st.metric("30d Revenue (point)", _fmt_usd(rev30["point"]))
                        st.caption(f"CI: {_fmt_usd(rev30['lower'])} – {_fmt_usd(rev30['upper'])}")
                        st.metric("30d ROAS (point)", f"{roas30['point']:.2f}×")
                        st.caption(f"CI: {roas30['lower']:.2f}× – {roas30['upper']:.2f}×")

                st.divider()

                # Revenue share pie
                ch_revs = {ch: r["revenue_forecasts"][0]["point"] for ch, r in valid_channels.items()}
                fig_pie = go.Figure(go.Pie(
                    labels=list(ch_revs.keys()),
                    values=list(ch_revs.values()),
                    hole=0.42,
                    marker_colors=["#4f46e5", "#7c3aed", "#a855f7"],
                ))
                fig_pie.update_layout(
                    title="Projected 30d Revenue Share",
                    paper_bgcolor="#0f0f1a", font=dict(color="#d1d5db"),
                    height=300, margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                st.subheader("All Windows")
                rows = []
                for ch, result in valid_channels.items():
                    for rev, roas in zip(result["revenue_forecasts"], result["roas_forecasts"]):
                        rows.append({
                            "Channel": ch.title(), "Window": f"{rev['days']}d",
                            "Revenue": _fmt_usd(rev["point"]),
                            "Revenue CI": f"{_fmt_usd(rev['lower'])} – {_fmt_usd(rev['upper'])}",
                            "ROAS": f"{roas['point']:.2f}×",
                        })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with subtab_type:
        if ct_forecasts:
            rows = []
            for label, result in ct_forecasts.items():
                rev30  = result["revenue_forecasts"][0]
                roas30 = result["roas_forecasts"][0]
                rows.append({
                    "Campaign Type":   label,
                    "30d Revenue":     _fmt_usd(rev30["point"]),
                    "Revenue CI":      f"{_fmt_usd(rev30['lower'])} – {_fmt_usd(rev30['upper'])}",
                    "30d ROAS":        f"{roas30['point']:.2f}×",
                    "ROAS CI":         f"{roas30['lower']:.2f}× – {roas30['upper']:.2f}×",
                })
            df_ct = pd.DataFrame(rows).sort_values("30d Revenue", ascending=False)
            st.dataframe(df_ct, hide_index=True, use_container_width=True)

            # Bar chart of campaign types
            fig_ct = go.Figure(go.Bar(
                x=[r["Campaign Type"] for r in rows],
                y=[result["revenue_forecasts"][0]["point"] for result in ct_forecasts.values()],
                marker_color="#6366f1",
                text=[_fmt_usd(result["revenue_forecasts"][0]["point"])
                      for result in ct_forecasts.values()],
                textposition="outside",
            ))
            fig_ct.update_layout(
                title="30d Revenue by Campaign Type (point estimate)",
                paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                font=dict(color="#d1d5db"), height=300,
                margin=dict(t=50, b=60, l=40, r=20),
            )
            fig_ct.update_xaxes(gridcolor="#1f2937", tickangle=-30)
            fig_ct.update_yaxes(gridcolor="#1f2937")
            st.plotly_chart(fig_ct, use_container_width=True)
        else:
            st.info("No campaign-type forecasts available.")

    with subtab_camp:
        st.subheader("Top 20 Campaigns by Historical Revenue")
        st.dataframe(
            summary.head(20)[["channel", "campaign_name", "campaign_type",
                               "total_revenue", "total_spend", "roas",
                               "nonzero_revenue_days"]].rename(columns={
                "total_revenue": "Revenue", "total_spend": "Spend",
                "roas": "ROAS", "nonzero_revenue_days": "Active Days",
            }),
            hide_index=True, use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Budget Simulation
# ════════════════════════════════════════════════════════════════════════════

with tab_budget:
    st.header("Budget Simulation")
    st.caption("Adjust per-channel daily budgets in the sidebar then click **Run Budget Simulation**.")

    budget_map    = {"google": budget_google, "meta": budget_meta, "bing": budget_bing}
    total_daily   = sum(budget_map.values())
    trailing_daily = trailing["spend"] / 30
    delta_pct      = (total_daily / trailing_daily - 1) * 100 if trailing_daily else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Daily Budget", _fmt_usd(total_daily))
    c2.metric("Monthly Budget", _fmt_usd(total_daily * 30))
    c3.metric("vs Trailing Daily Spend", _fmt_usd(trailing_daily), delta=f"{delta_pct:+.1f}%")

    if run_sim:
        with st.spinner("Running budget simulation…"):
            sim = simulate_budget(channel_data, budget_map, uncertainty_samples=300)
            st.session_state["sim_result"]  = sim
            st.session_state["sim_budgets"] = dict(budget_map)

    if "sim_result" not in st.session_state:
        st.info("Set budgets in the sidebar and click **Run Budget Simulation** to see projections.")
    else:
        sim = st.session_state["sim_result"]
        st.divider()
        st.subheader("Portfolio Projection")

        base_revs = {r["days"]: r["point"] for r in forecast["revenue_forecasts"]}
        rows = []
        for days, p in sim["portfolio"].items():
            base = base_revs.get(days, 0)
            uplift = (p["revenue_point"] / base - 1) * 100 if base else 0
            rows.append({
                "Window":     f"{days}d",
                "Revenue":    _fmt_usd(p["revenue_point"]),
                "Revenue CI": f"{_fmt_usd(p['revenue_lower'])} – {_fmt_usd(p['revenue_upper'])}",
                "ROAS":       f"{p['roas_point']:.2f}×",
                "vs Baseline": f"{uplift:+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        base_30 = base_revs.get(30, 0)
        sim_30  = sim["portfolio"][30]["revenue_point"]
        sim_lo  = sim["portfolio"][30]["revenue_lower"]
        sim_hi  = sim["portfolio"][30]["revenue_upper"]

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            x=["Baseline (30d)", "Simulated (30d)"],
            y=[base_30, sim_30],
            error_y=dict(type="data", symmetric=False,
                         array=[0, sim_hi - sim_30],
                         arrayminus=[0, sim_30 - sim_lo],
                         color="#6366f1", thickness=2.5, width=8),
            marker_color=["#374151", "#4f46e5"],
            text=[_fmt_usd(base_30), _fmt_usd(sim_30)],
            textposition="outside",
        ))
        fig_sim.update_layout(
            title="Baseline vs Simulated 30d Revenue",
            paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
            font=dict(color="#d1d5db"), height=280,
            margin=dict(t=50, b=30, l=40, r=20),
        )
        fig_sim.update_xaxes(gridcolor="#1f2937")
        fig_sim.update_yaxes(gridcolor="#1f2937")
        st.plotly_chart(fig_sim, use_container_width=True)
        st.caption("Diminishing returns are modelled via log(spend) regressor — doubling spend does not double revenue.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Anomalies
# ════════════════════════════════════════════════════════════════════════════

with tab_anom:
    st.header("Anomaly Detection")
    st.caption("Rolling 90-day lookback · Revenue z-score ≥ 2.5σ · ROAS collapse · Spend-revenue decoupling")

    if not anomalies:
        st.success("No significant anomalies detected in the last 90 days.")
    else:
        high   = [a for a in anomalies if a["severity"] == "high"]
        medium = [a for a in anomalies if a["severity"] == "medium"]
        c1, c2, c3 = st.columns(3)
        c1.metric("High severity",   len(high))
        c2.metric("Medium severity", len(medium))
        c3.metric("Total anomalies", len(anomalies))
        st.divider()
        for a in anomalies:
            sev  = a["severity"]
            pill = f'<span class="pill-{sev}">{sev.upper()}</span>'
            with st.expander(f"{a['date']}  ·  {a['channel'].title()}  ·  {a['type'].replace('_', ' ').title()}"):
                st.markdown(f"{pill} &nbsp; {a['description']}", unsafe_allow_html=True)
                st.caption(f"Magnitude: {a['magnitude']:.2f}σ")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI Insights
# ════════════════════════════════════════════════════════════════════════════

with tab_ai:
    st.header("AI Insights — Powered by Gemini")

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("Paste your **Google API Key** in the sidebar to enable AI insights.")
        from src.llm import get_insights_dry_run
        dry = get_insights_dry_run(forecast, trailing, anomalies,
                                   ch_forecasts if ch_forecasts else None)
        with st.expander("Preview — forecast context that will be sent to Gemini"):
            st.code(dry["forecast_explanation_prompt"], language="json")
        st.caption(f"Model: `{dry['model']}` · {dry['anomaly_count']} anomalies detected")
    else:
        sim_result      = st.session_state.get("sim_result")
        budget_by_ch    = st.session_state.get("sim_budgets")

        col_gen, _ = st.columns([1, 3])
        with col_gen:
            gen_btn = st.button("Generate Insights", type="primary", use_container_width=True)

        if gen_btn:
            from src.llm import get_insights
            with st.spinner("Calling Gemini (3–4 API calls)…"):
                insights = get_insights(
                    forecast_result=forecast, trailing=trailing,
                    anomaly_list=anomalies,
                    channel_results=ch_forecasts if ch_forecasts else None,
                    sim_result=sim_result, budget_by_channel=budget_by_ch,
                )
            st.session_state["insights"] = insights

        if "insights" in st.session_state:
            ins = st.session_state["insights"]
            st.subheader("Forecast Explanation")
            st.write(ins.get("forecast_explanation", "—"))
            if "budget_explanation" in ins:
                st.subheader("Budget Simulation Commentary")
                st.write(ins["budget_explanation"])
            st.subheader("Anomaly Interpretation")
            st.write(ins.get("anomaly_interpretation", "—"))
            st.subheader("Risk Flags")
            st.markdown(ins.get("risk_flags", "—"))


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Validation / Backtest
# ════════════════════════════════════════════════════════════════════════════

with tab_val:
    st.header("Model Validation — Walk-Forward Backtest")
    st.caption(
        "4-point walk-forward validation: train on data before each cutoff, "
        "predict 30/60/90 days forward, compare against actuals. "
        "Run offline; results are from actual model predictions."
    )

    # Pre-computed walk-forward results (computed offline, see README)
    wf_data = [
        {"Cutoff":"Apr 2025","Window":"30d","Actual":424835,"Predicted":346768,"MAPE":18.4,"CI Covers":"Yes"},
        {"Cutoff":"Apr 2025","Window":"60d","Actual":840034,"Predicted":563752,"MAPE":32.9,"CI Covers":"Yes"},
        {"Cutoff":"Apr 2025","Window":"90d","Actual":965680,"Predicted":842002,"MAPE":12.8,"CI Covers":"Yes"},
        {"Cutoff":"Jul 2025","Window":"30d","Actual":148409,"Predicted":338124,"MAPE":127.8,"CI Covers":"No"},
        {"Cutoff":"Jul 2025","Window":"60d","Actual":290782,"Predicted":743538,"MAPE":155.7,"CI Covers":"No"},
        {"Cutoff":"Jul 2025","Window":"90d","Actual":484565,"Predicted":1230752,"MAPE":154.0,"CI Covers":"No"},
        {"Cutoff":"Oct 2025","Window":"30d","Actual":233890,"Predicted":271472,"MAPE":16.1,"CI Covers":"Yes"},
        {"Cutoff":"Oct 2025","Window":"60d","Actual":783531,"Predicted":622568,"MAPE":20.5,"CI Covers":"Yes"},
        {"Cutoff":"Oct 2025","Window":"90d","Actual":2688006,"Predicted":1039278,"MAPE":61.3,"CI Covers":"No"},
        {"Cutoff":"Jan 2026","Window":"30d","Actual":255198,"Predicted":249703,"MAPE":2.2,"CI Covers":"Yes"},
        {"Cutoff":"Jan 2026","Window":"60d","Actual":590724,"Predicted":505634,"MAPE":14.4,"CI Covers":"Yes"},
        {"Cutoff":"Jan 2026","Window":"90d","Actual":958142,"Predicted":731210,"MAPE":23.7,"CI Covers":"Yes"},
    ]
    wf_df = pd.DataFrame(wf_data)
    wf_df["Actual"]    = wf_df["Actual"].apply(_fmt_usd)
    wf_df["Predicted"] = wf_df["Predicted"].apply(_fmt_usd)
    wf_df["MAPE"]      = wf_df["MAPE"].apply(lambda x: f"{x:.1f}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("30d Coverage (target 80%)", "75%")
    c2.metric("60d Coverage (target 80%)", "75%")
    c3.metric("90d Coverage (target 80%)", "50%")

    st.divider()

    # Coverage bar chart
    fig_cov = go.Figure()
    fig_cov.add_trace(go.Bar(
        x=["30d", "60d", "90d"],
        y=[75, 75, 50],
        name="Observed coverage",
        marker_color=["#22c55e", "#22c55e", "#f59e0b"],
        text=["75%", "75%", "50%"],
        textposition="outside",
    ))
    fig_cov.add_trace(go.Scatter(
        x=["30d", "60d", "90d"], y=[80, 80, 80],
        mode="lines", name="Target (80% CI)",
        line=dict(color="#6366f1", width=2, dash="dash"),
    ))
    fig_cov.update_layout(
        title="CI Coverage Probability vs Target (80% CI → expect 80% coverage)",
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font=dict(color="#d1d5db"), height=280,
        margin=dict(t=50, b=30, l=40, r=20),
        yaxis=dict(range=[0, 100], title="% of actuals inside CI"),
        legend=dict(orientation="h", y=1.15),
    )
    fig_cov.update_xaxes(gridcolor="#1f2937")
    fig_cov.update_yaxes(gridcolor="#1f2937")
    st.plotly_chart(fig_cov, use_container_width=True)

    st.subheader("Walk-Forward Results (all 12 tests)")
    st.dataframe(wf_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Why Coverage Probability > MAPE for Probabilistic Forecasts")
    st.markdown("""
**MAPE measures point accuracy.** But this tool produces *ranges*, not single numbers.
Evaluating a range by comparing its midpoint to the actual is like grading a weather
forecast "40% chance of rain" by asking whether it rained exactly 40% that day.

**Coverage probability** is the right metric: what fraction of actual outcomes fall
inside the stated prediction interval?

| Window | Coverage | Target | Assessment |
|---|---|---|---|
| 30 days | 75% | 80% | Close to target — reliable for tactical planning |
| 60 days | 75% | 80% | Close to target — directional planning |
| 90 days | 50% | 80% | Undercoverage driven by two structural causes (see below) |

**Two structural causes of 90d undercoverage:**

1. **Jul 2025 cutoff** — After a strong Apr–May 2025 peak (≈$418K/month), the model
   extrapolated growth momentum. Reality: Jul–Sep 2025 dipped to ≈$148–207K/month
   (summer slowdown). Any model trained on < 3 years of data will struggle with
   mid-year trend reversals that contradict the recent trend.

2. **Oct 2025 cutoff** — Nov–Dec 2025 produced $2.4M out of $2.7M in that 90-day window.
   With only one Q4 season in training data, Prophet cannot reliably estimate holiday spike
   *magnitude* even if it knows the *timing*. The CI upper correctly reached $1.82M —
   still below the actual $2.69M, indicating one season of Q4 data is insufficient.

**Practical advice for agencies:** Use 30d forecasts for media budget decisions (coverage ≈ 75%).
Treat 90d forecasts as a strategic range, not a commitment. Flag Q4 windows separately —
the model's CI upper bound is the conservative budget planning figure for Nov–Dec.
""")

    with st.spinner("Running live single backtest (Oct 2025 cutoff)…"):
        bt = _backtest(daily_json)

    st.divider()
    st.subheader("Live Check — Oct 2025 Cutoff, 90d")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual (Oct–Dec 2025)", _fmt_usd(bt["actual_90d"]))
    c2.metric("Predicted", _fmt_usd(bt["predicted_90d"]))
    c3.metric("MAPE", f"{bt['mape']:.1f}%")
    c4.metric("CI covers actual", "Yes" if bt["ci_covers"] else "No")


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Submission Structure
# ════════════════════════════════════════════════════════════════════════════

with tab_sub:
    st.header("Submission Structure & How to Run")
    st.caption("Repository layout, scoring pipeline, and output format.")

    st.subheader("Repository Layout")
    st.code("""
netexlir/
├── run.sh                        ← Entry point (required)
├── requirements.txt              ← Pinned dependencies (required)
├── data/                         ← Input CSVs — replaced by test data at eval time
│   ├── bing_campaign_stats.csv
│   ├── google_ads_campaign_stats.csv
│   └── meta_ads_campaign_stats.csv
├── pickle/
│   └── model.pkl                 ← Pre-trained Prophet models + hyperparameters
├── output/
│   └── predictions.csv           ← Generated by run.sh (not committed)
├── src/
│   ├── loader.py                 ← CSV ingestion, daily aggregation
│   ├── forecaster.py             ← Prophet models, cache, 30/60/90d forecast
│   ├── budget_sim.py             ← Budget simulation with diminishing returns
│   ├── anomaly.py                ← Rolling z-score anomaly detection
│   ├── llm.py                    ← Google Gemini integration
│   ├── generate_features.py      ← Step 1: CSVs → parquet
│   ├── predict.py                ← Step 2: parquet + pkl → predictions.csv
│   └── train.py                  ← One-time: fits models, saves pickle
├── app.py                        ← Streamlit demo UI
├── notes/
│   ├── plan.md
│   ├── data_profile.md
│   ├── methodology.md
│   └── forecast_validation.md
└── README.md
""", language="text")

    st.divider()
    st.subheader("Scoring Pipeline")

    st.markdown("Clone the repo, install dependencies, replace the sample data with yours, and run one command.")

    st.code("""# 1. Clone
git clone <your-repo-url>
cd netexlir

# 2. Install
pip install -r requirements.txt

# 3. (Judges do this) Replace data/ with held-out test data
#    cp /their/test/data/*.csv data/

# 4. Run — single command, no interaction needed
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv

# 5. Read output
head output/predictions.csv
""", language="bash")

    st.divider()
    st.subheader("Output Format — `predictions.csv`")

    st.markdown("""
Each row is one forecast cell. The scoring pipeline reads these columns:
""")

    sample_cols = {
        "window_days":   ["30", "60", "90", "30", "30", "0"],
        "level":         ["aggregate", "aggregate", "aggregate", "channel", "campaign_type", "trailing_30d"],
        "entity":        ["all", "all", "all", "google", "google/SEARCH", "all"],
        "revenue_lower": ["131K", "271K", "425K", "153K", "—", "343K"],
        "revenue_point": ["242K", "499K", "779K", "299K", "—", "343K"],
        "revenue_upper": ["445K", "917K", "1.4M", "591K", "—", "343K"],
        "roas_point":    ["4.35", "3.79", "3.67", "5.96", "—", "5.02"],
    }
    st.dataframe(pd.DataFrame(sample_cols), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Running the Demo UI")
    st.code("""# Start the Streamlit dashboard
streamlit run app.py

# Opens at http://localhost:8501
# Sidebar: load data, set API key, set budgets
# Tabs: Forecast · Channels & Types · Budget Sim · Anomalies · AI Insights · Validation · Submission
""", language="bash")

    st.divider()
    st.subheader("One-time: Retrain & Pickle the Models")
    st.code("""# Run this if you change the data or model config
python3 src/train.py --output ./pickle/model.pkl

# Trains 5 Prophet models (aggregate + 3 channels)
# Saves to pickle/model.pkl (~444 KB)
""", language="bash")

    st.divider()
    st.subheader("Environment Variables")
    st.code("""# Required for AI Insights tab
export GOOGLE_API_KEY=AIza...

# Optional: change the Gemini model
export NETEXLIR_MODEL=gemini-2.5-flash-preview-05-20

# Optional: point to a different data directory
export DATA_DIR=/path/to/your/csvs

# Python version used
python3 --version   # Python 3.12
""", language="bash")

    st.divider()
    st.subheader("Submission Checklist")
    checks = {
        "run.sh at root, executable, works with single command":         True,
        "run.sh accepts DATA_DIR, MODEL_PATH, OUTPUT_PATH as args":      True,
        "requirements.txt with pinned versions":                          True,
        "data/ folder with sample CSVs":                                  True,
        "pickle/model.pkl committed (~444 KB)":                           True,
        "output/predictions.csv generated fresh by run.sh":              True,
        "No absolute paths in source code":                               True,
        "No internet calls at prediction time (LLM is UI-only)":         True,
        "Random seeds set (SEED=42) in predict.py and train.py":         True,
        "README.md with architecture and run instructions":               True,
    }
    for item, ok in checks.items():
        status = "Pass" if ok else "Fail"
        st.write(f"{status}  —  {item}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — Demo Guide (2-minute walkthrough script)
# ════════════════════════════════════════════════════════════════════════════

with tab_demo:
    st.header("2-Minute Demo Walkthrough")
    st.caption("A suggested flow for walking through the app during the presentation.")

    st.info(
        "Before presenting: open the app at http://localhost:8501, "
        "paste your Google API Key in the sidebar, "
        "and set budgets to Google $4,000 · Meta $800 · Bing $200."
    )

    steps = [
        {
            "time": "0:00 – 0:20",
            "tab": "Forecast",
            "title": "The Problem & The Forecast",
            "say": (
                "We're solving the hardest question a digital marketing agency faces: "
                "*what will revenue look like next month?* "
                "Most tools give you a single number — ours gives you a range. "
                "Here's the 30-day forecast: **$131K–$445K** with an 80% prediction interval. "
                "The point estimate is $242K, but we're honest that it could be 3× higher or lower."
            ),
            "click": "Point at the revenue bar chart. Hover over the 90-day bar to show the wide CI.",
        },
        {
            "time": "0:20 – 0:35",
            "tab": "Forecast",
            "title": "Why the Forecast Looks Lower Than Last Month",
            "say": (
                "Notice the blue info box explaining why the forecast is lower than trailing actuals. "
                "The trailing 30 days captured a May peak ($418K). June is seasonally weaker — "
                "our 2025 data confirms June was only $127K. "
                "The model isn't predicting a business problem; it's predicting seasonality correctly."
            ),
            "click": "Point at the monthly revenue bar chart. Highlight the purple Nov–Dec bars.",
        },
        {
            "time": "0:35 – 0:50",
            "tab": "Channels & Types",
            "title": "Channel & Campaign-Type Breakdown",
            "say": (
                "Google drives 95% of forecast revenue at a 5.96× ROAS. "
                "Meta is smaller but 12× ROAS — extremely efficient at current spend levels. "
                "The Campaign Type tab breaks this down further: Performance Max vs Search vs Demand Gen."
            ),
            "click": "Switch to Campaign Type subtab. Show the bar chart.",
        },
        {
            "time": "0:50 – 1:10",
            "tab": "Budget Simulation",
            "title": "Budget Simulation with Diminishing Returns",
            "say": (
                "Now for the most valuable feature for agencies: budget scenario planning. "
                "I've set a total daily budget of $5,000 — about 10% more than current spend. "
                "Watch what happens when I click Run Budget Simulation."
            ),
            "click": "Click 'Run Budget Simulation'. Wait for result. Point at the baseline vs simulated chart.",
        },
        {
            "time": "1:10 – 1:20",
            "tab": "Budget Simulation",
            "title": "Diminishing Returns Encoded",
            "say": (
                "The uplift is positive but modest — not linear with the budget increase. "
                "That's because we model spend with a log transform inside Prophet. "
                "Doubling spend does not double revenue. This is realistic. "
                "Most tools assume linearity; ours doesn't."
            ),
            "click": "Point at the 'vs Baseline' column in the portfolio table.",
        },
        {
            "time": "1:20 – 1:35",
            "tab": "AI Insights",
            "title": "Gemini-Powered Causal Summaries",
            "say": (
                "Click Generate Insights. Gemini makes 3–4 API calls: "
                "a forecast explanation, anomaly interpretation, and 4 prioritised risk flags. "
                "All prompts are structured — not 'summarise this data' but specific analytical questions."
            ),
            "click": "Click 'Generate Insights'. While loading, mention the anomaly count in the sidebar.",
        },
        {
            "time": "1:35 – 1:50",
            "tab": "Validation",
            "title": "We Measured Ourselves",
            "say": (
                "We ran a 4-point walk-forward backtest. "
                "30-day coverage probability is 75% — close to our stated 80% CI. "
                "The 90-day undercoverage comes from the Q4 holiday spike; "
                "with only one Q4 season in training data, no model handles this well. "
                "We're transparent about this limitation."
            ),
            "click": "Point at the coverage chart. Show the dashed 80% target line.",
        },
        {
            "time": "1:50 – 2:00",
            "tab": "Submission",
            "title": "Fully Submission-Ready",
            "say": (
                "The submission checklist is all green. "
                "Clone the repo, pip install, drop in test data, run './run.sh' — "
                "that's it. The output is a structured predictions.csv with "
                "aggregate, channel, campaign-type, and campaign-level forecasts."
            ),
            "click": "Point at the green checklist.",
        },
    ]

    for i, step in enumerate(steps, 1):
        with st.expander(f"Step {i} · {step['time']} · {step['title']}", expanded=(i == 1)):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(step['say'])
            with col_b:
                st.caption(f"Tab: {step['tab']}")
                st.caption(step['click'])

    st.divider()
    st.subheader("Key Numbers to Know by Heart")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("All-time Revenue", "$11.09M")
        st.metric("Blended ROAS", "5.09×")
        st.metric("Channels", "3 (Google, Meta, Bing)")
    with col2:
        st.metric("30d Forecast Range", "$131K – $445K")
        st.metric("30d Coverage", "75%  (target 80%)")
        st.metric("Anomalies (90d)", "18 detected")
    with col3:
        st.metric("Budget Sim", "Diminishing returns via log(spend)")
        st.metric("LLM Calls per Run", "3–4 Gemini API calls")
        st.metric("Scoring Pipeline", "./run.sh → predictions.csv")

