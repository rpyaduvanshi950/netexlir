"""
Netexlir — AI-Assisted Ecommerce Forecasting
Streamlit prototype: ingest → forecast → budget sim → AI insights
"""

import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.loader import load_daily_aggregate, load_daily_by_channel, campaign_summary
from src.forecaster import run_aggregate_forecast, run_channel_forecasts, trailing_actuals
from src.anomaly import detect_anomalies
from src.budget_sim import simulate_budget

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Netexlir · Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-box { background:#1e1e2e; border-radius:8px; padding:14px 18px; margin:4px 0; }
  .metric-label { color:#9ca3af; font-size:0.78rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
  .metric-value { color:#f3f4f6; font-size:1.5rem; font-weight:700; margin-top:2px; }
  .metric-delta { font-size:0.8rem; margin-top:2px; }
  .pill-high   { background:#7f1d1d; color:#fca5a5; border-radius:12px; padding:2px 8px; font-size:.75rem; font-weight:600; }
  .pill-medium { background:#78350f; color:#fcd34d; border-radius:12px; padding:2px 8px; font-size:.75rem; font-weight:600; }
  .pill-low    { background:#1e3a5f; color:#93c5fd; border-radius:12px; padding:2px 8px; font-size:.75rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Cached data & forecast helpers ────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_data():
    daily = load_daily_aggregate()
    channel_data = load_daily_by_channel()
    summary = campaign_summary()
    trailing = trailing_actuals(daily, days=30)
    return daily, channel_data, summary, trailing


@st.cache_data(show_spinner=False)
def _run_forecast(daily_json: str):
    daily = pd.read_json(daily_json, orient="split", convert_dates=["ds"])
    return run_aggregate_forecast(daily, uncertainty_samples=500)


@st.cache_data(show_spinner=False)
def _run_channel_forecasts(channel_json: dict):
    channel_data = {ch: pd.read_json(v, orient="split", convert_dates=["ds"])
                    for ch, v in channel_json.items()}
    return run_channel_forecasts(channel_data, uncertainty_samples=300)


@st.cache_data(show_spinner=False)
def _detect_anomalies(daily_json: str, channel_json: dict):
    daily = pd.read_json(daily_json, orient="split", convert_dates=["ds"])
    channel_data = {ch: pd.read_json(v, orient="split", convert_dates=["ds"])
                    for ch, v in channel_json.items()}
    return detect_anomalies(daily, channel_data=channel_data)


def _fmt_usd(val: float) -> str:
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if val >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.0f}"


def _range_bar_chart(forecasts: list[dict], title: str, y_label: str, is_roas=False) -> go.Figure:
    days = [f["days"] for f in forecasts]
    points = [f["point"] for f in forecasts]
    lowers = [f["lower"] for f in forecasts]
    uppers = [f["upper"] for f in forecasts]
    err_minus = [p - l for p, l in zip(points, lowers)]
    err_plus  = [u - p for p, u in zip(points, uppers)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{d}d" for d in days],
        y=points,
        error_y=dict(type="data", symmetric=False,
                     array=err_plus, arrayminus=err_minus,
                     color="#6366f1", thickness=2.5, width=8),
        marker_color=["#4f46e5", "#7c3aed", "#a855f7"],
        text=[f"{'×' if is_roas else '$'}{p:,.2f}" if is_roas else _fmt_usd(p) for p in points],
        textposition="outside",
        name="Point estimate",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=y_label,
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font=dict(color="#d1d5db"),
        margin=dict(t=50, b=30, l=40, r=20),
        showlegend=False,
        height=280,
    )
    fig.update_xaxes(gridcolor="#1f2937")
    fig.update_yaxes(gridcolor="#1f2937")
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Netexlir")
    st.caption("AI-Assisted Ecommerce Forecasting")
    st.divider()

    # Load data once
    with st.spinner("Loading data…"):
        daily, channel_data, summary, trailing = _load_data()

    st.success(f"Data loaded · {daily['ds'].min().date()} → {daily['ds'].max().date()}")
    st.caption(f"{len(daily):,} daily rows · 3 channels · {len(summary)} campaigns")
    st.divider()

    st.subheader("AI Insights")
    gemini_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        placeholder="AIza…",
        help="Set GOOGLE_API_KEY env var or paste here",
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

    st.caption("Interval: 80% prediction interval (Prophet Monte Carlo)")


# ── Serialise DataFrames for cache keys ───────────────────────────────────────

daily_json = daily.to_json(orient="split", date_format="iso")
channel_json = {ch: df.to_json(orient="split", date_format="iso")
                for ch, df in channel_data.items()}


# ── Run core forecasts ────────────────────────────────────────────────────────

with st.spinner("Training Prophet models (first run takes ~45 s)…"):
    forecast = _run_forecast(daily_json)
    ch_forecasts = _run_channel_forecasts(channel_json)
    anomalies = _detect_anomalies(daily_json, channel_json)


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_fcst, tab_ch, tab_budget, tab_anom, tab_ai = st.tabs([
    "📊 Forecast", "📡 Channels", "💰 Budget Sim", "🚨 Anomalies", "🤖 AI Insights"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Aggregate Forecast
# ════════════════════════════════════════════════════════════════════════════

with tab_fcst:
    st.header("Aggregate Revenue & ROAS Forecast")

    # Trailing actuals row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trailing 30d Revenue", _fmt_usd(trailing["revenue"]))
    with col2:
        st.metric("Trailing 30d Spend", _fmt_usd(trailing["spend"]))
    with col3:
        st.metric("Trailing 30d ROAS", f"{trailing['roas']:.2f}×")
    with col4:
        st.metric("Data through", str(trailing["end"].date()))

    st.divider()

    col_rev, col_roas = st.columns(2)

    with col_rev:
        st.plotly_chart(
            _range_bar_chart(forecast["revenue_forecasts"],
                             "Revenue Forecast (30/60/90 days)", "USD"),
            use_container_width=True,
        )
        # Table view
        rows = []
        for rev in forecast["revenue_forecasts"]:
            rows.append({
                "Window": f"{rev['days']}d",
                "Lower (80% CI)": _fmt_usd(rev["lower"]),
                "Point Estimate": _fmt_usd(rev["point"]),
                "Upper (80% CI)": _fmt_usd(rev["upper"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col_roas:
        st.plotly_chart(
            _range_bar_chart(forecast["roas_forecasts"],
                             "ROAS Forecast (30/60/90 days)", "× ROAS", is_roas=True),
            use_container_width=True,
        )
        rows = []
        for roas in forecast["roas_forecasts"]:
            rows.append({
                "Window": f"{roas['days']}d",
                "Lower": f"{roas['lower']:.2f}×",
                "Point": f"{roas['point']:.2f}×",
                "Upper": f"{roas['upper']:.2f}×",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    # Historical revenue chart
    st.subheader("Historical Daily Revenue")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=daily["ds"], y=daily["revenue"].rolling(7).mean(),
        mode="lines", name="7-day rolling avg",
        line=dict(color="#6366f1", width=2),
    ))
    fig_hist.add_trace(go.Scatter(
        x=daily["ds"], y=daily["revenue"],
        mode="lines", name="Daily revenue",
        line=dict(color="#4f46e5", width=0.8),
        opacity=0.4,
    ))
    fig_hist.update_layout(
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font=dict(color="#d1d5db"), height=220,
        margin=dict(t=20, b=30, l=40, r=20),
        legend=dict(orientation="h", y=1.1),
        yaxis_title="Revenue (USD)",
    )
    fig_hist.update_xaxes(gridcolor="#1f2937")
    fig_hist.update_yaxes(gridcolor="#1f2937")
    st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Channel Breakdown
# ════════════════════════════════════════════════════════════════════════════

with tab_ch:
    st.header("Channel-Level Forecasts (30-day)")

    if ch_forecasts:
        cols = st.columns(len(ch_forecasts))
        for col, (ch, result) in zip(cols, ch_forecasts.items()):
            rev30 = result["revenue_forecasts"][0]
            roas30 = result["roas_forecasts"][0]
            with col:
                st.subheader(ch.title())
                st.metric("Revenue (point)", _fmt_usd(rev30["point"]))
                st.caption(f"CI: {_fmt_usd(rev30['lower'])} – {_fmt_usd(rev30['upper'])}")
                st.metric("ROAS (point)", f"{roas30['point']:.2f}×")
                st.caption(f"CI: {roas30['lower']:.2f}× – {roas30['upper']:.2f}×")

        st.divider()

        # Revenue share pie
        ch_revs = {ch: r["revenue_forecasts"][0]["point"] for ch, r in ch_forecasts.items()}
        fig_pie = go.Figure(go.Pie(
            labels=list(ch_revs.keys()),
            values=list(ch_revs.values()),
            hole=0.42,
            marker_colors=["#4f46e5", "#7c3aed", "#a855f7"],
        ))
        fig_pie.update_layout(
            title="Projected 30d Revenue Share",
            paper_bgcolor="#0f0f1a",
            font=dict(color="#d1d5db"),
            height=320,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        st.subheader("90-day Channel Comparison")
        rows = []
        for ch, result in ch_forecasts.items():
            for rev, roas in zip(result["revenue_forecasts"], result["roas_forecasts"]):
                rows.append({
                    "Channel": ch.title(),
                    "Window": f"{rev['days']}d",
                    "Revenue (point)": _fmt_usd(rev["point"]),
                    "Revenue CI": f"{_fmt_usd(rev['lower'])} – {_fmt_usd(rev['upper'])}",
                    "ROAS (point)": f"{roas['point']:.2f}×",
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No channel forecasts available — channels may lack sufficient data.")

    st.divider()
    st.subheader("Top Campaigns by Total Revenue")
    st.dataframe(
        summary.head(15)[["channel", "campaign_name", "campaign_type",
                           "total_revenue", "total_spend", "roas", "nonzero_revenue_days"]]
        .rename(columns={"total_revenue": "Revenue", "total_spend": "Spend",
                         "roas": "ROAS", "nonzero_revenue_days": "Active Days"}),
        hide_index=True,
        use_container_width=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Budget Simulation
# ════════════════════════════════════════════════════════════════════════════

with tab_budget:
    st.header("Budget Simulation")
    st.caption("Adjust per-channel daily budgets in the sidebar and click **Run Budget Simulation**.")

    budget_map = {"google": budget_google, "meta": budget_meta, "bing": budget_bing}
    total_daily = sum(budget_map.values())

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Daily Budget", _fmt_usd(total_daily))
        st.metric("Projected Monthly Budget", _fmt_usd(total_daily * 30))
    with col_b:
        trailing_spend_daily = trailing["spend"] / 30
        delta_pct = (total_daily / trailing_spend_daily - 1) * 100 if trailing_spend_daily else 0
        st.metric("vs Trailing Daily Spend", _fmt_usd(trailing_spend_daily),
                  delta=f"{delta_pct:+.1f}%")

    if "sim_result" not in st.session_state:
        st.info("Set budgets in the sidebar and click **Run Budget Simulation** to see projections.")

    if run_sim:
        with st.spinner("Running budget simulation…"):
            sim = simulate_budget(channel_data, budget_map, uncertainty_samples=300)
            st.session_state["sim_result"] = sim
            st.session_state["sim_budgets"] = dict(budget_map)

    if "sim_result" in st.session_state:
        sim = st.session_state["sim_result"]
        st.divider()
        st.subheader("Portfolio Projection")

        port_rows = []
        baseline_revs = {r["days"]: r["point"] for r in forecast["revenue_forecasts"]}
        for days, p in sim["portfolio"].items():
            base = baseline_revs.get(days, 0)
            uplift = (p["revenue_point"] / base - 1) * 100 if base else 0
            port_rows.append({
                "Window": f"{days}d",
                "Revenue (point)": _fmt_usd(p["revenue_point"]),
                "Revenue CI": f"{_fmt_usd(p['revenue_lower'])} – {_fmt_usd(p['revenue_upper'])}",
                "ROAS (point)": f"{p['roas_point']:.2f}×",
                "vs Baseline": f"{uplift:+.1f}%",
            })
        st.dataframe(pd.DataFrame(port_rows), hide_index=True, use_container_width=True)

        # Baseline vs simulation bar chart
        base_30 = baseline_revs.get(30, 0)
        sim_30  = sim["portfolio"][30]["revenue_point"]
        sim_30_lo = sim["portfolio"][30]["revenue_lower"]
        sim_30_hi = sim["portfolio"][30]["revenue_upper"]

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            x=["Baseline (30d)", "Simulated (30d)"],
            y=[base_30, sim_30],
            error_y=dict(type="data", symmetric=False,
                         array=[0, sim_30_hi - sim_30],
                         arrayminus=[0, sim_30 - sim_30_lo],
                         color="#6366f1", thickness=2.5, width=8),
            marker_color=["#374151", "#4f46e5"],
            text=[_fmt_usd(base_30), _fmt_usd(sim_30)],
            textposition="outside",
        ))
        fig_sim.update_layout(
            title="Baseline vs Simulated 30-day Revenue",
            paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
            font=dict(color="#d1d5db"), height=280,
            margin=dict(t=50, b=30, l=40, r=20),
        )
        fig_sim.update_xaxes(gridcolor="#1f2937")
        fig_sim.update_yaxes(gridcolor="#1f2937")
        st.plotly_chart(fig_sim, use_container_width=True)

        st.caption(
            "Diminishing returns are modelled via a log(spend) regressor in Prophet — "
            "doubling spend does not double revenue."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Anomalies
# ════════════════════════════════════════════════════════════════════════════

with tab_anom:
    st.header("Anomaly Detection")
    st.caption(
        "Scans the last 90 days for revenue spikes/drops (rolling 14-day z-score ≥ 2.5σ), "
        "ROAS collapses, and spend-revenue decoupling."
    )

    if not anomalies:
        st.success("No significant anomalies detected in the last 90 days.")
    else:
        high   = [a for a in anomalies if a["severity"] == "high"]
        medium = [a for a in anomalies if a["severity"] == "medium"]
        low    = [a for a in anomalies if a["severity"] == "low"]

        col1, col2, col3 = st.columns(3)
        col1.metric("High severity", len(high))
        col2.metric("Medium severity", len(medium))
        col3.metric("Low severity", len(low))

        st.divider()

        for a in anomalies:
            sev = a["severity"]
            pill = f'<span class="pill-{sev}">{sev.upper()}</span>'
            with st.expander(f"{a['date']} · {a['channel'].title()} · {a['type'].replace('_', ' ').title()}"):
                st.markdown(f"{pill} &nbsp; {a['description']}", unsafe_allow_html=True)
                st.caption(f"Magnitude: {a['magnitude']:.2f}σ")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI Insights
# ════════════════════════════════════════════════════════════════════════════

with tab_ai:
    st.header("AI Insights")

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("Paste your **Google API Key** in the sidebar to enable AI insights.")
    else:
        sim_result = st.session_state.get("sim_result")
        budget_by_channel = st.session_state.get("sim_budgets")

        col_gen, _ = st.columns([1, 3])
        with col_gen:
            gen_btn = st.button("Generate Insights", type="primary", use_container_width=True)

        if "insights" not in st.session_state or gen_btn:
            if gen_btn:
                from src.llm import get_insights
                with st.spinner("Calling Gemini API (3–4 calls)…"):
                    insights = get_insights(
                        forecast_result=forecast,
                        trailing=trailing,
                        anomaly_list=anomalies,
                        channel_results=ch_forecasts if ch_forecasts else None,
                        sim_result=sim_result,
                        budget_by_channel=budget_by_channel,
                    )
                st.session_state["insights"] = insights

        if "insights" in st.session_state:
            insights = st.session_state["insights"]

            st.subheader("Forecast Explanation")
            st.write(insights.get("forecast_explanation", "—"))

            if "budget_explanation" in insights:
                st.subheader("Budget Simulation Commentary")
                st.write(insights["budget_explanation"])

            st.subheader("Anomaly Interpretation")
            st.write(insights.get("anomaly_interpretation", "—"))

            st.subheader("Risk Flags")
            st.markdown(insights.get("risk_flags", "—"))

        else:
            # Show dry run preview
            from src.llm import get_insights_dry_run
            dry = get_insights_dry_run(forecast, trailing, anomalies, ch_forecasts)
            with st.expander("Preview — forecast context that will be sent to Gemini"):
                st.code(dry["forecast_explanation_prompt"], language="json")
            st.caption(f"Model: `{dry['model']}` · {dry['anomaly_count']} anomalies detected")
