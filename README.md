# Netexlir — AI-Assisted Ecommerce Forecasting

A hackathon prototype that turns raw ad-platform CSVs into probabilistic revenue and ROAS forecasts, budget simulations, anomaly detection, and plain-English AI insights — all in one Streamlit dashboard.

---

## What We Built

We built a full forecasting pipeline for ecommerce marketing teams who need to answer three questions:

1. **What will revenue and ROAS look like in the next 30, 60, and 90 days?**
2. **What happens to revenue if we increase or decrease our ad budgets?**
3. **What is going wrong in our data right now, and why?**

The tool ingests historical ad data from three platforms (Google Ads, Meta Ads, Bing Ads), trains probabilistic time-series models, simulates budget scenarios with diminishing returns, detects anomalies in recent performance, and generates plain-English explanations using the Google Gemini API.

---

## The Problem We Solved

Most forecasting tools give a single-number prediction ("you will make $300K next month"). That is misleading — there is genuine uncertainty in ad performance, and marketers need to know the range of outcomes to make good budget decisions.

We also found that budget simulation tools usually assume linear returns: spend 2× more → earn 2× more. That is wrong. Ad platforms have diminishing returns at scale.

**We solved both problems:**

- All forecasts are **probabilistic ranges** (80% prediction intervals), not single numbers.
- Budget simulation uses a **log(spend) regressor** inside Prophet, so the model naturally captures diminishing returns — doubling spend produces less than double revenue.

---

## Dataset

| Platform | File | Date Range | Campaigns |
|---|---|---|---|
| Google Ads | `data/google_ads_campaign_stats.csv` | Jan 2024 – Jun 2026 | 84 |
| Meta Ads | `data/meta_ads_campaign_stats.csv` | May 2024 – Jun 2026 | 30 |
| Bing Ads | `data/bing_campaign_stats.csv` | May 2024 – Jun 2026 | 22 |

**Combined:** 887 daily rows · 136 campaigns · $11.09M total revenue · $2.18M total spend

Key assumptions made during data analysis:
- Meta's `conversion` column is attributed revenue in USD (confirmed via cost-per-conversion analysis).
- Google's `metrics_cost_micros` divided by 1,000,000 gives spend in USD.
- Attribution used as-is — no cross-platform deduplication was applied.
- No GA4 or Shopify data was present; all revenue comes from ad-platform reports only.

---

## Our Approach

### Forecasting Engine — Facebook Prophet

We chose Prophet over SARIMA/XGBoost because:
- It handles **weekly and yearly seasonality** out of the box.
- It has explicit **holiday/promo event dampening** (Black Friday, Cyber Monday, Prime Day, etc.).
- It produces native **Monte Carlo prediction intervals** (80% CI via 500–1000 uncertainty samples).
- It works on 887 data points without overfitting.

**Key implementation decisions:**

- **Log-transform target**: We fit on `log1p(revenue)` to prevent negative extrapolation. Back-transformed with `expm1()`.
- **Spend regressor**: `log1p(daily_spend)` added as a regressor so budget changes influence forecasts.
- **Revenue floor**: `max(10th-pct of rolling window sums, 5% of avg-rate × days)` prevents the lower CI from collapsing to $0.
- **ROAS CI**: Derived from historical rolling-window ROAS coefficient of variation, not raw Prophet bounds (which produced 0×–136× intervals).
- **In-memory model cache**: Keyed by `(slice_label, y_col, add_spend_regressor, log_transform)` + MD5 data hash so Prophet never retrains on the same data twice in a session.

### Anomaly Detection

Three types detected on a rolling 90-day lookback:

1. **Revenue spike/drop** — rolling 14-day z-score ≥ 2.5σ
2. **ROAS collapse** — ROAS drops > 2.5σ below 14-day mean
3. **Spend-revenue decoupling** — spend rises ≥ 20% WoW but revenue falls ≥ 10% WoW

### LLM Layer — Google Gemini

Four prompt functions generate plain-English output from the structured forecast data:
- `explain_forecast()` — what the numbers mean and what's driving them
- `explain_budget_simulation()` — what the budget change implies for ROI
- `interpret_anomalies()` — likely cause and whether each anomaly needs action
- `flag_risks()` — 4 operational risks with mitigations

All functions are independent and wrapped in try/except so a single API failure does not block the others.

---

## Results

### Historical Performance (All Time)

| Metric | Value |
|---|---|
| Total Revenue | $11,095,456 |
| Total Ad Spend | $2,181,943 |
| Blended ROAS | 5.09× |
| Date Range | Jan 2024 – Jun 2026 |

### Trailing 30-Day Actuals

| Metric | Value |
|---|---|
| Revenue | $342,801 |
| Spend | $68,260 |
| ROAS | 5.02× |

### Channel Breakdown (All Time)

| Channel | Revenue | Spend | ROAS |
|---|---|---|---|
| Google Ads | $9,266,678 | $1,946,126 | 4.76× |
| Meta Ads | $1,656,751 | $196,387 | 8.44× |
| Bing Ads | $172,028 | $39,430 | 4.36× |

### Aggregate Forecast (80% Prediction Interval)

| Window | Lower CI | Point Estimate | Upper CI | ROAS (point) |
|---|---|---|---|---|
| 30 days | $130,768 | $242,283 | $442,702 | 4.35× |
| 60 days | $272,153 | $499,420 | $916,049 | 3.79× |
| 90 days | $427,236 | $779,316 | $1,431,021 | 3.67× |

### Channel 30-Day Forecast

| Channel | Lower CI | Point Estimate | Upper CI | ROAS (point) |
|---|---|---|---|---|
| Google Ads | $153,109 | $299,256 | $590,922 | 5.96× |
| Meta Ads | $4,537 | $15,298 | $49,866 | 12.33× |
| Bing Ads | — | — | — | — |

> Bing is excluded from channel forecasts due to insufficient non-zero revenue days (below the 90-day minimum required for a stable Prophet fit).

### Anomalies Detected (Last 90 Days)

| Severity | Count |
|---|---|
| High | 7 |
| Medium | 11 |
| **Total** | **18** |

Types include revenue spikes/drops, ROAS collapses, and spend-revenue decoupling events.

---

## Model Validation — Why Coverage Probability Beats MAPE

### Walk-Forward Backtest Results

We ran a 4-point walk-forward validation: train on data before a cutoff, predict the next 30/60/90 days, compare against actuals.

| Cutoff | Horizon | Actual | Predicted | MAPE | CI Covers Actual |
|---|---|---|---|---|---|
| Apr 2025 | 30d | $424,835 | $346,768 | 18.4% | ✅ Yes |
| Apr 2025 | 60d | $840,034 | $563,752 | 32.9% | ✅ Yes |
| Apr 2025 | 90d | $965,680 | $842,002 | 12.8% | ✅ Yes |
| Jul 2025 | 30d | $148,409 | $338,124 | 127.8% | ❌ No |
| Jul 2025 | 60d | $290,782 | $743,538 | 155.7% | ❌ No |
| Jul 2025 | 90d | $484,565 | $1,230,752 | 154.0% | ❌ No |
| Oct 2025 | 30d | $233,890 | $271,472 | 16.1% | ✅ Yes |
| Oct 2025 | 60d | $783,531 | $622,568 | 20.5% | ✅ Yes |
| Oct 2025 | 90d | $2,688,006 | $1,039,278 | 61.3% | ❌ No |
| Jan 2026 | 30d | $255,198 | $249,703 | 2.2% | ✅ Yes |
| Jan 2026 | 60d | $590,724 | $505,634 | 14.4% | ✅ Yes |
| Jan 2026 | 90d | $958,142 | $731,210 | 23.7% | ✅ Yes |

**Summary:** 30d MAPE 41%, coverage 75% · 60d MAPE 56%, coverage 75% · 90d MAPE 63%, coverage 50%

### Why MAPE Is the Wrong Metric for Probabilistic Forecasts

This model produces **ranges**, not single numbers. Evaluating a range forecast by comparing only the point estimate to the actual is like grading a confidence interval on whether its midpoint was exactly right — it ignores the entire point of the tool.

**The right metric is coverage probability:** what fraction of actual values fall inside the stated prediction interval? For an 80% CI, the target is 80%.

| Window | Observed Coverage | Target (80% CI) |
|---|---|---|
| 30 days | **75%** | 80% |
| 60 days | **75%** | 80% |
| 90 days | **50%** | 80% |

The 30d and 60d coverage is close to target. The 90d undercoverage has two structural causes:

1. **The Jul 2025 cutoff** — After a strong Apr–May 2025 (≈$418K/month), the model extrapolated upward momentum. In reality Jul–Sep 2025 dipped to ≈$148–207K/month (summer slowdown). Any model trained on < 3 years of data will struggle with mid-year trend reversals.

2. **The Oct 2025 cutoff** — Nov–Dec 2025 produced $2.4M of the $2.7M in that 90-day window. With only one historical Q4 season in training data, Prophet underestimates the holiday spike magnitude. The point estimate was $1.04M; the actual was $2.69M.

**Practical implication for agencies:** Use 30-day forecasts for tactical planning (coverage ≈ 75%). Treat 90-day forecasts as directional ranges, not targets. The wide CI upper bound correctly signals high uncertainty for Q4.

---

## Project Structure

```
netexlir/
├── run.sh                  # Scoring entry point — ./run.sh [DATA_DIR] [MODEL] [OUTPUT]
├── requirements.txt        # Pinned dependencies
├── app.py                  # Streamlit dashboard (8 tabs)
├── README.md
├── DEMO.md                 # 2-minute presentation script
│
├── data/                   # Sample CSVs (replaced by judges at eval time)
│   ├── bing_campaign_stats.csv
│   ├── google_ads_campaign_stats.csv
│   └── meta_ads_campaign_stats.csv
│
├── pickle/
│   └── model.pkl           # Pre-trained Prophet models (444 KB)
│
├── src/
│   ├── generate_features.py  # Step 1: CSVs → parquet feature store
│   ├── predict.py            # Step 2: features + pkl → predictions.csv
│   ├── train.py              # One-time model training (regenerates pkl)
│   ├── loader.py             # Schema-robust CSV ingestion with alias detection
│   ├── forecaster.py         # Prophet models, 30/60/90d forecasts, model cache
│   ├── budget_sim.py         # Budget simulation with log(spend) diminishing returns
│   ├── anomaly.py            # Rolling z-score anomaly detection
│   └── llm.py                # Google Gemini API integration (4 prompt functions)
│
└── output/                 # Generated at run time (not committed)
    └── predictions.csv
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Google API key (for AI insights)

```bash
export GOOGLE_API_KEY=AIza...
```

---

## How to Run

### Run the scoring pipeline (judges use this)

```bash
./run.sh                                          # uses defaults: ./data  ./pickle/model.pkl  ./output/predictions.csv
./run.sh /path/to/test_data ./pickle/model.pkl ./output/predictions.csv   # custom paths
```

Output: `output/predictions.csv` — 211 rows with columns:
`window_days, level, entity, revenue_lower, revenue_point, revenue_upper, spend_point, roas_lower, roas_point, roas_upper`

Levels produced: `aggregate`, `channel`, `campaign_type`, `campaign`, `trailing_30d`, `anomaly`

---

### Run the full Streamlit dashboard (recommended)

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

- The sidebar auto-loads all three CSV files from `data/`.
- Forecast training runs on first load (~45 seconds). Results are cached for the session.
- Enter your Google API Key in the sidebar and click **Generate Insights** in the AI Insights tab.
- Set per-channel daily budgets in the sidebar and click **Run Budget Simulation**.

---

### Run individual pipeline modules from Python

#### Load and inspect data

```python
from src.loader import load_daily_aggregate, load_daily_by_channel, campaign_summary

daily = load_daily_aggregate()
print(daily.tail())

channel_data = load_daily_by_channel()  # {'google': df, 'meta': df, 'bing': df}

summary = campaign_summary()            # all 136 campaigns ranked by revenue
print(summary.head(10))
```

#### Run aggregate forecast

```python
from src.loader import load_daily_aggregate
from src.forecaster import run_aggregate_forecast, trailing_actuals

daily = load_daily_aggregate()
trailing = trailing_actuals(daily, days=30)

forecast = run_aggregate_forecast(daily, uncertainty_samples=500)

for rev in forecast['revenue_forecasts']:
    print(f"{rev['days']}d: ${rev['lower']:,.0f} – ${rev['upper']:,.0f}  (point: ${rev['point']:,.0f})")
```

#### Run channel-level forecasts

```python
from src.loader import load_daily_by_channel
from src.forecaster import run_channel_forecasts

channel_data = load_daily_by_channel()
ch_forecasts = run_channel_forecasts(channel_data, uncertainty_samples=300)

for ch, result in ch_forecasts.items():
    rev30 = result['revenue_forecasts'][0]
    print(f"{ch}: ${rev30['point']:,.0f}  (CI: ${rev30['lower']:,.0f}–${rev30['upper']:,.0f})")
```

#### Run budget simulation

```python
from src.loader import load_daily_by_channel
from src.budget_sim import simulate_budget

channel_data = load_daily_by_channel()

# Daily budgets per channel in USD
budget = {'google': 3000.0, 'meta': 500.0, 'bing': 200.0}

sim = simulate_budget(channel_data, budget, uncertainty_samples=300)

for days, p in sim['portfolio'].items():
    print(f"{days}d: ${p['revenue_point']:,.0f}  ROAS {p['roas_point']:.2f}x")
```

#### Detect anomalies

```python
from src.loader import load_daily_aggregate, load_daily_by_channel
from src.anomaly import detect_anomalies

daily = load_daily_aggregate()
channel_data = load_daily_by_channel()

anomalies = detect_anomalies(daily, channel_data=channel_data, lookback_days=90)

for a in anomalies:
    print(f"[{a['date']}] {a['severity'].upper()} | {a['channel']} | {a['description']}")
```

#### Generate AI insights with Gemini

```python
import os
os.environ['GOOGLE_API_KEY'] = 'AIza...'   # or set in shell before running

from src.loader import load_daily_aggregate, load_daily_by_channel
from src.forecaster import run_aggregate_forecast, run_channel_forecasts, trailing_actuals
from src.anomaly import detect_anomalies
from src.llm import get_insights

daily = load_daily_aggregate()
channel_data = load_daily_by_channel()
trailing = trailing_actuals(daily, days=30)
forecast = run_aggregate_forecast(daily, uncertainty_samples=500)
ch_forecasts = run_channel_forecasts(channel_data, uncertainty_samples=300)
anomalies = detect_anomalies(daily, channel_data=channel_data)

insights = get_insights(
    forecast_result=forecast,
    trailing=trailing,
    anomaly_list=anomalies,
    channel_results=ch_forecasts,
)

print(insights['forecast_explanation'])
print(insights['risk_flags'])
```

#### Dry run (no API key required — previews prompts only)

```python
from src.llm import get_insights_dry_run
result = get_insights_dry_run(forecast, trailing, anomalies, ch_forecasts)
print(result['note'])
print(result['forecast_explanation_prompt'])
```

#### Override the Gemini model

```bash
export NETEXLIR_MODEL=gemini-2.5-flash-preview-05-20
```

Default is `gemini-2.0-flash`.

---

## Architecture

```
CSV files (Google / Meta / Bing)
        │
        ▼
  src/loader.py
  ├── load_daily_aggregate()        → single daily ds/revenue/spend DataFrame
  ├── load_daily_by_channel()       → {channel: daily_df}
  ├── load_daily_by_campaign_type() → {channel/type: daily_df}
  └── load_daily_by_campaign()      → {channel/campaign: daily_df}
        │
        ▼
  src/forecaster.py
  ├── run_aggregate_forecast()      → 30/60/90d revenue + ROAS with 80% CI
  ├── run_channel_forecasts()       → per-channel forecast dicts
  ├── run_slice_forecast()          → generic Prophet fit for any slice
  └── _MODEL_CACHE                  → in-memory cache (MD5 hash keyed)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  src/budget_sim.py               src/anomaly.py
  simulate_budget()               detect_anomalies()
  marginal_roas_curve()           anomaly_summary()
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              src/llm.py  (Google Gemini)
              ├── explain_forecast()
              ├── explain_budget_simulation()
              ├── interpret_anomalies()
              ├── flag_risks()
              └── get_insights()  ← unified entry point
                       │
                       ▼
                  app.py  (Streamlit UI)
```

---

## Limitations

- **Attribution is as-is** — no cross-channel deduplication. A single sale may appear in both Google and Meta reports.
- **No external signals** — the model does not ingest GA4, Shopify, email, or macro-economic data.
- **Bing excluded from channel forecasts** — too few non-zero revenue days for a stable Prophet fit.
- **Prophet does not extrapolate well beyond the training range** — 90-day forecasts carry significantly more uncertainty than 30-day.
- **Gemini API required** for AI insights — dry-run mode shows prompts without calling the API.
