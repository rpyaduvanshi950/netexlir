# Netexlir — AI-Assisted Ecommerce Forecasting

**Live Demo:** https://netexlir.streamlit.app

**GitHub:** https://github.com/rpyaduvanshi950/netexlir

A forecasting tool that turns raw ad-platform CSVs into probabilistic revenue and ROAS forecasts, budget simulations, anomaly detection, and plain-English AI insights — built for the AIgnition 3.0 Hackathon by NetElixir.

---

## The Problem

Digital marketing agencies managing ecommerce ad spend face three questions every week that existing tools answer poorly:

1. **What will revenue look like next month?** Most tools return a single number. That single number is almost always wrong and gives no sense of the range of outcomes.
2. **What happens if we shift the budget?** Most simulators assume linear returns — spend 2× more, earn 2× more. Ad platforms do not work that way.
3. **What is going wrong right now, and why?** Anomalies get buried in dashboards. Plain-English explanations are nowhere.

We built a system that answers all three with honest uncertainty quantification, a realistic budget model, and AI-generated explanations.

---

## What We Built

### Probabilistic Forecasting (not point estimates)

We use Facebook Prophet with a log-transformed revenue target and a log(spend) regressor. Every forecast is a range — an 80% prediction interval generated via 500 Monte Carlo samples. The tool reports lower, point, and upper estimates for 30, 60, and 90-day horizons at four levels: aggregate, channel, campaign type, and individual campaign.

Key engineering decisions:
- **Log-transform target** (`log1p(revenue)`) prevents negative extrapolation and stabilises variance across the data range.
- **Spend regressor** (`log1p(daily_spend)`) encodes diminishing returns directly into the model. When you change the budget in the simulator, the model uses the learned log-linear relationship — not a linear extrapolation.
- **Revenue floor** (`max(10th-pct rolling window, 5% avg-rate × days)`) prevents the lower CI from collapsing to zero on low-revenue channels.
- **ROAS CI** derived from historical rolling-window coefficient of variation, not raw Prophet bounds (which produced 0×–136× intervals on our data).
- **In-memory model cache** keyed by slice label + MD5 data hash — Prophet never retrains on the same data twice in a session.
- **Promo holiday calendar** covers Black Friday, Cyber Monday, Prime Day, Memorial Day, Labor Day, and Christmas week.

### Budget Simulation with Diminishing Returns

The simulator takes per-channel daily budgets, runs them through the trained Prophet model as regressor inputs, and returns projected revenue and ROAS at 30/60/90 days. Because spend enters the model as `log1p(spend)`, the relationship is inherently sub-linear. Increasing the Google budget by 50% produces a smaller-than-50% revenue lift, which matches how ad platforms actually behave at scale.

### Anomaly Detection

Three anomaly types are detected on a rolling 90-day lookback:

- **Revenue spike or drop** — rolling 14-day z-score ≥ 2.5σ
- **ROAS collapse** — ROAS drops more than 2.5σ below the 14-day mean
- **Spend-revenue decoupling** — weekly spend rises ≥ 20% but revenue falls ≥ 10%

Results are ranked by severity and surfaced in a structured table with dates, channels, and magnitudes.

### AI Insights via Google Gemini

Four structured prompt functions call the Gemini API (`gemini-2.0-flash`):

- `explain_forecast()` — interprets the 30/60/90d forecast in business terms
- `explain_budget_simulation()` — explains what a budget shift means for ROAS and efficiency
- `interpret_anomalies()` — gives a likely cause and recommended action for each anomaly
- `flag_risks()` — surfaces 4 prioritised operational risks with mitigations

Each function is independent and wrapped in a try/except — one API failure does not block the others. The model is configurable via the `NETEXLIR_MODEL` environment variable.

### Schema-Robust Data Loader

The loader detects column names from alias lists so the pipeline survives variations in held-out test data. For example, a date column named `TimePeriod`, `date_start`, `segments_date`, `date`, or `Date` all resolve to the same canonical field. Google spend in micros is auto-detected and converted. Missing channels are skipped with a warning rather than crashing the pipeline.

---

## Dataset

| Platform | File | Date Range | Campaigns |
|---|---|---|---|
| Google Ads | `data/google_ads_campaign_stats.csv` | Jan 2024 – Jun 2026 | 84 |
| Meta Ads | `data/meta_ads_campaign_stats.csv` | May 2024 – Jun 2026 | 30 |
| Bing Ads | `data/bing_campaign_stats.csv` | May 2024 – Jun 2026 | 22 |

**Combined:** 887 daily rows · 136 campaigns · $11.09M total revenue · $2.18M total spend

Data assumptions:
- Meta's `conversion` column is attributed revenue in USD, confirmed via cost-per-conversion analysis against known benchmarks.
- Google's `metrics_cost_micros` is divided by 1,000,000 to get USD spend. Auto-detected when the median column value exceeds 10,000.
- No cross-platform attribution deduplication was applied.
- No GA4 or Shopify data was available — all revenue is sourced from ad-platform reports.

---

## Results

### Historical Performance

| Metric | Value |
|---|---|
| Total Revenue | $11,095,456 |
| Total Ad Spend | $2,181,943 |
| Blended ROAS | 5.09× |
| Date Range | Jan 2024 – Jun 2026 |

### Channel Breakdown (All Time)

| Channel | Revenue | Spend | ROAS |
|---|---|---|---|
| Google Ads | $9,266,678 | $1,946,126 | 4.76× |
| Meta Ads | $1,656,751 | $196,387 | 8.44× |
| Bing Ads | $172,028 | $39,430 | 4.36× |

### Trailing 30-Day Actuals

| Metric | Value |
|---|---|
| Revenue | $342,801 |
| Spend | $68,260 |
| ROAS | 5.02× |

### Aggregate Forecast (80% Prediction Interval)

| Window | Lower | Point Estimate | Upper | ROAS (point) |
|---|---|---|---|---|
| 30 days | $130,768 | $242,283 | $442,702 | 4.35× |
| 60 days | $272,153 | $499,420 | $916,049 | 3.79× |
| 90 days | $427,236 | $779,316 | $1,431,021 | 3.67× |

### Channel 30-Day Forecast

| Channel | Lower | Point Estimate | Upper | ROAS (point) |
|---|---|---|---|---|
| Google Ads | $153,109 | $299,256 | $590,922 | 5.96× |
| Meta Ads | $4,537 | $15,298 | $49,866 | 12.33× |
| Bing Ads | — | — | — | insufficient data |

Bing is excluded from channel forecasts — fewer than 90 non-zero revenue days, which is the minimum required for a stable Prophet seasonal decomposition.

### Anomalies Detected (Last 90 Days)

| Severity | Count |
|---|---|
| High | 7 |
| Medium | 11 |
| Total | 18 |

---

## Model Validation

### Walk-Forward Backtest

We ran a 4-point walk-forward validation: train on all data before each cutoff date, forecast 30/60/90 days forward, compare against actuals. All 12 tests were run offline with the same model code used in production.

| Cutoff | Horizon | Actual | Predicted | MAPE | CI Covers Actual |
|---|---|---|---|---|---|
| Apr 2025 | 30d | $424,835 | $346,768 | 18.4% | Yes |
| Apr 2025 | 60d | $840,034 | $563,752 | 32.9% | Yes |
| Apr 2025 | 90d | $965,680 | $842,002 | 12.8% | Yes |
| Jul 2025 | 30d | $148,409 | $338,124 | 127.8% | No |
| Jul 2025 | 60d | $290,782 | $743,538 | 155.7% | No |
| Jul 2025 | 90d | $484,565 | $1,230,752 | 154.0% | No |
| Oct 2025 | 30d | $233,890 | $271,472 | 16.1% | Yes |
| Oct 2025 | 60d | $783,531 | $622,568 | 20.5% | Yes |
| Oct 2025 | 90d | $2,688,006 | $1,039,278 | 61.3% | No |
| Jan 2026 | 30d | $255,198 | $249,703 | 2.2% | Yes |
| Jan 2026 | 60d | $590,724 | $505,634 | 14.4% | Yes |
| Jan 2026 | 90d | $958,142 | $731,210 | 23.7% | Yes |

**Coverage summary:** 30d = 75% · 60d = 75% · 90d = 50%

### Why Coverage Probability Matters More Than MAPE

This tool produces ranges, not single numbers. Evaluating a range forecast by comparing its midpoint to the actual is like grading a weather forecast on whether it rained exactly 40% that day. MAPE measures point accuracy; it tells you nothing about whether the stated interval was honest.

Coverage probability is the right metric: what fraction of actual outcomes fall inside the stated prediction interval? For an 80% CI, the target is 80%.

| Window | Observed Coverage | Target |
|---|---|---|
| 30 days | 75% | 80% |
| 60 days | 75% | 80% |
| 90 days | 50% | 80% |

The 30d and 60d coverage is close to target. The 90d undercoverage has two structural causes:

**Jul 2025 cutoff:** After a strong Apr–May 2025 (≈$418K/month), the model extrapolated growth momentum into summer. In reality, Jul–Sep 2025 dipped to ≈$148–207K/month. Any model trained on fewer than 3 years of data will struggle with mid-year trend reversals that contradict the recent trend direction.

**Oct 2025 cutoff:** Nov–Dec 2025 produced $2.4M out of $2.7M in that 90-day window. With only one historical Q4 season in training data, Prophet correctly identifies the timing of the holiday spike but cannot reliably estimate its magnitude. The CI upper bound reached $1.82M — still below the actual $2.69M.

For agencies, this means: use 30-day forecasts for media budget decisions (coverage ≈ 75%). Treat 90-day forecasts as directional ranges. Flag Q4 separately and widen the planning buffer to account for holiday spike uncertainty.

---

## Architecture

```
CSV files (Google / Meta / Bing)
        │
        ▼
  src/loader.py
  ├── load_daily_aggregate()          → single daily ds/revenue/spend DataFrame
  ├── load_daily_by_channel()         → {channel: daily_df}
  ├── load_daily_by_campaign_type()   → {channel/type: daily_df}
  └── load_daily_by_campaign()        → {channel/campaign: daily_df}
        │
        ▼
  src/forecaster.py
  ├── run_aggregate_forecast()        → 30/60/90d revenue + ROAS with 80% CI
  ├── run_channel_forecasts()         → per-channel forecast dicts
  ├── run_campaign_type_forecasts()   → per-campaign-type forecast dicts
  ├── run_slice_forecast()            → generic Prophet fit for any slice
  └── _MODEL_CACHE                    → in-memory cache (MD5 hash keyed)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
  src/budget_sim.py                   src/anomaly.py
  simulate_budget()                   detect_anomalies()
  marginal_roas_curve()               anomaly_summary()
        │                                      │
        └─────────────────┬────────────────────┘
                          ▼
                 src/llm.py  (Google Gemini API)
                 ├── explain_forecast()
                 ├── explain_budget_simulation()
                 ├── interpret_anomalies()
                 ├── flag_risks()
                 └── get_insights()
                          │
                          ▼
                     app.py  (Streamlit UI — 8 tabs)
```

### Scoring Pipeline (separate from the UI)

```
run.sh
  │
  ├── src/generate_features.py   reads CSVs → parquet feature store
  └── src/predict.py             loads pkl + features → predictions.csv
```

---

## Project Structure

```
netexlir/
├── run.sh                        # Scoring entry point
├── requirements.txt              # Dependencies (Streamlit app)
├── requirements-pipeline.txt     # Additional deps for scoring pipeline
├── .python-version               # Pins Python 3.12
├── packages.txt                  # System packages for Streamlit Cloud
├── app.py                        # Streamlit dashboard (8 tabs)
├── README.md
├── DEMO.md                       # 2-minute presentation script
│
├── data/                         # Sample CSVs (replaced at eval time)
│   ├── bing_campaign_stats.csv
│   ├── google_ads_campaign_stats.csv
│   └── meta_ads_campaign_stats.csv
│
├── pickle/
│   └── model.pkl                 # Pre-trained Prophet models (444 KB)
│
├── src/
│   ├── generate_features.py      # Step 1: CSVs → parquet
│   ├── predict.py                # Step 2: parquet + pkl → predictions.csv
│   ├── train.py                  # One-time: trains and saves pkl
│   ├── loader.py                 # Schema-robust CSV ingestion
│   ├── forecaster.py             # Prophet models and cache
│   ├── budget_sim.py             # Budget simulation
│   ├── anomaly.py                # Anomaly detection
│   └── llm.py                    # Gemini API integration
│
└── output/                       # Generated at run time (not committed)
    └── predictions.csv
```

---

## Setup and Running

### Streamlit Dashboard (recommended for demos)

The live app is at **https://netexlir.streamlit.app**

To run locally:

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=AIza...
streamlit run app.py
```

Opens at http://localhost:8501. Prophet models train on first load (~45 seconds), then results are cached for the session. Set per-channel daily budgets in the sidebar and click Run Budget Simulation to see projections.

### Scoring Pipeline

```bash
pip install -r requirements.txt
pip install -r requirements-pipeline.txt

# Run with defaults (data/ model/ output/)
./run.sh

# Run with custom paths
./run.sh /path/to/test_data ./pickle/model.pkl ./output/predictions.csv
```

Output: `output/predictions.csv` with 211 rows.

Columns: `window_days, level, entity, revenue_lower, revenue_point, revenue_upper, spend_point, roas_lower, roas_point, roas_upper`

Levels: `aggregate`, `channel`, `campaign_type`, `campaign`, `trailing_30d`, `anomaly`

### Retrain the Model

```bash
python3 src/train.py --output ./pickle/model.pkl
```

Trains 5 Prophet models (aggregate revenue, aggregate spend, Bing, Meta, Google channel revenue) and saves them to `pickle/model.pkl` (~444 KB). Set `SEED=42` is applied before training for reproducibility.

### Environment Variables

```bash
export GOOGLE_API_KEY=AIza...          # required for AI Insights tab
export NETEXLIR_MODEL=gemini-2.0-flash # optional: change Gemini model
export DATA_DIR=/path/to/csvs          # optional: override data directory
```

---

## Limitations

- **Attribution is as-is.** No cross-channel deduplication. A single conversion may appear in both Google and Meta reports.
- **No external signals.** The model does not use GA4, Shopify, email, or macroeconomic data. All signals come from the ad-platform CSVs.
- **Bing excluded from channel forecasts.** Fewer than 90 non-zero revenue days — insufficient for stable seasonal decomposition.
- **90-day forecasts carry high uncertainty.** Especially around Q4, where one season of training data is not enough to reliably estimate holiday spike magnitude.
- **Gemini API required for AI Insights.** A dry-run mode is available that previews prompts without making API calls.
