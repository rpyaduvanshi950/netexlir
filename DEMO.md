# Demo Walkthrough — 2 Minutes

## Setup (before presenting)

```bash
streamlit run app.py          # opens http://localhost:8501
```

In the sidebar:
- Paste your `GOOGLE_API_KEY`
- Set budgets: Google $4,000 · Meta $800 · Bing $200

---

## Step 1 · 0:00–0:20 · Tab: 📊 Forecast

**Say:** "We're solving the hardest question a digital marketing agency faces: *what will revenue look like next month?* Most tools give you a single number — ours gives you a range backed by 500 Monte Carlo simulations. Here's the 30-day forecast: **$131K–$445K** at 80% confidence. The point estimate is $242K, but the range is the honest answer."

**Do:** Hover over the 90-day revenue bar to show the wide CI.

---

## Step 2 · 0:20–0:35 · Tab: 📊 Forecast (scroll down)

**Say:** "You'll notice the forecast looks lower than last month's actual. There's an explanation right here in the UI — the trailing 30 days captured a May peak. June is seasonally 70% weaker based on 2025 data. The model isn't predicting a problem; it's predicting seasonality correctly."

**Do:** Point at the monthly revenue bar chart. Highlight the purple Nov–Dec holiday spikes (53% of annual revenue).

---

## Step 3 · 0:35–0:50 · Tab: 📡 Channels & Types

**Say:** "Google drives 95% of forecast revenue. Meta is smaller but 12× ROAS — extremely efficient at current scale. The Campaign Type subtab breaks it further: Performance Max accounts for the bulk of Google revenue."

**Do:** Click "Campaign Type" subtab. Show the bar chart.

---

## Step 4 · 0:50–1:10 · Tab: 💰 Budget Sim

**Say:** "Now the most valuable feature for agencies. I've set a $5,000 daily budget — about 10% above current spend. Click Run Budget Simulation and watch what happens."

**Do:** Click **Run Budget Simulation**. Point at the "vs Baseline" column.

**Say:** "The uplift is positive but not proportional to the budget increase. That's because we model spend with a log transform — doubling spend does not double revenue. Most tools assume linearity. Ours doesn't."

---

## Step 5 · 1:10–1:25 · Tab: 🤖 AI Insights

**Say:** "Click Generate Insights. Gemini makes 3–4 structured API calls — not 'summarise this data', but specific analytical questions about the forecast, each anomaly, and operational risks."

**Do:** Click **Generate Insights**. While it loads: "We detected 18 anomalies in the last 90 days — revenue spikes, ROAS collapses, and spend-revenue decoupling events. Gemini interprets each one."

---

## Step 6 · 1:25–1:45 · Tab: ✅ Validation

**Say:** "We measured ourselves. 4-point walk-forward backtest. 30-day coverage probability is 75% — close to our stated 80% CI. We're transparent about the 90-day undercoverage: with only one Q4 season in training data, no model reliably estimates the November–December holiday spike magnitude."

**Do:** Point at the coverage probability chart and the dashed 80% target line.

---

## Step 7 · 1:45–2:00 · Tab: 📋 Submission

**Say:** "The pipeline is fully submission-ready. Clone, install, drop in test data, run './run.sh'. One command. The output is a structured CSV with aggregate, channel, campaign-type, and campaign-level probabilistic forecasts."

**Do:** Point at the green checklist. End.

---

## Key Numbers

| Metric | Value |
|---|---|
| All-time revenue | $11.09M |
| Blended ROAS | 5.09× |
| 30d forecast range | $131K – $445K (80% CI) |
| 30d coverage probability | 75% (target 80%) |
| Anomalies detected | 18 (last 90 days) |
| LLM calls per run | 3–4 Gemini |
| Scoring command | `./run.sh` |

---

## Likely Judge Questions

**"Why is the forecast lower than trailing actuals?"**
Trailing 30d captured the May peak. June is seasonally 70% lower (2025: May $418K → June $127K). Prophet correctly predicts the dip.

**"How did you handle diminishing returns?"**
`log₁p(daily_spend)` is added as a regressor inside Prophet. The model learns a sub-linear spend→revenue relationship from historical data.

**"What's the model accuracy?"**
30d coverage probability 75% in walk-forward backtest. Coverage probability is the right metric for probabilistic range forecasts — not MAPE, which only measures point accuracy.

**"Why Prophet?"**
Handles weekly + yearly seasonality and promo holidays natively. Produces Monte Carlo prediction intervals out of the box. Supports additional regressors (spend). 887 training rows — Prophet works well with < 1,000 observations; tree-based models don't.

**"Can it handle new/different data?"**
Yes. Column names are auto-detected from aliases. Missing channels are skipped with warnings, not crashes.
