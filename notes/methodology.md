# Methodology Proposal — netexlir Forecasting Prototype

_Phase 2 — On paper only. Awaiting approval before building._

---

## (a) Forecasting Model

**Chosen model: Facebook Prophet (additive time-series decomposition)**

**Why Prophet:**
- Handles daily data with multiple seasonality levels (weekly, annual) out of the box — critical given our observed weekly volatility and ~25-month history.
- Robust to missing days and outliers (important: 84.8% zero-revenue rows in Bing; spike weeks in all channels).
- Natively produces forecast intervals without extra work.
- No GPU needed, no heavy dependencies beyond `prophet` (pip-installable).
- Outputs aggregate-period totals easily (sum the daily forecasts over 30/60/90 days).
- Simpler than LSTM / XGBoost given our data volume and prototype scope.

**Fallback (if Prophet overfits):** SARIMA (statsmodels). Will evaluate residuals and switch if needed.

**Input series:**
1. **Aggregate daily revenue** (sum across Bing + Meta + Google, all campaigns) → primary forecast.
2. **Aggregate daily spend** (same sum) → second series for ROAS computation.
3. Channel-level series (Bing / Meta / Google separately) for Phase 4 breakdowns.
4. Campaign-type series in Phase 4.

**Preprocessing:**
- Google: `metrics_cost_micros` ÷ 1,000,000; revenue = `metrics_conversions_value`.
- Meta: revenue = `conversion`; spend = `spend`.
- Bing: revenue = `Revenue`; spend = `Spend`.
- Drop unnamed index columns; parse dates; fill the 7+14 budget nulls with 0 (no budget = inactive).
- Align all to a common date index (2024-05-25 → 2026-06-04). Google pre-overlap days used only for seasonality if Prophet ingests full history.

---

## (b) Uncertainty Ranges — Prediction Intervals

**Chosen method: Prophet's built-in Monte Carlo uncertainty intervals**

Prophet generates uncertainty by sampling from the posterior of trend changepoints via Monte Carlo simulation (`uncertainty_samples=1000` default). This produces a credible-interval band (default 80%; we will expose 50%, 80%, 95% to the user).

**Why not quantile regression or pure bootstrap:**
- Prophet intervals already account for trend uncertainty + seasonality noise.
- Quantile regression would require choosing a model type separately for each quantile.
- Bootstrap would require us to implement residual resampling manually.
- For a prototype, Prophet's built-in approach is defensible and fast.

**Output:** For each 30/60/90-day window, we report:
- `yhat_sum`: point estimate (sum of daily `yhat` over window)
- `yhat_lower_sum` / `yhat_upper_sum`: summed lower/upper bounds (80% interval)
- Optionally 50% and 95% bands.

**ROAS interval:** Computed as `revenue_interval / spend_interval`. We forecast revenue and spend independently, then divide. Note: this may underestimate ROAS uncertainty (covariance between revenue and spend is ignored). We will flag this as a limitation.

---

## (c) Budget → Revenue Mapping (Diminishing Returns)

**Approach: Scaled spend injection as a regressor in Prophet**

1. **Historical spend-to-revenue relationship:** Fit a simple diminishing-returns curve (Hill function or log curve: `revenue = a * log(1 + b * spend)`) on historical daily aggregate data per channel.
2. **Budget simulation:** When the user inputs a future daily budget for each channel, we compute the implied revenue multiplier relative to historical average spend, then pass `future_spend` as an additional regressor in Prophet.
3. **Prophet regressor:** Add `spend_usd` (normalized) as an additional regressor column so the model learns spend→revenue correlation during fitting.

**Diminishing returns handling:**
- Use a log transform of spend as the regressor (`log1p(spend)`). Log-spend is a well-known approximation of diminishing returns without requiring full Michaelis-Menten fitting on sparse data.
- This means doubling spend does NOT double revenue in the simulation.

**Assumption:** Historical spend allocation patterns across campaigns reflect approximately constant media mix. The simulation adjusts aggregate spend only; it does not re-optimize channel allocation.

---

## (d) Seasonality Handling

Prophet handles three seasonality levels:
1. **Weekly seasonality** (`weekly_seasonality=True`): ~25 months of daily data gives ~100 weekly cycles — enough to fit reliably.
2. **Yearly seasonality** (`yearly_seasonality=True`): ~1.5 annual cycles — modest but usable. Yearly seasonality mode set to `'auto'` (Prophet will add it if data length is sufficient).
3. **Custom promo spikes:** The observed mid-May 2026 spike (and similar outliers) will be added as Prophet **holidays/events** if dates can be identified (e.g., Black Friday, Cyber Monday). We will check the spike dates against retail calendar events and add a `holidays` DataFrame.

**Limitations:** Only 1.5 annual cycles limits confidence in year-over-year pattern. Year 2 seasonal estimates will have wider uncertainty.

---

## (e) LLM Integration Plan

**Model:** Claude API (claude-sonnet-4-6 or haiku-4-5 for cost) via `anthropic` Python SDK.

**When LLM is invoked:**
- After each forecast run, a structured JSON summary (point estimates, intervals, ROAS, key anomalies) is assembled in Python.
- The JSON is passed to the LLM as a system + user prompt.
- LLM output is shown in the UI as a plain-English narrative.

**Three prompt types:**

### 1. Forecast Explanation
```
System: You are a digital marketing analyst assistant. You receive structured ad performance forecast data.
User: Here is the 30/60/90-day forecast for [brand]:
  - Predicted revenue: $X (80% CI: $Y–$Z)
  - Predicted ROAS: N× (80% CI: A–B×)
  - vs. prior 30 days actual: ...
Write a plain-English paragraph (4–6 sentences) explaining these forecasts to a non-technical marketing manager.
Focus on trend direction, confidence, and what's driving it.
```

### 2. Anomaly Interpretation
```
System: You are a data analyst. Flag unusual patterns in ad campaign data.
User: The following anomalies were detected in the historical data: [list].
For each anomaly, write 1–2 sentences explaining a likely cause and whether it's a concern.
```

### 3. Risk Flags
```
System: You are a marketing risk analyst.
User: [Forecast summary + current budget levels + trend data].
List 3–5 operational risks (e.g., over-reliance on single channel, declining ROAS trend,
spend near budget ceiling) with a brief mitigation suggestion for each.
```

**Inputs to LLM (structured):** forecast point estimates, confidence intervals, ROAS, trailing actuals, anomaly list, channel breakdown, budget utilization %.

**Outputs:** Plain-English strings displayed in UI. Not used for any computation.

**Cost control:** LLM called once per forecast run (not streaming per token). Use `max_tokens=500` per call. Cache system prompts if making repeated calls.

---

## Assumptions & Limitations

1. **Meta `conversion` = revenue (USD).** If this is a conversion count, all Meta figures are wrong by ~3-5 orders of magnitude.
2. **Attribution double-counting accepted** per brief. Combined revenue may overstate true revenue.
3. **No GA4 or Shopify data** — cannot cross-validate with true sales revenue.
4. **ROAS uncertainty is underestimated** (revenue and spend intervals multiplied/divided independently).
5. **Diminishing returns curve is log-linear** — a simplification. True curve may differ.
6. **Seasonal pattern based on ~1.5 years** — Q4 2024 is the only observed holiday season.
7. **Forecasts are aggregate totals, not daily.** Daily shape within the window is not exposed.
8. **Budget simulation assumes historical channel mix is maintained.** No channel reallocation optimization.
9. **All monetary values in USD** — no currency conversion logic.
10. **Prophet changepoints fitted automatically** — aggressive trend changes (ramp-ups/shutdowns) may require manual changepoint tuning in later phases.

---

## Proposed Stack

| Component | Library |
|-----------|---------|
| Data loading + transforms | `pandas`, `numpy` |
| Forecasting | `prophet` (Meta Prophet) |
| Diminishing-returns curve | `scipy.optimize.curve_fit` |
| LLM integration | `anthropic` Python SDK |
| Prototype UI | `streamlit` |
| Visualization | `plotly` |

**Heavy dependencies to confirm before installing:** `prophet` (requires `pystan` or `cmdstanpy`). Will ask before installing.
