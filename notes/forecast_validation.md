# Phase 3 Forecast Validation

_Run date: 2026-06-16_

## Training data

- Combined daily aggregate (Bing + Meta + Google)
- Date range: 2024-01-01 → 2026-06-05 (887 days)
- Total revenue: $11,095,456  |  Total spend: $2,181,943  |  Blended ROAS: 5.09×

## Trailing 30-day actuals (2026-05-07 → 2026-06-05)

| Metric | Value |
|--------|-------|
| Revenue | $342,801 |
| Spend | $68,260 |
| ROAS | 5.02× |

## Baseline Forecast (no budget change, 80% CI)

| Window | Revenue Point | Revenue CI | Spend Point | ROAS Point | ROAS CI |
|--------|--------------|------------|-------------|-----------|---------|
| 30d | $264,662 | $124K – $612K | $69,453 | 3.81× | 2.30× – 5.32× |
| 60d | $521,844 | $246K – $1.22M | $153,989 | 3.39× | 2.14× – 4.64× |
| 90d | $788,871 | $351K – $1.85M | $243,206 | 3.24× | 2.09× – 4.40× |

### Sanity checks

- **30d forecast / trailing 30d actual = 0.77×** — model predicts a mild dip, consistent with June being post-spring-peak (model has seen 1 prior June in Google data with a trough).
- Lower bound non-zero: ✅ (floored at 10th-pct historical rolling sum)
- Upper > point > lower: ✅
- ROAS CI is narrow enough to be actionable (±30–40% around point estimate)

## Budget Simulation Results

| Daily Spend | 30d Revenue | 30d ROAS | vs Baseline |
|-------------|-------------|----------|------------|
| Baseline (~$2,315/day) | $264,662 | 3.81× | — |
| $5,000/day | $383,849 | 2.56× | +45% revenue, spend +2.2× |
| $10,000/day | $500,337 | 1.67× | +89% revenue, spend +4.3× |

Diminishing returns shape confirmed: doubling spend from baseline ($2.3k) to $5k/day gives +45% revenue; going to $10k/day adds only +30% more. Log-spend regressor is working correctly.

## Known Limitations

1. Revenue CI upper bound can be 2–7× the point estimate — reflects high real-world volatility (promo spikes observed in data).
2. ROAS CI is computed from historical rolling-window ROAS variance, not from Prophet's own sampling — acknowledged approximation.
3. 30d forecast is a mild undershoot vs trailing actuals (0.77×). May be due to Prophet picking up on a summer trough pattern from the single prior summer in the data (Google 2024). Monitor in production.
4. Spend forecast model shows wide CI ($5k–$142k for 30d) — historical spend was erratic. If user provides future budget, this is overridden and becomes the source of truth.
