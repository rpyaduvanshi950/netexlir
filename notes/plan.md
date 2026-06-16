# Project Plan — netexlir Forecasting Prototype

## Phase Checklist

- [x] **Phase 0** — Token discipline rules in place
- [x] **Phase 1** — Data profile written → `notes/data_profile.md`
- [x] **Phase 2** — Methodology proposal → `notes/methodology.md` _(awaiting approval)_
- [x] **Phase 3** — Forecasting core: aggregate revenue + ROAS, 30/60/90-day ranges → `src/loader.py`, `src/forecaster.py`, `notes/forecast_validation.md`
- [x] **Phase 4** — Breadth: channel / campaign-type / campaign-level + budget simulation → `src/loader.py` (extended), `src/forecaster.py` (slice/channel/type functions + model cache), `src/budget_sim.py`
- [x] **Phase 5** — LLM layer: explanations, anomaly flags, risk narratives → `src/anomaly.py`, `src/llm.py` (get_insights + 4 prompt functions)
- [x] **Phase 6** — Prototype UI + technical docs → `app.py`

## Key Decisions / Assumptions (log)

1. Meta `conversion` column treated as **revenue in USD** (not a count).
2. Google `metrics_cost_micros` ÷ 1,000,000 = spend in USD.
3. Attribution used AS-IS — no cross-platform deduplication.
4. No GA4 or Shopify data present; all revenue from ad-platform reports only.
5. Forecasts are aggregate-period totals (not daily), per brief.
6. Bing+Google share 27 campaign names → usable for cross-platform campaign grouping.
7. Revenue in overlap period (May 2024–Jun 2026) used as primary training window;
   Google's extra 5 months (Jan–May 2024) used for better seasonality estimation.
