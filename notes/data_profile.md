# Data Profile — netexlir Forecasting Prototype

_Generated: Phase 1 profiling run_

---

## 1. Files & Shape

| File | Rows | Cols | Date Range | Unique Dates |
|------|------|------|-----------|--------------|
| `dataset/bing_campaign_stats.csv` | 2,873 | 11 | 2024-05-25 → 2026-06-05 | 742 |
| `dataset/meta_ads_campaign_stats.csv` | 3,417 | 13 | 2024-05-23 → 2026-06-05 | 576 |
| `dataset/google_ads_campaign_stats.csv` | 19,272 | 12 | 2024-01-01 → 2026-06-04 | 886 |

**Granularity:** All files are daily × campaign rows (one row per campaign per day).  
**Common date overlap:** 2024-05-25 → 2026-06-04 (~25 months). Google has ~5 months extra history (Jan–May 2024).

---

## 2. Column Dictionary

### Bing (`bing_campaign_stats.csv`)

| Column | Type | Notes |
|--------|------|-------|
| `CampaignId` | int64 | Platform-specific ID |
| `TimePeriod` | date | Daily date |
| `Revenue` | float | **Attributed revenue (USD)** |
| `Spend` | float | Ad spend (USD) |
| `Clicks` | int | |
| `Impressions` | int | |
| `Conversions` | float | Conversion count |
| `CampaignType` | str | Search / PerformanceMax / Shopping / Audience |
| `DailyBudget` | float | Budget cap per day (USD) |
| `CampaignName` | str | Human-readable name |

### Meta (`meta_ads_campaign_stats.csv`)

| Column | Type | Notes |
|--------|------|-------|
| `campaign_id` | int64 | Platform-specific ID |
| `date_start` | date | Daily date |
| `cpc` | float | Cost per click |
| `cpm` | float | Cost per 1000 impressions |
| `ctr` | float | Click-through rate (%) |
| `reach` | float | Unique users reached |
| `spend` | float | Ad spend (USD) |
| `clicks` | float | Click count |
| `impressions` | float | |
| `conversion` | float | **Attributed revenue (USD)** — NOT a count; values up to ~26,500 confirm this |
| `daily_budget` | float | Budget cap (USD); 7 nulls (early rows) |
| `campaign_name` | str | Human-readable name |

**Note:** Meta has NO explicit "revenue" or "conversions_value" column. `conversion` is used as the revenue proxy based on magnitude and pattern (e.g., spend=$85 → conversion=$163–$3,964). This is a critical assumption.

### Google (`google_ads_campaign_stats.csv`)

| Column | Type | Notes |
|--------|------|-------|
| `campaign_id` | int64 | Platform-specific ID |
| `segments_date` | date | Daily date |
| `metrics_clicks` | int | |
| `metrics_conversions` | float | Conversion count (fractional due to model attribution) |
| `metrics_cost_micros` | int | **Spend in micros — divide by 1,000,000 for USD** |
| `metrics_impressions` | int | |
| `metrics_video_views` | int | |
| `metrics_conversions_value` | float | **Attributed revenue (USD)** |
| `campaign_advertising_channel_type` | str | SEARCH / PERFORMANCE_MAX / VIDEO / DEMAND_GEN / SHOPPING / DISPLAY |
| `campaign_budget_amount` | float | Daily budget (USD); 14 nulls |
| `campaign_name` | str | Human-readable name |

---

## 3. Revenue & Spend Summary (totals across full date range)

| Channel | Total Revenue (USD) | Total Spend (USD) | Blended ROAS |
|---------|--------------------|--------------------|--------------|
| Bing | ~172,028 | ~39,430 | ~4.4× |
| Meta | ~1,655,941 | ~196,284 | ~8.4× |
| Google | ~9,265,677 | ~1,945,125 | ~4.8× |
| **Combined** | **~11,093,646** | **~2,180,839** | **~5.1×** |

**Google dominates** — ~83% of total revenue and ~89% of spend. PERFORMANCE_MAX accounts for the vast majority of Google revenue ($5.87M, 63% of Google total).

---

## 4. Campaign Types / Channel Breakdown

### Bing
- Search: $170K revenue, $31.6K spend (dominant, >98% of revenue)
- PerformanceMax: $1.6K revenue, $7.1K spend (low ROAS)
- Shopping: $466 revenue, $26 spend
- Audience: $0 revenue, $693 spend

### Google
| Type | Revenue | Spend | ROAS |
|------|---------|-------|------|
| PERFORMANCE_MAX | $5.87M | $1.28M | 4.6× |
| SEARCH | $2.32M | $379K | 6.1× |
| SHOPPING | $1.05M | $244K | 4.3× |
| DEMAND_GEN | $10.8K | $12.7K | 0.85× |
| VIDEO | $4.7K | $25.6K | 0.18× |
| DISPLAY | $0 | $276 | 0× |

### Meta
- Campaign name patterns: Prospecting_DPA, Prospecting_Brand, Remarketing_DPA, Remarketing_Brand, Generic
- No explicit type column; inferred from name prefix

---

## 5. Join Keys & Cross-Platform Linkage

- **No shared campaign_id** across platforms (each is platform-native).
- **Campaign name** is the only potential cross-platform join key.
  - Bing ∩ Google: **27 campaign names shared** (e.g., `Pmax_NTM_Campaign_01..19`, `Search_TM_Campaign_01..05`)
  - Bing ∩ Meta: **0 shared names**
  - Meta ∩ Google: **0 shared names**
- **Implication:** For aggregate forecasting, we will sum revenue/spend across platforms by date (no join needed). For campaign-level, Bing+Google can be grouped under shared campaign names; Meta is standalone.

---

## 6. Data Quality Notes

| Issue | File | Severity | Action |
|-------|------|----------|--------|
| 84.8% zero-revenue rows | Bing | Medium | Normal for campaign-level daily data; aggregation hides this |
| 31.5% zero-revenue rows | Meta | Low | |
| 34.4% zero-revenue rows | Google | Low | |
| `conversion` column meaning ambiguous | Meta | **High** | Treating as revenue (USD). If it's a count, all Meta revenue figures are wrong |
| `metrics_cost_micros` needs ÷1M | Google | High | Must transform before use |
| 7 null `daily_budget` rows | Meta | Low | Earliest rows; impute or ignore |
| 14 null `campaign_budget_amount` rows | Google | Low | Same — early rows |
| Unnamed index column in all files | All | Low | Drop on load |
| Possible cross-platform double-counting | All | **Medium** | Each platform reports own attributed revenue. At aggregate, this means 1 purchase may be counted 3×. We accept AS-IS per brief instructions ("use attribution as-is") |

---

## 7. Seasonality Signals

Weekly revenue (recent 8 weeks, combined channels) shows **notable volatility** — a +50% swing between consecutive weeks is common. Visible patterns:
- Week of 2026-05-11 is a spike across all channels simultaneously (suggesting a real-world event or promotion).
- Week of 2026-04-27 is a trough across all channels.
- Suggests **strong promo-driven spikes** rather than smooth seasonal patterns.
- With 25 months of daily data, there is enough history for ~1.5 full annual seasonal cycles.

---

## 8. What's Missing vs. Brief

- **No GA4 data** — brief mentions GA4 source/medium data; absent from the dataset folder.
- **No Shopify data** — brief mentions Shopify revenue; absent.
- **No impression-share or auction-insight data.**
- **No creative/ad-level data** — only campaign level.

**Consequence:** All revenue comes from ad-platform-reported attributed values. We cannot reconcile against Shopify ground truth. The blended ROAS is computed from platform-reported numbers only.
