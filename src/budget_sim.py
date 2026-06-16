"""
Budget simulation — given per-channel daily budgets, projects portfolio
revenue and ROAS for 30/60/90-day windows.

Uses channel-level Prophet models (with log-spend regressor) so that
diminishing returns are encoded: doubling spend does not double revenue.
"""

import numpy as np
import pandas as pd
from src.forecaster import run_channel_forecasts, FORECAST_HORIZONS


def simulate_budget(
    channel_data: dict[str, pd.DataFrame],
    budget_by_channel: dict[str, float],
    uncertainty_samples: int = 500,
) -> dict:
    """
    Run channel-level forecasts with user-specified future daily budgets,
    then roll up to a portfolio view.

    Parameters
    ----------
    channel_data : {channel: daily_df} from loader.load_daily_by_channel()
    budget_by_channel : {'google': 3000.0, 'meta': 500.0, 'bing': 200.0}
                        Daily spend budget per channel (USD).
                        Omitted channels use their historical average spend.
    uncertainty_samples : Prophet MC samples (lower = faster)

    Returns
    -------
    dict with:
      channel_results   : {channel: forecast_result}
      portfolio         : {days: {point, lower, upper, roas_point, ...}}
      budget_by_channel : the input budgets (echoed for reference)
      total_daily_budget: sum of all channel budgets
    """
    channel_results = run_channel_forecasts(
        channel_data,
        future_daily_spend_by_channel=budget_by_channel,
        uncertainty_samples=uncertainty_samples,
    )

    portfolio = {}
    for days in FORECAST_HORIZONS:
        rev_point = rev_lower = rev_upper = 0.0
        spend_total = 0.0

        for ch, result in channel_results.items():
            rev = next(r for r in result["revenue_forecasts"] if r["days"] == days)
            spd = next(s for s in result["spend_forecasts"] if s["days"] == days)
            rev_point += rev["point"]
            rev_lower += rev["lower"]
            rev_upper += rev["upper"]
            spend_total += spd["point"]

        # For channels not in channel_results (insufficient data), use historical ROAS × budget
        for ch in budget_by_channel:
            if ch not in channel_results:
                daily = channel_data.get(ch)
                if daily is not None and not daily.empty:
                    hist_roas = daily["revenue"].sum() / max(daily["spend"].sum(), 1)
                    fallback_spend = budget_by_channel[ch] * days
                    fallback_rev = fallback_spend * hist_roas
                    rev_point += fallback_rev
                    rev_lower += fallback_rev * 0.6
                    rev_upper += fallback_rev * 1.4
                    spend_total += fallback_spend

        portfolio[days] = {
            "days": days,
            "revenue_point": rev_point,
            "revenue_lower": rev_lower,
            "revenue_upper": rev_upper,
            "spend": spend_total,
            "roas_point": rev_point / max(spend_total, 1),
            "roas_lower": rev_lower / max(spend_total, 1),
            "roas_upper": rev_upper / max(spend_total, 1),
        }

    return {
        "channel_results": channel_results,
        "portfolio": portfolio,
        "budget_by_channel": budget_by_channel,
        "total_daily_budget": sum(budget_by_channel.values()),
    }


def marginal_roas_curve(
    channel_data: dict[str, pd.DataFrame],
    channel: str,
    budget_range: list[float],
    horizon_days: int = 30,
    uncertainty_samples: int = 200,
) -> pd.DataFrame:
    """
    Compute projected 30-day (or other horizon) revenue for a range of daily
    budgets on a single channel — produces the diminishing-returns curve.

    Returns a DataFrame: daily_budget, revenue_point, roas_point.
    """
    rows = []
    daily = channel_data.get(channel)
    if daily is None or daily.empty:
        return pd.DataFrame()

    for budget in budget_range:
        from src.forecaster import run_slice_forecast
        result = run_slice_forecast(
            daily,
            label=f"marginal/{channel}/{budget}",
            future_daily_spend=float(budget),
            uncertainty_samples=uncertainty_samples,
        )
        if result:
            rev = next(r for r in result["revenue_forecasts"] if r["days"] == horizon_days)
            rows.append({
                "daily_budget": budget,
                "revenue_point": rev["point"],
                "revenue_lower": rev["lower"],
                "revenue_upper": rev["upper"],
                "roas_point": rev["point"] / max(budget * horizon_days, 1),
            })

    return pd.DataFrame(rows)
