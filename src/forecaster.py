"""
Core forecasting module — fits Prophet on aggregate revenue and spend,
returns probabilistic 30/60/90-day forecasts with prediction intervals.

Model cache: trained Prophet models are stored in _MODEL_CACHE keyed by
(slice_name, data_hash) so repeated UI renders don't retrain from scratch.
Training happens once per unique dataset fingerprint, then results are reused.
"""

import hashlib
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

# ── In-memory model cache ─────────────────────────────────────────────────────
# Key: (slice_label, y_col, use_spend_regressor) → (data_hash, fitted_Prophet, forecast_df)
_MODEL_CACHE: dict = {}


def _data_hash(daily: pd.DataFrame) -> str:
    """Fingerprint a DataFrame so cache is invalidated when data changes."""
    return hashlib.md5(
        pd.util.hash_pandas_object(daily[["ds", "revenue", "spend"]], index=False)
        .values.tobytes()
    ).hexdigest()[:12]


def clear_cache() -> None:
    """Evict all cached models (call after loading fresh data)."""
    _MODEL_CACHE.clear()

# Known retail/promo calendar events to dampen Prophet's sensitivity
# to spikes on these dates (so it doesn't extrapolate them as baseline)
_PROMO_HOLIDAYS = pd.DataFrame(
    {
        "holiday": [
            "black_friday", "cyber_monday",
            "black_friday", "cyber_monday",
            "valentines_day", "valentines_day",
            "mothers_day", "mothers_day",
            "prime_day", "prime_day",
        ],
        "ds": pd.to_datetime([
            "2024-11-29", "2024-12-02",
            "2025-11-28", "2025-12-01",
            "2025-02-14", "2026-02-14",
            "2025-05-11", "2026-05-10",
            "2025-07-08", "2024-07-16",  # approximate Prime Day dates
        ]),
        "lower_window": [-2, -1, -2, -1, -2, -2, -2, -2, -2, -2],
        "upper_window": [1, 2, 1, 2, 2, 2, 1, 1, 2, 2],
    }
)

FORECAST_HORIZONS = [30, 60, 90]


def _build_prophet(
    uncertainty_samples: int = 1000,
    interval_width: float = 0.80,
    add_spend_regressor: bool = False,
) -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=_PROMO_HOLIDAYS,
        interval_width=interval_width,
        uncertainty_samples=uncertainty_samples,
        changepoint_prior_scale=0.05,   # conservative — avoids wild extrapolation
        seasonality_prior_scale=10.0,
    )
    if add_spend_regressor:
        m.add_regressor("log_spend", standardize=True)
    return m


def _fit_series(
    daily: pd.DataFrame,
    y_col: str,
    add_spend_regressor: bool = False,
    uncertainty_samples: int = 1000,
    interval_width: float = 0.80,
    cache_key: str = "",
    log_transform: bool = True,
) -> tuple[Prophet, bool]:
    """
    Fit a Prophet model, using in-memory cache to avoid redundant retraining.
    log_transform=True fits on log1p(y) to prevent negative extrapolation.
    Returns (model, log_transform_used).
    """
    dhash = _data_hash(daily)
    ck = (cache_key, y_col, add_spend_regressor, log_transform)
    cached = _MODEL_CACHE.get(ck)
    if cached and cached[0] == dhash:
        return cached[1], log_transform

    df = daily[["ds", y_col]].copy().rename(columns={y_col: "y"})
    df["y"] = df["y"].clip(lower=0)
    if log_transform:
        df["y"] = np.log1p(df["y"])
    if add_spend_regressor:
        df["log_spend"] = np.log1p(daily["spend"].values)

    m = _build_prophet(
        uncertainty_samples=uncertainty_samples,
        interval_width=interval_width,
        add_spend_regressor=add_spend_regressor,
    )
    m.fit(df)
    _MODEL_CACHE[ck] = (dhash, m)
    return m, log_transform


def _future_df(
    m: Prophet,
    horizon_days: int,
    future_daily_spend: float | None = None,
    historical_avg_spend: float | None = None,
) -> pd.DataFrame:
    """Build future DataFrame for Prophet prediction."""
    future = m.make_future_dataframe(periods=horizon_days, freq="D")
    if "log_spend" in m.extra_regressors:
        # Use provided future spend or fall back to historical average
        spend_val = future_daily_spend if future_daily_spend is not None else (historical_avg_spend or 0)
        n_hist = len(future) - horizon_days
        future_spend_val = future_daily_spend if future_daily_spend is not None else historical_avg_spend
        hist_spend_val = historical_avg_spend if historical_avg_spend is not None else 0.0
        spend_values = np.array(
            [future_spend_val if i >= n_hist else hist_spend_val
             for i in range(len(future))],
            dtype=float,
        )
        future["log_spend"] = np.log1p(spend_values)
    return future


def _aggregate_window(
    forecast: pd.DataFrame,
    last_train_date: pd.Timestamp,
    days: int,
    revenue_floor: float = 0.0,
    log_transform: bool = False,
) -> dict:
    """Sum forecast values over a future window of `days` days."""
    window = forecast[forecast["ds"] > last_train_date].head(days).copy()
    if log_transform:
        # back-transform daily log-space predictions, then sum
        point = np.expm1(window["yhat"].clip(lower=0)).sum()
        lower = max(np.expm1(window["yhat_lower"].clip(lower=0)).sum(), revenue_floor)
        upper = np.expm1(window["yhat_upper"].clip(lower=0)).sum()
    else:
        point = window["yhat"].clip(lower=0).sum()
        lower = max(window["yhat_lower"].clip(lower=0).sum(), revenue_floor)
        upper = window["yhat_upper"].clip(lower=0).sum()
    upper = max(upper, point, lower)  # upper must be >= both point and lower
    return {
        "days": days,
        "point": point,
        "lower": lower,
        "upper": upper,
    }


def run_aggregate_forecast(
    daily: pd.DataFrame,
    future_daily_spend: float | None = None,
    uncertainty_samples: int = 1000,
    interval_width: float = 0.80,
    use_spend_regressor: bool = True,
) -> dict:
    """
    Fit Prophet on aggregate revenue and spend, return 30/60/90-day
    probabilistic forecasts and ROAS intervals.

    Parameters
    ----------
    daily : output of loader.load_daily_aggregate()
    future_daily_spend : if provided, used as the assumed daily spend
                         for the future windows (for budget simulation)
    uncertainty_samples : Monte Carlo samples for intervals
    interval_width : CI width (0.80 = 80%)
    use_spend_regressor : whether to include log(spend) as a regressor

    Returns
    -------
    dict with keys: revenue_forecasts, spend_forecasts, roas_forecasts,
                    last_date, model_revenue, model_spend, forecast_rev_df, forecast_spend_df
    """
    hist_avg_spend = daily["spend"].mean()
    hist_avg_daily_rev = daily["revenue"].mean()

    # Revenue floor: max(10th-pct of rolling window sums, 5% of avg-rate projection)
    # prevents CI lower bound from collapsing to 0 on volatile-but-nonzero series
    rev_floors = {
        d: max(
            np.percentile(daily["revenue"].rolling(d, min_periods=d // 2).sum().dropna(), 10),
            hist_avg_daily_rev * d * 0.05,
        )
        for d in FORECAST_HORIZONS
    }

    # --- Fit revenue model ---
    m_rev, log_rev = _fit_series(
        daily, "revenue",
        add_spend_regressor=use_spend_regressor,
        uncertainty_samples=uncertainty_samples,
        interval_width=interval_width,
        cache_key="aggregate",
        log_transform=True,
    )
    future_rev = _future_df(m_rev, max(FORECAST_HORIZONS), future_daily_spend, hist_avg_spend)
    fc_rev = m_rev.predict(future_rev)

    # --- Fit spend model (for projecting baseline spend) ---
    m_spend, log_spd = _fit_series(
        daily, "spend",
        add_spend_regressor=False,
        uncertainty_samples=uncertainty_samples,
        interval_width=interval_width,
        cache_key="aggregate",
        log_transform=True,
    )
    future_spend = m_spend.make_future_dataframe(periods=max(FORECAST_HORIZONS), freq="D")
    fc_spend = m_spend.predict(future_spend)

    last_date = daily["ds"].max()

    revenue_forecasts, spend_forecasts, roas_forecasts = [], [], []
    for days in FORECAST_HORIZONS:
        rev = _aggregate_window(fc_rev, last_date, days, revenue_floor=rev_floors[days], log_transform=log_rev)
        spd = _aggregate_window(fc_spend, last_date, days, log_transform=log_spd)

        # If user provided a future budget, override spend projection
        if future_daily_spend is not None:
            total_future_spend = future_daily_spend * days
            spd["point"] = total_future_spend
            spd["lower"] = total_future_spend
            spd["upper"] = total_future_spend

        # ROAS CI: use point estimates for point, and bound by ±historical ROAS CV
        # rather than the extreme low/high combination which produces 0x–∞x ranges
        hist_roas = daily["revenue"].sum() / max(daily["spend"].sum(), 1)
        # historical ROAS coefficient of variation on rolling windows
        rolling_roas = (
            daily["revenue"].rolling(days, min_periods=days // 2).sum()
            / daily["spend"].rolling(days, min_periods=days // 2).sum().replace(0, np.nan)
        ).dropna()
        roas_cv = rolling_roas.std() / max(rolling_roas.mean(), 1e-6)

        roas_point = rev["point"] / max(spd["point"], 1)
        roas_range = roas_point * roas_cv
        roas = {
            "days": days,
            "point": roas_point,
            "lower": max(roas_point - roas_range, 0.1),
            "upper": roas_point + roas_range,
        }

        revenue_forecasts.append(rev)
        spend_forecasts.append(spd)
        roas_forecasts.append(roas)

    return {
        "revenue_forecasts": revenue_forecasts,
        "spend_forecasts": spend_forecasts,
        "roas_forecasts": roas_forecasts,
        "last_date": last_date,
        "model_revenue": m_rev,
        "model_spend": m_spend,
        "forecast_rev_df": fc_rev,
        "forecast_spend_df": fc_spend,
        "historical_avg_daily_spend": hist_avg_spend,
        "interval_width": interval_width,
        "revenue_floors": rev_floors,
    }


def run_slice_forecast(
    daily: pd.DataFrame,
    label: str,
    future_daily_spend: float | None = None,
    uncertainty_samples: int = 500,
    interval_width: float = 0.80,
) -> dict:
    """
    Same logic as run_aggregate_forecast but for a named slice
    (channel, campaign type, or campaign). Uses label as cache key.
    Returns None if the slice has insufficient data.
    """
    if daily.empty or len(daily) < 90 or (daily["revenue"] > 0).sum() < 60:
        return None

    hist_avg_spend = daily["spend"].mean()
    hist_avg_daily_rev = daily["revenue"].mean()
    rev_floors = {
        d: max(
            np.percentile(daily["revenue"].rolling(d, min_periods=d // 2).sum().dropna(), 10),
            hist_avg_daily_rev * d * 0.05,  # never below 5% of average-rate projection
        )
        for d in FORECAST_HORIZONS
    }

    m_rev, log_rev = _fit_series(
        daily, "revenue",
        add_spend_regressor=True,
        uncertainty_samples=uncertainty_samples,
        interval_width=interval_width,
        cache_key=label,
        log_transform=True,
    )
    future_rev = _future_df(m_rev, max(FORECAST_HORIZONS), future_daily_spend, hist_avg_spend)
    fc_rev = m_rev.predict(future_rev)

    m_spend, log_spd = _fit_series(
        daily, "spend",
        add_spend_regressor=False,
        uncertainty_samples=uncertainty_samples,
        interval_width=interval_width,
        cache_key=label,
        log_transform=True,
    )
    fc_spend = m_spend.predict(
        m_spend.make_future_dataframe(periods=max(FORECAST_HORIZONS), freq="D")
    )

    last_date = daily["ds"].max()
    revenue_forecasts, spend_forecasts, roas_forecasts = [], [], []

    for days in FORECAST_HORIZONS:
        rev = _aggregate_window(fc_rev, last_date, days, revenue_floor=rev_floors[days], log_transform=log_rev)
        spd = _aggregate_window(fc_spend, last_date, days, log_transform=log_spd)

        if future_daily_spend is not None:
            total = future_daily_spend * days
            spd["point"] = spd["lower"] = spd["upper"] = total

        rolling_roas = (
            daily["revenue"].rolling(days, min_periods=days // 2).sum()
            / daily["spend"].rolling(days, min_periods=days // 2).sum().replace(0, np.nan)
        ).dropna()
        roas_cv = rolling_roas.std() / max(rolling_roas.mean(), 1e-6) if len(rolling_roas) > 1 else 0.3

        roas_point = rev["point"] / max(spd["point"], 1)
        roas_range = roas_point * roas_cv
        roas_forecasts.append({
            "days": days,
            "point": roas_point,
            "lower": max(roas_point - roas_range, 0.1),
            "upper": roas_point + roas_range,
        })
        revenue_forecasts.append(rev)
        spend_forecasts.append(spd)

    return {
        "label": label,
        "revenue_forecasts": revenue_forecasts,
        "spend_forecasts": spend_forecasts,
        "roas_forecasts": roas_forecasts,
        "last_date": last_date,
        "historical_avg_daily_spend": hist_avg_spend,
        "interval_width": interval_width,
    }


def run_channel_forecasts(
    channel_data: dict[str, pd.DataFrame],
    future_daily_spend_by_channel: dict[str, float] | None = None,
    uncertainty_samples: int = 500,
) -> dict[str, dict]:
    """
    Run forecasts for each channel slice.
    channel_data: output of loader.load_daily_by_channel()
    future_daily_spend_by_channel: {'google': 3000.0, 'meta': 500.0, ...}
    Returns {channel: forecast_result}.
    """
    results = {}
    for ch, daily in channel_data.items():
        future_spend = (
            future_daily_spend_by_channel.get(ch) if future_daily_spend_by_channel else None
        )
        r = run_slice_forecast(daily, label=f"channel/{ch}",
                               future_daily_spend=future_spend,
                               uncertainty_samples=uncertainty_samples)
        if r:
            results[ch] = r
    return results


def run_campaign_type_forecasts(
    type_data: dict[str, pd.DataFrame],
    uncertainty_samples: int = 500,
) -> dict[str, dict]:
    """
    Run forecasts for each channel/campaign-type slice.
    type_data: output of loader.load_daily_by_campaign_type()
    Returns {label: forecast_result}.
    """
    results = {}
    for label, daily in type_data.items():
        r = run_slice_forecast(daily, label=f"type/{label}",
                               uncertainty_samples=uncertainty_samples)
        if r:
            results[label] = r
    return results


def run_campaign_forecasts(
    campaign_data: dict[str, pd.DataFrame],
    uncertainty_samples: int = 300,
) -> dict[str, dict]:
    """
    Run forecasts for individual campaigns that have sufficient data.
    campaign_data: output of loader.load_daily_by_campaign()
    Returns {label: forecast_result}.
    """
    results = {}
    for label, daily in campaign_data.items():
        r = run_slice_forecast(daily, label=f"campaign/{label}",
                               uncertainty_samples=uncertainty_samples)
        if r:
            results[label] = r
    return results


def trailing_actuals(daily: pd.DataFrame, days: int = 30) -> dict:
    """Return trailing actual revenue/spend/ROAS for the last `days` days."""
    window = daily.tail(days)
    rev = window["revenue"].sum()
    spd = window["spend"].sum()
    return {
        "days": days,
        "revenue": rev,
        "spend": spd,
        "roas": rev / spd if spd > 0 else 0,
        "start": window["ds"].min(),
        "end": window["ds"].max(),
    }
