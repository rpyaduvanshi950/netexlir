"""
Data loader — reads the three ad-platform CSVs and returns a single
daily aggregate DataFrame with columns: ds, revenue, spend, roas.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "dataset"


def _load_bing() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "bing_campaign_stats.csv", index_col=0)
    df["ds"] = pd.to_datetime(df["TimePeriod"])
    df["revenue"] = df["Revenue"].clip(lower=0)
    df["spend"] = df["Spend"].clip(lower=0)
    return df[["ds", "revenue", "spend", "CampaignName", "CampaignType"]].rename(
        columns={"CampaignName": "campaign_name", "CampaignType": "campaign_type"}
    )


def _load_meta() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "meta_ads_campaign_stats.csv", index_col=0)
    df["ds"] = pd.to_datetime(df["date_start"])
    # `conversion` is attributed revenue in USD (confirmed by cost-per-conversion analysis)
    df["revenue"] = df["conversion"].clip(lower=0)
    df["spend"] = df["spend"].clip(lower=0)
    # Infer campaign type from name prefix
    df["campaign_type"] = df["campaign_name"].str.split("_").str[0]
    return df[["ds", "revenue", "spend", "campaign_name", "campaign_type"]]


def _load_google() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "google_ads_campaign_stats.csv", index_col=0)
    df["ds"] = pd.to_datetime(df["segments_date"])
    df["revenue"] = df["metrics_conversions_value"].clip(lower=0)
    # cost_micros → USD
    df["spend"] = (df["metrics_cost_micros"] / 1_000_000).clip(lower=0)
    df = df.rename(
        columns={
            "campaign_name": "campaign_name",
            "campaign_advertising_channel_type": "campaign_type",
        }
    )
    return df[["ds", "revenue", "spend", "campaign_name", "campaign_type"]]


def load_all_raw() -> pd.DataFrame:
    """Return all rows from all three platforms with a 'channel' column."""
    bing = _load_bing().assign(channel="bing")
    meta = _load_meta().assign(channel="meta")
    google = _load_google().assign(channel="google")
    return pd.concat([bing, meta, google], ignore_index=True)


def load_daily_aggregate(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Return a daily aggregate DataFrame (sum across all channels + campaigns).
    Columns: ds, revenue, spend, roas
    """
    raw = load_all_raw()
    daily = (
        raw.groupby("ds")[["revenue", "spend"]]
        .sum()
        .reset_index()
        .sort_values("ds")
    )

    if start_date:
        daily = daily[daily["ds"] >= pd.to_datetime(start_date)]
    if end_date:
        daily = daily[daily["ds"] <= pd.to_datetime(end_date)]

    # Fill any gaps in date range with zeros (no ads running = 0 revenue/spend)
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    daily = daily.rename(columns={"index": "ds"})

    daily["roas"] = daily["revenue"] / daily["spend"].replace(0, np.nan)
    return daily


def _to_daily(sub: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a raw sub-DataFrame to daily revenue/spend/roas."""
    daily = sub.groupby("ds")[["revenue", "spend"]].sum().reset_index().sort_values("ds")
    if daily.empty:
        return daily
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    daily = daily.rename(columns={"index": "ds"})
    daily["roas"] = daily["revenue"] / daily["spend"].replace(0, np.nan)
    return daily


def load_daily_by_channel() -> dict[str, pd.DataFrame]:
    """Return {channel: daily_df} for each platform."""
    raw = load_all_raw()
    return {ch: _to_daily(raw[raw["channel"] == ch]) for ch in ["bing", "meta", "google"]}


def load_daily_by_campaign_type() -> dict[str, pd.DataFrame]:
    """
    Return {channel/type: daily_df} keyed as 'bing/Search', 'google/PERFORMANCE_MAX', etc.
    Only includes slices with >= 60 days of non-zero revenue (enough for Prophet).
    """
    raw = load_all_raw()
    result = {}
    for (ch, ct), grp in raw.groupby(["channel", "campaign_type"]):
        daily = _to_daily(grp)
        nonzero_days = (daily["revenue"] > 0).sum()
        if nonzero_days >= 60:
            result[f"{ch}/{ct}"] = daily
    return result


def load_daily_by_campaign(min_nonzero_days: int = 60) -> dict[str, pd.DataFrame]:
    """
    Return {channel/campaign_name: daily_df}.
    Only includes campaigns with >= min_nonzero_days of non-zero revenue.
    Sparse campaigns are excluded (insufficient data for Prophet).
    """
    raw = load_all_raw()
    result = {}
    for (ch, name), grp in raw.groupby(["channel", "campaign_name"]):
        daily = _to_daily(grp)
        nonzero_days = (daily["revenue"] > 0).sum()
        if nonzero_days >= min_nonzero_days:
            key = f"{ch}/{name}"
            result[key] = daily
    return result


def campaign_summary() -> pd.DataFrame:
    """
    Return a summary table: channel, campaign_name, campaign_type,
    total_revenue, total_spend, roas, active_days, nonzero_revenue_days.
    """
    raw = load_all_raw()
    rows = []
    for (ch, name), grp in raw.groupby(["channel", "campaign_name"]):
        ct = grp["campaign_type"].mode().iloc[0] if not grp.empty else ""
        rev = grp["revenue"].sum()
        spd = grp["spend"].sum()
        active = grp["ds"].nunique()
        nz = (grp.groupby("ds")["revenue"].sum() > 0).sum()
        rows.append({
            "channel": ch,
            "campaign_name": name,
            "campaign_type": ct,
            "total_revenue": rev,
            "total_spend": spd,
            "roas": rev / spd if spd > 0 else 0,
            "active_days": active,
            "nonzero_revenue_days": nz,
        })
    return pd.DataFrame(rows).sort_values("total_revenue", ascending=False).reset_index(drop=True)
