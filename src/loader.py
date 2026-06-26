"""
Data loader — reads the three ad-platform CSVs and returns a single
daily aggregate DataFrame with columns: ds, revenue, spend, roas.

Schema is auto-detected from column aliases so the pipeline survives
minor column-name variations in held-out test data.
"""

import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path


def _default_data_dir() -> Path:
    """Resolve data directory: DATA_DIR env var → data/ → dataset/ (legacy)."""
    env = os.getenv("DATA_DIR")
    if env:
        return Path(env)
    root = Path(__file__).parent.parent
    if (root / "data").exists():
        return root / "data"
    return root / "dataset"


DATASET_DIR = _default_data_dir()

# ── Schema aliases ─────────────────────────────────────────────────────────────
# Each key is our canonical name; values are possible column names in the CSV.
_DATE_ALIASES    = ["TimePeriod", "date_start", "segments_date", "date", "Date",
                    "day", "Day", "report_date", "time_period", "period"]
_REVENUE_ALIASES = ["Revenue", "revenue", "conversion", "metrics_conversions_value",
                    "conv_value", "ConvValue", "total_revenue", "sales",
                    "total_sales", "Total sales", "conversion_value"]
_SPEND_ALIASES   = ["Spend", "spend", "metrics_cost_micros", "cost", "Cost",
                    "total_spend", "Total spend", "amount_spent"]
_CAMP_ALIASES    = ["CampaignName", "campaign_name", "campaign", "Campaign",
                    "ad_set_name", "adset_name", "campaign_id"]
_TYPE_ALIASES    = ["CampaignType", "campaign_type", "campaign_advertising_channel_type",
                    "type", "Type", "objective", "Objective"]


def _pick(df: pd.DataFrame, aliases: list[str], default=None):
    """Return the first alias that exists as a column, or default."""
    for a in aliases:
        if a in df.columns:
            return a
    return default


def validate_data_dir(data_dir: Path) -> dict:
    """
    Check which CSV files exist and what schemas they have.
    Returns a report dict — does NOT raise; caller decides what to do.
    """
    report = {"data_dir": str(data_dir), "files": {}, "warnings": []}
    expected = {
        "bing":   "bing_campaign_stats.csv",
        "meta":   "meta_ads_campaign_stats.csv",
        "google": "google_ads_campaign_stats.csv",
    }
    for channel, fname in expected.items():
        path = data_dir / fname
        if not path.exists():
            # Try fuzzy match (e.g. bing_ads.csv, bing.csv)
            candidates = list(data_dir.glob(f"{channel}*.csv"))
            if candidates:
                path = candidates[0]
                report["warnings"].append(
                    f"Expected '{fname}' but found '{path.name}' — using it."
                )
            else:
                report["files"][channel] = {"found": False}
                report["warnings"].append(f"Missing: {fname} — {channel} channel will be skipped.")
                continue
        try:
            sample = pd.read_csv(path, index_col=0, nrows=3)
            report["files"][channel] = {
                "found": True,
                "path": str(path),
                "columns": list(sample.columns),
                "date_col":    _pick(sample, _DATE_ALIASES),
                "revenue_col": _pick(sample, _REVENUE_ALIASES),
                "spend_col":   _pick(sample, _SPEND_ALIASES),
            }
            missing = [k for k, v in report["files"][channel].items()
                       if k.endswith("_col") and v is None]
            if missing:
                report["warnings"].append(
                    f"{channel}: could not detect columns for {missing}. "
                    f"Available: {list(sample.columns)}"
                )
        except Exception as e:
            report["files"][channel] = {"found": True, "error": str(e)}
            report["warnings"].append(f"{channel}: failed to read — {e}")
    return report


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV trying index_col=0 first, then no index_col."""
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return pd.read_csv(path)


def _load_bing(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / "bing_campaign_stats.csv"
    if not path.exists():
        candidates = list(data_dir.glob("bing*.csv"))
        if not candidates:
            warnings.warn("bing_campaign_stats.csv not found — skipping Bing channel.")
            return None
        path = candidates[0]

    df = _read_csv_safe(path)
    date_col    = _pick(df, _DATE_ALIASES)
    revenue_col = _pick(df, _REVENUE_ALIASES)
    spend_col   = _pick(df, _SPEND_ALIASES)
    camp_col    = _pick(df, _CAMP_ALIASES,  default="campaign_name")
    type_col    = _pick(df, _TYPE_ALIASES,  default="campaign_type")

    if date_col is None or revenue_col is None or spend_col is None:
        warnings.warn(f"Bing CSV schema unrecognised (cols={list(df.columns)}) — skipping.")
        return None

    df["ds"]      = pd.to_datetime(df[date_col])
    df["revenue"] = pd.to_numeric(df[revenue_col], errors="coerce").clip(lower=0).fillna(0)
    df["spend"]   = pd.to_numeric(df[spend_col],   errors="coerce").clip(lower=0).fillna(0)
    df["campaign_name"] = df[camp_col].astype(str) if camp_col in df.columns else "unknown"
    df["campaign_type"] = df[type_col].astype(str) if type_col in df.columns else "unknown"
    return df[["ds", "revenue", "spend", "campaign_name", "campaign_type"]]


def _load_meta(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / "meta_ads_campaign_stats.csv"
    if not path.exists():
        candidates = list(data_dir.glob("meta*.csv")) + list(data_dir.glob("facebook*.csv"))
        if not candidates:
            warnings.warn("meta_ads_campaign_stats.csv not found — skipping Meta channel.")
            return None
        path = candidates[0]

    df = _read_csv_safe(path)
    date_col    = _pick(df, _DATE_ALIASES)
    revenue_col = _pick(df, _REVENUE_ALIASES)
    spend_col   = _pick(df, _SPEND_ALIASES)
    camp_col    = _pick(df, _CAMP_ALIASES,  default="campaign_name")
    type_col    = _pick(df, _TYPE_ALIASES,  default="campaign_type")

    if date_col is None or revenue_col is None or spend_col is None:
        warnings.warn(f"Meta CSV schema unrecognised (cols={list(df.columns)}) — skipping.")
        return None

    df["ds"]      = pd.to_datetime(df[date_col])
    df["revenue"] = pd.to_numeric(df[revenue_col], errors="coerce").clip(lower=0).fillna(0)
    df["spend"]   = pd.to_numeric(df[spend_col],   errors="coerce").clip(lower=0).fillna(0)
    df["campaign_name"] = df[camp_col].astype(str) if camp_col in df.columns else "unknown"
    if type_col in df.columns:
        df["campaign_type"] = df[type_col].astype(str)
    else:
        df["campaign_type"] = df["campaign_name"].str.split("_").str[0]
    return df[["ds", "revenue", "spend", "campaign_name", "campaign_type"]]


def _load_google(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / "google_ads_campaign_stats.csv"
    if not path.exists():
        candidates = list(data_dir.glob("google*.csv"))
        if not candidates:
            warnings.warn("google_ads_campaign_stats.csv not found — skipping Google channel.")
            return None
        path = candidates[0]

    df = _read_csv_safe(path)
    date_col    = _pick(df, _DATE_ALIASES)
    revenue_col = _pick(df, _REVENUE_ALIASES)
    spend_col   = _pick(df, _SPEND_ALIASES)
    camp_col    = _pick(df, _CAMP_ALIASES,  default="campaign_name")
    type_col    = _pick(df, _TYPE_ALIASES,  default="campaign_type")

    if date_col is None or revenue_col is None or spend_col is None:
        warnings.warn(f"Google CSV schema unrecognised (cols={list(df.columns)}) — skipping.")
        return None

    df["ds"]      = pd.to_datetime(df[date_col])
    rev_raw       = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0)
    # If values look like micros (median > 10_000), divide by 1e6
    spend_raw     = pd.to_numeric(df[spend_col], errors="coerce").fillna(0)
    if spend_raw.median() > 10_000:
        spend_raw = spend_raw / 1_000_000
    df["revenue"] = rev_raw.clip(lower=0)
    df["spend"]   = spend_raw.clip(lower=0)
    df["campaign_name"] = df[camp_col].astype(str) if camp_col in df.columns else "unknown"
    df["campaign_type"] = df[type_col].astype(str) if type_col in df.columns else "unknown"
    return df[["ds", "revenue", "spend", "campaign_name", "campaign_type"]]


def load_all_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """Return all rows from all available platforms with a 'channel' column."""
    d = Path(data_dir) if data_dir else DATASET_DIR
    frames = []
    for ch, loader in [("bing", _load_bing), ("meta", _load_meta), ("google", _load_google)]:
        df = loader(d)
        if df is not None and not df.empty:
            frames.append(df.assign(channel=ch))
    if not frames:
        raise RuntimeError(f"No usable data files found in {d}. "
                           f"Expected: bing_campaign_stats.csv, meta_ads_campaign_stats.csv, "
                           f"google_ads_campaign_stats.csv")
    return pd.concat(frames, ignore_index=True)


def load_daily_aggregate(
    start_date: str | None = None,
    end_date: str | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Return daily aggregate DataFrame. Columns: ds, revenue, spend, roas."""
    raw = load_all_raw(data_dir)
    daily = (
        raw.groupby("ds")[["revenue", "spend"]]
        .sum().reset_index().sort_values("ds")
    )
    if start_date:
        daily = daily[daily["ds"] >= pd.to_datetime(start_date)]
    if end_date:
        daily = daily[daily["ds"] <= pd.to_datetime(end_date)]

    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    daily = daily.rename(columns={"index": "ds"})
    daily["roas"] = daily["revenue"] / daily["spend"].replace(0, np.nan)
    return daily


def _to_daily(sub: pd.DataFrame) -> pd.DataFrame:
    daily = sub.groupby("ds")[["revenue", "spend"]].sum().reset_index().sort_values("ds")
    if daily.empty:
        return daily
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    daily = daily.rename(columns={"index": "ds"})
    daily["roas"] = daily["revenue"] / daily["spend"].replace(0, np.nan)
    return daily


def load_daily_by_channel(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    raw = load_all_raw(data_dir)
    return {ch: _to_daily(raw[raw["channel"] == ch])
            for ch in raw["channel"].unique()}


def load_daily_by_campaign_type(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    raw = load_all_raw(data_dir)
    result = {}
    for (ch, ct), grp in raw.groupby(["channel", "campaign_type"]):
        daily = _to_daily(grp)
        if (daily["revenue"] > 0).sum() >= 60:
            result[f"{ch}/{ct}"] = daily
    return result


def load_daily_by_campaign(
    min_nonzero_days: int = 60,
    data_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    raw = load_all_raw(data_dir)
    result = {}
    for (ch, name), grp in raw.groupby(["channel", "campaign_name"]):
        daily = _to_daily(grp)
        if (daily["revenue"] > 0).sum() >= min_nonzero_days:
            result[f"{ch}/{name}"] = daily
    return result


def campaign_summary(data_dir: Path | None = None) -> pd.DataFrame:
    raw = load_all_raw(data_dir)
    rows = []
    for (ch, name), grp in raw.groupby(["channel", "campaign_name"]):
        ct  = grp["campaign_type"].mode().iloc[0] if not grp.empty else ""
        rev = grp["revenue"].sum()
        spd = grp["spend"].sum()
        active = grp["ds"].nunique()
        nz = (grp.groupby("ds")["revenue"].sum() > 0).sum()
        rows.append({
            "channel": ch, "campaign_name": name, "campaign_type": ct,
            "total_revenue": rev, "total_spend": spd,
            "roas": rev / spd if spd > 0 else 0,
            "active_days": active, "nonzero_revenue_days": nz,
        })
    return pd.DataFrame(rows).sort_values("total_revenue", ascending=False).reset_index(drop=True)
