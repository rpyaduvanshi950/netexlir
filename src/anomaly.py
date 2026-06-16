"""
Anomaly detection on historical daily data.

Detects three classes of anomaly:
  1. Revenue spikes / drops  (z-score on rolling 14-day window)
  2. ROAS collapse           (ROAS drops > 2 std below recent mean)
  3. Spend-revenue decoupling (spend rises but revenue falls week-over-week)

Returns a list of dicts ready to pass to the LLM.
"""

import numpy as np
import pandas as pd


def _rolling_zscore(series: pd.Series, window: int = 14) -> pd.Series:
    roll_mean = series.rolling(window, min_periods=window // 2).mean()
    roll_std = series.rolling(window, min_periods=window // 2).std()
    return (series - roll_mean) / roll_std.replace(0, np.nan)


def detect_anomalies(
    daily: pd.DataFrame,
    channel_data: dict[str, pd.DataFrame] | None = None,
    lookback_days: int = 90,
    z_threshold: float = 2.5,
) -> list[dict]:
    """
    Scan the last `lookback_days` of aggregate (and optionally per-channel) data
    for anomalies. Returns a list of anomaly dicts, most recent first.

    Each dict has:
      date, type, channel, description, magnitude, severity ('low'/'medium'/'high')
    """
    anomalies = []

    # ── Aggregate anomalies ────────────────────────────────────────────────
    agg = daily.copy().sort_values("ds")
    agg["roas"] = agg["revenue"] / agg["spend"].replace(0, np.nan)
    agg["rev_zscore"] = _rolling_zscore(agg["revenue"])
    agg["roas_zscore"] = _rolling_zscore(agg["roas"].fillna(0))

    recent = agg[agg["ds"] >= agg["ds"].max() - pd.Timedelta(days=lookback_days)]

    for _, row in recent.iterrows():
        z = row["rev_zscore"]
        if pd.isna(z) or abs(z) < z_threshold:
            continue
        direction = "spike" if z > 0 else "drop"
        sev = "high" if abs(z) > 4 else "medium"
        anomalies.append({
            "date": row["ds"].date().isoformat(),
            "type": f"revenue_{direction}",
            "channel": "aggregate",
            "description": (
                f"Revenue {direction} of {abs(row['revenue'] - agg['revenue'].rolling(14).mean().loc[row.name]):.0f} USD "
                f"({abs(z):.1f}σ from 14-day rolling mean)."
            ),
            "magnitude": float(abs(z)),
            "severity": sev,
        })

    # ROAS collapse (only flag drops, not spikes — a ROAS spike is usually good)
    for _, row in recent.iterrows():
        z = row["roas_zscore"]
        if pd.isna(z) or z > -z_threshold:
            continue
        sev = "high" if z < -4 else "medium"
        anomalies.append({
            "date": row["ds"].date().isoformat(),
            "type": "roas_collapse",
            "channel": "aggregate",
            "description": (
                f"ROAS dropped to {row['roas']:.2f}x ({abs(z):.1f}σ below 14-day mean). "
                f"Spend was ${row['spend']:.0f}, revenue ${row['revenue']:.0f}."
            ),
            "magnitude": float(abs(z)),
            "severity": sev,
        })

    # Spend-revenue decoupling: week-over-week spend up ≥20% but revenue down ≥10%
    agg["rev_7d"] = agg["revenue"].rolling(7, min_periods=4).sum()
    agg["spd_7d"] = agg["spend"].rolling(7, min_periods=4).sum()
    agg["rev_wow"] = agg["rev_7d"].pct_change(7)
    agg["spd_wow"] = agg["spd_7d"].pct_change(7)
    decouple = agg[
        (agg["spd_wow"] >= 0.20) & (agg["rev_wow"] <= -0.10)
        & (agg["ds"] >= agg["ds"].max() - pd.Timedelta(days=lookback_days))
    ]
    for _, row in decouple.iterrows():
        anomalies.append({
            "date": row["ds"].date().isoformat(),
            "type": "spend_revenue_decoupling",
            "channel": "aggregate",
            "description": (
                f"Spend rose {row['spd_wow']:.0%} WoW but revenue fell {abs(row['rev_wow']):.0%} WoW. "
                f"Potential audience saturation or tracking issue."
            ),
            "magnitude": float(abs(row["rev_wow"])),
            "severity": "high",
        })

    # ── Per-channel anomalies ──────────────────────────────────────────────
    if channel_data:
        for ch, ch_daily in channel_data.items():
            cd = ch_daily.copy().sort_values("ds")
            cd["rev_zscore"] = _rolling_zscore(cd["revenue"])
            recent_ch = cd[cd["ds"] >= cd["ds"].max() - pd.Timedelta(days=lookback_days)]
            for _, row in recent_ch.iterrows():
                z = row["rev_zscore"]
                if pd.isna(z) or abs(z) < z_threshold:
                    continue
                direction = "spike" if z > 0 else "drop"
                sev = "high" if abs(z) > 4 else "medium"
                anomalies.append({
                    "date": row["ds"].date().isoformat(),
                    "type": f"revenue_{direction}",
                    "channel": ch,
                    "description": (
                        f"{ch.title()} revenue {direction} ({abs(z):.1f}σ). "
                        f"Revenue: ${row['revenue']:.0f}, spend: ${row['spend']:.0f}."
                    ),
                    "magnitude": float(abs(z)),
                    "severity": sev,
                })

    # Deduplicate same-date same-type-same-channel, sort by date desc then magnitude
    seen = set()
    unique = []
    for a in sorted(anomalies, key=lambda x: (-x["magnitude"], x["date"]), reverse=False):
        key = (a["date"], a["type"], a["channel"])
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return sorted(unique, key=lambda x: x["date"], reverse=True)


def anomaly_summary(anomalies: list[dict], max_items: int = 8) -> str:
    """Compact text summary of top anomalies for LLM context."""
    if not anomalies:
        return "No significant anomalies detected in the lookback window."
    lines = []
    for a in anomalies[:max_items]:
        lines.append(f"- [{a['date']}] {a['severity'].upper()} | {a['channel']} | {a['description']}")
    return "\n".join(lines)
