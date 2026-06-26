"""
predict.py — Step 2 of the scoring pipeline.

Loads features from the parquet directory produced by generate_features.py,
fits Prophet models (using hyperparameters from pickle/model.pkl if present),
generates 30/60/90-day probabilistic forecasts, and writes predictions.csv.

Usage:
    python src/predict.py \
        --features features.parquet \
        --model    ./pickle/model.pkl \
        --output   ./output/predictions.csv
"""

import argparse
import pickle
import sys
import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forecaster import (
    run_aggregate_forecast,
    run_slice_forecast,
    trailing_actuals,
    FORECAST_HORIZONS,
)
from src.anomaly import detect_anomalies


def _load_features(features_dir: Path) -> dict:
    """Load all parquet slices from the features directory."""
    data = {}

    agg_path = features_dir / "aggregate.parquet"
    if not agg_path.exists():
        raise FileNotFoundError(f"aggregate.parquet not found in {features_dir}")
    data["aggregate"] = pd.read_parquet(agg_path)

    channels = {}
    for p in sorted(features_dir.glob("channel__*.parquet")):
        ch = p.stem.replace("channel__", "")
        channels[ch] = pd.read_parquet(p)
    data["channels"] = channels

    camp_types = {}
    for p in sorted(features_dir.glob("camptype__*.parquet")):
        label = p.stem.replace("camptype__", "").replace("__", "/")
        camp_types[label] = pd.read_parquet(p)
    data["campaign_types"] = camp_types

    campaigns = {}
    for p in sorted(features_dir.glob("campaign__*.parquet")):
        label = p.stem.replace("campaign__", "").replace("__", "/", 1)
        campaigns[label] = pd.read_parquet(p)
    data["campaigns"] = campaigns

    return data


def _load_model_config(model_path: Path) -> dict:
    """Load pickled model config. Returns defaults if file missing."""
    defaults = {
        "uncertainty_samples": 500,
        "interval_width": 0.80,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
    }
    if not model_path.exists():
        print(f"[predict] model.pkl not found at {model_path} — using defaults")
        return defaults
    with open(model_path, "rb") as f:
        config = pickle.load(f)
    return {**defaults, **config.get("hyperparameters", {})}


def _rows_from_result(result: dict, level: str, entity: str) -> list[dict]:
    rows = []
    for rev, spd, roas in zip(
        result["revenue_forecasts"],
        result["spend_forecasts"],
        result["roas_forecasts"],
    ):
        rows.append({
            "window_days":     rev["days"],
            "level":           level,
            "entity":          entity,
            "revenue_lower":   round(rev["lower"], 2),
            "revenue_point":   round(rev["point"], 2),
            "revenue_upper":   round(rev["upper"], 2),
            "spend_point":     round(spd["point"], 2),
            "roas_lower":      round(roas["lower"], 4),
            "roas_point":      round(roas["point"], 4),
            "roas_upper":      round(roas["upper"], 4),
        })
    return rows


def predict(features_dir: str, model_path: str, output_path: str) -> None:
    features_dir = Path(features_dir)
    model_path   = Path(model_path)
    output_path  = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[predict] Loading features from {features_dir}")
    data   = _load_features(features_dir)
    config = _load_model_config(model_path)

    daily        = data["aggregate"]
    channels     = data["channels"]
    camp_types   = data["campaign_types"]
    campaigns    = data["campaigns"]

    us  = config["uncertainty_samples"]
    iw  = config["interval_width"]

    all_rows = []

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print("[predict] Fitting aggregate forecast…")
    agg_result = run_aggregate_forecast(
        daily, uncertainty_samples=us, interval_width=iw
    )
    all_rows.extend(_rows_from_result(agg_result, "aggregate", "all"))

    trailing = trailing_actuals(daily, days=30)
    all_rows.append({
        "window_days": 0, "level": "trailing_30d", "entity": "all",
        "revenue_lower": round(trailing["revenue"], 2),
        "revenue_point": round(trailing["revenue"], 2),
        "revenue_upper": round(trailing["revenue"], 2),
        "spend_point":   round(trailing["spend"], 2),
        "roas_lower":    round(trailing["roas"], 4),
        "roas_point":    round(trailing["roas"], 4),
        "roas_upper":    round(trailing["roas"], 4),
    })

    # ── Channel ────────────────────────────────────────────────────────────────
    print("[predict] Fitting channel forecasts…")
    for ch, df in channels.items():
        result = run_slice_forecast(
            df, label=f"channel/{ch}",
            uncertainty_samples=min(us, 300), interval_width=iw,
        )
        if result:
            all_rows.extend(_rows_from_result(result, "channel", ch))

    # ── Campaign type ──────────────────────────────────────────────────────────
    print("[predict] Fitting campaign-type forecasts…")
    for label, df in camp_types.items():
        result = run_slice_forecast(
            df, label=f"type/{label}",
            uncertainty_samples=min(us, 300), interval_width=iw,
        )
        if result:
            all_rows.extend(_rows_from_result(result, "campaign_type", label))

    # ── Campaign ───────────────────────────────────────────────────────────────
    print("[predict] Fitting campaign-level forecasts…")
    for label, df in campaigns.items():
        result = run_slice_forecast(
            df, label=f"campaign/{label}",
            uncertainty_samples=min(us, 200), interval_width=iw,
        )
        if result:
            all_rows.extend(_rows_from_result(result, "campaign", label))

    # ── Anomaly summary ────────────────────────────────────────────────────────
    anomalies = detect_anomalies(daily, channel_data=channels)
    for a in anomalies[:20]:
        all_rows.append({
            "window_days":   0,
            "level":         "anomaly",
            "entity":        f"{a['channel']}/{a['type']}",
            "revenue_lower": a["date"],
            "revenue_point": a["severity"],
            "revenue_upper": round(a["magnitude"], 3),
            "spend_point":   0,
            "roas_lower":    0,
            "roas_point":    0,
            "roas_upper":    0,
        })

    # ── Write output ───────────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(output_path, index=False)
    print(f"[predict] Done. {len(df_out)} rows written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--model",    default="./pickle/model.pkl")
    parser.add_argument("--output",   default="./output/predictions.csv")
    args = parser.parse_args()
    predict(args.features, args.model, args.output)
