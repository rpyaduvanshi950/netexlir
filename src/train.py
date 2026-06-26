"""
train.py — Pre-trains Prophet models and saves them to pickle/model.pkl.

Run this once before submitting to produce the required pickle artifact:
    python src/train.py

The pickle stores:
  - hyperparameters used for training
  - the promo-holiday calendar
  - fitted Prophet models for aggregate + channels (used as warm-start cache)
"""

import pickle
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import load_daily_aggregate, load_daily_by_channel
from src.forecaster import (
    _fit_series,
    _PROMO_HOLIDAYS,
    _MODEL_CACHE,
)

HYPERPARAMETERS = {
    "uncertainty_samples": 500,
    "interval_width": 0.80,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
}


def train_and_save(output_path: str = "./pickle/model.pkl") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[train] Loading data…")
    daily        = load_daily_aggregate()
    channel_data = load_daily_by_channel()

    us = HYPERPARAMETERS["uncertainty_samples"]
    iw = HYPERPARAMETERS["interval_width"]

    fitted = {}

    # Aggregate revenue + spend
    print("[train] Fitting aggregate revenue model…")
    m_rev, _ = _fit_series(
        daily, "revenue",
        add_spend_regressor=True, uncertainty_samples=us,
        interval_width=iw, cache_key="aggregate", log_transform=True,
    )
    fitted["aggregate_revenue"] = m_rev

    print("[train] Fitting aggregate spend model…")
    m_spd, _ = _fit_series(
        daily, "spend",
        add_spend_regressor=False, uncertainty_samples=us,
        interval_width=iw, cache_key="aggregate", log_transform=True,
    )
    fitted["aggregate_spend"] = m_spd

    # Channel models
    for ch, ch_daily in channel_data.items():
        if ch_daily.empty or len(ch_daily) < 90 or (ch_daily["revenue"] > 0).sum() < 60:
            print(f"[train] Skipping {ch} — insufficient data")
            continue
        print(f"[train] Fitting {ch} revenue model…")
        m, _ = _fit_series(
            ch_daily, "revenue",
            add_spend_regressor=True, uncertainty_samples=min(us, 300),
            interval_width=iw, cache_key=f"channel/{ch}", log_transform=True,
        )
        fitted[f"channel_{ch}_revenue"] = m

    payload = {
        "hyperparameters": HYPERPARAMETERS,
        "promo_holidays":  _PROMO_HOLIDAYS,
        "fitted_models":   fitted,
        "training_last_date": str(daily["ds"].max().date()),
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / 1_000_000
    print(f"[train] Saved {len(fitted)} models → {output_path} ({size_mb:.1f} MB)")
    print(f"[train] Training last date: {payload['training_last_date']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./pickle/model.pkl")
    args = parser.parse_args()
    train_and_save(args.output)
