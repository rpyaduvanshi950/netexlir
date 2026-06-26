"""
generate_features.py — Step 1 of the scoring pipeline.

Reads raw CSVs from DATA_DIR and writes a features parquet file containing
all daily aggregations needed by predict.py.

Usage:
    python src/generate_features.py --data-dir ./data --out features.parquet
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import (
    load_daily_aggregate,
    load_daily_by_channel,
    load_daily_by_campaign_type,
    load_daily_by_campaign,
    campaign_summary,
)


def generate(data_dir: str, out_path: str) -> None:
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[generate_features] Reading from {data_dir}")

    # Schema validation — warn but continue on partial data
    from src.loader import validate_data_dir
    report = validate_data_dir(data_dir)
    for w in report["warnings"]:
        print(f"  [WARN] {w}")
    for ch, info in report["files"].items():
        if info.get("found"):
            print(f"  {ch}: date={info.get('date_col')}  rev={info.get('revenue_col')}  spend={info.get('spend_col')}")

    daily       = load_daily_aggregate(data_dir=data_dir)
    channel     = load_daily_by_channel(data_dir=data_dir)
    camp_type   = load_daily_by_campaign_type(data_dir=data_dir)
    campaigns   = load_daily_by_campaign(data_dir=data_dir)
    summary     = campaign_summary(data_dir=data_dir)

    print(f"  Aggregate rows : {len(daily)}")
    print(f"  Channels       : {list(channel.keys())}")
    print(f"  Campaign types : {len(camp_type)}")
    print(f"  Campaigns      : {len(campaigns)}")

    # Serialise everything into one parquet via a multi-level index convention.
    # Each slice is stored as a separate parquet inside a directory.
    out_path.mkdir(parents=True, exist_ok=True)

    daily.to_parquet(out_path / "aggregate.parquet", index=False)

    for ch, df in channel.items():
        df.to_parquet(out_path / f"channel__{ch}.parquet", index=False)

    for label, df in camp_type.items():
        safe = label.replace("/", "__")
        df.to_parquet(out_path / f"camptype__{safe}.parquet", index=False)

    for label, df in campaigns.items():
        safe = label.replace("/", "__").replace(" ", "_")[:120]
        df.to_parquet(out_path / f"campaign__{safe}.parquet", index=False)

    summary.to_parquet(out_path / "campaign_summary.parquet", index=False)

    print(f"[generate_features] Written to {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="features.parquet")
    args = parser.parse_args()
    generate(args.data_dir, args.out)
