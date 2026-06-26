#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

FEATURES_DIR="$(dirname "$OUTPUT_PATH")/features.parquet"

echo "=== Netexlir Forecasting Pipeline ==="
echo "  DATA_DIR    : $DATA_DIR"
echo "  MODEL_PATH  : $MODEL_PATH"
echo "  OUTPUT_PATH : $OUTPUT_PATH"
echo ""

mkdir -p "$(dirname "$OUTPUT_PATH")"

# Install pipeline-only deps (pyarrow, scikit-learn, joblib) if not present
pip3 install -q -r requirements-pipeline.txt 2>/dev/null || true

# Step 1 — Generate features from raw CSVs
echo "[1/2] Generating features…"
python3 src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out      "$FEATURES_DIR"

# Step 2 — Load model config + fit + predict
echo "[2/2] Running predictions…"
python3 src/predict.py \
    --features "$FEATURES_DIR" \
    --model    "$MODEL_PATH" \
    --output   "$OUTPUT_PATH"

echo ""
echo "Done. Predictions written to $OUTPUT_PATH"
