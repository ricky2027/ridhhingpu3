#!/usr/bin/env bash
#
# run.sh -- build (if needed), ensure test data exists, run the batch processor.
#
# 1. Builds bin/npp_processor when it is missing.
# 2. Generates 200 synthetic test images if data/input/ is empty.
# 3. Runs the full pipeline with verbose intermediate output.
# 4. Points you at the results CSV.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BINARY="bin/npp_processor"
INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
RESULTS_CSV="data/results/execution_log.csv"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "data/results"

# --- 1. Build if necessary -------------------------------------------------
if [[ ! -x "$BINARY" ]]; then
  echo "==> Binary not found; building with make..."
  make
fi

# --- 2. Generate test data if the input directory is empty -----------------
if [[ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]]; then
  echo "==> $INPUT_DIR is empty; generating 200 synthetic test images..."
  if ! python3 -c "import PIL" >/dev/null 2>&1; then
    echo "    Pillow not found; installing (pip install Pillow)..."
    pip install Pillow
  fi
  python3 scripts/generate_test_images.py --count 200 --out "$INPUT_DIR"
fi

# --- 3. Run the batch processor --------------------------------------------
echo "==> Running batch processor (full pipeline, verbose)..."
"./$BINARY" -i "$INPUT_DIR" -o "$OUTPUT_DIR" -p full -v

# --- 4. Summary reminder ---------------------------------------------------
echo ""
echo "==> Done. Outputs are in $OUTPUT_DIR/"
echo "    Timing log: $RESULTS_CSV"
