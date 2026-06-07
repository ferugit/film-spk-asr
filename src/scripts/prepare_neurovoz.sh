#!/usr/bin/env bash
# Prepare Neurovoz dataset (TSV + HF dataset) - bash wrapper translated from prepare_neurovoz.py

set -euo pipefail

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_DIR="data/neurovoz_raw"
OUTPUT_DIR="data/neurovoz"
SKIP_TSV=0
SKIP_HF=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dataset_dir PATH] [--output_dir PATH] [--skip_tsv] [--skip_hf]

Options:
  --dataset_dir PATH   Path to Neurovoz dataset (default: data/neurovoz_raw)
  --output_dir PATH    Output directory for prepared files (default: data/neurovoz)
  --skip_tsv           Skip TSV creation
  --skip_hf            Skip HF dataset creation
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_dir)
      DATASET_DIR="$2"; shift 2;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2;;
    --skip_tsv)
      SKIP_TSV=1; shift;;
    --skip_hf)
      SKIP_HF=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

mkdir -p "$OUTPUT_DIR"

if [[ "$SKIP_TSV" -eq 0 ]]; then
  echo "============================================================"
  echo "Creating TSV file for Neurovoz..."
  echo "============================================================"
  "$PYTHON" "$SCRIPT_DIR/neurovoz_create_tsv.py" --dataset_dir "$DATASET_DIR" --output_dir "$OUTPUT_DIR"
fi

if [[ "$SKIP_HF" -eq 0 ]]; then
  echo
  echo "============================================================"
  echo "Creating Hugging Face dataset for Neurovoz..."
  echo "============================================================"
  TSV_FILE="$OUTPUT_DIR/neurovoz.tsv"
  "$PYTHON" "$SCRIPT_DIR/neurovoz_create_hf_dataset.py" --tsv_file "$TSV_FILE" --output_dir "$OUTPUT_DIR" --speaker_aware
fi

echo
echo "============================================================"
echo "✓ Neurovoz preparation complete"
echo "============================================================"
