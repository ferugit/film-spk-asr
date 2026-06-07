#!/usr/bin/env bash
# Prepare TORGO dataset (TSV + HF dataset with train/val/test splits) - bash wrapper

set -euo pipefail

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_DATASET="abnerh/TORGO-database"
CACHE_DIR="data/"
OUTPUT_DIR="data/torgo"
SKIP_TSV=0
SKIP_HF=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--hf_dataset NAME] [--cache_dir PATH] [--output_dir PATH] [--skip_tsv] [--skip_hf]

Options:
  --hf_dataset NAME    Hugging Face dataset name (default: abnerh/TORGO-database)
  --cache_dir PATH     Cache directory for HF dataset (default: data/)
  --output_dir PATH    Output directory for prepared files (default: data/torgo)
  --skip_tsv           Skip TSV creation
  --skip_hf            Skip HF dataset creation with splits
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf_dataset)
      HF_DATASET="$2"; shift 2;;
    --cache_dir)
      CACHE_DIR="$2"; shift 2;;
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
  echo "Creating TSV file from HF dataset for TORGO..."
  echo "============================================================"
  "$PYTHON" "$SCRIPT_DIR/torgo_get_tsv_from_hf.py"
fi

if [[ "$SKIP_HF" -eq 0 ]]; then
  echo
  echo "============================================================"
  echo "Creating Hugging Face dataset with splits for TORGO..."
  echo "============================================================"
  TSV_FILE="$OUTPUT_DIR/torgo.tsv"
  "$PYTHON" "$SCRIPT_DIR/torgo_create_hf_dataset.py" \
    --tsv_file "$TSV_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --hf_dataset "$HF_DATASET" \
    --cache_dir "$CACHE_DIR" \
    --stratify_by speech_type
fi

echo
echo "============================================================"
echo "✓ TORGO preparation complete"
echo "============================================================"
echo
echo "Created files:"
echo "  - $OUTPUT_DIR/torgo.tsv (full dataset metadata)"
echo "  - $OUTPUT_DIR/train.tsv (training split metadata)"
echo "  - $OUTPUT_DIR/validation.tsv (validation split metadata)"
echo "  - $OUTPUT_DIR/test.tsv (test split metadata)"
echo "  - $OUTPUT_DIR/torgo_dataset/ (HF dataset with splits)"
echo
