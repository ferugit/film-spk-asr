#!/bin/bash

# Evaluation script for Speaker-Conditioned Voxtral (FiLM + SiAMResNet34)
# Evaluates on NeuroVoz (Spanish) and TORGO (English) test sets separately
# Produces per-sample TSV files + evaluation summary JSON with WER breakdown

set -e

echo "========================================"
echo "Speaker-Conditioned Voxtral Evaluation"
echo "FiLM + SiAMResNet34 x-vectors"
echo "Test Sets: NeuroVoz + TORGO"
echo "========================================"

# Use a single GPU for inference
export CUDA_VISIBLE_DEVICES=0,1,2
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Evaluation will use CPU (very slow!)"
fi

# Configuration
BASE_MODEL="mistralai/Voxtral-Mini-3B-2507"
DATASET_PATH="data/combined_neurovoz_torgo_cv"
OUTPUT_DIR="results/evaluation/spk_cond_voxtral"
CACHE_DIR="./models"

# Models to evaluate
FINETUNED_MODELS=(
    "models/voxtral-spk-cond-neurovoz-torgo-cv"
)

##############################################################

echo ""
echo "========================================"
echo "EVALUATING SPEAKER-CONDITIONED MODELS"
echo "========================================"

for FINETUNED_MODEL in "${FINETUNED_MODELS[@]}"; do
    echo ""
    echo "────────────────────────────────────────"
    echo "Model: ${FINETUNED_MODEL}"
    echo "────────────────────────────────────────"

    # Extract model name from path:
    #   models/voxtral-spk-cond-neurovoz-torgo-cv -> spk-cond-neurovoz-torgo-cv
    MODEL_BASENAME=$(basename "${FINETUNED_MODEL}")
    MODEL_NAME=${MODEL_BASENAME#voxtral-}

    # Check if model exists
    if [ ! -d "${FINETUNED_MODEL}" ]; then
        echo "WARNING: Model not found at ${FINETUNED_MODEL}"
        echo "Skipping. Train the model first with: bash train_spk_cond_voxtral.sh"
        continue
    fi

    # Check for required checkpoint file
    if [ ! -f "${FINETUNED_MODEL}/spk_cond_voxtral.pt" ]; then
        echo "WARNING: spk_cond_voxtral.pt not found in ${FINETUNED_MODEL}"
        echo "Skipping. This doesn't look like a valid spk_cond_voxtral checkpoint."
        continue
    fi

    EVAL_START=$(date +%s)

    # ── Step 1: Inference ────────────────────────────────────────────
    echo ""
    echo "Step 1/2: Running inference..."
    python -u src/finetune/spk_cond_voxtral/eval.py \
        --model_path "${FINETUNED_MODEL}" \
        --model_name "voxtral_${MODEL_NAME}" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --base_model_id "${BASE_MODEL}" \
        --cache_dir "${CACHE_DIR}"

    # ── Step 2: WER computation + summary ────────────────────────────
    echo ""
    echo "Step 2/2: Computing WER and generating summary..."
    python -u src/scripts/generate_eval_summary.py \
        --model_name "voxtral_${MODEL_NAME}" \
        --model_path "${FINETUNED_MODEL}" \
        --neurovoz_tsv "${OUTPUT_DIR}/voxtral_${MODEL_NAME}_neurovoz_test_results.tsv" \
        --torgo_tsv "${OUTPUT_DIR}/voxtral_${MODEL_NAME}_torgo_test_results.tsv" \
        --output_dir "${OUTPUT_DIR}"

    EVAL_END=$(date +%s)
    EVAL_TIME=$((EVAL_END - EVAL_START))

    echo ""
    echo "✓ ${MODEL_NAME} evaluated in $(($EVAL_TIME / 60))m $(($EVAL_TIME % 60))s"

done

echo ""
echo "========================================"
echo "EVALUATION COMPLETED!"
echo "========================================"
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Output files:"
echo "  - *_neurovoz_test_results.tsv    (per-sample predictions)"
echo "  - *_torgo_test_results.tsv       (per-sample predictions)"
echo "  - *_neurovoz_test_results.json   (NeuroVoz WER breakdown)"
echo "  - *_torgo_test_results.json      (TORGO WER breakdown)"
echo "  - *_evaluation_summary.json      (combined summary)"
echo ""
