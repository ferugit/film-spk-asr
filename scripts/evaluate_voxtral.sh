#!/bin/bash

# Evaluation script for Voxtral models (base and fine-tuned)
# Evaluates on NeuroVoz (Spanish) and TORGO (English) test sets separately

set -e

echo "========================================"
echo "Voxtral Model Evaluation Script"
echo "Test Sets: NeuroVoz + TORGO"
echo "========================================"

# Use GPU 1
export CUDA_VISIBLE_DEVICES=5
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
DATASET_PATH="data/combined_neurovoz_torgo"
OUTPUT_DIR="results/evaluation/voxtral"
FINETUNED_MODELS=(
    #"models/voxtral-finetuned-neurovoz-torgo"
    #"models/voxtral-encoder-finetuned-neurovoz-torgo"
    #"models/voxtral-encoder-lora-finetuned-neurovoz-torgo"
    "models/voxtral-lora-finetuned-neurovoz-torgo"
    "models/voxtral-encoder-lora-finetuned-neurovoz-torgo"
)

EVALUATE_BASE=false
EVALUATE_FINETUNED=true

##############################################################


if [ "$EVALUATE_BASE" = true ]; then
    echo ""
    echo "========================================"
    echo "EVALUATING BASE MODEL"
    echo "========================================"
    echo "Model: ${BASE_MODEL}"
    echo ""

    python -u src/finetune/voxtral/eval.py \
        --model_path "${BASE_MODEL}" \
        --model_name "voxtral_base" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --base_model_id "${BASE_MODEL}"

    python -u src/scripts/generate_eval_summary.py \
        --model_name "voxtral_base" \
        --model_path "${BASE_MODEL}" \
        --neurovoz_tsv "${OUTPUT_DIR}/voxtral_base_neurovoz_test_results.tsv" \
        --torgo_tsv "${OUTPUT_DIR}/voxtral_base_torgo_test_results.tsv" \
        --output_dir "${OUTPUT_DIR}"
fi

echo ""
echo "========================================"
echo "EVALUATING FINE-TUNED MODELS"
echo "========================================"

if [ "$EVALUATE_FINETUNED" = false ]; then
    echo "Skipping fine-tuned model evaluation."
    exit 0
fi

for FINETUNED_MODEL in "${FINETUNED_MODELS[@]}"; do
    echo "Model: ${FINETUNED_MODEL}"
    echo ""

    # Extract from model name:
    # models/voxtral-finetuned-neurovoz-torgo -> finetuned
    # models/voxtral-encoder-finetuned-neurovoz-torgo -> encoder-finetuned
    # models/voxtral-encoder-lora-finetuned-neurovoz-torgo -> encoder-lora-finetuned
    MODEL_BASENAME=$(basename "${FINETUNED_MODEL}")
    MODEL_NAME=${MODEL_BASENAME#voxtral-}
    MODEL_NAME=${MODEL_NAME%-neurovoz-torgo}

    # Check if fine-tuned model exists
    if [ ! -d "${FINETUNED_MODEL}" ]; then
        echo "WARNING: Fine-tuned model not found at ${FINETUNED_MODEL}"
        echo "Skipping fine-tuned model evaluation."
    else
        python -u src/finetune/voxtral/eval.py \
            --model_path "${FINETUNED_MODEL}" \
            --model_name "voxtral_${MODEL_NAME}" \
            --dataset_path "${DATASET_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --base_model_id "${BASE_MODEL}"

        python -u src/scripts/generate_eval_summary.py \
            --model_name "voxtral_${MODEL_NAME}" \
            --model_path "${FINETUNED_MODEL}" \
            --neurovoz_tsv "${OUTPUT_DIR}/voxtral_${MODEL_NAME}_neurovoz_test_results.tsv" \
            --torgo_tsv "${OUTPUT_DIR}/voxtral_${MODEL_NAME}_torgo_test_results.tsv" \
            --output_dir "${OUTPUT_DIR}"
    fi

done

echo ""
echo "========================================"
echo "EVALUATION COMPLETED!"
echo "========================================"
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
