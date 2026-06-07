#!/bin/bash

# Training script for Speaker-Conditioned Voxtral
# FiLM conditioning of Voxtral encoder with SiAMResNet34 x-vectors
#
# Dataset: combined_neurovoz_torgo_cv (NeuroVoz + TORGO + CommonVoice ES)
# Architecture: Frozen Voxtral encoder + trainable FiLM layers + trainable SiAMResNet34

set -e

echo "========================================"
echo "Speaker-Conditioned Voxtral Training"
echo "FiLM + SiAMResNet34 x-vectors"
echo "Dataset: NeuroVoz + TORGO + CommonVoice"
echo "========================================"

# GPU configuration
export CUDA_VISIBLE_DEVICES=6,7
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader
    echo ""
fi

# Navigate to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo ""

# Run training
START_TIME=$(date +%s)

python -u finetuning.py \
    --config_file config/film_conditioning.yaml

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "Training completed!"
echo "Time: $(($ELAPSED / 3600))h $(($ELAPSED % 3600 / 60))m $(($ELAPSED % 60))s"
echo "Model saved to: ./models/voxtral-spk-cond-neurovoz-torgo-cv"
echo "========================================"
