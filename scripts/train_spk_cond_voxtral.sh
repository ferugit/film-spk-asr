#!/bin/bash

# Training script for Speaker-Conditioned Voxtral (FiLM + SiAMResNet34)
# Conditions Voxtral audio encoder on pathological speaker embeddings via FiLM layers
#
# Architecture:
#   - Voxtral encoder (frozen) + FiLM layers (trainable) + SiAMResNet34 x-vectors (trainable)
#   - Pathological speech (DYSARTHRIC/PARKINSON) → FiLM conditioning
#   - Normative speech (HC/Unknown) → identity (zero x-vector)
#
# Dataset: combined_neurovoz_torgo_cv (NeuroVoz + TORGO + CommonVoice ES)

set -e

echo "========================================"
echo "Speaker-Conditioned Voxtral Training"
echo "FiLM + SiAMResNet34 x-vectors"
echo "Dataset: NeuroVoz + TORGO + CommonVoice"
echo "========================================"

# GPU configuration
#export CUDA_VISIBLE_DEVICES=0,1,2
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Training will use CPU (very slow!)"
fi

# Set environment variables for better training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# Configuration
CONFIG_FILE="src/finetune/spk_cond_voxtral/config/film_conditioning_lang_balanced_mn5.yaml"

# Verify prerequisites
echo "Verifying prerequisites..."

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -d "data/combined_neurovoz_torgo_cv" ]; then
    echo "ERROR: Dataset not found at data/combined_neurovoz_torgo_cv"
    exit 1
fi

if [ ! -f "models/SiAMResNet34/samresnet34_w_features.jit" ]; then
    echo "ERROR: SiAMResNet34 model not found at models/SiAMResNet34/samresnet34_w_features.jit"
    exit 1
fi

echo "✓ All prerequisites found"
echo ""

echo "========================================"
echo "Training: FiLM Conditioning (per-layer)"
echo "========================================"
echo "  Config:        ${CONFIG_FILE}"
echo "  Base model:    mistralai/Voxtral-Mini-3B-2507"
echo "  x-vector:      SiAMResNet34 (trainable)"
echo "  FiLM:          per-layer, gated"
echo "  Trainable:     FiLM + SiAMResNet34 + Projector (~98.7M / 4.75B = 2.08%)"
echo "  Batch size:    2 × 8 acc = 16 effective"
echo "  Learning rate: 2e-4 (x-vector: 2e-5)"
echo "  Epochs:        10 (early stopping patience=5)"
echo ""

# Track training time
START_TIME=$(date +%s)

python -u src/finetune/spk_cond_voxtral/finetuning.py \
    --config_file "${CONFIG_FILE}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "TRAINING COMPLETED!"
echo "========================================"
echo ""
echo "Training Summary:"
echo "  Time:          $(($ELAPSED / 3600))h $(($ELAPSED % 3600 / 60))m $(($ELAPSED % 60))s"
echo "  Config:        ${CONFIG_FILE}"
echo "  Model saved:   ./models/voxtral-spk-cond-neurovoz-torgo-cv"
echo ""
echo "Saved artifacts:"
echo "  - spk_cond_voxtral.pt      (full state dict)"
echo "  - film_bank.pt              (FiLM layers only)"
echo "  - xvector_finetuned.pt      (fine-tuned SiAMResNet34)"
echo "  - training_config.yaml      (config used)"
echo "  - architecture.txt          (architecture description)"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir=./models/voxtral-spk-cond-neurovoz-torgo-cv"
echo ""
echo "Next steps: Run evaluation with evaluate_spk_cond_voxtral.sh"
echo ""
