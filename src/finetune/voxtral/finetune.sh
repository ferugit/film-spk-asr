#!/bin/bash

# Training script for Voxtral fine-tuning on NeuroVoz + TORGO dataset
# Combined dataset: Spanish speech with dysarthria and Parkinson's disease

# We will perform three training strategies sequentially:
# 1. Full model fine-tuning (all parameters) - maximum performance
# 2. Encoder-only fine-tuning (encoder + projector) - acoustic adaptation
# 3. Encoder LoRA fine-tuning (LoRA adapters) - ultra-efficient

set -e

echo "========================================"
echo "Voxtral Fine-tuning Training Script"
echo "Dataset: Combined NeuroVoz + TORGO"
echo "Sequential Training: 3 Strategies"
echo "========================================"

# Use GPUs 6 and 7
export CUDA_VISIBLE_DEVICES=6,7
echo "Using GPUs:" $CUDA_VISIBLE_DEVICES

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Training will use CPU (very slow!)"
fi

# Set environment variables for better training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# Track start time
SCRIPT_START_TIME=$(date +%s)

echo ""
echo "========================================"
echo "Strategy 1/3: Full Fine-Tuning"
echo "========================================"
echo "Training all model parameters..."
echo "Expected time: ~5-6 hours"
echo ""

STRATEGY1_START=$(date +%s)
#python -u src/finetune/voxtral/finetuning.py \
#    --config_file src/finetune/voxtral/config/full_finetuning.yaml

STRATEGY1_END=$(date +%s)
STRATEGY1_TIME=$((STRATEGY1_END - STRATEGY1_START))

echo ""
echo "✓ Full fine-tuning completed in $(($STRATEGY1_TIME / 3600))h $(($STRATEGY1_TIME % 3600 / 60))m"
echo "Model saved to: ./models/voxtral-finetuned-neurovoz-torgo"
echo ""

echo "========================================"
echo "Strategy 2/3: Encoder-Only Fine-Tuning"
echo "========================================"
echo "Training encoder + projector only..."
echo "Expected time: ~2-3 hours"
echo ""

STRATEGY2_START=$(date +%s)
#python -u src/finetune/voxtral/finetuning.py \
#    --config_file src/finetune/voxtral/config/encoder_only.yaml

STRATEGY2_END=$(date +%s)
STRATEGY2_TIME=$((STRATEGY2_END - STRATEGY2_START))

echo ""
echo "✓ Encoder-only fine-tuning completed in $(($STRATEGY2_TIME / 3600))h $(($STRATEGY2_TIME % 3600 / 60))m"
echo "Model saved to: ./models/voxtral-encoder-finetuned-neurovoz-torgo"
echo ""

echo "========================================"
echo "Strategy 3/3: LoRA Fine-Tuning"
echo "========================================"
echo "Training with LoRA adapters..."
echo "Expected time: ~1-2 hours"
echo ""

STRATEGY3_START=$(date +%s)
python -u src/finetune/voxtral/finetuning.py \
    --config_file src/finetune/voxtral/config/encoder_lora.yaml

STRATEGY3_END=$(date +%s)
STRATEGY3_TIME=$((STRATEGY3_END - STRATEGY3_START))

echo ""
echo "✓ LoRA fine-tuning completed in $(($STRATEGY3_TIME / 3600))h $(($STRATEGY3_TIME % 3600 / 60))m"
echo "Model saved to: ./models/voxtral-encoder-lora-finetuned-neurovoz-torgo"
echo ""

# Calculate total time
SCRIPT_END_TIME=$(date +%s)
TOTAL_TIME=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

echo ""
echo "========================================"
echo "ALL TRAINING COMPLETED!"
echo "========================================"
echo ""
echo "Training Summary:"
echo "  1. Full Fine-Tuning:        $(($STRATEGY1_TIME / 3600))h $(($STRATEGY1_TIME % 3600 / 60))m"
echo "  2. Encoder-Only Fine-Tuning: $(($STRATEGY2_TIME / 3600))h $(($STRATEGY2_TIME % 3600 / 60))m"
echo "  3. LoRA Fine-Tuning:         $(($STRATEGY3_TIME / 3600))h $(($STRATEGY3_TIME % 3600 / 60))m"
echo "  ─────────────────────────────────────"
echo "  Total Time:                 $(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m"
echo ""
echo "Models saved:"
echo "  1. ./models/voxtral-finetuned-neurovoz-torgo"
echo "  2. ./models/voxtral-encoder-finetuned-neurovoz-torgo"
echo "  3. ./models/voxtral-encoder-lora-finetuned-neurovoz-torgo"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir=./models/voxtral-finetuned-neurovoz-torgo/runs"
echo "  tensorboard --logdir=./models/voxtral-encoder-finetuned-neurovoz-torgo/runs"
echo "  tensorboard --logdir=./models/voxtral-encoder-lora-finetuned-neurovoz-torgo/runs"
echo ""
echo "Next steps: Evaluate all three models on test sets"
echo ""

