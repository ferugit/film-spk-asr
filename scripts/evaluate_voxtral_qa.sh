#!/usr/bin/env bash
#SBATCH --job-name=voxtral-qa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm_%j_voxtral_qa.out
#SBATCH --error=logs/slurm_%j_voxtral_qa.err
# Run Voxtral QA inference + evaluation pipeline.
# Must be run from the project root: sbatch evaluate_voxtral_qa.sh [model_path]
set -euo pipefail

# Run from the repository root (this script lives in scripts/).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Activate your environment if needed, e.g.:
# source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${1:-mistralai/Voxtral-Mini-3B-2507}"
MODEL_PATH="${MODEL_PATH%/}"
MODEL_NAME="$(basename "$MODEL_PATH")"

echo "============================================"
echo "  Voxtral QA Evaluation Pipeline"
echo "============================================"
echo "Model:      ${MODEL_PATH}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

# Step 1: Inference
echo ""
echo "[Step 1/2] Running inference ..."
python src/inference/voxtral_qa_inference.py \
    --model-path "${MODEL_PATH}" \
    --questions pathological-speech-questions/questions.json \
    --output "results/qa/${MODEL_NAME}_predictions.json"

# Step 2: Evaluation
echo ""
echo "[Step 2/2] Evaluating predictions ..."
python src/scripts/evaluate_qa.py \
    --input "results/qa/${MODEL_NAME}_predictions.json" \
    --output "results/qa/${MODEL_NAME}_eval_summary.json"

echo ""
echo "Done. Results in results/qa/"
