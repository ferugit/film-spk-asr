#!/bin/bash

# Removes the maximum file size that can be created
ulimit -f unlimited

# Removes the maximum amount of virtual memory available
ulimit -v unlimited

MODEL_NAME="mistralai/Voxtral-Mini-3B-2507"
CACHE_DIR="models/"

# Download the model
python -u src/scripts/download_voxtral.py \
    --model_name $MODEL_NAME \
    --cache_dir $CACHE_DIR 