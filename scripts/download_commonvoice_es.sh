#!/bin/bash
# =============================================================================
# Download Common Voice Scripted Speech 24.0 - Spanish
# from Mozilla Data Collective
#
# Dataset page: https://datacollective.mozillafoundation.org/datasets/cmj8u3p26007dnxxbwyo07lb8
# File: mcv-scripted-es-v24.0.tar.gz (47.99 GB)
#
# Prerequisites:
#   1. Create an API key at https://datacollective.mozillafoundation.org/profile/credentials
#   2. Accept the dataset terms on the dataset page
#   3. Set your API key below or export it: export MDC_API_KEY="your_key"
#
# Usage:
#   bash download_commonvoice_es.sh
# =============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
API_KEY="${MDC_API_KEY:-YOUR_API_KEY}"
DATASET_ID="cmj8u3p26007dnxxbwyo07lb8"
DST_DIR="./data/commonvoice_es"
TARBALL="mcv-scripted-es-v24.0.tar.gz"

DOWNLOAD=true
UNCOMPRESS=true
# ── End configuration ─────────────────────────────────────────────────────────

if [ "$API_KEY" = "YOUR_API_KEY" ]; then
    echo "ERROR: Set your Mozilla Data Collective API key."
    echo "  export MDC_API_KEY=\"your_key\""
    echo "  Or edit this script and replace YOUR_API_KEY."
    exit 1
fi

if [ "$DOWNLOAD" = true ]; then
    MAX_RETRIES=10
    RETRY_DELAY=30

    for attempt in $(seq 1 $MAX_RETRIES); do
        echo ""
        echo "=== Attempt ${attempt}/${MAX_RETRIES}: Requesting download URL ==="
        RESPONSE=$(curl -s -X POST \
            "https://datacollective.mozillafoundation.org/api/datasets/${DATASET_ID}/download" \
            -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json")

        DOWNLOAD_URL=$(echo "$RESPONSE" | jq -r '.downloadUrl')

        if [ "$DOWNLOAD_URL" = "null" ] || [ -z "$DOWNLOAD_URL" ]; then
            echo "ERROR: Could not get download URL. Response:"
            echo "$RESPONSE"
            exit 1
        fi

        echo "=== Downloading ${TARBALL} (resume enabled) ==="
        # -C - resumes from where a previous partial download left off
        if curl -L -C - --connect-timeout 60 --max-time 0 \
                -o "${TARBALL}" "$DOWNLOAD_URL"; then
            echo "=== Download complete ==="
            break
        else
            echo "WARNING: Download interrupted on attempt ${attempt}."
            if [ "$attempt" -lt "$MAX_RETRIES" ]; then
                echo "  Partial file kept for resume. Retrying in ${RETRY_DELAY}s..."
                echo "  (Will request a fresh download URL and resume from byte $(stat -c%s "${TARBALL}" 2>/dev/null || echo 0))"
                sleep "$RETRY_DELAY"
            else
                echo "ERROR: Download failed after ${MAX_RETRIES} attempts."
                echo "  Partial file '${TARBALL}' kept. Re-run the script to resume."
                exit 1
            fi
        fi
    done
else
    echo "Skipping download step."
fi

if [ "$UNCOMPRESS" = true ]; then
    echo "=== Uncompressing dataset to ${DST_DIR} ==="
    mkdir -p "$DST_DIR"
    tar -xzf "${TARBALL}" -C "$DST_DIR"
    echo "=== Uncompression complete ==="
    echo ""
    echo "Expected structure in ${DST_DIR}:"
    echo "  cv-corpus-24.0-2025-12-03/es/"
    echo "    ├── clips/          (MP3 audio files)"
    echo "    ├── train.tsv"
    echo "    ├── dev.tsv"
    echo "    ├── test.tsv"
    echo "    ├── validated.tsv"
    echo "    └── ..."
    echo ""
    # Find the actual extracted path
    CV_PATH=$(find "$DST_DIR" -name "train.tsv" -path "*/es/*" | head -1 | xargs dirname 2>/dev/null || true)
    if [ -n "$CV_PATH" ]; then
        echo "Detected CV root: ${CV_PATH}"
        echo "Clips directory: ${CV_PATH}/clips/"
        echo ""
        echo "Train samples: $(tail -n +2 "${CV_PATH}/train.tsv" | wc -l)"
        echo "Dev samples:   $(tail -n +2 "${CV_PATH}/dev.tsv" | wc -l)"
    fi
fi

echo ""
echo "=== Done! ==="
echo "Next step: run the merge script to combine with NeuroVoz + TORGO:"
echo "  python src/scripts/merge_cv_neurovoz_torgo.py --cv_dir ${DST_DIR}"
