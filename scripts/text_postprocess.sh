#!/bin/bash

##############################################################################
# Text Post-Processing Pipeline for Hallucination Reduction
#
# Applies text-based deduplication to remove repetitive hallucinations
# from ASR outputs, without requiring audio alignment.
#
# Three-step approach:
#   1. Error filtering  – words > max_word_length with internal repetition
#   2. Word-level dedup – consecutive repeated words collapsed
#   3. Phrase-level dedup – consecutive repeated phrases collapsed
#
# Usage: ./text_postprocess.sh [OPTIONS]
#
# Options:
#   --help                    Show this help message
#   --max-word-length N       Max word length before flagging (default: 15)
#   --max-word-repeats N      Max consecutive word repeats allowed (default: 2)
#   --models MODEL...         Models to process: whisper, voxtral, finetuned, all (default)
#   --skip-wer                Skip WER calculation
#   --debug                   Enable debug output
#
# Examples:
#   ./text_postprocess.sh
#   ./text_postprocess.sh --max-word-repeats 1
#   ./text_postprocess.sh --models whisper voxtral --debug
#   ./text_postprocess.sh --models finetuned --skip-wer
#
##############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PP_SCRIPT="src/scripts/text_postprocess_reduce_hallucinations.py"
WER_SCRIPT="src/scripts/calculate_wer.py"

# Directories
INFERENCE_DIR="results/inference"
OUTPUT_DIR="results/evaluation/pp"

# Defaults
MAX_WORD_LENGTH=15
MAX_WORD_REPEATS=2 # Max allowed consecutive repeats of the same word before flagging as hallucination
MODELS="all"
SKIP_WER=false
DEBUG=""

SKIP_RAW_INFERENCE=false   # Set to true to skip re-processing raw inference files (if already done)
SKIP_BASE_EVAL=true       # Set to true to skip re-processing base model evaluation files (if already done)
SKIP_FINETUNED_EVAL=true  # Set to true to skip re-processing finetuned model evaluation files (if already done)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            head -30 "$0" | grep -E "^#" | sed 's/^# \?//'
            exit 0
            ;;
        --max-word-length)
            MAX_WORD_LENGTH="$2"; shift 2 ;;
        --max-word-repeats)
            MAX_WORD_REPEATS="$2"; shift 2 ;;
        --models)
            MODELS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS="$MODELS $1"; shift
            done
            MODELS=$(echo "$MODELS" | xargs)  # trim
            ;;
        --skip-wer)
            SKIP_WER=true; shift ;;
        --debug)
            DEBUG="--debug"; shift ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "Text Post-Processing Pipeline"
echo "========================================"
echo "Max word length:     $MAX_WORD_LENGTH"
echo "Max word repeats:    $MAX_WORD_REPEATS"
echo "Models:              $MODELS"
echo "Skip WER:            $SKIP_WER"
echo "Output dir:          $OUTPUT_DIR"
echo "========================================"
echo ""

cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Helper: run post-processing + optional WER
# ---------------------------------------------------------------------------
run_postprocess() {
    local input_tsv="$1"
    local output_tsv="$2"
    local hyp_col="$3"
    local label="$4"

    if [ ! -f "$input_tsv" ]; then
        echo "  [SKIP] $input_tsv not found"
        return
    fi

    echo "  Processing: $label"
    echo "    Input:  $input_tsv"
    echo "    Output: $output_tsv"

    python3 "$PP_SCRIPT" \
        --input_tsv "$input_tsv" \
        --output_tsv "$output_tsv" \
        --hypothesis_column "$hyp_col" \
        --max_word_length "$MAX_WORD_LENGTH" \
        --max_word_repeats "$MAX_WORD_REPEATS" \
        $DEBUG

    echo ""
}

run_wer() {
    local input_tsv="$1"
    local hyp_col="$2"
    local label="$3"
    local ref_file="${4:-}"          # optional: path to reference file
    local trans_col="${5:-}"         # optional: transcription column name
    local no_merge="${6:-}"          # optional: "true" to skip merge (ref in input TSV)

    if [ "$SKIP_WER" = true ]; then
        return
    fi

    if [ ! -f "$input_tsv" ]; then
        return
    fi

    echo "  Computing WER for: $label (column: $hyp_col)"

    local extra_args=()
    if [ "$no_merge" = "true" ]; then
        extra_args+=(--no_merge)
    fi
    if [ -n "$ref_file" ]; then
        extra_args+=(--reference_file "$ref_file")
    fi
    if [ -n "$trans_col" ]; then
        extra_args+=(--transcription_column "$trans_col")
    fi

    python3 "$WER_SCRIPT" \
        --input_tsv "$input_tsv" \
        --output_dir "$OUTPUT_DIR" \
        --hypothesis_column "$hyp_col" \
        "${extra_args[@]}" \
        2>&1 | tail -5
    echo ""
}

# ---------------------------------------------------------------------------
# Determine which model files to process
# ---------------------------------------------------------------------------
should_process() {
    local filename="$1"
    if [ "$MODELS" = "all" ]; then
        return 0
    fi
    for model in $MODELS; do
        case $model in
            whisper)
                if echo "$filename" | grep -qi "whisper"; then return 0; fi
                ;;
            voxtral)
                if echo "$filename" | grep -qi "voxtral\|Voxtral"; then return 0; fi
                ;;
            finetuned)
                if echo "$filename" | grep -qi "finetuned"; then return 0; fi
                ;;
            omni*)
                if echo "$filename" | grep -qi "omni"; then return 0; fi
                ;;
            scribe*)
                if echo "$filename" | grep -qi "scribe"; then return 0; fi
                ;;
        esac
    done
    return 1
}

# ===================================================================
# PART 1: Post-process raw inference files (Hypothesis column)
# ===================================================================
# echo "========================================"
# echo "PART 1: Raw Inference Files"
# echo "========================================"

if [ "$SKIP_RAW_INFERENCE" = true ]; then
    echo "Skipping raw inference post-processing (already done)"

else
    echo "========================================"
    echo "PART 1: Raw Inference Files"
    echo "========================================"

    for tsv in "$INFERENCE_DIR"/dss_*_hypothesis.tsv; do
    [ -f "$tsv" ] || continue

    basename=$(basename "$tsv")
    if ! should_process "$basename"; then
        continue
    fi

    # Output filename: same name under results/pp/
    output_tsv="$OUTPUT_DIR/$basename"

    run_postprocess "$tsv" "$output_tsv" "Hypothesis" "$basename"
    run_wer "$output_tsv" "postprocessed_Hypothesis" "$basename"
    done

fi



# ===================================================================
# PART 3: Post-process NeuroVoz & Torgo base model results
# ===================================================================

if [ "$SKIP_BASE_EVAL" = true ]; then
    echo "Skipping base model evaluation post-processing (already done)"

else

    echo "========================================"
    echo "PART 3: NeuroVoz & Torgo (base models)"
    echo "========================================"

    EVAL_WHISPER_DIR="results/evaluation/whisper"
    EVAL_VOXTRAL_DIR="results/evaluation/voxtral"

    BASE_EVAL_FILES=(
        "$EVAL_WHISPER_DIR/whisper_base_neurovoz_test_results.tsv"
        "$EVAL_WHISPER_DIR/whisper_base_torgo_test_results.tsv"
        "$EVAL_VOXTRAL_DIR/voxtral_base_neurovoz_test_results.tsv"
        "$EVAL_VOXTRAL_DIR/voxtral_base_torgo_test_results.tsv"
    )

    for tsv in "${BASE_EVAL_FILES[@]}"; do
        if [ ! -f "$tsv" ]; then
            echo "  [SKIP] $tsv not found"
            continue
        fi

        basename=$(basename "$tsv")
        if ! should_process "$basename"; then
            continue
        fi

        output_tsv="$OUTPUT_DIR/$basename"

        run_postprocess "$tsv" "$output_tsv" "hypothesis" "$basename"
        # These TSVs already contain a 'reference' column — use --no_merge
        # so calculate_wer.py reads references directly from the input TSV.
        run_wer "$output_tsv" "postprocessed_hypothesis" "$basename" "" "reference" "true"
    done

fi

# ===================================================================
# PART 4: Finetunede models (if applicable)
# ===================================================================

if [ "$SKIP_FINETUNED_EVAL" = true ]; then
    echo "Skipping finetuned model evaluation post-processing (already done)"

else
    echo "========================================"
    echo "PART 4: NeuroVoz + Torgo Finetuned Models: voxtral encoder lora finetuned"
    echo "========================================"

    EVAL_VOXTRAL_DIR="results/evaluation/voxtral"
    EVAL_WHISPER_DIR="results/evaluation/whisper"
    EVAL_SPK_COND_DIR="results/evaluation/spk_cond_voxtral"

    EVAL_FILES=(
        # Full Finetuned models
        # "$EVAL_VOXTRAL_DIR/voxtral_finetuned_neurovoz_test_results.tsv"
        # "$EVAL_VOXTRAL_DIR/voxtral_finetuned_torgo_test_results.tsv"
        # "$EVAL_WHISPER_DIR/whisper-large-v3-finetuned_neurovoz_test_results.tsv"
        # "$EVAL_WHISPER_DIR/whisper-large-v3-finetuned_torgo_test_results.tsv"
        # Full lora finetuned models
        # "$EVAL_VOXTRAL_DIR/voxtral_lora-finetuned_neurovoz_test_results.tsv"
        # "$EVAL_VOXTRAL_DIR/voxtral_lora-finetuned_torgo_test_results.tsv"
        # "$EVAL_WHISPER_DIR/whisper-large-v3-lora-finetuned_neurovoz_test_results.tsv"
        # "$EVAL_WHISPER_DIR/whisper-large-v3-lora-finetuned_torgo_test_results.tsv"
        # Encoder finetuned models
        # "$EVAL_VOXTRAL_DIR/voxtral_encoder-finetuned_neurovoz_test_results.tsv"
        # "$EVAL_VOXTRAL_DIR/voxtral_encoder-finetuned_torgo_test_results.tsv"
        # "$EVAL_WHISPER_DIR/whisper-large-v3-encoder-finetuned_neurovoz_test_results.tsv"
        # "$EVAL_WHISPER_DIR/whisper-large-v3-encoder-finetuned_torgo_test_results.tsv"
        # Encoder lora finetuned models
        # "$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned_neurovoz_test_results.tsv"
        # "$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned_torgo_test_results.tsv"
        # "$EVAL_WHISPER_DIR/whisper-large-v3-encoder-lora-finetuned_neurovoz_test_results.tsv"
        # "$EVAL_WHISPER_DIR/whisper-large-v3-encoder-lora-finetuned_torgo_test_results.tsv"
        # Data experiments
        # Balance with CV
        # Balance with CV
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_torgo_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-finetuned-neurovoz-torgo-cv_torgo_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_lora-finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_lora-finetuned-neurovoz-torgo-cv_torgo_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_finetuned-neurovoz-torgo-cv_torgo_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_torgo_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv"
        # NV-only
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_torgo_test_results.tsv"
        # NV-only
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned-neurovoz_neurovoz_test_results.tsv"
        #"$EVAL_VOXTRAL_DIR/voxtral_encoder-lora-finetuned-neurovoz_torgo_test_results.tsv"
        # Speaker-conditioned (FiLM)
        "$EVAL_SPK_COND_DIR/voxtral_spk-cond-neurovoz-torgo-cv_neurovoz_test_results.tsv"
        "$EVAL_SPK_COND_DIR/voxtral_spk-cond-neurovoz-torgo-cv_torgo_test_results.tsv"
    )

    for tsv in "${EVAL_FILES[@]}"; do
        if [ ! -f "$tsv" ]; then
            echo "  [SKIP] $tsv not found"
            continue
        fi

        basename=$(basename "$tsv")
        if ! should_process "$basename"; then
            continue
        fi

        output_tsv="$OUTPUT_DIR/$basename"

        run_postprocess "$tsv" "$output_tsv" "hypothesis" "$basename"
        # These TSVs already contain a 'reference' column — use --no_merge
        # so calculate_wer.py reads references directly from the input TSV.
        run_wer "$output_tsv" "postprocessed_hypothesis" "$basename" "" "reference" "true"
    done

fi

echo "========================================"
echo "Post-processing complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "========================================"
echo ""
ls -lh "$OUTPUT_DIR/"
