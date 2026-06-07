#!/usr/bin/env python3
"""
Generate an evaluation summary JSON from NeuroVoz and TORGO prediction TSVs.

This script reads the TSV files produced by the inference-only eval scripts
(src/finetune/voxtral/eval.py or src/finetune/whisper/eval.py), computes WER
with proper text normalisation (punctuation removal, digit-to-word conversion
in the correct language), and produces:
  - A per-dataset JSON for NeuroVoz
  - A per-dataset JSON for TORGO (with single/multi-word breakdown)
  - A combined *_evaluation_summary.json

The summary JSON is consumed by aggregate_all_results.py.

Usage:
    python src/scripts/generate_eval_summary.py \
        --model_name voxtral_base \
        --model_path mistralai/Voxtral-Mini-3B-2507 \
        --neurovoz_tsv results/evaluation/voxtral/voxtral_base_neurovoz_test_results.tsv \
        --torgo_tsv results/evaluation/voxtral/voxtral_base_torgo_test_results.tsv \
        --output_dir results/evaluation/voxtral
"""

import os
import json
import string
import re
import argparse

import pandas as pd
import jiwer
from num2words import num2words

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\b\d+\b")


def digits_to_words(text: str, lang: str = "es") -> str:
    """Convert digit tokens to words in the specified language."""
    def repl(m):
        return num2words(int(m.group(0)), lang=lang)
    return _NUM_RE.sub(repl, text)


def normalize_text(text: str, lang: str = "es") -> str:
    """Lowercase, remove <unk>, convert digits, strip punctuation."""
    text = str(text).lower().replace("<unk>", "")
    text = digits_to_words(text, lang=lang)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------
def compute_wer(references: list, hypotheses: list) -> float:
    """Return WER as a percentage."""
    if not references:
        return float("nan")
    out = jiwer.process_words(references, hypotheses)
    total_ref_words = out.hits + out.substitutions + out.deletions
    if total_ref_words == 0:
        return 0.0
    return (out.substitutions + out.deletions + out.insertions) / total_ref_words * 100


def classify_utterance(text: str) -> str:
    """single-word if ≤1 word after basic cleaning, else multi-word."""
    clean = str(text).strip()
    for ch in ".,!?;:":
        clean = clean.replace(ch, "")
    words = [w for w in clean.split() if w]
    return "single-word" if len(words) <= 1 else "multi-word"


# ---------------------------------------------------------------------------
# Process a single TSV
# ---------------------------------------------------------------------------
def process_tsv(tsv_path: str, lang: str) -> dict:
    """
    Read a predictions TSV and compute total / single-word / multi-word WER.

    Parameters
    ----------
    tsv_path : str
        Path to the TSV with columns: sample_id, reference, hypothesis.
    lang : str
        Language code for normalisation ('es' or 'en').

    Returns
    -------
    dict  with keys: wer, num_samples, model_output_path, utterance_analysis (optional)
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df["reference"] = df["reference"].fillna("")
    df["hypothesis"] = df["hypothesis"].fillna("")

    norm = lambda t: normalize_text(t, lang=lang)
    refs = df["reference"].apply(norm).tolist()
    hyps = df["hypothesis"].apply(norm).tolist()

    total_wer = compute_wer(refs, hyps)

    # Classify each utterance (on the raw reference, before normalisation)
    utt_types = [classify_utterance(r) for r in df["reference"].tolist()]

    sw_refs = [r for r, t in zip(refs, utt_types) if t == "single-word"]
    sw_hyps = [h for h, t in zip(hyps, utt_types) if t == "single-word"]
    mw_refs = [r for r, t in zip(refs, utt_types) if t == "multi-word"]
    mw_hyps = [h for h, t in zip(hyps, utt_types) if t == "multi-word"]

    result = {
        "wer": total_wer,
        "num_samples": len(refs),
        "model_output_path": tsv_path,
    }

    # Add utterance breakdown if there are both types
    if sw_refs or mw_refs:
        result["utterance_analysis"] = {
            "single_word": {
                "count": len(sw_refs),
                "wer": compute_wer(sw_refs, sw_hyps),
                "percentage": len(sw_refs) / len(refs) * 100 if refs else 0,
            },
            "multi_word": {
                "count": len(mw_refs),
                "wer": compute_wer(mw_refs, mw_hyps),
                "percentage": len(mw_refs) / len(refs) * 100 if refs else 0,
            },
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation summary JSON from prediction TSVs."
    )
    parser.add_argument("--model_name", required=True,
                        help="Short model name (e.g. voxtral_base).")
    parser.add_argument("--model_path", required=True,
                        help="Original model path / ID.")
    parser.add_argument("--neurovoz_tsv", required=True,
                        help="Path to NeuroVoz predictions TSV.")
    parser.add_argument("--torgo_tsv", required=True,
                        help="Path to TORGO predictions TSV.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write JSON outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── NeuroVoz (Spanish) ───────────────────────────────────────────────
    print(f"Processing NeuroVoz: {args.neurovoz_tsv}")
    nv_results = process_tsv(args.neurovoz_tsv, lang="es")
    nv_results["dataset"] = f"NeuroVoz Test (Spanish) - {args.model_name}"

    nv_json_path = os.path.join(args.output_dir,
                                f"{args.model_name}_neurovoz_test_results.json")
    with open(nv_json_path, "w") as f:
        json.dump(nv_results, f, indent=2)
    print(f"  WER: {nv_results['wer']:.2f}%  ({nv_results['num_samples']} samples)")
    print(f"  → {nv_json_path}")

    # ── TORGO (English) ──────────────────────────────────────────────────
    print(f"Processing TORGO: {args.torgo_tsv}")
    tg_results = process_tsv(args.torgo_tsv, lang="en")
    tg_results["dataset"] = f"TORGO Test (English) - {args.model_name}"

    tg_json_path = os.path.join(args.output_dir,
                                f"{args.model_name}_torgo_test_results.json")
    with open(tg_json_path, "w") as f:
        json.dump(tg_results, f, indent=2)
    print(f"  WER: {tg_results['wer']:.2f}%  ({tg_results['num_samples']} samples)")
    if "utterance_analysis" in tg_results:
        ua = tg_results["utterance_analysis"]
        print(f"    Single-word: {ua['single_word']['wer']:.2f}% ({ua['single_word']['count']} samples)")
        print(f"    Multi-word:  {ua['multi_word']['wer']:.2f}% ({ua['multi_word']['count']} samples)")
    print(f"  → {tg_json_path}")

    # ── Combined summary ─────────────────────────────────────────────────
    total_n = nv_results["num_samples"] + tg_results["num_samples"]
    combined_wer = (
        nv_results["wer"] * nv_results["num_samples"]
        + tg_results["wer"] * tg_results["num_samples"]
    ) / total_n if total_n > 0 else float("nan")

    summary = {
        "model": args.model_path,
        "model_name": args.model_name,
        "neurovoz_test": nv_results,
        "torgo_test": tg_results,
        "combined_wer": combined_wer,
        "total_samples": total_n,
    }

    summary_path = os.path.join(args.output_dir,
                                f"{args.model_name}_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY — {args.model_name}")
    print(f"{'=' * 60}")
    print(f"  NeuroVoz WER: {nv_results['wer']:.2f}%  ({nv_results['num_samples']} samples)")
    print(f"  TORGO WER:    {tg_results['wer']:.2f}%  ({tg_results['num_samples']} samples)")
    print(f"  Combined WER: {combined_wer:.2f}%  ({total_n} samples)")
    print(f"  → {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
