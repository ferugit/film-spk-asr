#!/usr/bin/env python3
"""
Aggregate evaluation results from all Voxtral and Whisper models into a single
consolidated report.

For each model it reports WER (%) broken down by:
  - NeuroVoz: total WER (all utterances are multi-word)
  - TORGO:    single-word | multi-word | total
  - Overall (combined NeuroVoz + TORGO)

Usage:
    python src/scripts/aggregate_all_results.py \
        --results_dir results/evaluation \
        --output results/evaluation/all_models_results.csv
"""

import os
import json
import argparse
from collections import OrderedDict

import pandas as pd


def discover_models(results_dir: str) -> list:
    """
    Walk voxtral/ and whisper/ subdirectories and discover evaluated models
    by looking for *_evaluation_summary.json files.

    Returns a list of dicts with keys: family, model_name, summary_path,
        neurovoz_tsv, torgo_tsv.
    """
    models = []

    for family in ("voxtral", "whisper"):
        family_dir = os.path.join(results_dir, family)
        if not os.path.isdir(family_dir):
            continue

        for fname in sorted(os.listdir(family_dir)):
            if not fname.endswith("_evaluation_summary.json"):
                continue

            summary_path = os.path.join(family_dir, fname)
            with open(summary_path) as f:
                summary = json.load(f)

            model_name = summary.get("model_name", fname.replace("_evaluation_summary.json", ""))
            model_path = summary.get("model", "")

            # Locate the per-dataset TSV files from the summary
            nv_tsv = summary.get("neurovoz_test", {}).get("model_output_path", "")
            torgo_tsv = summary.get("torgo_test", {}).get("model_output_path", "")

            models.append({
                "family": family.capitalize(),
                "model_name": model_name,
                "model_path": model_path,
                "summary_path": summary_path,
                "summary": summary,
                "neurovoz_tsv": nv_tsv,
                "torgo_tsv": torgo_tsv,
            })

    return models


def build_report(results_dir: str) -> pd.DataFrame:
    """Build the consolidated results DataFrame.

    Reads WER metrics directly from the *_evaluation_summary.json files
    produced by the eval scripts, rather than recomputing from TSV files
    (which would require matching the exact same normalisation pipeline).
    """
    models = discover_models(results_dir)

    rows = []
    for m in models:
        row = OrderedDict()
        row["Family"] = m["family"]
        row["Model"] = m["model_name"]
        row["Model Path"] = m["model_path"]

        summary = m["summary"]

        # --- NeuroVoz ---
        nv_info = summary.get("neurovoz_test", {})
        nv_wer = nv_info.get("wer", float("nan"))
        nv_n = nv_info.get("num_samples", 0)

        row["NV WER (%)"] = nv_wer
        row["NV N"] = nv_n

        # --- TORGO ---
        tg_info = summary.get("torgo_test", {})
        tg_wer = tg_info.get("wer", float("nan"))
        tg_n = tg_info.get("num_samples", 0)
        tg_utt = tg_info.get("utterance_analysis", {})

        sw = tg_utt.get("single_word", {})
        mw = tg_utt.get("multi_word", {})

        row["TORGO Total WER (%)"] = tg_wer
        row["TORGO Total N"] = tg_n
        row["TORGO Single-Word WER (%)"] = sw.get("wer", float("nan"))
        row["TORGO Single-Word N"] = sw.get("count", 0)
        row["TORGO Multi-Word WER (%)"] = mw.get("wer", float("nan"))
        row["TORGO Multi-Word N"] = mw.get("count", 0)

        # --- Overall weighted WER ---
        total_n = nv_n + tg_n
        if total_n > 0:
            nv_wer_safe = nv_wer if not pd.isna(nv_wer) else 0
            tg_wer_safe = tg_wer if not pd.isna(tg_wer) else 0
            overall = (nv_wer_safe * nv_n + tg_wer_safe * tg_n) / total_n
        else:
            overall = float("nan")
        row["Overall WER (%)"] = overall

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_summary_table(df: pd.DataFrame) -> None:
    """Pretty-print the consolidated results table."""
    # Select display columns
    display_cols = [
        "Family", "Model",
        "NV WER (%)",
        "TORGO Total WER (%)", "TORGO Single-Word WER (%)", "TORGO Multi-Word WER (%)",
        "Overall WER (%)",
    ]
    dfd = df[display_cols].copy()

    # Format floats
    float_cols = [c for c in dfd.columns if "WER" in c]
    for c in float_cols:
        dfd[c] = dfd[c].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")

    print("\n" + "=" * 140)
    print("CONSOLIDATED EVALUATION RESULTS — ALL MODELS")
    print("=" * 140)
    print(dfd.to_string(index=False))
    print("=" * 140 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results from all models into a single report."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/evaluation",
        help="Root directory containing voxtral/ and whisper/ result subdirs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation/all_models_results.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    df = build_report(args.results_dir)

    # Sort: base models first, then alphabetically
    df = df.sort_values(["Family", "Model"]).reset_index(drop=True)

    # Print to console
    print_summary_table(df)

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_json(args.output, orient="records", indent=2)
    print(f"✓ Consolidated JSON saved to: {args.output}")


if __name__ == "__main__":
    main()
