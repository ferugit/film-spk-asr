#!/usr/bin/env python3
"""Print a comparison table of all QA eval summaries in results/qa/."""

import json
from pathlib import Path

RESULTS_DIR = Path("results/qa")

# Friendly display names
MODEL_NAMES = {
    "Voxtral-Mini-3B-2507": "Base (Voxtral-Mini-3B)",
    "voxtral-finetuned-neurovoz-torgo-cv": "Full fine-tune",
    "voxtral-encoder-finetuned-neurovoz-torgo-cv": "Encoder fine-tune",
    "voxtral-lora-finetuned-neurovoz-torgo-cv": "LoRA fine-tune",
    "voxtral-encoder-lora-finetuned-neurovoz-torgo-cv": "Encoder LoRA",
    "voxtral-spk-cond-neurovoz-torgo-cv": "Speaker-cond (FiLM)",
}

# Display order
MODEL_ORDER = [
    "Voxtral-Mini-3B-2507",
    "voxtral-finetuned-neurovoz-torgo-cv",
    "voxtral-encoder-finetuned-neurovoz-torgo-cv",
    "voxtral-lora-finetuned-neurovoz-torgo-cv",
    "voxtral-encoder-lora-finetuned-neurovoz-torgo-cv",
    "voxtral-spk-cond-neurovoz-torgo-cv",
]

# Metrics to show (key in JSON → column header)
METRICS = [
    ("all", "Overall"),
    ("sex", "Sex"),
    ("sex_neurovoz", "Sex (NV)"),
    ("sex_torgo", "Sex (TG)"),
    ("age", "Age"),
    ("age_neurovoz", "Age (NV)"),
]


def load_summaries():
    summaries = {}
    for f in RESULTS_DIR.glob("*_eval_summary.json"):
        model_key = f.name.replace("_eval_summary.json", "")
        with open(f) as fh:
            summaries[model_key] = json.load(fh)
    return summaries


def main():
    summaries = load_summaries()

    # Determine column widths
    name_col = max(len(MODEL_NAMES.get(k, k)) for k in MODEL_ORDER) + 2
    metric_col = 10

    # Header
    header = f"{'Model':<{name_col}}"
    for _, label in METRICS:
        header += f" {label:>{metric_col}}"
    print(header)
    print("-" * len(header))

    # Rows
    for model_key in MODEL_ORDER:
        if model_key not in summaries:
            continue
        data = summaries[model_key]
        display = MODEL_NAMES.get(model_key, model_key)
        row = f"{display:<{name_col}}"
        for metric_key, _ in METRICS:
            if metric_key in data:
                acc = data[metric_key]["accuracy"] * 100
                n = data[metric_key]["total"]
                row += f" {acc:>8.1f}% "
            else:
                row += f" {'N/A':>{metric_col}}"
        print(row)

    print()

    # Also print any models found but not in MODEL_ORDER
    extra = set(summaries.keys()) - set(MODEL_ORDER)
    if extra:
        print("(Additional models found:)")
        for model_key in sorted(extra):
            data = summaries[model_key]
            display = model_key
            row = f"{display:<{name_col}}"
            for metric_key, _ in METRICS:
                if metric_key in data:
                    acc = data[metric_key]["accuracy"] * 100
                    row += f" {acc:>8.1f}% "
                else:
                    row += f" {'N/A':>{metric_col}}"
            print(row)


if __name__ == "__main__":
    main()
