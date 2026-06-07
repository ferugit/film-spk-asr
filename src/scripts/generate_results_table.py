"""
Generate a summary WER table for all Voxtral model variants.

Reads JSON WER reports from the evaluation directories and produces
a Markdown (and optionally LaTeX) table with columns:
  Model | NeuroVoz | NeuroVoz+PP | TORGO | TORGO+PP | DSS | DSS+PP

Usage:
    python src/scripts/generate_results_table.py
    python src/scripts/generate_results_table.py --latex
    python src/scripts/generate_results_table.py --output results/voxtral_results_table.md
"""

import argparse
import json
import os

# ── Model definitions ────────────────────────────────────────────────────────
# Each entry: (display_name, {
#   "neurovoz": (directory, filename),
#   "neurovoz_pp": (directory, filename),
#   "torgo": ..., "torgo_pp": ...
# })
#
# All models trained on combined NeuroVoz + TORGO + CommonVoice ES.
#
# Directories
EVAL_VOX = "results/evaluation/voxtral"
EVAL_SPK = "results/evaluation/spk_cond_voxtral"
EVAL_SPK_ADAPTER = "results/evaluation/spk_cond_voxtral_adapter"
EVAL_PP  = "results/evaluation/pp"

MODELS = [
    ("Full fine-tune", {
        "neurovoz":    (EVAL_VOX, "voxtral_finetuned-neurovoz-torgo-cv_neurovoz_test_results.json"),
        "neurovoz_pp": (EVAL_PP,  "wer_report_voxtral_finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv.json"),
        "torgo":       (EVAL_VOX, "voxtral_finetuned-neurovoz-torgo-cv_torgo_test_results.json"),
        "torgo_pp":    (EVAL_PP,  "wer_report_voxtral_finetuned-neurovoz-torgo-cv_torgo_test_results.tsv.json"),
    }),
    ("LoRA fine-tune", {
        "neurovoz":    (EVAL_VOX, "voxtral_lora-finetuned-neurovoz-torgo-cv_neurovoz_test_results.json"),
        "neurovoz_pp": (EVAL_PP,  "wer_report_voxtral_lora-finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv.json"),
        "torgo":       (EVAL_VOX, "voxtral_lora-finetuned-neurovoz-torgo-cv_torgo_test_results.json"),
        "torgo_pp":    (EVAL_PP,  "wer_report_voxtral_lora-finetuned-neurovoz-torgo-cv_torgo_test_results.tsv.json"),
    }),
    ("Encoder fine-tune", {
        "neurovoz":    (EVAL_VOX, "voxtral_encoder-finetuned-neurovoz-torgo-cv_neurovoz_test_results.json"),
        "neurovoz_pp": (EVAL_PP,  "wer_report_voxtral_encoder-finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv.json"),
        "torgo":       (EVAL_VOX, "voxtral_encoder-finetuned-neurovoz-torgo-cv_torgo_test_results.json"),
        "torgo_pp":    (EVAL_PP,  "wer_report_voxtral_encoder-finetuned-neurovoz-torgo-cv_torgo_test_results.tsv.json"),
    }),
    ("Encoder LoRA", {
        "neurovoz":    (EVAL_VOX, "voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_neurovoz_test_results.json"),
        "neurovoz_pp": (EVAL_PP,  "wer_report_voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_neurovoz_test_results.tsv.json"),
        "torgo":       (EVAL_VOX, "voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_torgo_test_results.json"),
        "torgo_pp":    (EVAL_PP,  "wer_report_voxtral_encoder-lora-finetuned-neurovoz-torgo-cv_torgo_test_results.tsv.json"),
    }),
    ("Spk-Cond FiLM", {
        "neurovoz":    (EVAL_SPK, "voxtral_spk-cond-neurovoz-torgo-cv_neurovoz_test_results.json"),
        "neurovoz_pp": (EVAL_PP,  "wer_report_voxtral_spk-cond-neurovoz-torgo-cv_neurovoz_test_results.tsv.json"),
        "torgo":       (EVAL_SPK, "voxtral_spk-cond-neurovoz-torgo-cv_torgo_test_results.json"),
        "torgo_pp":    (EVAL_PP,  "wer_report_voxtral_spk-cond-neurovoz-torgo-cv_torgo_test_results.tsv.json"),
    }),
    ("Spk-Cond FiLM + Adapter", {
        "neurovoz":    (EVAL_SPK_ADAPTER, "voxtral_spk-cond-neurovoz-torgo-cv_neurovoz_test_results.json"),
        "torgo":       (EVAL_SPK_ADAPTER, "voxtral_spk-cond-neurovoz-torgo-cv_torgo_test_results.json"),
    }),
]

COLUMNS = ["neurovoz", "neurovoz_pp", "torgo", "torgo_pp", "torgo_sw", "torgo_sw_pp", "torgo_mw", "torgo_mw_pp"]
HEADERS = ["Model", "NV", "NV+PP", "TORGO", "TORGO+PP", "T-SW", "T-SW+PP", "T-MW", "T-MW+PP"]


def read_wer(directory, filename, utterance_type=None):
    """Read WER from a JSON report file.

    If utterance_type is given ("single_word" or "multi_word"), returns
    the WER for that subset instead of the overall WER.
    """
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)

    if utterance_type is None:
        overall = d.get("overall", d)
        return overall.get("wer", None)

    # Raw eval files: utterance_analysis with single_word / multi_word
    ua = d.get("utterance_analysis", {})
    if utterance_type in ua:
        return ua[utterance_type].get("wer", None)

    # PP files: by_utterance_type with single-word / multi-word
    but = d.get("by_utterance_type", {})
    key_dash = utterance_type.replace("_", "-")
    if key_dash in but:
        return but[key_dash].get("wer", None)

    return None


# Map column names to utterance_type parameter for read_wer
_UTTERANCE_TYPE = {
    "torgo_sw": "single_word",
    "torgo_sw_pp": "single_word",
    "torgo_mw": "multi_word",
    "torgo_mw_pp": "multi_word",
}

# Map sub-columns to their source TORGO column
_SOURCE_COL = {
    "torgo_sw": "torgo",
    "torgo_sw_pp": "torgo_pp",
    "torgo_mw": "torgo",
    "torgo_mw_pp": "torgo_pp",
}


def build_rows():
    """Build list of (name, [wer_values]) tuples."""
    rows = []
    for name, files in MODELS:
        values = []
        for col in COLUMNS:
            src = _SOURCE_COL.get(col, col)
            ut = _UTTERANCE_TYPE.get(col, None)
            if src in files:
                directory, filename = files[src]
                wer = read_wer(directory, filename, utterance_type=ut)
                values.append(wer)
            else:
                values.append(None)
        rows.append((name, values))
    return rows


def fmt(val):
    """Format a WER value for display."""
    if val is None:
        return "—"
    return f"{val:.2f}"


def bold_best(rows):
    """Find the best (lowest) WER per column for markdown bolding."""
    best = []
    for ci in range(len(COLUMNS)):
        vals = [r[1][ci] for r in rows if r[1][ci] is not None]
        best.append(min(vals) if vals else None)
    return best


def generate_markdown(rows):
    """Generate a Markdown table string."""
    best = bold_best(rows)
    lines = []

    # Header
    lines.append("| " + " | ".join(HEADERS) + " |")
    lines.append("|" + "|".join(["---"] + [":---:"] * len(COLUMNS)) + "|")

    for name, values in rows:
        cells = [name]
        for ci, v in enumerate(values):
            s = fmt(v)
            if v is not None and best[ci] is not None and abs(v - best[ci]) < 0.005:
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_latex(rows):
    """Generate a LaTeX tabular string."""
    best = bold_best(rows)
    lines = []

    ncols = len(HEADERS)
    col_spec = "l" + "c" * (ncols - 1)
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{WER (\%) of Voxtral model variants across evaluation sets. PP = post-processing. Best result per column in \textbf{bold}.}")
    lines.append(r"\label{tab:voxtral_results}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Multi-column header
    lines.append(
        r"\multirow{2}{*}{\textbf{Model}} & "
        r"\multicolumn{2}{c}{\textbf{NeuroVoz}} & "
        r"\multicolumn{2}{c}{\textbf{TORGO}} & "
        r"\multicolumn{2}{c}{\textbf{TORGO (SW)}} & "
        r"\multicolumn{2}{c}{\textbf{TORGO (MW)}} \\"
    )
    lines.append(
        r" & Raw & +PP & Raw & +PP & Raw & +PP & Raw & +PP \\"
    )
    lines.append(r"\midrule")

    for name, values in rows:
        cells = [name]
        for ci, v in enumerate(values):
            s = fmt(v)
            if v is not None and best[ci] is not None and abs(v - best[ci]) < 0.005:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Voxtral results table")
    parser.add_argument("--latex", action="store_true", help="Also generate LaTeX table")
    parser.add_argument("--output", type=str, default=None, help="Save Markdown table to file")
    args = parser.parse_args()

    rows = build_rows()

    md = generate_markdown(rows)
    print(md)

    if args.latex:
        print("\n\n% LaTeX version:\n")
        latex = generate_latex(rows)
        print(latex)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(md + "\n")
            if args.latex:
                f.write("\n\n% LaTeX version:\n\n")
                f.write(generate_latex(rows) + "\n")
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
