#!/usr/bin/env python3
"""Evaluate MCQA accuracy from Voxtral QA inference output using exact match."""

import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path

CHOICE_LETTERS = list(string.ascii_uppercase)


def exact_match(answer: str, prediction: str, choices: list[str] | None = None) -> bool:
    """Check if prediction matches the answer after stripping choice letters.

    If the prediction is only a choice letter like ``(A)`` and *choices* is
    provided, the letter is resolved to the corresponding choice text before
    comparing.
    """
    prediction = str(prediction).strip()
    answer = str(answer).strip()

    # Try to resolve a bare choice letter, e.g. "(A)" or "(B)"
    letter_match = re.match(r'^\s*\(([a-zA-Z])\)\s*$', prediction)
    if letter_match and choices:
        idx = CHOICE_LETTERS.index(letter_match.group(1).upper())
        if idx < len(choices):
            prediction = choices[idx]

    # Strip leading choice letter prefix like "(A) Male" -> "Male"
    prediction = re.sub(r'^\s*\([a-zA-Z]\)\s*', '', prediction).strip()

    return answer.lower() == prediction.lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA exact-match accuracy")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the inference output JSON (with model_output field)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to write evaluation summary JSON",
    )
    return parser.parse_args()


def evaluate_group(items: list) -> dict:
    """Compute accuracy for a group of items."""
    correct = sum(
        1 for q in items
        if exact_match(q["answer"], q["model_output"], q.get("choices"))
    )
    total = len(items)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def main():
    args = parse_args()

    with open(args.input, encoding="utf-8") as f:
        results = json.load(f)

    # Group by type and source
    groups = defaultdict(list)
    for q in results:
        qtype = "sex" if q["id"].endswith("_sex") else "age"
        groups["all"].append(q)
        groups[qtype].append(q)
        groups[f"{qtype}_{q['source']}"].append(q)

    # Print results
    print("=" * 60)
    print("MCQA EXACT-MATCH ACCURACY")
    print("=" * 60)

    summary = {}
    for group_name in sorted(groups):
        metrics = evaluate_group(groups[group_name])
        summary[group_name] = metrics
        print(
            f"  {group_name:25s}: {metrics['accuracy']*100:6.2f}%"
            f"  ({metrics['correct']}/{metrics['total']})"
        )

    # Print some example predictions
    print()
    print("=" * 60)
    print("SAMPLE PREDICTIONS (first 10)")
    print("=" * 60)
    for q in results[:10]:
        match = exact_match(q["answer"], q["model_output"], q.get("choices"))
        status = "OK" if match else "WRONG"
        print(f"  [{status:5s}] answer=\"{q['answer']}\"  pred=\"{q['model_output']}\"")

    # Save summary
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    main()
