#!/usr/bin/env python3
"""Measure random-selection baseline performance on the pathological speech questions."""

import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)
NUM_RUNS = 10000  # Monte Carlo runs for stable estimates

# questions.json is written to the repo root by generate_questions.py
QUESTIONS_PATH = Path(__file__).resolve().parents[2] / "pathological-speech-questions" / "questions.json"

def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    # Group questions by type
    groups = defaultdict(list)
    for q in questions:
        qtype = "sex" if q["id"].endswith("_sex") else "age"
        groups[qtype].append(q)
        groups["all"].append(q)
        groups[f"{qtype}_{q['source']}"].append(q)

    print(f"Total questions: {len(questions)}")
    print(f"  Sex questions:  {len(groups['sex'])}")
    print(f"  Age questions:  {len(groups['age'])}")
    print()

    # Analytical expected accuracy (1 / num_choices per question)
    print("=" * 60)
    print("ANALYTICAL RANDOM BASELINE (expected accuracy)")
    print("=" * 60)
    for group_name in sorted(groups):
        group_qs = groups[group_name]
        expected = sum(1.0 / len(q["choices"]) for q in group_qs) / len(group_qs)
        print(f"  {group_name:25s}: {expected*100:6.2f}% ({len(group_qs)} questions)")
    print()

    # Monte Carlo simulation
    print("=" * 60)
    print(f"MONTE CARLO RANDOM BASELINE ({NUM_RUNS} runs)")
    print("=" * 60)
    for group_name in sorted(groups):
        group_qs = groups[group_name]
        run_accs = []
        for _ in range(NUM_RUNS):
            correct = sum(
                1 for q in group_qs
                if random.choice(q["choices"]) == q["answer"]
            )
            run_accs.append(correct / len(group_qs))
        mean_acc = sum(run_accs) / len(run_accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in run_accs) / len(run_accs)) ** 0.5
        print(f"  {group_name:25s}: {mean_acc*100:6.2f}% ± {std_acc*100:.2f}%")

    # Distribution of answers (to show class imbalance)
    print()
    print("=" * 60)
    print("ANSWER DISTRIBUTION")
    print("=" * 60)
    for qtype in ("sex", "age"):
        print(f"\n  {qtype.upper()} questions:")
        answer_counts = defaultdict(int)
        for q in groups[qtype]:
            answer_counts[q["answer"]] += 1
        total = len(groups[qtype])
        for answer, count in sorted(answer_counts.items(), key=lambda x: -x[1]):
            print(f"    {answer:40s}: {count:5d} ({count/total*100:5.1f}%)")

    # Majority-class baseline
    print()
    print("=" * 60)
    print("MAJORITY-CLASS BASELINE")
    print("=" * 60)
    for qtype in ("sex", "age"):
        answer_counts = defaultdict(int)
        for q in groups[qtype]:
            answer_counts[q["answer"]] += 1
        majority = max(answer_counts, key=answer_counts.get)
        majority_acc = answer_counts[majority] / len(groups[qtype])
        print(f"  {qtype:25s}: {majority_acc*100:6.2f}% (always predict '{majority}')")


if __name__ == "__main__":
    main()
