#!/usr/bin/env python3
"""Generate pathological speech questions JSON and symlink audio files."""

import csv
import json
import os
import random
from pathlib import Path

random.seed(42)

# Repo root (two levels up from src/scripts/). Override with REPO_ROOT if needed.
BASE_DIR = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT_DIR = BASE_DIR / "pathological-speech-questions"
AUDIO_DIR = OUTPUT_DIR / "audio"

NEUROVOZ_TSV = BASE_DIR / "data" / "combined_neurovoz_torgo_cv" / "test_neurovoz.tsv"
TORGO_TSV = BASE_DIR / "data" / "torgo" / "test_full_paths.tsv"

SEX_CHOICES = ["Male", "Female", "I do not know"]
AGE_CHOICES = ["Less than 20 years old", "Between 20 and 40 years old",
               "Between 40 and 60 years old", "More than 60 years old"]


def age_to_range(age: int) -> str:
    """Map numeric age to the corresponding range label."""
    if age < 20:
        return "Less than 20 years old"
    elif age <= 40:
        return "Between 20 and 40 years old"
    elif age <= 60:
        return "Between 40 and 60 years old"
    else:
        return "More than 60 years old"


def shuffled(choices: list) -> list:
    """Return a shuffled copy of the choices list."""
    c = choices.copy()
    random.shuffle(c)
    return c


def read_tsv(path: Path, source_default: str = None):
    """Read a TSV file and yield rows as dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if source_default and "dataset_source" not in row:
                row["dataset_source"] = source_default
            yield row


def make_audio_filename(sample_id: str) -> str:
    """Create a unique audio filename from the sample id."""
    return f"{sample_id}.wav"


def main():
    # Create output dirs
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    questions = []

    # --- Process NeuroVoz ---
    for row in read_tsv(NEUROVOZ_TSV):
        sample_id = row["sample_id"]
        audio_src = row["audio_path"]
        sex = row["sex"]
        age_str = row["age"]
        source = row.get("dataset_source", "neurovoz")

        audio_filename = make_audio_filename(sample_id)
        audio_rel = f"./audio/{audio_filename}"

        # Symlink audio
        dst = AUDIO_DIR / audio_filename
        if not dst.exists():
            os.symlink(audio_src, dst)

        # Sex question (skip Unknown)
        if sex in ("Male", "Female"):
            questions.append({
                "id": f"{sample_id}_sex",
                "audio_path": audio_rel,
                "question": "What is the sex of the speaker?",
                "choices": shuffled(SEX_CHOICES),
                "answer": sex,
                "modality": "speech",
                "category": "Semantic Layer",
                "sub-category": "Speaker Analysis",
                "language": "es",
                "source": source,
            })

        # Age question (skip unknown age = -1)
        age = int(age_str)
        if age > 0:
            answer = age_to_range(age)
            questions.append({
                "id": f"{sample_id}_age",
                "audio_path": audio_rel,
                "question": "What is the age range of the speaker?",
                "choices": shuffled(AGE_CHOICES),
                "answer": answer,
                "modality": "speech",
                "category": "Semantic Layer",
                "sub-category": "Speaker Analysis",
                "language": "es",
                "source": source,
            })

    # --- Process TORGO ---
    for row in read_tsv(TORGO_TSV, source_default="torgo"):
        sample_id = row["sample_id"]
        audio_src = row["audio_path"]
        sex = row["sex"]
        source = row.get("dataset_source", "torgo")

        audio_filename = make_audio_filename(sample_id)
        audio_rel = f"./audio/{audio_filename}"

        # Symlink audio
        dst = AUDIO_DIR / audio_filename
        if not dst.exists():
            os.symlink(audio_src, dst)

        # Sex question only (TORGO has no age data)
        if sex in ("Male", "Female"):
            questions.append({
                "id": f"{sample_id}_sex",
                "audio_path": audio_rel,
                "question": "What is the sex of the speaker?",
                "choices": shuffled(SEX_CHOICES),
                "answer": sex,
                "modality": "speech",
                "category": "Semantic Layer",
                "sub-category": "Speaker Analysis",
                "language": "en",
                "source": source,
            })

    # Shuffle all questions to avoid dataset-order bias
    random.shuffle(questions)

    # Write JSON
    output_json = OUTPUT_DIR / "questions.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=4, ensure_ascii=False)

    # Summary
    sex_qs = sum(1 for q in questions if "sex" in q["id"])
    age_qs = sum(1 for q in questions if "age" in q["id"])
    neurovoz_qs = sum(1 for q in questions if q["source"] == "neurovoz")
    torgo_qs = sum(1 for q in questions if q["source"] == "torgo")
    print(f"Generated {len(questions)} questions ({sex_qs} sex, {age_qs} age)")
    print(f"  NeuroVoz: {neurovoz_qs} questions")
    print(f"  TORGO:    {torgo_qs} questions")
    print(f"  Audio symlinks in: {AUDIO_DIR}")
    print(f"  JSON written to:   {output_json}")


if __name__ == "__main__":
    main()
