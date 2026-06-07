#!/usr/bin/env python3
"""
Merge Common Voice Spanish with the existing NeuroVoz + TORGO combined dataset.

This script:
1. Reads Common Voice Spanish TSV + clips (standard CV format).
2. Filters a target number of hours from CV train and dev splits.
3. Loads the existing combined_neurovoz_torgo HF dataset.
4. Merges everything into a new dataset at data/combined_neurovoz_torgo_cv.

The output dataset has the same schema as combined_neurovoz_torgo:
  - Splits: train, validation, test_neurovoz, test_torgo
  - Columns: audio, transcription, speaker_id, sample_id, duration,
             audio_path, sex, age, speech_type, dataset_source

Common Voice samples are labelled:
  - dataset_source = "commonvoice_es"
  - speech_type    = "HC"
  - age            = mapped from CV age bands or -1 if unknown
  - sex            = mapped from CV gender field or "Unknown"

Usage:
  python src/scripts/merge_cv_neurovoz_torgo.py \\
      --cv_dir data/commonvoice_es/cv-corpus-24.0-2025-12-03/es \\
      --target_hours 10.0

If you're unsure of the exact CV path, run:
  find data/commonvoice_es -name "train.tsv" -path "*/es/*"
"""

import os
import argparse
import random
import pandas as pd
from datasets import (
    load_from_disk,
    Dataset,
    DatasetDict,
    Audio,
    concatenate_datasets,
)


# ── CV age-band → approximate numeric age ─────────────────────────────────────
CV_AGE_MAP = {
    "teens": 16,
    "twenties": 25,
    "thirties": 35,
    "fourties": 45,
    "fifties": 55,
    "sixties": 65,
    "seventies": 75,
    "eighties": 85,
    "nineties": 95,
}

# CV gender → sex
CV_GENDER_MAP = {
    "male_masculine": "Male",
    "male": "Male",
    "female_feminine": "Female",
    "female": "Female",
}


def _pick_duration_reader():
    """Return the fastest available function to read MP3 duration in seconds."""
    # 1. mutagen – reads MP3 headers only, fastest
    try:
        from mutagen.mp3 import MP3

        def _dur_mutagen(path):
            try:
                return MP3(path).info.length
            except Exception:
                return 0.0

        print("    Using mutagen for MP3 duration reading (fast).")
        return _dur_mutagen
    except ImportError:
        pass

    # 2. torchaudio.info – no full decode
    try:
        import torchaudio

        def _dur_torchaudio(path):
            try:
                info = torchaudio.info(path)
                return info.num_frames / info.sample_rate
            except Exception:
                return 0.0

        print("    Using torchaudio for MP3 duration reading.")
        return _dur_torchaudio
    except ImportError:
        pass

    # 3. librosa – slowest but always works
    import librosa

    def _dur_librosa(path):
        try:
            return librosa.get_duration(filename=path)
        except Exception:
            return 0.0

    print("    Using librosa for MP3 duration reading (slow, consider: pip install mutagen).")
    return _dur_librosa


def load_cv_split(cv_dir: str, split_name: str) -> pd.DataFrame:
    """Load a Common Voice TSV split and normalise columns."""
    tsv_path = os.path.join(cv_dir, f"{split_name}.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"CV TSV not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # Build absolute audio path
    clips_dir = os.path.join(cv_dir, "clips")
    df["audio_path"] = df["path"].apply(lambda p: os.path.join(clips_dir, p))

    # Keep only rows whose audio file actually exists (safety check for a sample)
    # We skip the full existence check for speed; the HF Audio feature will
    # raise an error at load time if a file is missing.

    # Map columns to our schema
    df["transcription"] = df["sentence"]
    df["speaker_id"] = df["client_id"].str[:12]  # shorten hash
    df["sample_id"] = "cv_" + df["path"].str.replace(".mp3", "", regex=False)
    df["sex"] = df["gender"].str.lower().map(CV_GENDER_MAP).fillna("Unknown")
    df["age"] = (
        df["age"].str.lower().map(CV_AGE_MAP).fillna(-1).astype(int)
    )
    df["speech_type"] = "HC"
    df["dataset_source"] = "commonvoice_es"

    return df


def filter_hours(
    df: pd.DataFrame,
    target_seconds: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Select a random subset of rows that sum to approximately target_seconds.

    We need to compute durations first (CV doesn't provide them in the TSV),
    so we use a heuristic: pick random rows until we hit the budget.
    If the TSV already contains a 'duration' column we use it, otherwise we
    fall back to estimating from file size or simply selecting greedily.
    """
    if "duration" not in df.columns or df["duration"].isna().all():
        # CV doesn't include duration; compute on-the-fly while selecting.
        # We shuffle first, then compute durations only for rows we need
        # (avoids reading 300K+ files when we only need ~10h ≈ a few thousand).
        print("    Shuffling and computing durations on-the-fly...")
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Pick the fastest available MP3 duration reader
        _get_duration = _pick_duration_reader()

        durations = []
        cumulative = 0.0
        cutoff_idx = len(df) - 1
        for i, path in enumerate(df["audio_path"]):
            dur = _get_duration(path)
            durations.append(dur)
            cumulative += dur
            if cumulative >= target_seconds:
                cutoff_idx = i
                break
            if i % 2000 == 0 and i > 0:
                print(f"      {i} files scanned, {cumulative/3600:.2f}h accumulated...")

        # We only computed durations up to cutoff_idx
        df = df.iloc[: cutoff_idx + 1].copy()
        df["duration"] = durations
        df = df[df["duration"] > 0].copy()
        return df

    # If durations already exist, use standard approach
    df = df[df["duration"] > 0].copy()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cumsum = df["duration"].cumsum()
    cutoff = cumsum.searchsorted(target_seconds, side="right")
    selected = df.iloc[: cutoff + 1].copy()

    return selected


def cv_df_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Convert a filtered CV DataFrame to an HF Dataset with Audio feature."""
    keep_cols = [
        "sample_id",
        "audio_path",
        "transcription",
        "speaker_id",
        "duration",
        "sex",
        "age",
        "speech_type",
        "dataset_source",
    ]
    df = df[keep_cols].reset_index(drop=True)

    # Create the HF dataset
    ds = Dataset.from_pandas(df)

    # Add audio column from file paths
    ds = ds.add_column("audio", ds["audio_path"])
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Merge Common Voice Spanish with NeuroVoz + TORGO."
    )
    parser.add_argument(
        "--cv_dir",
        type=str,
        required=True,
        help="Path to the extracted CV Spanish directory (the one containing "
             "train.tsv, dev.tsv, clips/, etc.).",
    )
    parser.add_argument(
        "--existing_dataset",
        type=str,
        default="data/combined_neurovoz_torgo",
        help="Path to the existing NeuroVoz+TORGO HF dataset on disk.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/combined_neurovoz_torgo_cv",
        help="Output directory for the new merged dataset.",
    )
    parser.add_argument(
        "--target_hours",
        type=float,
        default=7.31,
        help="Total hours to sample from Common Voice (split across train + val). "
             "Default 7.31h balances Spanish to match TORGO's 9.42h in train+val.",
    )
    parser.add_argument(
        "--train_val_ratio",
        type=float,
        default=0.7632,
        help="Fraction of CV hours allocated to train (rest goes to validation). "
             "Default 0.7632 matches TORGO's train/val ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MERGE COMMON VOICE (ES) + NEUROVOZ + TORGO")
    print("=" * 70)

    # ── Step 1: Load & filter Common Voice ────────────────────────────────────
    target_train_sec = args.target_hours * 3600 * args.train_val_ratio
    target_val_sec = args.target_hours * 3600 * (1 - args.train_val_ratio)

    print(f"\nStep 1: Loading Common Voice Spanish from {args.cv_dir}")
    print(f"  Target: {args.target_hours}h total "
          f"({target_train_sec/3600:.1f}h train, {target_val_sec/3600:.1f}h val)")

    cv_train_df = load_cv_split(args.cv_dir, "train")
    print(f"  CV train loaded: {len(cv_train_df)} samples")

    cv_val_df = load_cv_split(args.cv_dir, "dev")
    print(f"  CV dev loaded:   {len(cv_val_df)} samples")

    print("\nStep 2: Filtering Common Voice to target hours...")
    cv_train_filtered = filter_hours(cv_train_df, target_train_sec, seed=args.seed)
    print(f"  CV train filtered: {len(cv_train_filtered)} samples, "
          f"{cv_train_filtered['duration'].sum()/3600:.2f}h")

    cv_val_filtered = filter_hours(cv_val_df, target_val_sec, seed=args.seed + 1)
    print(f"  CV val filtered:   {len(cv_val_filtered)} samples, "
          f"{cv_val_filtered['duration'].sum()/3600:.2f}h")

    # ── Step 3: Convert to HF Datasets ────────────────────────────────────────
    print("\nStep 3: Converting CV to HF Dataset format...")
    cv_train_ds = cv_df_to_hf_dataset(cv_train_filtered)
    cv_val_ds = cv_df_to_hf_dataset(cv_val_filtered)
    print(f"  CV train HF dataset: {len(cv_train_ds)} samples")
    print(f"  CV val HF dataset:   {len(cv_val_ds)} samples")

    # ── Step 4: Load existing dataset ─────────────────────────────────────────
    print(f"\nStep 4: Loading existing dataset from {args.existing_dataset}...")
    existing_ds = load_from_disk(args.existing_dataset)
    for split_name in existing_ds:
        print(f"  {split_name}: {len(existing_ds[split_name])} samples")

    # ── Step 5: Merge ─────────────────────────────────────────────────────────
    print("\nStep 5: Merging datasets...")

    merged_train = concatenate_datasets([existing_ds["train"], cv_train_ds])
    print(f"  Merged train: {len(merged_train)} samples")

    merged_val = concatenate_datasets([existing_ds["validation"], cv_val_ds])
    print(f"  Merged validation: {len(merged_val)} samples")

    merged_dataset = DatasetDict({
        "train": merged_train,
        "validation": merged_val,
        "test_neurovoz": existing_ds["test_neurovoz"],
        "test_torgo": existing_ds["test_torgo"],
    })

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    print(f"\nStep 6: Saving merged dataset to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_dataset.save_to_disk(args.output_dir)

    # ── Step 7: Create TSV files ──────────────────────────────────────────────
    print("\nStep 7: Creating TSV files...")
    tsv_columns = [
        "sample_id", "audio_path", "transcription", "speaker_id",
        "duration", "sex", "age", "speech_type", "dataset_source",
    ]
    for split_name in merged_dataset:
        split_data = merged_dataset[split_name]
        df = split_data.to_pandas()
        df_tsv = df[[c for c in tsv_columns if c in df.columns]]
        out_path = os.path.join(args.output_dir, f"{split_name}.tsv")
        df_tsv.to_csv(out_path, sep="\t", index=False)
        print(f"  {out_path}: {len(df_tsv)} samples")

    # ── Step 8: Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MERGED DATASET SUMMARY")
    print("=" * 70)

    for split_name, split_data in merged_dataset.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total samples: {len(split_data):,}")
        if "dataset_source" in split_data.column_names:
            sources = pd.Series(split_data["dataset_source"]).value_counts()
            for src, cnt in sources.items():
                print(f"    {src}: {cnt:,} samples ({cnt/len(split_data)*100:.1f}%)")
        if "duration" in split_data.column_names:
            total_dur = sum(split_data["duration"])
            print(f"  Duration: {total_dur/3600:.2f}h")
        if "speaker_id" in split_data.column_names:
            print(f"  Unique speakers: {len(set(split_data['speaker_id']))}")

    # Language balance summary
    print("\n" + "-" * 70)
    print("LANGUAGE BALANCE (train + validation)")
    print("-" * 70)
    train_df = merged_dataset["train"].to_pandas()
    val_df = merged_dataset["validation"].to_pandas()
    all_tv = pd.concat([train_df, val_df])

    spanish_mask = all_tv["dataset_source"].isin(["neurovoz", "commonvoice_es"])
    english_mask = all_tv["dataset_source"] == "torgo"

    esp_dur = all_tv.loc[spanish_mask, "duration"].sum() / 3600
    eng_dur = all_tv.loc[english_mask, "duration"].sum() / 3600
    print(f"  Spanish (NeuroVoz + CV): {esp_dur:.2f}h")
    print(f"  English (TORGO):         {eng_dur:.2f}h")
    print(f"  Ratio (ES/EN):           {esp_dur/eng_dur:.2f}")

    print("\n" + "=" * 70)
    print("✓ DATASET MERGING COMPLETE!")
    print("=" * 70)
    print(f"\nMerged dataset: {args.output_dir}")
    print("Splits: train, validation, test_neurovoz, test_torgo")


if __name__ == "__main__":
    main()
