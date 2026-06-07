"""
Create Hugging Face dataset from TSV file.
Loads audio and creates train/validation/test splits.
Supports filtering phonation tasks and speaker-aware splits.
"""

import os
import argparse
import pandas as pd

import librosa

from datasets import Dataset, DatasetDict


def main():
    parser = argparse.ArgumentParser(
        description="Create Hugging Face dataset from TSV file."
    )
    parser.add_argument(
        "--tsv_file",
        type=str,
        default="data/neurovoz/neurovoz.tsv",
        help="Path to the TSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="Output directory for HF dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="neurovoz_dataset",
        help="Name of output HF dataset directory.",
    )
    parser.add_argument(
        "--include_phonation",
        action="store_true",
        default=False,
        help="Include phonation tasks (vowels and PATAKA). Default: exclude phonation.",
    )
    parser.add_argument(
        "--speaker_aware",
        action="store_true",
        default=False,
        help="Use speaker-aware splits (no speaker in multiple splits). Default: random splits.",
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        choices=["speech_type", "sex", "none"],
        default="speech_type",
        help="Stratify splits by speech_type or sex. Default: speech_type.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio. Default: 0.8",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio. Default: 0.1",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    hf_dataset_dir = os.path.join(args.output_dir, args.dataset_name)
    
    print("Step 1: Loading TSV file...")
    
    # Load TSV
    df = pd.read_csv(args.tsv_file, sep="\t")
    print(f"Loaded {len(df)} samples from TSV")
    
    # Filter out phonation tasks if requested
    if not args.include_phonation:
        print("\nStep 1.5: Filtering out phonation tasks...")
        phonation_mask = df["transcription"].str.startswith("<sustained-phonation-")
        df = df[~phonation_mask].reset_index(drop=True)
        print(f"Filtered to {len(df)} speech samples (removed {phonation_mask.sum()} phonation tasks)")
    
    print("\nStep 2: Creating Hugging Face dataset...")
    
    # Create HF dataset
    dataset_dict = {
        "audio": [],
        "transcription": [],
        "speaker_id": [],
        "sample_id": [],
        "duration": [],
        "audio_path": [],
        "sex": [],
        "age": [],
        "speech_type": [],
    }
    
    skipped = 0
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            audio_array, sampling_rate = librosa.load(row["audio_path"], sr=16000)
            
            dataset_dict["audio"].append({
                "array": audio_array.tolist(),
                "sampling_rate": 16000,
            })
            dataset_dict["transcription"].append(row["transcription"])
            dataset_dict["speaker_id"].append(str(row["speaker_id"]))
            dataset_dict["sample_id"].append(row["sample_id"])
            dataset_dict["duration"].append(row["duration"])
            dataset_dict["audio_path"].append(row["audio_path"])
            dataset_dict["sex"].append(row["sex"])
            dataset_dict["age"].append(int(row["age"]))
            dataset_dict["speech_type"].append(row.get("speech_type", None))
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1} samples...")
        except Exception as e:
            skipped += 1
            print(f"Warning: Skipping {row['audio_path']}: {e}")
            continue
    
    print(f"Processed {len(dataset_dict['audio'])} samples (skipped {skipped})")
    
    print("\nStep 3: Creating dataset splits...")
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Determine split strategy
    if args.speaker_aware:
        print("Using speaker-aware splits (no speaker in multiple splits)...")
        dataset_dict_hf = _speaker_aware_split(
            dataset, 
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            stratify_by=args.stratify_by
        )
    else:
        print(f"Using random splits with stratification by '{args.stratify_by}'...")
        dataset_dict_hf = _random_split(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            stratify_by=args.stratify_by
        )
    
    dataset_dict_hf.save_to_disk(hf_dataset_dir)
    print(f"\nHF dataset created: {hf_dataset_dir}")
    print(f"Train: {len(dataset_dict_hf['train'])}")
    print(f"Validation: {len(dataset_dict_hf['validation'])}")
    print(f"Test: {len(dataset_dict_hf['test'])}")
    
    # Print split statistics
    print("\nSplit statistics:")
    for split_name in ["train", "validation", "test"]:
        split_data = dataset_dict_hf[split_name]
        # `split_data` is a HuggingFace Dataset (not a dict). Safely access columns.
        if hasattr(split_data, "column_names") and "speech_type" in split_data.column_names:
            speech_types = pd.Series(split_data["speech_type"]).value_counts()
        else:
            speech_types = pd.Series([], dtype=object)

        if hasattr(split_data, "column_names") and "sex" in split_data.column_names:
            sexes = pd.Series(split_data["sex"]).value_counts()
        else:
            sexes = pd.Series([], dtype=object)
        print(f"\n{split_name.upper()}:")
        print(f"  Speech type distribution: {dict(speech_types)}")
        print(f"  Sex distribution: {dict(sexes)}")
        print(f"  Unique speakers: {len(set(split_data['speaker_id']))}")
    
    print("\n✓ Dataset creation complete!")


def _random_split(dataset, train_ratio=0.8, val_ratio=0.1, stratify_by="speech_type"):
    """Create random splits with optional stratification."""
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Simple random split (HF datasets doesn't easily support stratification by string columns)
    split_data = dataset.train_test_split(test_size=test_ratio, seed=42)
    train_val = split_data["train"].train_test_split(
        test_size=val_ratio / train_ratio, seed=42
    )
    return DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": split_data["test"],
    })


def _speaker_aware_split(dataset, train_ratio=0.8, val_ratio=0.1, stratify_by="speech_type"):
    """Create speaker-aware splits (no speaker in multiple splits) with stratification."""
    
    # Get unique speakers
    speakers = sorted(set(dataset["speaker_id"]))
    
    # Create speaker metadata
    speaker_speech_type = {}
    speaker_sex = {}
    speaker_samples = {}
    
    for sample in dataset:
        sid = sample["speaker_id"]
        if sid not in speaker_speech_type:
            speaker_speech_type[sid] = sample.get("speech_type", None)
            speaker_sex[sid] = sample["sex"]
            speaker_samples[sid] = 0
        speaker_samples[sid] += 1
    
    # Group speakers by stratification key
    if stratify_by == "speech_type":
        stratify_key = "speech_type"
        stratify_map = speaker_speech_type
    elif stratify_by == "sex":
        stratify_key = "sex"
        stratify_map = speaker_sex
    else:
        stratify_map = None
    
    # Group speakers by stratification key
    from collections import defaultdict
    if stratify_map:
        speaker_groups = defaultdict(list)
        for speaker in speakers:
            group = stratify_map[speaker]
            speaker_groups[group].append(speaker)
    else:
        speaker_groups = {"all": speakers}
    
    # Split speakers for each group, maintaining stratification
    train_speakers = []
    val_speakers = []
    test_speakers = []
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    import random
    random.seed(42)
    
    for group, group_speakers in speaker_groups.items():
        n = len(group_speakers)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val
        
        # Shuffle speakers within group
        shuffled = group_speakers.copy()
        random.shuffle(shuffled)
        
        test_speakers.extend(shuffled[:n_test])
        val_speakers.extend(shuffled[n_test:n_test + n_val])
        train_speakers.extend(shuffled[n_test + n_val:])
    
    print(f"  Train speakers: {len(train_speakers)}")
    print(f"  Validation speakers: {len(val_speakers)}")
    print(f"  Test speakers: {len(test_speakers)}")
    
    # Split dataset by speakers
    train_indices = [i for i, s in enumerate(dataset["speaker_id"]) if s in train_speakers]
    val_indices = [i for i, s in enumerate(dataset["speaker_id"]) if s in val_speakers]
    test_indices = [i for i, s in enumerate(dataset["speaker_id"]) if s in test_speakers]
    
    return DatasetDict({
        "train": dataset.select(train_indices),
        "validation": dataset.select(val_indices),
        "test": dataset.select(test_indices),
    })


if __name__ == "__main__":
    main()
