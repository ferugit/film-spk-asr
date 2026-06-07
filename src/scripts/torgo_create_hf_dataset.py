"""
Create Hugging Face dataset with train/validation/test splits for TORGO.
Uses speaker-aware splits to ensure no speaker appears in multiple splits.
Loads directly from the original HF dataset to skip audio processing.
"""

import os
import argparse
import pandas as pd

from datasets import load_dataset, Dataset, DatasetDict


def main():
    parser = argparse.ArgumentParser(
        description="Create Hugging Face dataset with splits for TORGO."
    )
    parser.add_argument(
        "--tsv_file",
        type=str,
        default="data/torgo/torgo.tsv",
        help="Path to the TSV file with metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/torgo",
        help="Output directory for HF dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="torgo_dataset",
        help="Name of output HF dataset directory.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="abnerh/TORGO-database",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/",
        help="Cache directory for HF dataset.",
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
    
    print("Step 1: Loading original HF dataset...")
    
    # Load original HF dataset
    original_dataset = load_dataset(args.hf_dataset, cache_dir=args.cache_dir)
    torgo_train = original_dataset["train"]
    print(f"Loaded {len(torgo_train)} samples from HF dataset")
    
    print("\nStep 2: Loading TSV metadata...")
    
    # Load TSV with metadata
    df = pd.read_csv(args.tsv_file, sep="\t")
    print(f"Loaded {len(df)} samples from TSV")
    
    # Convert the original dataset to pandas to easily access audio paths
    print("\nStep 3: Converting HF dataset to pandas...")
    torgo_df = torgo_train.to_pandas()
    
    # Create a mapping from audio filename to metadata
    metadata_map = {}
    for _, row in df.iterrows():
        filename = row["audio_filename"]
        metadata_map[filename] = {
            "sample_id": row["sample_id"],
            "speaker_id": row["speaker_id"],
            "sex": row["sex"],
            "age": int(row["age"]),
            "speech_type": row["speech_type"],
            "audio_path": row["audio_path"],
        }
    
    print("\nStep 4: Enriching dataset with metadata...")
    
    # Add metadata columns to the dataset
    sample_ids = []
    speaker_ids = []
    sexes = []
    ages = []
    speech_types = []
    audio_paths = []
    
    for _, row in torgo_df.iterrows():
        audio_path = row["audio"]["path"]
        filename = os.path.basename(audio_path)
        
        if filename in metadata_map:
            meta = metadata_map[filename]
            sample_ids.append(meta["sample_id"])
            speaker_ids.append(meta["speaker_id"])
            sexes.append(meta["sex"])
            ages.append(meta["age"])
            speech_types.append(meta["speech_type"])
            audio_paths.append(meta["audio_path"])
        else:
            # Fallback: extract from filename
            speaker_id = filename.split("_")[0]
            sample_ids.append(f"torgo_{os.path.splitext(filename)[0]}")
            speaker_ids.append(speaker_id)
            sexes.append("Unknown")
            ages.append(-1)
            speech_types.append("Unknown")
            audio_paths.append(audio_path)
    
    # Add new columns to the dataset
    enriched_dataset = torgo_train.add_column("sample_id", sample_ids)
    enriched_dataset = enriched_dataset.add_column("speaker_id", speaker_ids)
    enriched_dataset = enriched_dataset.add_column("sex", sexes)
    enriched_dataset = enriched_dataset.add_column("age", ages)
    enriched_dataset = enriched_dataset.add_column("speech_type", speech_types)
    enriched_dataset = enriched_dataset.add_column("audio_path", audio_paths)
    
    print(f"Enriched dataset with {len(enriched_dataset)} samples")
    
    print("\nStep 5: Creating speaker-aware splits...")
    
    dataset_dict_hf = _speaker_aware_split(
        enriched_dataset, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stratify_by=args.stratify_by
    )
    
    print("\nStep 6: Saving dataset...")
    
    dataset_dict_hf.save_to_disk(hf_dataset_dir)
    print(f"\nHF dataset created: {hf_dataset_dir}")
    print(f"Train: {len(dataset_dict_hf['train'])}")
    print(f"Validation: {len(dataset_dict_hf['validation'])}")
    print(f"Test: {len(dataset_dict_hf['test'])}")
    
    # Print split statistics
    print("\nSplit statistics:")
    for split_name in ["train", "validation", "test"]:
        split_data = dataset_dict_hf[split_name]
        
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
    
    # Group speakers by stratification key AND sex for gender-balanced splits
    from collections import defaultdict
    
    # Group speakers by sex first to ensure gender balance
    male_speakers = [s for s in speakers if speaker_sex[s] == 'Male']
    female_speakers = [s for s in speakers if speaker_sex[s] == 'Female']
    
    print(f"  Male speakers: {len(male_speakers)}, Female speakers: {len(female_speakers)}")
    
    # Now stratify within each sex group
    def stratify_speakers_by_group(speaker_list, stratify_map):
        """Group speakers by stratification key."""
        groups = defaultdict(list)
        if stratify_map:
            for speaker in speaker_list:
                group = stratify_map[speaker]
                groups[group].append(speaker)
        else:
            groups["all"] = speaker_list
        return groups
    
    male_groups = stratify_speakers_by_group(male_speakers, stratify_map)
    female_groups = stratify_speakers_by_group(female_speakers, stratify_map)
    
    # Split speakers for each group, maintaining stratification AND gender balance
    train_speakers = []
    val_speakers = []
    test_speakers = []
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    import random
    random.seed(42)
    
    # Split male speakers
    for group, group_speakers in male_groups.items():
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
    
    # Split female speakers
    for group, group_speakers in female_groups.items():
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
    
    print(f"  Train speakers: {len(train_speakers)} - {train_speakers}")
    print(f"  Validation speakers: {len(val_speakers)} - {val_speakers}")
    print(f"  Test speakers: {len(test_speakers)} - {test_speakers}")
    
    # Verify gender balance
    train_male = sum(1 for s in train_speakers if speaker_sex[s] == 'Male')
    train_female = sum(1 for s in train_speakers if speaker_sex[s] == 'Female')
    val_male = sum(1 for s in val_speakers if speaker_sex[s] == 'Male')
    val_female = sum(1 for s in val_speakers if speaker_sex[s] == 'Female')
    test_male = sum(1 for s in test_speakers if speaker_sex[s] == 'Male')
    test_female = sum(1 for s in test_speakers if speaker_sex[s] == 'Female')
    
    print(f"  Train: {train_male} male, {train_female} female")
    print(f"  Validation: {val_male} male, {val_female} female")
    print(f"  Test: {test_male} male, {test_female} female")
    
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
