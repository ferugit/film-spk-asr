"""
In this script, we merge multiple Hugging Face datasets into a single dataset.

Especially useful for combining NeuroVoz and TORGO datasets for diverse speech understanding tasks.

We take train/val splits from each dataset and concatenate them to form unified splits.

For the test, we maintain separate test sets for each dataset to evaluate performance individually.

Datasets:
- NeuroVoz: data/neurovoz/neurovoz_dataset
- TORGO: data/torgo/torgo_dataset

are merged into:
- Merged Dataset: data/combined_neurovoz_torgo_dataset

After running this script, the merged dataset will have the following structure:
data/combined_neurovoz_torgo_dataset/
    ├── train/
    ├── validation/
    ├── test_neurovoz/
    └── test_torgo/

The script also prints out the number of samples in each split for verification.
Also prints the distributions of samples per dataset in each split.

"""

import os
import argparse
import pandas as pd
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Audio


def normalize_columns(dataset, dataset_name):
    """
    Normalize column names and add dataset source identifier.
    
    Common columns: audio, transcription, speaker_id, sample_id, duration, 
                   audio_path, sex, age, speech_type
    """
    # Add dataset source column
    dataset = dataset.add_column("dataset_source", [dataset_name] * len(dataset))
    
    # Rename columns if needed to match common schema
    if "gender" in dataset.column_names:
        # TORGO uses 'gender' instead of 'sex' (already has 'sex' added during creation)
        # Remove redundant 'gender' column
        dataset = dataset.remove_columns(["gender"])
    
    if "speech_status" in dataset.column_names:
        # Remove TORGO's original 'speech_status' column (we use 'speech_type')
        dataset = dataset.remove_columns(["speech_status"])
    
    # Cast audio to a common format if needed
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Merge NeuroVoz and TORGO HF datasets."
    )
    parser.add_argument(
        "--neurovoz_dataset",
        type=str,
        default="data/neurovoz/neurovoz_dataset",
        help="Path to NeuroVoz HF dataset.",
    )
    parser.add_argument(
        "--torgo_dataset",
        type=str,
        default="data/torgo/torgo_dataset",
        help="Path to TORGO HF dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/combined_neurovoz_torgo",
        help="Output directory for merged dataset.",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MERGING NEUROVOZ AND TORGO DATASETS")
    print("=" * 60)
    
    # Load datasets
    print("\nStep 1: Loading NeuroVoz dataset...")
    neurovoz_ds = load_from_disk(args.neurovoz_dataset)
    print(f"  Train: {len(neurovoz_ds['train'])} samples")
    print(f"  Validation: {len(neurovoz_ds['validation'])} samples")
    print(f"  Test: {len(neurovoz_ds['test'])} samples")
    
    print("\nStep 2: Loading TORGO dataset...")
    torgo_ds = load_from_disk(args.torgo_dataset)
    print(f"  Train: {len(torgo_ds['train'])} samples")
    print(f"  Validation: {len(torgo_ds['validation'])} samples")
    print(f"  Test: {len(torgo_ds['test'])} samples")
    
    # Normalize datasets
    print("\nStep 3: Normalizing column schemas...")
    neurovoz_train = normalize_columns(neurovoz_ds['train'], "neurovoz")
    neurovoz_val = normalize_columns(neurovoz_ds['validation'], "neurovoz")
    neurovoz_test = normalize_columns(neurovoz_ds['test'], "neurovoz")
    
    torgo_train = normalize_columns(torgo_ds['train'], "torgo")
    torgo_val = normalize_columns(torgo_ds['validation'], "torgo")
    torgo_test = normalize_columns(torgo_ds['test'], "torgo")
    
    print(f"  NeuroVoz columns: {neurovoz_train.column_names}")
    print(f"  TORGO columns: {torgo_train.column_names}")
    
    # Merge train and validation splits
    print("\nStep 4: Merging train splits...")
    merged_train = concatenate_datasets([neurovoz_train, torgo_train])
    print(f"  Merged train: {len(merged_train)} samples")
    
    print("\nStep 5: Merging validation splits...")
    merged_val = concatenate_datasets([neurovoz_val, torgo_val])
    print(f"  Merged validation: {len(merged_val)} samples")
    
    # Keep test splits separate for individual evaluation
    print("\nStep 6: Keeping test splits separate...")
    
    # Create merged dataset
    merged_dataset = DatasetDict({
        "train": merged_train,
        "validation": merged_val,
        "test_neurovoz": neurovoz_test,
        "test_torgo": torgo_test,
    })
    
    # Save merged dataset
    print(f"\nStep 7: Saving merged dataset to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_dataset.save_to_disk(args.output_dir)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("MERGED DATASET STATISTICS")
    print("=" * 60)
    
    for split_name, split_data in merged_dataset.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total samples: {len(split_data)}")
        
        # Dataset source distribution
        if "dataset_source" in split_data.column_names:
            sources = pd.Series(split_data["dataset_source"]).value_counts()
            print(f"  Dataset distribution: {dict(sources)}")
        
        # Speech type distribution
        if "speech_type" in split_data.column_names:
            speech_types = pd.Series(split_data["speech_type"]).value_counts()
            print(f"  Speech type distribution: {dict(speech_types)}")
        
        # Sex distribution
        if "sex" in split_data.column_names:
            sexes = pd.Series(split_data["sex"]).value_counts()
            print(f"  Sex distribution: {dict(sexes)}")
        
        # Unique speakers
        if "speaker_id" in split_data.column_names:
            print(f"  Unique speakers: {len(set(split_data['speaker_id']))}")
    
    # Create TSV files for merged splits
    print("\n" + "=" * 60)
    print("CREATING TSV FILES")
    print("=" * 60)
    
    for split_name in merged_dataset.keys():
        split_data = merged_dataset[split_name]
        df = split_data.to_pandas()
        
        # Select relevant columns for TSV
        tsv_columns = ['sample_id', 'audio_path', 'transcription', 'speaker_id', 
                      'duration', 'sex', 'age', 'speech_type', 'dataset_source']
        df_tsv = df[tsv_columns]
        
        # Save to TSV
        output_path = os.path.join(args.output_dir, f"{split_name}.tsv")
        df_tsv.to_csv(output_path, sep='\t', index=False)
        print(f"  Created {output_path} with {len(df_tsv)} samples")
    
    print("\n" + "=" * 60)
    print("✓ DATASET MERGING COMPLETE!")
    print("=" * 60)
    print(f"\nMerged dataset saved to: {args.output_dir}")
    print("\nDataset structure:")
    print("  - train/ (combined NeuroVoz + TORGO training data)")
    print("  - validation/ (combined NeuroVoz + TORGO validation data)")
    print("  - test_neurovoz/ (NeuroVoz test data only)")
    print("  - test_torgo/ (TORGO test data only)")
    print("\nTSV files:")
    print("  - train.tsv")
    print("  - validation.tsv")
    print("  - test_neurovoz.tsv")
    print("  - test_torgo.tsv")
    print()


if __name__ == "__main__":
    main()