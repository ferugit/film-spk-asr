"""
Example script showing how to use the prepared Neurovoz dataset
for ASR fine-tuning or evaluation.
"""

import argparse
from datasets import load_from_disk
import pandas as pd


def example_tsv_usage():
    """Example: Loading and using the TSV file."""
    print("=" * 60)
    print("Example 1: Using the TSV file with Pandas")
    print("=" * 60)
    
    tsv_path = "data/neurovoz/neurovoz_dataset.tsv"
    
    # Load TSV
    df = pd.read_csv(tsv_path, sep="\t")
    
    print(f"Total samples: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst sample:")
    print(df.iloc[0])
    
    # Get statistics
    print(f"\n\nDataset Statistics:")
    print(f"  - Audio duration range: {df['duration'].min():.2f}s - {df['duration'].max():.2f}s")
    print(f"  - Average duration: {df['duration'].mean():.2f}s")
    print(f"  - Total speakers: {df['speaker_id'].nunique()}")
    print(f"  - Unique speakers: {df['speaker_id'].unique()[:5]}")
    
    # Disease distribution
    print(f"\nDisease Distribution:")
    print(f"  {df['disease'].value_counts().to_dict()}")
    
    # Sex distribution
    print(f"\nSex Distribution:")
    sex_map = {1.0: "Male", 0.0: "Female"}
    if 'sex' in df.columns:
        sex_counts = df['sex'].value_counts()
        for sex_val, count in sex_counts.items():
            print(f"  - {sex_map.get(sex_val, 'Unknown')}: {count}")
    
    # Age statistics
    print(f"\nAge Statistics:")
    if 'age' in df.columns:
        print(f"  - Average age: {df['age'].mean():.1f} years")
        print(f"  - Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
    
    return df


def example_hf_dataset_usage():
    """Example: Loading and using the Hugging Face dataset."""
    print("\n" + "=" * 60)
    print("Example 2: Using the Hugging Face Dataset")
    print("=" * 60)
    
    hf_dataset_path = "datasets/neurovoz_dataset"
    
    # Load HF dataset
    dataset = load_from_disk(hf_dataset_path)
    
    print(f"Dataset splits: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f"  - {split}: {len(dataset[split])} samples")
    
    # Access a sample
    print(f"\nFirst sample from training set:")
    sample = dataset["train"][0]
    print(f"  - Audio array shape: {len(sample['audio']['array'])}")
    print(f"  - Sampling rate: {sample['audio']['sampling_rate']}")
    print(f"  - Transcription: {sample['transcription']}")
    print(f"  - Speaker ID: {sample['speaker_id']}")
    print(f"  - Duration: {sample['duration']}s")
    print(f"  - Sex: {sample['sex']}")
    print(f"  - Age: {sample['age']}")
    print(f"  - Disease: {sample['disease']}")
    
    return dataset


def example_dataset_loading_for_training():
    """Example: Loading dataset for model fine-tuning."""
    print("\n" + "=" * 60)
    print("Example 3: Dataset usage for fine-tuning")
    print("=" * 60)
    
    hf_dataset_path = "data/neurovoz/neurovoz_hf_dataset"
    dataset = load_from_disk(hf_dataset_path)
    
    # Get training data
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Example iteration
    print(f"\nIterating through first 3 training samples:")
    for i, sample in enumerate(train_dataset.take(3)):
        print(f"\n  Sample {i+1}:")
        print(f"    - Text: {sample['transcription'][:50]}...")
        print(f"    - Speaker: {sample['speaker_id']}")
        print(f"    - Duration: {sample['duration']}s")
        print(f"    - Sex: {sample['sex']}")
        print(f"    - Age: {sample['age']}")
        print(f"    - Disease: {sample['disease']}")
    
    # Example: filtering by disease
    print(f"\n\nFiltering samples by disease (HC only):")
    hc_samples = train_dataset.filter(lambda x: x["disease"] == "HC")
    print(f"  - Original: {len(train_dataset)} samples")
    print(f"  - HC only: {len(hc_samples)} samples")
    
    # Example: filtering by duration
    print(f"\nFiltering samples by duration (< 5 seconds):")
    short_samples = train_dataset.filter(lambda x: x["duration"] < 5.0)
    print(f"  - Original: {len(train_dataset)} samples")
    print(f"  - Filtered: {len(short_samples)} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Example usage of prepared Neurovoz dataset"
    )
    parser.add_argument(
        "--example",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Which example to run (0=all, 1=TSV, 2=HF Dataset, 3=Training prep)",
    )
    
    args = parser.parse_args()
    
    if args.example == 0 or args.example == 1:
        example_tsv_usage()
    
    if args.example == 0 or args.example == 2:
        example_hf_dataset_usage()
    
    if args.example == 0 or args.example == 3:
        example_dataset_loading_for_training()


if __name__ == "__main__":
    main()
