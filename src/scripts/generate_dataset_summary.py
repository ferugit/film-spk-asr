"""
Generate a summary text file for the combined NeuroVoz + TORGO dataset.
"""

import argparse
from datasets import load_from_disk
import pandas as pd
from datetime import datetime


def generate_summary(dataset_path, output_file):
    """Generate dataset summary and save to text file."""
    
    # Load dataset
    ds = load_from_disk(dataset_path)
    
    # Open output file
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("COMBINED NEUROVOZ + TORGO DATASET SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset path: {dataset_path}\n")
        
        # Basic info
        f.write(f"\nAvailable splits: {', '.join(ds.keys())}\n")
        f.write(f"Column names: {', '.join(ds['train'].column_names)}\n")
        
        # Detailed split information
        f.write("\n" + "-" * 70 + "\n")
        f.write("SPLIT DETAILS\n")
        f.write("-" * 70 + "\n")
        
        for split_name in ds.keys():
            split_data = ds[split_name]
            f.write(f"\n{split_name.upper()}:\n")
            f.write(f"  Total samples: {len(split_data):,}\n")
            
            # Convert to pandas for analysis
            df = pd.DataFrame(split_data[:])
            
            # Dataset distribution
            dataset_dist = df['dataset_source'].value_counts()
            f.write(f"  Dataset sources:\n")
            for source, count in dataset_dist.items():
                percentage = count / len(split_data) * 100
                f.write(f"    - {source}: {count:,} samples ({percentage:.1f}%)\n")
            
            # Speech type distribution
            speech_dist = df['speech_type'].value_counts()
            f.write(f"  Speech types:\n")
            for stype, count in speech_dist.items():
                percentage = count / len(split_data) * 100
                f.write(f"    - {stype}: {count:,} samples ({percentage:.1f}%)\n")
            
            # Sex distribution
            sex_dist = df['sex'].value_counts()
            f.write(f"  Sex distribution:\n")
            for sex, count in sex_dist.items():
                percentage = count / len(split_data) * 100
                f.write(f"    - {sex}: {count:,} samples ({percentage:.1f}%)\n")
            
            # Unique speakers
            unique_speakers = df['speaker_id'].nunique()
            f.write(f"  Unique speakers: {unique_speakers}\n")
            
            # Duration statistics
            total_duration = df['duration'].sum()
            avg_duration = df['duration'].mean()
            f.write(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes, {total_duration/3600:.2f} hours)\n")
            f.write(f"  Average duration: {avg_duration:.2f} seconds\n")
        
        # Overall summary
        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 70 + "\n")
        
        total_samples = sum(len(ds[split]) for split in ds.keys())
        f.write(f"\nTotal samples across all splits: {total_samples:,}\n")
        
        # Combine all data for overall statistics
        all_data = []
        for split_name in ds.keys():
            df_split = pd.DataFrame(ds[split_name][:])
            df_split['split'] = split_name
            all_data.append(df_split)
        
        df_all = pd.concat(all_data, ignore_index=True)
        
        f.write(f"\nOverall dataset sources:\n")
        overall_source_dist = df_all['dataset_source'].value_counts()
        for source, count in overall_source_dist.items():
            percentage = count / len(df_all) * 100
            f.write(f"  - {source}: {count:,} samples ({percentage:.1f}%)\n")
        
        f.write(f"\nOverall speech types:\n")
        overall_speech_dist = df_all['speech_type'].value_counts()
        for stype, count in overall_speech_dist.items():
            percentage = count / len(df_all) * 100
            f.write(f"  - {stype}: {count:,} samples ({percentage:.1f}%)\n")
        
        f.write(f"\nOverall sex distribution:\n")
        overall_sex_dist = df_all['sex'].value_counts()
        for sex, count in overall_sex_dist.items():
            percentage = count / len(df_all) * 100
            f.write(f"  - {sex}: {count:,} samples ({percentage:.1f}%)\n")
        
        total_duration_all = df_all['duration'].sum()
        f.write(f"\nTotal audio duration: {total_duration_all:.2f} seconds ({total_duration_all/60:.2f} minutes, {total_duration_all/3600:.2f} hours)\n")
        
        unique_speakers_all = df_all['speaker_id'].nunique()
        f.write(f"Total unique speakers: {unique_speakers_all}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("✓ Summary generated successfully!\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary file for combined dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/combined_neurovoz_torgo",
        help="Path to the combined dataset.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path. If not provided, will save to dataset_path/DATASET_SUMMARY.txt",
    )
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = f"{args.dataset_path}/DATASET_SUMMARY.txt"
    
    generate_summary(args.dataset_path, args.output_file)


if __name__ == "__main__":
    main()
