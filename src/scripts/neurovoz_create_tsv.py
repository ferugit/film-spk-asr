import argparse
import csv
import os
from pathlib import Path
import pandas as pd
import librosa


def extract_speaker_id_from_filename(filename):
    """Extract numeric speaker ID from filename.
    
    Handles filenames like: PD_O3_0023 or HC_ABLANDADA_0034
    Returns the numeric part (0023 -> 23)
    """
    parts = filename.split("_")
    numeric_id_str = parts[-1]  # e.g., "0034" or "0023"
    numeric_id = str(int(numeric_id_str))  # Remove leading zeros
    return numeric_id


def get_transcription(base, transcription_files, phonation_map):
    """Get transcription for a sample.
    
    If transcription file exists, read it. Otherwise, determine from phonation type.
    Returns (transcription, is_phonation)
    """
    # Check if transcription file exists
    if base in transcription_files:
        try:
            with open(transcription_files[base], "r", encoding="utf-8") as f:
                transcription = f.read().strip()
                # Clean up newlines and carriage returns
                transcription = transcription.replace("\r", " ").replace("\n", " ")
                # Remove extra spaces
                transcription = " ".join(transcription.split())
            return transcription, False
        except:
            return None, False
    
    # No transcription file - determine if it's a phonation task
    parts = base.split("_")
    phonation_task = parts[-2] if len(parts) >= 2 else None
    
    if phonation_task in phonation_map:
        transcription = phonation_map[phonation_task]
        return transcription, True
    
    # Unknown phonation task
    return None, True


def get_metadata(speaker_id, metadata_map):
    """Get metadata for a speaker."""
    if speaker_id in metadata_map:
        return (
            metadata_map[speaker_id]["sex"],
            metadata_map[speaker_id]["age"],
            metadata_map[speaker_id]["speech_type"],
        )
    return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Create TSV file from Neurovoz dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/neurovoz_raw",
        help="Path to the Neurovoz dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/neurovoz",
        help="Output directory for TSV file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="neurovoz.tsv",
        help="Name of output TSV file.",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths
    audio_dir = os.path.join(args.dataset_dir, "audios")
    transcription_dir = os.path.join(args.dataset_dir, "transcriptions")
    metadata_dir = os.path.join(args.dataset_dir, "metadata")
    
    tsv_path = os.path.join(args.output_dir, args.output_file)
    
    print("Step 0: Loading metadata...")
    
    # Load metadata
    metadata_hc = pd.read_csv(os.path.join(metadata_dir, "metadata_hc.csv"))
    metadata_pd = pd.read_csv(os.path.join(metadata_dir, "metadata_pd.csv"))
    
    # Add disease label
    # Use 'speech_type' to indicate speaker group (was previously named 'disease')
    metadata_hc["speech_type"] = "HC"
    metadata_pd["speech_type"] = "PARKINSON"
    
    # Combine metadata
    metadata = pd.concat([metadata_hc, metadata_pd], ignore_index=True)
    metadata["ID"] = metadata["ID"].astype(str)
    
    # Create a mapping from ID to metadata (take first occurrence since all rows have same metadata)
    metadata_map = {}
    for _, row in metadata.iterrows():
        speaker_id = str(int(row["ID"]))
        if speaker_id not in metadata_map:
            # Convert sex: 1.0 = Male, 0.0 = Female
            sex_value = row["Sex"]
            if pd.isna(sex_value):
                sex_label = None
            elif sex_value == 1.0 or sex_value == 1:
                sex_label = "Male"
            elif sex_value == 0.0 or sex_value == 0:
                sex_label = "Female"
            else:
                sex_label = None
            
            metadata_map[speaker_id] = {
                "sex": sex_label,
                "age": row["Age"],
                "speech_type": row.get("speech_type", None),
            }
    
    print(f"Loaded metadata for {len(metadata_map)} speakers")
    
    print("\nStep 1: Matching audio files to transcriptions...")
    
    # Load all audio and transcription files
    audio_files = {Path(f).stem: os.path.join(audio_dir, f) 
                   for f in os.listdir(audio_dir) if f.endswith(".wav")}
    
    transcription_files = {Path(f).stem: os.path.join(transcription_dir, f)
                          for f in os.listdir(transcription_dir) if f.endswith(".txt")}
    
    # Count audio-transcription pairs
    matched_count = sum(1 for base in audio_files if base in transcription_files)
    print(f"Found {matched_count} audio-transcription pairs")
    print(f"Non-matching audio files: {len(audio_files) - matched_count}")
    
    # Define phonation task mapping
    phonation_map = {
        # Vowels (sustained, no transcription files)
        "A1": "<sustained-phonation-vowel-a>",
        "A2": "<sustained-phonation-vowel-a>",
        "A3": "<sustained-phonation-vowel-a>",
        "E1": "<sustained-phonation-vowel-e>",
        "E2": "<sustained-phonation-vowel-e>",
        "E3": "<sustained-phonation-vowel-e>",
        "I1": "<sustained-phonation-vowel-i>",
        "I2": "<sustained-phonation-vowel-i>",
        "I3": "<sustained-phonation-vowel-i>",
        "O1": "<sustained-phonation-vowel-o>",
        "O2": "<sustained-phonation-vowel-o>",
        "O3": "<sustained-phonation-vowel-o>",
        "U1": "<sustained-phonation-vowel-u>",
        "U2": "<sustained-phonation-vowel-u>",
        "U3": "<sustained-phonation-vowel-u>",
        # Consonant cluster (no transcription files)
        "PATAKA": "<sustained-phonation-consonant-cluster-pataka>",
    }
    
    print("\nStep 2: Processing all audio files...")
    
    rows = []
    speech_count = 0
    phonation_count = 0
    skipped_count = 0
    
    # Process all audio files in a single loop
    for base, audio_path in audio_files.items():
        try:
            # Get transcription
            transcription, is_phonation = get_transcription(base, transcription_files, phonation_map)
            
            if transcription is None:
                skipped_count += 1
                continue
            
            # Get audio duration
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Extract speaker ID from filename
            speaker_id = extract_speaker_id_from_filename(base)
            
            # Get metadata
            sex, age, speech_type = get_metadata(speaker_id, metadata_map)
            
            # Create row
            rows.append({
                "sample_id": base,
                "audio_path": audio_path,
                "audio_filename": Path(audio_path).name,
                "transcription": transcription,
                "speaker_id": speaker_id,
                "duration": round(duration, 2),
                "sex": sex,
                "age": age,
                "speech_type": speech_type,
            })
            
            if is_phonation:
                phonation_count += 1
            else:
                speech_count += 1
                
        except Exception as e:
            skipped_count += 1
            continue
    
    print(f"Processed {len(rows)} files")
    print(f"  - Speech with transcriptions: {speech_count}")
    print(f"  - Phonation tasks: {phonation_count}")
    print(f"  - Skipped: {skipped_count}")
    
    print("\nStep 3: Writing TSV file...")
    
    df = pd.DataFrame(rows)
    
    # Replace NaN values with "Unknown" or appropriate defaults
    df["sex"] = df["sex"].fillna("Unknown")
    df["age"] = df["age"].fillna(-1)  # Use -1 to indicate unknown age
    # Replace NaN values for the renamed column
    df["speech_type"] = df["speech_type"].fillna("Unknown")
    
    df.to_csv(tsv_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    
    print(f"\nTSV file created: {tsv_path}")
    print(f"Total samples: {len(df)}")
    print(f"  - Speech with transcriptions: {(df['transcription'].str.startswith('<') == False).sum()}")
    print(f"  - Phonation tasks: {(df['transcription'].str.startswith('<')).sum()}")
    print(f"\nColumns: {list(df.columns)}")
    
    return tsv_path


if __name__ == "__main__":
    main()
