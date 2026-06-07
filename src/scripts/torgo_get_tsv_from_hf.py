
import os
import pandas as pd

from datasets import load_dataset


def get_audio_path(a):
    if isinstance(a, dict):
        return a.get("path", "")
    return ""


def extract_metadata_from_filename(path):
    filename = os.path.basename(path)
    parts = filename.split("_")
    if len(parts) < 2:
        return "Unknown", ""
    speaker_code = parts[0]
    if speaker_code.startswith("MC"):
        sex = "Male"
        speech_type = "HC"
    elif speaker_code.startswith("FC"):
        sex = "Female"
        speech_type = "HC"
    elif speaker_code.startswith("M"):
        sex = "Male"
        speech_type = "DYSARTHRIC"
    elif speaker_code.startswith("F"):
        sex = "Female"
        speech_type = "DYSARTHRIC"
    else:
        sex = "Unknown"
        speech_type = "Unknown"
    return sex, speech_type


def normalize_torgo_df(df: pd.DataFrame, id_prefix: str = "torgo") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["sample_id", "audio_path", "transcription", "speaker_id", "duration", "sex", "age", "speech_type"])

    audio_col = df.get("audio")
    if audio_col is None:
        df["audio_path"] = ""
    else:
        df["audio_path"] = df["audio"].apply(get_audio_path)

    df["audio_filename"] = df["audio_path"].apply(lambda p: os.path.basename(p) if p else "")

    # From audio path we can extract metadata
    # For instance: MC03_2_arrayMic_0085.wav
    #   MC -> Male Control (HC)
    #   M03 -> Male with dysarthria
    #   03 --> speaker number, so speaker_id is just the audio prefix up to first underscore
    #   arrayMic -> microphone_type, store for the tsv

    # Get sex and speech_type from filename
    df[["sex", "speech_type"]] = df["audio_path"].apply(
        lambda p: pd.Series(extract_metadata_from_filename(p))
    )

    # speaker_id: get from filename prefix
    df["speaker_id"] = df["audio_path"].apply(
        lambda p: os.path.basename(p).split("_")[0] if p else ""
    )
    
    # sample id: use filename without extension with prefix
    df["sample_id"] = df["audio_path"].apply(
        lambda p: f"{id_prefix}_{os.path.splitext(os.path.basename(p))[0]}" if p else ""
    )

    # Age is unknown in TORGO dataset
    df["age"] = -1

    out = df.loc[:, ["sample_id", "audio_path", "audio_filename", "transcription", "speaker_id", "duration", "sex", "age", "speech_type"]].copy()
    return out

def main():
    print("Loading TORGO dataset...")
    torgo_dataset = load_dataset("abnerh/TORGO-database", cache_dir="data/")
    torgo_train = torgo_dataset["train"]

    print(f"Loaded TORGO dataset with {len(torgo_train)} training samples.")
    torgo_df = torgo_train.to_pandas()

    print(f"TORGO DataFrame columns: {list(torgo_df.columns)}")

    normalized_torgo_df = normalize_torgo_df(torgo_df, id_prefix="torgo")
    
    print(f"Normalized TORGO DataFrame with columns: {list(normalized_torgo_df.columns)}")
    print(normalized_torgo_df.head())

    # Sotore tsv file
    tsv_path = "data/torgo/torgo.tsv"
    normalized_torgo_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"TSV file created: {tsv_path}")
    
if __name__ == "__main__":
    main()