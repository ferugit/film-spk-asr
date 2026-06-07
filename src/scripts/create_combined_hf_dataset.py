#!/usr/bin/env python3
"""
Create a combined Hugging Face-style dataset from local `neurovoz` and `torgo` data.

Normalization rules (to match neurovoz fields):
- sample_id
- audio_path
- audio_filename
- transcription
- speaker_id (for torgo: concat sex + speech_type)
- duration
- sex (torgo -> 'Unknown')
- age (torgo -> -1)
- speech_type (torgo -> speech_status)

Torgo only has a 'train' split; all Torgo rows will be added to the combined `train` split.

Output: creates `data/combined_neurovoz_torgo/` with `train.tsv`, `validation.tsv`, `test.tsv` and
`dataset_dict.json` listing splits.
"""
import glob
import json
import os
import sys
from typing import Dict

try:
	import pandas as pd
	import pyarrow.ipc as ipc
	from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
except Exception as e:
	print("Missing dependency:", e)
	print("Please install requirements: pip install pandas pyarrow")
	raise


def read_arrow_files_in_dir(dir_path: str) -> pd.DataFrame:
	"""Read all .arrow files in a directory and return a concatenated DataFrame."""
	arrow_files = sorted(glob.glob(os.path.join(dir_path, "*.arrow")))
	dfs = []
	for fpath in arrow_files:
		# Try multiple readers: Arrow IPC, Feather, Parquet, TSV fallback
		read_succeeded = False
		# 1) try Arrow IPC
		try:
			with open(fpath, "rb") as f:
				reader = ipc.RecordBatchFileReader(f)
				table = reader.read_all()
			dfs.append(table.to_pandas())
			read_succeeded = True
		except Exception:
			pass

		if not read_succeeded:
			# 2) try pyarrow.feather
			try:
				import pyarrow.feather as feather

				table = feather.read_table(fpath)
				dfs.append(table.to_pandas())
				read_succeeded = True
			except Exception:
				pass

		if not read_succeeded:
			# 3) try pandas.read_parquet (some HF shards are parquet)
			try:
				dfs.append(pd.read_parquet(fpath))
				read_succeeded = True
			except Exception:
				pass

		if not read_succeeded:
			# 4) last resort: try as TSV/CSV
			try:
				dfs.append(pd.read_csv(fpath, sep="\t", encoding="utf-8"))
				read_succeeded = True
			except Exception as e:
				print(f"Warning: failed reading shard {fpath} with any reader: {e}")
	if not dfs:
		return pd.DataFrame()
	return pd.concat(dfs, ignore_index=True)


def load_neurovoz(neurovoz_dataset_dir: str) -> Dict[str, pd.DataFrame]:
	"""Load neurovoz splits (train/validation/test) from `neurovoz_dataset` folder."""
	# First try to load an HF Dataset saved on disk (robust fallback for arrow shards)
	try:
		ds = load_from_disk(neurovoz_dataset_dir)
		if isinstance(ds, DatasetDict):
			print(f"Loaded neurovoz HF dataset from disk: {neurovoz_dataset_dir}")
			return {s: ds[s].to_pandas() for s in ds.keys()}
	except Exception:
		# fall through to shard readers
		pass

	splits = {}
	for split in ["train", "validation", "test"]:
		split_dir = os.path.join(neurovoz_dataset_dir, split)
		if os.path.isdir(split_dir):
			splits[split] = read_arrow_files_in_dir(split_dir)
		else:
			splits[split] = pd.DataFrame()
	# If any split is empty (e.g. arrow shards unreadable), try fallback to the original neurovoz TSV
	if any((splits[s] is None or splits[s].empty) for s in ["train", "validation", "test"]):
		parent = os.path.dirname(neurovoz_dataset_dir)
		tsv_path = os.path.join(parent, "neurovoz.tsv")
		if os.path.isfile(tsv_path):
			try:
				print(f"Falling back to reading TSV file for neurovoz: {tsv_path}")
				tsv_df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")
				# Assign all rows to train split as fallback (original split info unavailable)
				splits["train"] = tsv_df
				splits["validation"] = pd.DataFrame()
				splits["test"] = pd.DataFrame()
				return splits
			except Exception as e:
				print(f"Warning: failed reading fallback neurovoz TSV {tsv_path}: {e}")
	return splits


def load_torgo(torgo_cache_root: str) -> pd.DataFrame:
	"""Load torgo arrow files (train only) from the cache folder that contains .arrow files.

	torgo_cache_root should be the directory that contains the .arrow shards and dataset_info.json
	we auto-detect it from `data/abnerh___torgo-database/default/0.0.0/`.
	"""
	# Prefer loading any HF dataset saved on disk (datasets.save_to_disk style)
	try:
		ds = load_from_disk(torgo_cache_root)
		# take train split if available
		if isinstance(ds, DatasetDict) and "train" in ds:
			print(f"Loaded torgo HF dataset from disk: {torgo_cache_root}")
			return ds["train"].to_pandas()
		# if it's a single Dataset, convert to pandas
		if not isinstance(ds, DatasetDict):
			return ds.to_pandas()
	except Exception:
		pass

	return read_arrow_files_in_dir(torgo_cache_root)


def normalize_neurovoz_df(df: pd.DataFrame) -> pd.DataFrame:
	if df is None or df.empty:
		return pd.DataFrame(columns=["sample_id", "audio_path", "audio_filename", "transcription", "speaker_id", "duration", "sex", "age", "speech_type"])

	# audio_path may already exist
	if "audio_path" not in df.columns and "audio" in df.columns:
		def get_path(a):
			if isinstance(a, dict):
				return a.get("path", "")
			return ""

		df["audio_path"] = df["audio"].apply(get_path)

	if "audio_filename" not in df.columns:
		df["audio_filename"] = df["audio_path"].apply(lambda p: os.path.basename(p) if p else "")

	# Ensure required columns exist
	defaults = {
		"sample_id": "",
		"transcription": "",
		"speaker_id": "",
		"duration": -1.0,
		"sex": "Unknown",
		"age": -1,
		"speech_type": "",
	}
	for col, val in defaults.items():
		if col not in df.columns:
			df[col] = val

	out = df.loc[:, ["sample_id", "audio_path", "audio_filename", "transcription", "speaker_id", "duration", "sex", "age", "speech_type"]].copy()
	return out


def normalize_torgo_df(df: pd.DataFrame, id_prefix: str = "torgo") -> pd.DataFrame:
	if df is None or df.empty:
		return pd.DataFrame(columns=["sample_id", "audio_path", "audio_filename", "transcription", "speaker_id", "duration", "sex", "age", "speech_type"])

	def get_audio_path(a):
		if isinstance(a, dict):
			return a.get("path", "")
		return ""

	audio_col = df.get("audio")
	if audio_col is None:
		df["audio_path"] = ""
	else:
		df["audio_path"] = df["audio"].apply(get_audio_path)

	df["audio_filename"] = df["audio_path"].apply(lambda p: os.path.basename(p) if p else "")
	# transcription may exist
	if "transcription" not in df.columns:
		df["transcription"] = ""

	# sex unknown
	df["sex"] = "Unknown"

	# speech_type comes from speech_status (per dataset_info.json)
	if "speech_status" in df.columns:
		df["speech_type"] = df["speech_status"].fillna("")
	else:
		df["speech_type"] = ""

	# speaker_id: concat sex + speech_type
	df["speaker_id"] = df.apply(lambda r: f"{r['sex']}_{r['speech_type']}" if r.get("speech_type", "") else r["sex"], axis=1)

	# sample id: generate unique ids
	df = df.reset_index(drop=True)
	df["sample_id"] = [f"{id_prefix}_{i}" for i in range(len(df))]

	if "duration" not in df.columns:
		df["duration"] = -1.0

	df["age"] = -1

	out = df.loc[:, ["sample_id", "audio_path", "audio_filename", "transcription", "speaker_id", "duration", "sex", "age", "speech_type"]].copy()
	return out


def write_splits(out_dir: str, splits: Dict[str, pd.DataFrame]):
	os.makedirs(out_dir, exist_ok=True)
	for split, df in splits.items():
		out_path = os.path.join(out_dir, f"{split}.tsv")
		df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
		print(f"Wrote {len(df)} rows to {out_path}")
	# dataset dict
	ds_dict = {"splits": list(splits.keys())}
	with open(os.path.join(out_dir, "dataset_dict.json"), "w", encoding="utf-8") as fh:
		json.dump(ds_dict, fh, indent=2)


def find_single_subdir(root: str):
	# choose the first non-lock directory
	items = sorted(os.listdir(root))
	for it in items:
		full = os.path.join(root, it)
		if os.path.isdir(full) and not it.endswith(".lock") and not it.endswith(".incomplete_info.lock"):
			return full
	raise FileNotFoundError(f"No suitable dataset subdir found in {root}")


def main():
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	data_root = os.path.join(repo_root, "data")

	neurovoz_dataset_dir = os.path.join(data_root, "neurovoz", "neurovoz_dataset")
	if not os.path.isdir(neurovoz_dataset_dir):
		print(f"neurovoz dataset directory not found: {neurovoz_dataset_dir}")
		sys.exit(1)

	neurovoz_splits = load_neurovoz(neurovoz_dataset_dir)
	# normalize neurovoz
	for k in list(neurovoz_splits.keys()):
		neurovoz_splits[k] = normalize_neurovoz_df(neurovoz_splits[k])

	# load torgo from the Hub (do not parse local shards)
	print("Loading TORGO dataset...")
	torgo_dataset = load_dataset("abnerh/TORGO-database", cache_dir="data/")
	torgo_train = torgo_dataset["train"]
	print(f"Dataset loaded successfully!")
	# convert to pandas and normalize
	torgo_df = torgo_train.to_pandas()
	torgo_norm = normalize_torgo_df(torgo_df)

	# Combine: neurovoz train + torgo -> combined train; neurovoz validation/test unchanged
	combined_splits = {}
	combined_splits["train"] = pd.concat([neurovoz_splits.get("train", pd.DataFrame()), torgo_norm], ignore_index=True)
	combined_splits["validation"] = neurovoz_splits.get("validation", pd.DataFrame())
	combined_splits["test"] = neurovoz_splits.get("test", pd.DataFrame())

	out_dir = os.path.join(data_root, "combined_neurovoz_torgo")
	write_splits(out_dir, combined_splits)

	print("Combined dataset created at:", out_dir)
	print("Counts:")
	for k, v in combined_splits.items():
		print(f"  {k}: {len(v)}")


if __name__ == "__main__":
	main()
