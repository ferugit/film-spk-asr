"""
Pre-compute x-vectors for all samples in the dataset.

This is useful for:
  - Offline analysis / visualization of speaker embeddings
  - Faster training if you want to skip on-the-fly extraction
    (though the main training script does on-the-fly for gradient flow)

Usage:
    python src/finetune/spk_cond_voxtral/precompute_xvectors.py \
        --dataset_path data/combined_neurovoz_torgo_cv \
        --jit_path models/SiAMResNet34/samresnet34_w_features.jit \
        --output_path data/xvectors_neurovoz_torgo_cv.pt \
        --batch_size 32
"""

import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, Audio


def main():
    parser = argparse.ArgumentParser(description="Pre-compute x-vectors for dataset")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--jit_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading SiAMResNet34 from {args.jit_path}...")
    model = torch.jit.load(args.jit_path)
    model = model.to(device)
    model.eval()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    results = {}

    for split_name in dataset.keys():
        split = dataset[split_name]
        print(f"\nProcessing split '{split_name}' ({len(split)} samples)...")

        all_embeddings = []
        all_sample_ids = []
        all_speech_types = []

        for i in tqdm(range(0, len(split), args.batch_size)):
            batch_indices = range(i, min(i + args.batch_size, len(split)))
            batch = split.select(batch_indices)

            # Get audio arrays and pad to same length
            arrays = [np.asarray(s["array"], dtype=np.float32) for s in batch["audio"]]
            max_len = max(len(a) for a in arrays)
            padded = np.zeros((len(arrays), max_len), dtype=np.float32)
            for j, a in enumerate(arrays):
                padded[j, :len(a)] = a

            waveforms = torch.from_numpy(padded).to(device)

            with torch.no_grad():
                embeddings = model(waveforms)  # (B, 256)

            all_embeddings.append(embeddings.cpu())
            all_sample_ids.extend(batch["sample_id"])
            all_speech_types.extend(batch["speech_type"])

        all_embeddings = torch.cat(all_embeddings, dim=0)

        results[split_name] = {
            "embeddings": all_embeddings,
            "sample_ids": all_sample_ids,
            "speech_types": all_speech_types,
        }

        print(f"  Embeddings shape: {all_embeddings.shape}")

    # Save
    print(f"\nSaving to {args.output_path}...")
    torch.save(results, args.output_path)
    print("Done!")


if __name__ == "__main__":
    main()
