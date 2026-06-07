"""
Evaluation / inference script for Speaker-Conditioned Voxtral.

Loads a trained SpeakerConditionedVoxtral checkpoint and runs inference on
NeuroVoz (Spanish) and TORGO (English) test sets, producing per-sample TSV
files compatible with generate_eval_summary.py.

Usage:
    python src/finetune/spk_cond_voxtral/eval.py \
        --model_path models/voxtral-spk-cond-neurovoz-torgo-cv \
        --model_name spk_cond_voxtral \
        --dataset_path data/combined_neurovoz_torgo_cv \
        --output_dir results/evaluation/spk_cond_voxtral \
        --base_model_id mistralai/Voxtral-Mini-3B-2507
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_from_disk, Audio
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))
from model import SpeakerConditionedVoxtral

# Project root: three levels up from this file (src/finetune/spk_cond_voxtral/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Normative speech types (identity conditioning)
NORMATIVE_SPEECH_TYPES = {"HC", "Unknown"}


def build_model_from_checkpoint(
    checkpoint_dir: str,
    base_model_id: str,
    cache_dir: str = None,
):
    """
    Rebuild a SpeakerConditionedVoxtral from a training checkpoint.

    The checkpoint directory must contain:
      - spk_cond_voxtral.pt    (full state dict)
      - training_config.yaml   (architecture / training config)
    """
    import yaml

    if cache_dir is None:
        cache_dir = os.path.join(PROJECT_ROOT, "models")

    config_path = os.path.join(checkpoint_dir, "training_config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"training_config.yaml not found in {checkpoint_dir}. "
            "Is this a valid spk_cond_voxtral checkpoint?"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading base Voxtral model: {base_model_id}")
    voxtral_model = VoxtralForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )

    # Build the speaker-conditioned wrapper
    print(f"Building SpeakerConditionedVoxtral from checkpoint: {checkpoint_dir}")
    model = SpeakerConditionedVoxtral(
        voxtral_model=voxtral_model,
        xvector_model_path=cfg.get("xvector_model_path",
                                    os.path.join(PROJECT_ROOT, "models", "SiAMResNet34", "samresnet34_w_features.jit")),
        film_config={
            "hidden_dim": cfg.get("film_hidden_dim", 512),
            "mode": cfg.get("film_mode", "per_layer"),
            "use_gate": cfg.get("film_use_gate", True),
        },
        train_xvector=False,   # Eval mode — no training
        train_encoder=False,
        train_projector=False,
    )

    # Load trained weights (FiLM + x-vector + projector)
    state_dict_path = os.path.join(checkpoint_dir, "spk_cond_voxtral.pt")
    if os.path.isfile(state_dict_path):
        print(f"Loading state dict from: {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (expected for frozen components)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"State dict not found: {state_dict_path}")

    model.eval()
    return model


def evaluate_model(
    model,
    processor,
    dataset,
    language: str = "es",
    model_id: str = None,
    device: str = "cuda",
):
    """
    Run inference on a dataset and return predictions with references.

    Parameters
    ----------
    model : SpeakerConditionedVoxtral
    processor : VoxtralProcessor
    dataset : HuggingFace dataset with audio, transcription, speech_type
    language : "es" or "en"
    model_id : Model ID for processor prompt
    device : Device for inputs

    Returns
    -------
    predictions, references, sample_ids : lists
    """
    predictions = []
    references = []
    sample_ids = []

    print(f"Evaluating {len(dataset)} samples (lang={language})...")

    for sample in tqdm(dataset, desc="Generating predictions"):
        try:
            audio_array = sample["audio"]["array"]
            speech_type = sample.get("speech_type", "Unknown")

            # Prepare standard Voxtral input
            inputs = processor.apply_transcription_request(
                language=language,
                audio=audio_array,
                model_id=model_id,
                format=["WAV"],
                return_tensors="pt",
            )
            inputs = inputs.to(device, dtype=torch.bfloat16)

            # Prepare raw waveform for x-vector extraction
            wav = np.asarray(audio_array, dtype=np.float32)
            raw_waveforms = torch.from_numpy(wav).unsqueeze(0).to(device)  # (1, T)

            # Normative flag
            is_normative = torch.tensor(
                [speech_type in NORMATIVE_SPEECH_TYPES], dtype=torch.bool
            ).to(device)

            # Generate with speaker conditioning
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    input_features=inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    raw_waveforms=raw_waveforms,
                    is_normative=is_normative,
                    max_new_tokens=256,
                )

            # Decode
            decoded_output = processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]

            predictions.append(decoded_output.strip())
            references.append(sample["transcription"].strip())
            sample_ids.append(sample["sample_id"])

        except Exception as e:
            print(f"\nError processing sample {sample.get('sample_id', 'unknown')}: {e}")
            predictions.append("")
            references.append(sample["transcription"].strip())
            sample_ids.append(sample.get("sample_id", "unknown"))

    return predictions, references, sample_ids


def save_predictions(predictions, references, sample_ids, output_path):
    """Save inference predictions to a TSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({
        "sample_id": sample_ids,
        "reference": references,
        "hypothesis": predictions,
    })
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved predictions to: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Speaker-Conditioned Voxtral (FiLM) on NeuroVoz and TORGO test sets"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to trained spk_cond_voxtral checkpoint directory",
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Short name for model (used for output file naming)",
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="data/combined_neurovoz_torgo_cv",
        help="Path to dataset directory (default: data/combined_neurovoz_torgo_cv)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="results/evaluation/spk_cond_voxtral",
        help="Output directory for results",
    )
    parser.add_argument(
        "--base_model_id", type=str,
        default="mistralai/Voxtral-Mini-3B-2507",
        help="Base Voxtral model ID (for loading base weights + processor)",
    )
    parser.add_argument(
        "--cache_dir", type=str,
        default=os.path.join(PROJECT_ROOT, "models"),
        help="Cache directory for models (default: <project_root>/models)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path
    model_name = args.model_name
    base_model_id = args.base_model_id
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    cache_dir = args.cache_dir

    # Validate checkpoint
    if not os.path.isdir(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train the model first with train_spk_cond_voxtral.sh")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model checkpoint: {model_path}")
    print(f"Model name:       {model_name}")
    print(f"Base model:       {base_model_id}")
    print(f"Using device:     {device}")
    print("=" * 80)

    # ---- Load processor ----
    print(f"Loading processor from: {model_path}")
    try:
        processor = VoxtralProcessor.from_pretrained(model_path)
    except Exception:
        print(f"Processor not found in checkpoint, loading from: {base_model_id}")
        processor = VoxtralProcessor.from_pretrained(base_model_id, cache_dir=cache_dir)

    # ---- Build model from checkpoint ----
    model = build_model_from_checkpoint(
        checkpoint_dir=model_path,
        base_model_id=base_model_id,
        cache_dir=cache_dir,
    )
    print("Model loaded successfully!")
    print("=" * 80)

    # ---- Load dataset ----
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    test_neurovoz = dataset["test_neurovoz"]
    test_torgo = dataset["test_torgo"]

    print(f"Test NeuroVoz samples: {len(test_neurovoz)}")
    print(f"Test TORGO samples:    {len(test_torgo)}")
    print("=" * 80)

    # ── Inference on NeuroVoz test set (Spanish) ─────────────────────────
    print("\n" + "=" * 80)
    print("INFERENCE ON NEUROVOZ TEST SET (SPANISH)")
    print("=" * 80)
    preds_nv, refs_nv, ids_nv = evaluate_model(
        model, processor, test_neurovoz,
        language="es",
        model_id=base_model_id,
        device=device,
    )
    nv_tsv = f"{output_dir}/{model_name}_neurovoz_test_results.tsv"
    save_predictions(preds_nv, refs_nv, ids_nv, nv_tsv)

    # ── Inference on TORGO test set (English) ────────────────────────────
    print("\n" + "=" * 80)
    print("INFERENCE ON TORGO TEST SET (ENGLISH)")
    print("=" * 80)
    preds_tg, refs_tg, ids_tg = evaluate_model(
        model, processor, test_torgo,
        language="en",
        model_id=base_model_id,
        device=device,
    )
    torgo_tsv = f"{output_dir}/{model_name}_torgo_test_results.tsv"
    save_predictions(preds_tg, refs_tg, ids_tg, torgo_tsv)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Model:              {model_name}")
    print(f"NeuroVoz predictions: {len(preds_nv)} samples → {nv_tsv}")
    print(f"TORGO predictions:    {len(preds_tg)} samples → {torgo_tsv}")
    print("=" * 80)
    print("\nInference completed. Run generate_eval_summary.py on the TSV files to compute WER.")


if __name__ == "__main__":
    main()
