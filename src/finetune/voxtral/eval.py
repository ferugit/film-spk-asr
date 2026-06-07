import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from datasets import load_from_disk, Audio
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor


def evaluate_model(model, processor, dataset, language="es", model_id=None, device="cuda"):
    """
    Evaluate model on a dataset and return predictions with references.
    
    Args:
        model: Voxtral model
        processor: Voxtral processor
        dataset: HuggingFace dataset with audio and transcription
        language: Language code ("es" for Spanish, "en" for English)
        model_id: Model ID for processor
        device: Device to run inference on
    
    Returns:
        predictions: List of predicted transcriptions
        references: List of ground truth transcriptions
        sample_ids: List of sample IDs
    """
    predictions = []
    references = []
    sample_ids = []
    
    print(f"Evaluating {len(dataset)} samples...")
    
    for sample in tqdm(dataset, desc="Generating predictions"):
        try:
            # Get audio array
            audio_array = sample["audio"]["array"]
            
            # Prepare input with transcription request
            inputs = processor.apply_transcription_request(
                language=language,
                audio=audio_array,
                model_id=model_id,
                format=["WAV"],
                return_tensors="pt",
            )
            inputs = inputs.to(device, dtype=torch.bfloat16)
            
            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256)
            
            # Decode output (skip the prompt part)
            decoded_output = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )[0]
            
            predictions.append(decoded_output.strip())
            references.append(sample["transcription"].strip())
            sample_ids.append(sample["sample_id"])
            
        except Exception as e:
            print(f"\nError processing sample {sample.get('sample_id', 'unknown')}: {e}")
            predictions.append("")
            references.append(sample["transcription"].strip())
            sample_ids.append(sample["sample_id"])
    
    return predictions, references, sample_ids


def save_predictions(predictions, references, sample_ids, output_path):
    """Save inference predictions to a TSV file.

    Columns: sample_id, reference, hypothesis
    """
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Voxtral model (base or fine-tuned) on NeuroVoz and TORGO test sets"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., 'mistralai/Voxtral-Mini-3B-2507' or 'models/voxtral-finetuned-neurovoz-torgo')"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Short name for model (e.g., 'base' or 'finetuned') - used for output file naming"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/combined_neurovoz_torgo",
        help="Path to dataset directory (default: data/combined_neurovoz_torgo)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results (default: results/evaluation)"
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="mistralai/Voxtral-Mini-3B-2507",
        help="Base model ID for processor (default: mistralai/Voxtral-Mini-3B-2507)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./models",
        help="Cache directory for models (default: ./models)"
    )
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    model_checkpoint = args.model_path
    model_name = args.model_name
    base_model_id = args.base_model_id
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    cache_dir = args.cache_dir
    
    # Check if model exists
    if not os.path.exists(model_checkpoint) and not model_checkpoint.startswith("mistralai/"):
        print(f"Error: Model not found at {model_checkpoint}")
        print("Please check the model path or train the model first.")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_checkpoint}")
    print(f"Model name: {model_name}")
    print(f"Using device: {device}")
    print("=" * 80)
    
    # Load processor and model
    print(f"Loading processor and model from: {model_checkpoint}")
    processor = VoxtralProcessor.from_pretrained(
        model_checkpoint,
        cache_dir=cache_dir,
    )
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
    print("=" * 80)
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Cast audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Get test splits
    test_neurovoz = dataset["test_neurovoz"]
    test_torgo = dataset["test_torgo"]
    
    print(f"Test NeuroVoz samples: {len(test_neurovoz)}")
    print(f"Test TORGO samples: {len(test_torgo)}")
    print("=" * 80)
    
    # ── Inference on NeuroVoz test set (Spanish) ─────────────────────────
    print("\n" + "=" * 80)
    print("INFERENCE ON NEUROVOZ TEST SET (SPANISH)")
    print("=" * 80)
    predictions_nv, references_nv, sample_ids_nv = evaluate_model(
        model, processor, test_neurovoz, 
        language="es",
        model_id=base_model_id,
        device=device
    )
    
    # Save NeuroVoz predictions
    nv_tsv = f"{output_dir}/{model_name}_neurovoz_test_results.tsv"
    save_predictions(predictions_nv, references_nv, sample_ids_nv, nv_tsv)
    
    # ── Inference on TORGO test set (English) ────────────────────────────
    print("\n" + "=" * 80)
    print("INFERENCE ON TORGO TEST SET (ENGLISH)")
    print("=" * 80)
    predictions_torgo, references_torgo, sample_ids_torgo = evaluate_model(
        model, processor, test_torgo,
        language="en",
        model_id=base_model_id,
        device=device
    )
    
    # Save TORGO predictions
    torgo_tsv = f"{output_dir}/{model_name}_torgo_test_results.tsv"
    save_predictions(predictions_torgo, references_torgo, sample_ids_torgo, torgo_tsv)
    
    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"NeuroVoz predictions: {len(predictions_nv)} samples → {nv_tsv}")
    print(f"TORGO predictions:    {len(predictions_torgo)} samples → {torgo_tsv}")
    print("=" * 80)
    print("\nInference completed. Run calculate_wer.py on the TSV files to compute WER.")


if __name__ == "__main__":
    main()
