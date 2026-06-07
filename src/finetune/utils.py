import torch
import evaluate as evaluate_lib
from torch.utils.data import WeightedRandomSampler


def compute_metrics(pred, tokenizer, metric, dataset_sources=None):
    """Compute WER metric for evaluation, with optional language-specific breakdown."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predictions and references
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute overall WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    metrics = {"wer": wer}
    
    # If dataset sources available, compute language-specific WER
    if dataset_sources is not None and len(dataset_sources) == len(pred_str):
        # Separate by language: neurovoz=Spanish, torgo=English
        spanish_preds, spanish_refs = [], []
        english_preds, english_refs = [], []
        
        for i, source in enumerate(dataset_sources):
            if source == "neurovoz":
                spanish_preds.append(pred_str[i])
                spanish_refs.append(label_str[i])
            else:  # torgo or other
                english_preds.append(pred_str[i])
                english_refs.append(label_str[i])
        
        # Compute Spanish WER if we have Spanish samples
        if spanish_preds:
            wer_es = 100 * metric.compute(predictions=spanish_preds, references=spanish_refs)
            metrics["wer_spanish"] = wer_es
        
        # Compute English WER if we have English samples
        if english_preds:
            wer_en = 100 * metric.compute(predictions=english_preds, references=english_refs)
            metrics["wer_english"] = wer_en
    
    return metrics


def compute_whisper_metrics(eval_preds, tokenizer, metric, dataset_sources=None, trainer=None):
    # Only rank 0 runs generation
    if trainer is not None and not trainer.is_world_process_zero():
        return {}

    model = trainer.model
    model.eval()

    dataloader = trainer.get_eval_dataloader()
    predictions, references = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = trainer._prepare_inputs(batch)
            generated_ids = model.generate(
                batch["input_features"],
                max_length=trainer.args.generation_max_length,
            )
            predictions.extend(generated_ids.cpu())
            references.extend(batch["labels"].cpu())

    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(references, skip_special_tokens=True)

    metrics = {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    # Optional language-specific WER
    if dataset_sources is not None and len(dataset_sources) == len(pred_str):
        spanish_preds, spanish_refs = [], []
        english_preds, english_refs = [], []

        for i, source in enumerate(dataset_sources):
            if source == "neurovoz":
                spanish_preds.append(pred_str[i])
                spanish_refs.append(label_str[i])
            else:
                english_preds.append(pred_str[i])
                english_refs.append(label_str[i])

        if spanish_preds:
            metrics["wer_spanish"] = 100 * metric.compute(predictions=spanish_preds, references=spanish_refs)
        if english_preds:
            metrics["wer_english"] = 100 * metric.compute(predictions=english_preds, references=english_refs)

    return metrics


def get_compute_metrics_fn(tokenizer, eval_dataset=None):
    """
    Factory function to create a compute_metrics function with WER metric.
    
    Args:
        tokenizer: The tokenizer to use for decoding predictions and labels
        eval_dataset: Optional evaluation dataset to extract dataset_source for language-specific metrics
        
    Returns:
        A compute_metrics function ready to be used by Trainer
    """
    wer_metric = evaluate_lib.load("wer")
    
    # Extract dataset sources if available
    dataset_sources = None
    if eval_dataset is not None and "dataset_source" in eval_dataset.column_names:
        dataset_sources = eval_dataset["dataset_source"]
    
    return lambda pred: compute_metrics(pred, tokenizer, wer_metric, dataset_sources)


def get_compute_whisper_metrics_fn(trainer, tokenizer, eval_dataset=None):
    """
    Factory function to create a compute_metrics function with WER metric.
    
    Args:
        tokenizer: The tokenizer to use for decoding predictions and labels
        eval_dataset: Optional evaluation dataset to extract dataset_source for language-specific metrics
        
    Returns:
        A compute_metrics function ready to be used by Trainer
    """
    wer_metric = evaluate_lib.load("wer")
    
    # Extract dataset sources if available
    dataset_sources = None
    if eval_dataset is not None and "dataset_source" in eval_dataset.column_names:
        dataset_sources = eval_dataset["dataset_source"]
    
    return lambda pred: compute_whisper_metrics(pred, tokenizer, wer_metric, dataset_sources, trainer)


def get_weighted_sampler(dataset, spanish_weight=3.0):
    """
    Create a weighted sampler to oversample Spanish (NeuroVoz) samples.
    
    Args:
        dataset: The training dataset with 'dataset_source' column
        spanish_weight: Weight multiplier for Spanish samples (default: 3.0)
                       Higher values = more Spanish samples in each epoch
                       
    Returns:
        WeightedRandomSampler for use in Trainer
        
    Example weights:
        - spanish_weight=1.0 : No oversampling (balanced by count)
        - spanish_weight=3.0 : Spanish samples 3x more likely to be selected
        - spanish_weight=6.0 : Spanish samples 6x more likely (aggressive balancing)
    """
    if "dataset_source" not in dataset.column_names:
        raise ValueError("Dataset must have 'dataset_source' column for weighted sampling")
    
    # Get dataset sources
    sources = dataset["dataset_source"]
    
    # Calculate weights for each sample
    weights = []
    for source in sources:
        if source == "neurovoz":  # Spanish
            weights.append(spanish_weight)
        else:  # English (TORGO)
            weights.append(1.0)
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True  # Allow sampling with replacement
    )
    
    return sampler


def print_sampler_stats(dataset, spanish_weight=3.0):
    """
    Print statistics about how the sampler will affect data distribution.
    
    Args:
        dataset: The training dataset
        spanish_weight: The weight being used for Spanish samples
    """
    sources = dataset["dataset_source"]
    
    # Count samples
    n_spanish = sum(1 for s in sources if s == "neurovoz")
    n_english = sum(1 for s in sources if s != "neurovoz")
    total = len(sources)
    
    # Calculate effective distribution with weighting
    total_weight = (n_spanish * spanish_weight) + (n_english * 1.0)
    effective_spanish_pct = (n_spanish * spanish_weight) / total_weight * 100
    effective_english_pct = (n_english * 1.0) / total_weight * 100
    
    print("\n" + "="*60)
    print("WEIGHTED SAMPLER STATISTICS")
    print("="*60)
    print(f"Original distribution:")
    print(f"  Spanish (NeuroVoz): {n_spanish:,} samples ({n_spanish/total*100:.1f}%)")
    print(f"  English (TORGO):    {n_english:,} samples ({n_english/total*100:.1f}%)")
    print(f"\nWith spanish_weight={spanish_weight}:")
    print(f"  Effective Spanish: {effective_spanish_pct:.1f}% of samples per epoch")
    print(f"  Effective English: {effective_english_pct:.1f}% of samples per epoch")
    print(f"\nExpected samples per epoch:")
    print(f"  Spanish: ~{int(total * effective_spanish_pct / 100):,} samples")
    print(f"  English: ~{int(total * effective_english_pct / 100):,} samples")
    print("="*60 + "\n")

