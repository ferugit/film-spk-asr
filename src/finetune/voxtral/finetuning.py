import os
import shutil
import argparse

import torch
from transformers import (
    EarlyStoppingCallback,
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from data import load_and_prepare_dataset, VoxtralDataCollator
from config import get_training_config

# Import utility functions from parent directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import get_compute_metrics_fn, get_weighted_sampler, print_sampler_stats


class WeightedSamplerTrainer(Seq2SeqTrainer):
    """Seq2Seq Trainer that supports weighted sampling."""
    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler
    
    def _get_train_sampler(self, train_dataset):
        if self._train_sampler is not None:
            return self._train_sampler
        return super()._get_train_sampler(train_dataset)


def freeze_decoder(model):
    """Freeze decoder parameters, keep encoder trainable."""
    frozen_params = 0
    
    # Freeze the language model (decoder + lm_head)
    if hasattr(model, 'language_model'):
        for param in model.language_model.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
    
    return model, frozen_params


def apply_lora(model, lora_config):
    """Apply LoRA adapters based on configuration."""
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=lora_config.get("lora_r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        modules_to_save=None,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Enable gradients flow on base model for LoRA
    model.enable_input_require_grads() 
    
    return model


def configure_model_for_training(model, strategy_config):
    """
    Configure model based on training strategy.
    
    Strategies:
        - "full": Train all parameters (default)
        - "encoder_only": Freeze decoder, train encoder
        - "lora": Apply LoRA adapters (most efficient)
    """
    strategy = strategy_config.get("training_strategy", "full")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    frozen_params = 0
    
    print("\n" + "="*60)
    print(f"TRAINING STRATEGY: {strategy.upper()}")
    print("="*60)
    
    if strategy == "full":
        # Full fine-tuning - all parameters trainable
        for param in model.parameters():
            param.requires_grad = True
            trainable_params += param.numel()
        
        print("Configuration: Full Fine-Tuning")
        print("  ✓ All model parameters trainable")
        
    elif strategy == "encoder_only":
        # Encoder-only fine-tuning - freeze decoder
        model, frozen = freeze_decoder(model)
        frozen_params = frozen
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("Configuration: Encoder-Only Fine-Tuning")
        print("  ✓ Audio Encoder trainable")
        print("  ✓ Audio Projector trainable")
        print("  ✗ Language Model Decoder frozen")
        print("  ✗ LM Head frozen")
        
    elif strategy == "lora":
        # LoRA fine-tuning - ultra efficient
        model = apply_lora(model, strategy_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print("Configuration: LoRA Fine-Tuning")
        print(f"  ✓ LoRA adapters (r={strategy_config.get('lora_r', 8)})")
        print(f"  ✓ Target modules: {strategy_config.get('lora_target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])}")
        print("  ✗ Base model frozen")
        
    else:
        raise ValueError(f"Unknown training strategy: {strategy}. Use 'full', 'encoder_only', or 'lora'")
    
    print(f"\nParameter Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print("="*60 + "\n")
    
    return model


def main():

    parser = argparse.ArgumentParser(
        description="Fine-tune Voxtral model on combined NeuroVoz + TORGO dataset with configurable strategy"
    )
        
    # Training configurations:
    parser.add_argument("--config_file", type=str, required=True,
                       help="Path to a YAML config file with training parameters")
    args = parser.parse_args()

    # Check if file exists
    if not os.path.isfile(args.config_file):
        print(f"Error: Config file not found at {args.config_file}")
        print("Please check the config file path.")
        return

    # Read config file and get training parameters
    training_config = get_training_config(args.config_file)
    
    # Set seed globally for reproducibility (affects all random operations)
    seed = training_config["seed"]
    set_seed(seed)  # Transformers function that sets all seeds (torch, numpy, random, etc.)
    print(f"\nRandom seed set to {seed} for reproducibility")

    # Configuration
    model_checkpoint = training_config["model_checkpoint"]
    output_dir = training_config["output_dir"]
    dataset_path = training_config["dataset_path"]
    strategy = training_config.get("training_strategy", "full")
    
    # Set device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")
    print(f"Fine-tuning Voxtral on combined NeuroVoz + TORGO dataset")
    print(f"  - NeuroVoz: Spanish speech with Parkinson's disease")
    print(f"  - TORGO: English speech with dysarthria")
    print(f"  - Strategy: {strategy}")
    
    # Load processor and model
    print("\nLoading processor and model...")
    processor = VoxtralProcessor.from_pretrained(model_checkpoint)

    # torch_dtype set to bfloat16 for memory efficiency
    torch_dtype = torch.bfloat16 if training_config.get("bf16", False) else torch.float32

    model = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=training_config["cache_dir"],
    )
    
    # Configure model based on strategy
    model = configure_model_for_training(model, training_config)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled to reduce memory usage")
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(dataset_path)
    
    # Setup data collator
    data_collator = VoxtralDataCollator(processor, model_checkpoint)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        bf16=training_config["bf16"],
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        eval_strategy=training_config["eval_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        save_total_limit=training_config["save_total_limit"],
        report_to=training_config["report_to"],
        remove_unused_columns=training_config["remove_unused_columns"],
        dataloader_num_workers=training_config["dataloader_num_workers"],
        warmup_steps=training_config["warmup_steps"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        weight_decay=training_config["weight_decay"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        max_grad_norm=training_config["max_grad_norm"],
        optim=training_config["optim"],
        predict_with_generate=False,  # Disabled: use loss for model selection instead of WER
        seed=training_config.get("seed", 42),  # Random seed for reproducibility
    )

    # Create weighted sampler
    train_sampler = None
    if training_config.get("use_weighted_sampler", False):
        spanish_weight = training_config.get("spanish_weight", 3.0)
        train_sampler = get_weighted_sampler(train_dataset, spanish_weight)
        print_sampler_stats(train_dataset, spanish_weight)
    
    # Setup trainer
    # Note: If using custom sampler, we can't shuffle
    trainer = WeightedSamplerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        #compute_metrics=get_compute_metrics_fn(processor.tokenizer, eval_dataset),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config["early_stopping_patience"]
            )
        ],
        train_sampler=train_sampler,
    )

    # Store the yaml config used for training
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.config_file, os.path.join(output_dir, "training_config.yaml"))
    
    # Save strategy information
    with open(os.path.join(output_dir, "training_strategy.txt"), "w") as f:
        f.write(f"TRAINING STRATEGY: {strategy.upper()}\n")
        f.write("="*60 + "\n\n")
        
        if strategy == "full":
            f.write("Configuration: Full Fine-Tuning\n")
            f.write("  - All model parameters trainable\n")
        elif strategy == "encoder_only":
            f.write("Configuration: Encoder-Only Fine-Tuning\n")
            f.write("  - Audio Encoder: trainable\n")
            f.write("  - Audio Projector: trainable\n")
            f.write("  - Decoder: frozen\n")
            f.write("  - LM Head: frozen\n")
        elif strategy == "lora":
            f.write("Configuration: LoRA Fine-Tuning\n")
            f.write(f"  - LoRA rank (r): {training_config.get('lora_r', 8)}\n")
            f.write(f"  - LoRA alpha: {training_config.get('lora_alpha', 16)}\n")
            f.write(f"  - LoRA dropout: {training_config.get('lora_dropout', 0.1)}\n")
            f.write(f"  - Target modules: {training_config.get('lora_target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])}\n")
    
    # Start training
    print(f"\nStarting {strategy} training...")
    trainer.train()

    # Save model and processor
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    # Final evaluation
    if eval_dataset:
        results = trainer.evaluate()
        print(f"\nFinal evaluation results: {results}")
    
    print(f"\n{strategy.upper()} fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
