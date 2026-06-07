"""
Fine-tuning script for Speaker-Conditioned Voxtral.

Usage:
    python src/finetune/spk_cond_voxtral/finetuning.py \
        --config_file src/finetune/spk_cond_voxtral/config/film_conditioning.yaml
"""

import os
import sys
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

from data import load_and_prepare_dataset, SpkCondVoxtralDataCollator
from config import get_training_config
from model import SpeakerConditionedVoxtral

# Import shared utilities from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import get_compute_metrics_fn, get_weighted_sampler, print_sampler_stats


# ---------------------------------------------------------------
# Custom Trainer that handles:
#   1) Weighted sampling
#   2) Separate learning rates for x-vector vs FiLM
#   3) Passing extra inputs (raw_waveforms, is_normative) to model
# ---------------------------------------------------------------

class SpkCondTrainer(Seq2SeqTrainer):
    """
    Trainer subclass for Speaker-Conditioned Voxtral.

    - Supports weighted random sampling.
    - Creates separate optimizer param groups with different LRs
      for the x-vector model vs FiLM / projector.
    """

    def __init__(self, *args, train_sampler=None,
                 xvector_lr_scale=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler
        self.xvector_lr_scale = xvector_lr_scale

    def _get_train_sampler(self, train_dataset):
        if self._train_sampler is not None:
            return self._train_sampler
        return super()._get_train_sampler(train_dataset)

    def create_optimizer(self):
        """Create optimizer with separate LR for x-vector model."""
        if self.optimizer is not None:
            return self.optimizer

        model = self.model

        # Separate parameter groups
        xvector_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("xvector_model."):
                xvector_params.append(param)
            else:
                other_params.append(param)

        base_lr = self.args.learning_rate
        xvec_lr = base_lr * self.xvector_lr_scale

        param_groups = []
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": base_lr,
                "name": "film_projector",
            })
        if xvector_params:
            param_groups.append({
                "params": xvector_params,
                "lr": xvec_lr,
                "name": "xvector",
            })

        print(f"\nOptimizer parameter groups:")
        print(f"  FiLM / Projector: {sum(p.numel() for p in other_params):,} params  @ lr={base_lr}")
        print(f"  SiAMResNet34:     {sum(p.numel() for p in xvector_params):,} params  @ lr={xvec_lr}")

        # Use the optimizer class from training args
        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(
            self.args, model
        )
        # Remove lr from kwargs since we set it per group
        optimizer_kwargs.pop("lr", None)

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override to ensure raw_waveforms and is_normative are passed to model.
        The default Trainer would drop them since they're not standard HF inputs.
        """
        # Extract our custom inputs
        raw_waveforms = inputs.pop("raw_waveforms", None)
        is_normative = inputs.pop("is_normative", None)

        # Move to correct device
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        if raw_waveforms is not None:
            raw_waveforms = raw_waveforms.to(device)
        if is_normative is not None:
            is_normative = is_normative.to(device)

        # Forward pass
        outputs = model(
            raw_waveforms=raw_waveforms,
            is_normative=is_normative,
            **inputs,
        )

        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # With device_map="auto" the LLM decoder (and thus the loss) may
        # live on a different GPU than cuda:0.  The Trainer requires the
        # loss to be on self.args.device (cuda:0), so we move it.
        if loss.device != self.args.device:
            loss = loss.to(self.args.device)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Speaker-Conditioned Voxtral (FiLM + SiAMResNet34)"
    )
    parser.add_argument(
        "--config_file", type=str, required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        return

    # ---- Config ----
    cfg = get_training_config(args.config_file)

    seed = cfg["seed"]
    set_seed(seed)
    print(f"\nRandom seed: {seed}")

    model_checkpoint = cfg["model_checkpoint"]
    output_dir = cfg["output_dir"]
    dataset_path = cfg["dataset_path"]

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Speaker-Conditioned Voxtral Fine-Tuning")
    print(f"  Dataset:   {dataset_path}")
    print(f"  x-vector:  {cfg['xvector_model_path']}")
    print(f"  FiLM mode: {cfg['film_mode']}, gate={cfg['film_use_gate']}")
    print(f"  Train x-vector: {cfg['train_xvector']}")
    print(f"  Train encoder:  {cfg['train_encoder']}")
    print(f"  Train projector: {cfg['train_projector']}")

    # ---- Load Voxtral processor & base model ----
    print("\nLoading Voxtral processor and model...")
    processor = VoxtralProcessor.from_pretrained(model_checkpoint)

    torch_dtype = torch.bfloat16 if cfg.get("bf16", False) else torch.float32
    voxtral_model = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=cfg["cache_dir"],
    )

    # ---- Build Speaker-Conditioned model ----
    print("\nBuilding Speaker-Conditioned Voxtral...")
    model = SpeakerConditionedVoxtral(
        voxtral_model=voxtral_model,
        xvector_model_path=cfg["xvector_model_path"],
        film_config={
            "hidden_dim": cfg["film_hidden_dim"],
            "mode": cfg["film_mode"],
            "use_gate": cfg["film_use_gate"],
        },
        train_xvector=cfg["train_xvector"],
        train_encoder=cfg["train_encoder"],
        train_projector=cfg["train_projector"],
    )

    # Enable gradient checkpointing
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # ---- Dataset ----
    train_dataset, eval_dataset = load_and_prepare_dataset(dataset_path)

    # ---- Data collator ----
    data_collator = SpkCondVoxtralDataCollator(processor, model_checkpoint)

    # ---- Training arguments ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg["report_to"],
        remove_unused_columns=False,  # Must be False — we have custom columns
        dataloader_num_workers=cfg["dataloader_num_workers"],
        warmup_steps=cfg["warmup_steps"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=cfg["weight_decay"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        predict_with_generate=False,
        seed=seed,
        dataloader_pin_memory=cfg.get("dataloader_pin_memory", True),
    )

    # ---- Weighted sampler ----
    train_sampler = None
    if cfg.get("use_weighted_sampler", False):
        spanish_weight = cfg.get("spanish_weight", 3.0)
        train_sampler = get_weighted_sampler(train_dataset, spanish_weight)
        print_sampler_stats(train_dataset, spanish_weight)

    # ---- Trainer ----
    trainer = SpkCondTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"]
            )
        ],
        train_sampler=train_sampler,
        xvector_lr_scale=cfg.get("xvector_lr_scale", 0.1),
    )

    # ---- Save config ----
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.config_file, os.path.join(output_dir, "training_config.yaml"))

    # Save architecture description
    with open(os.path.join(output_dir, "architecture.txt"), "w") as f:
        f.write("SPEAKER-CONDITIONED VOXTRAL (FiLM)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Base model:      {model_checkpoint}\n")
        f.write(f"x-vector model:  {cfg['xvector_model_path']}\n")
        f.write(f"FiLM mode:       {cfg['film_mode']}\n")
        f.write(f"FiLM hidden dim: {cfg['film_hidden_dim']}\n")
        f.write(f"FiLM gate:       {cfg['film_use_gate']}\n")
        f.write(f"Train x-vector:  {cfg['train_xvector']}\n")
        f.write(f"Train encoder:   {cfg['train_encoder']}\n")
        f.write(f"Train projector: {cfg['train_projector']}\n")
        f.write(f"Dataset:         {dataset_path}\n")

    # ---- Train ----
    print(f"\nStarting Speaker-Conditioned Voxtral training...")
    trainer.train()

    # ---- Save ----
    print(f"\nSaving model to {output_dir}")
    # Save the full model state dict (FiLM + x-vector + projector)
    torch.save(model.state_dict(), os.path.join(output_dir, "spk_cond_voxtral.pt"))
    # Also save FiLM bank separately for easy loading
    torch.save(model.film_bank.state_dict(), os.path.join(output_dir, "film_bank.pt"))
    # Save x-vector model if trained
    if cfg["train_xvector"]:
        torch.save(
            {k: v for k, v in model.state_dict().items() if k.startswith("xvector_model.")},
            os.path.join(output_dir, "xvector_finetuned.pt"),
        )
    # Save processor
    processor.save_pretrained(output_dir)

    # ---- Evaluate ----
    if eval_dataset:
        results = trainer.evaluate()
        print(f"\nFinal evaluation: {results}")

    print("\nSpeaker-Conditioned Voxtral training completed!")


if __name__ == "__main__":
    main()
