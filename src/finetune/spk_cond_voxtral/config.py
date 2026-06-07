"""
Configuration for Speaker-Conditioned Voxtral training.

Reads a YAML config and provides sensible defaults.
"""

import os
import yaml

# Project root: three levels up from this file (src/finetune/spk_cond_voxtral/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Paths that should be resolved relative to the project root
_PATH_KEYS = ("cache_dir", "output_dir", "dataset_path", "xvector_model_path")


def _resolve_paths(config: dict) -> dict:
    """Resolve relative paths in config against PROJECT_ROOT."""
    for key in _PATH_KEYS:
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.join(PROJECT_ROOT, config[key])
    return config


def read_yaml_config(config_path):
    """Read YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_training_config(config_path=None):
    """
    Get training configuration from YAML file with defaults.

    Adds speaker-conditioning-specific keys on top of the
    standard Voxtral fine-tuning defaults.
    """
    default_config = {
        # --- Model ---
        "model_checkpoint": "mistralai/Voxtral-Mini-3B-2507",
        "output_dir": "./models/voxtral-spk-cond-neurovoz-torgo-cv",
        "dataset_path": "data/combined_neurovoz_torgo_cv",
        "cache_dir": "./models",

        # --- SiAMResNet34 / x-vector ---
        "xvector_model_path": "models/SiAMResNet34/samresnet34_w_features.jit",
        "train_xvector": True,
        "xvector_lr_scale": 0.1,   # LR multiplier for x-vector model vs FiLM

        # --- FiLM ---
        "film_hidden_dim": 512,
        "film_mode": "per_layer",  # "per_layer" or "shared"
        "film_use_gate": True,

        # --- What to train ---
        "train_encoder": False,    # Voxtral audio encoder
        "train_projector": True,   # multi-modal projector

        # --- Weighted sampling ---
        "use_weighted_sampler": True,
        "spanish_weight": 3.0,

        # --- Training hyperparams ---
        "seed": 2026,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,  # effective batch = 16
        "learning_rate": 2e-4,
        "num_train_epochs": 10,

        "bf16": True,
        "optim": "adamw_torch_fused",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,

        "logging_steps": 50,
        "eval_steps": 200,
        "save_steps": 200,
        "eval_strategy": "steps",
        "save_strategy": "steps",

        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,

        "report_to": "tensorboard",
        "remove_unused_columns": False,
        "dataloader_num_workers": 4,
        "warmup_steps": 200,
        "lr_scheduler_type": "cosine",
        "early_stopping_patience": 5,
        "generation_max_length": 512,
        "dataloader_pin_memory": True,
    }

    if config_path:
        config = read_yaml_config(config_path)
        default_config.update(config)

        # Ensure learning_rate is float
        if "learning_rate" in default_config:
            default_config["learning_rate"] = float(default_config["learning_rate"])

    # Resolve all relative paths against the project root
    _resolve_paths(default_config)

    return default_config
