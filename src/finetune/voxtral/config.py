import yaml


# Read config file yaml and stablish default parameters if not set

def read_yaml_config(config_path):
    """Read YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_training_config(config_path=None):
    """Get training configuration from YAML file or use default parameters."""
    # Default training parameters
    default_config = {
        'model_checkpoint': "mistralai/Voxtral-Mini-3B-2507",
        'output_dir': "./models/voxtral-finetuned-neurovoz-torgo",
        'dataset_path': "data/combined_neurovoz_torgo",
        'cache_dir': "./models",
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': 16,
        'learning_rate': 1e-5,
        'num_train_epochs': 3,
        'bf16': True,
        'logging_steps': 50,
        'eval_steps': 200,
        'save_steps': 200,
        'eval_strategy': "steps",
        'save_strategy': "steps",
        'load_best_model_at_end': True,
        'metric_for_best_model': "wer",
        'greater_is_better': False,
        'save_total_limit': 3,
        'report_to': "tensorboard",
        'remove_unused_columns': False,
        'dataloader_num_workers': 2,
        'warmup_steps': 100,
        'lr_scheduler_type': "cosine",  # Cosine scheduler with warmup for smooth LR decay
        'weight_decay': 0.01,
        'gradient_checkpointing': True,
        'max_grad_norm': 1.0,
        'optim': "adamw_torch_fused",
        'early_stopping_patience': 5,
    }

    if config_path:
        config = read_yaml_config(config_path)
        # Update default config with values from the file
        default_config.update(config)
        
        # Ensure learning_rate is float
        if 'learning_rate' in default_config:
            default_config['learning_rate'] = float(default_config['learning_rate'])
    
    return default_config


