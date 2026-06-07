# Voxtral Fine-Tuning Configurations

This directory contains YAML configuration files for different fine-tuning strategies.

## Available Configurations

### 1. `full_finetuning.yaml` - Full Model Fine-Tuning
**Use for**: Maximum performance, complete model adaptation

**Key settings**:
- Batch size: 1 per device (effective: 16 with grad accumulation)
- Learning rate: 1e-5 (conservative)
- Epochs: 3
- All parameters trainable (~3B params)

**Run with**:
```bash
python src/fine-tune/voxtral/full_finetuning.py \
    --config_file src/fine-tune/voxtral/config/full_finetuning.yaml
```

---

### 2. `encoder_only.yaml` - Encoder-Only Fine-Tuning
**Use for**: Fast acoustic adaptation, memory-efficient training

**Key settings**:
- Batch size: 2 per device (effective: 16 with grad accumulation)
- Learning rate: 2e-5 (higher than full)
- Epochs: 5 (faster per epoch)
- Only encoder + projector trainable (~20-30% of params)

**Run with**:
```bash
python src/fine-tune/voxtral/enc_finetuning.py \
    --config_file src/fine-tune/voxtral/config/encoder_only.yaml
```

---

## Configuration Parameters

### Model & Data
- `model_checkpoint`: HuggingFace model identifier
- `output_dir`: Where to save fine-tuned model
- `dataset_path`: Path to training dataset
- `cache_dir`: Cache directory for models

### Training
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Accumulate gradients for larger effective batch
- `learning_rate`: Initial learning rate
- `num_train_epochs`: Total training epochs

### Optimization
- `bf16`: Use bfloat16 precision (recommended for modern GPUs)
- `optim`: Optimizer type
- `weight_decay`: L2 regularization
- `max_grad_norm`: Gradient clipping threshold
- `gradient_checkpointing`: Save memory at cost of speed

### Evaluation & Checkpointing
- `eval_steps`: Evaluate every N steps
- `save_steps`: Save checkpoint every N steps
- `metric_for_best_model`: Metric to select best model (e.g., "wer")
- `early_stopping_patience`: Stop after N evals without improvement

### Learning Rate Schedule
- `warmup_steps`: Linear warmup steps
- `lr_scheduler_type`: LR schedule type (e.g., "cosine")

---

## Creating Custom Configurations

You can create your own YAML config by copying one of these templates and modifying parameters:

```bash
cp src/fine-tune/voxtral/config/full_finetuning.yaml my_experiment.yaml
# Edit my_experiment.yaml
python src/fine-tune/voxtral/full_finetuning.py --config_file my_experiment.yaml
```

---

## Monitoring Training

All configurations log to TensorBoard:

```bash
# Monitor full fine-tuning
tensorboard --logdir models/voxtral-finetuned-neurovoz-torgo/runs

# Monitor encoder-only
tensorboard --logdir models/voxtral-encoder-finetuned-neurovoz-torgo/runs
```

**Metrics logged**:
- `eval/wer` - Overall Word Error Rate
- `eval/wer_spanish` - Spanish (NeuroVoz) WER
- `eval/wer_english` - English (TORGO) WER
- `eval/loss` - Validation loss
- `train/learning_rate` - Learning rate schedule
- `train/loss` - Training loss

---

## Tips

1. **Start with encoder-only** for faster experimentation
2. **Adjust batch size** based on your GPU memory
3. **Monitor validation WER** by language to see if model favors one language
4. **Use early stopping** to prevent overfitting
5. **Compare results** between strategies for your paper

For more details, see `FINETUNING_STRATEGIES.md`
