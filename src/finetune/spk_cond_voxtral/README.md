# Speaker-Conditioned Voxtral (FiLM + SiAMResNet34)

## Architecture

```
                            ┌────────────────────┐
                            │  SiAMResNet34 (JIT) │  ← trainable
                            │  x-vector extractor │
                            └────────┬───────────┘
                                     │
                              z ∈ R^256
                                     │
┌──────────────┐              ┌──────▼──────┐
│ Mel features │──►  Voxtral  │  FiLM Bank  │  ← trainable
│              │    Encoder   │  (per-layer) │
│              │   (frozen)   │  γ, β, α    │
│              │     + FiLM   └─────────────┘
│              │       │
│              │       ▼
│              │   Projector   ← trainable
│              │       │
│              │       ▼
│              │   LLM Decoder ← frozen
│              │       │
└──────────────┘       ▼
                     text
```

### FiLM Conditioning

Each Voxtral encoder layer (32 total) is conditioned via:

```
FiLM(h) = α · (γ ⊙ h + β) + (1 - α) · h
```

Where:
- `γ, β ∈ R^1280` are scale/shift from a small MLP conditioned on x-vector
- `α ∈ [0,1]` is a learnable gate (per layer)
- For **normative** speech (HC, Unknown): x-vector is zeroed → FiLM ≈ identity
- For **pathological** speech: FiLM adapts encoder representations

### Identity Conditioning

- Normative speakers (HC, Unknown `speech_type`) receive a zero x-vector
- FiLM generators are initialized near identity (γ≈1, β≈0, gate α≈0.12)
- The model learns when and how much to modulate for pathological speech

## Files

| File | Description |
|------|-------------|
| `film.py` | FiLM modules (FiLMGenerator, FiLMLayer, FiLMBank) |
| `model.py` | SpeakerConditionedVoxtral wrapper |
| `data.py` | Data collator with raw waveforms + normative mask |
| `config.py` | Configuration reader |
| `finetuning.py` | Main training script |
| `precompute_xvectors.py` | Offline x-vector extraction |
| `config/film_conditioning.yaml` | Training configuration |
| `finetune.sh` | Shell wrapper for training |

## Usage

### Training

```bash
# From project root
python src/finetune/spk_cond_voxtral/finetuning.py \
    --config_file src/finetune/spk_cond_voxtral/config/film_conditioning.yaml

# Or use the shell script
bash src/finetune/spk_cond_voxtral/finetune.sh
```

### Pre-compute x-vectors (optional, for analysis)

```bash
python src/finetune/spk_cond_voxtral/precompute_xvectors.py \
    --dataset_path data/combined_neurovoz_torgo_cv \
    --jit_path models/SiAMResNet34/samresnet34_w_features.jit \
    --output_path data/xvectors_neurovoz_torgo_cv.pt
```

## Trainable Parameters

With default config (encoder frozen, FiLM per-layer, x-vector trainable):

| Component | Params | Status |
|-----------|--------|--------|
| SiAMResNet34 | ~25.2M | ✓ trainable (lr × 0.1) |
| FiLM Bank (32 layers) | ~84.1M | ✓ trainable |
| Multi-Modal Projector | ~7.9M | ✓ trainable |
| Voxtral Encoder | ~637M | ✗ frozen |
| LLM Decoder | ~2.3B | ✗ frozen |

## Dataset

Uses `data/combined_neurovoz_torgo_cv`:
- **NeuroVoz** (es): Parkinson's disease speech
- **TORGO** (en): Dysarthric speech
- **CommonVoice ES** (es): Normative Spanish speech

Speech types:
- `HC` → normative (identity conditioning)
- `Unknown` → normative (identity conditioning)
- `PARKINSON` → pathological (FiLM conditioning)
- `DYSARTHRIC` → pathological (FiLM conditioning)
