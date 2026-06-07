#!/usr/bin/env python3
"""Print total and trainable parameter counts per fine-tuning technique."""

import torch
from transformers import VoxtralForConditionalGeneration

CACHE_DIR = "models/"
BASE_MODEL = "mistralai/Voxtral-Mini-3B-2507"


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def main():
    print("Loading base model...")
    model = VoxtralForConditionalGeneration.from_pretrained(
        BASE_MODEL, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
    )

    total_all = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.audio_tower.parameters())
    projector_params = sum(p.numel() for p in model.multi_modal_projector.parameters())
    decoder_params = sum(p.numel() for p in model.language_model.parameters())

    print(f"\nBase model component breakdown:")
    print(f"  Audio encoder:          {fmt(encoder_params):>10}  ({encoder_params:,})")
    print(f"  Multi-modal projector:  {fmt(projector_params):>10}  ({projector_params:,})")
    print(f"  Language model decoder:  {fmt(decoder_params):>10}  ({decoder_params:,})")
    print(f"  Total:                  {fmt(total_all):>10}  ({total_all:,})")

    # --- Compute trainable params per technique ---

    # 1. Full fine-tune: everything
    full_trainable = total_all

    # 2. Encoder fine-tune: encoder + projector
    enc_trainable = encoder_params + projector_params

    # 3. LoRA fine-tune (full model): encoder attn+linear + decoder attn + projector
    #    r=8, targets: q,k,v,o_proj, fc1,fc2, up/down/gate_proj, projector
    from peft import LoraConfig, get_peft_model
    import copy

    model_lora_full = copy.deepcopy(model)
    lora_config_full = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=[
            "down_proj", "o_proj", "multi_modal_projector.linear_1",
            "up_proj", "fc1", "v_proj", "multi_modal_projector.linear_2",
            "k_proj", "q_proj", "fc2", "gate_proj",
        ],
        task_type="FEATURE_EXTRACTION",
    )
    model_lora_full = get_peft_model(model_lora_full, lora_config_full)
    lora_full_total, lora_full_trainable = count_params(model_lora_full)
    del model_lora_full

    # 4. Encoder LoRA: encoder attn+linear + projector only
    model_lora_enc = copy.deepcopy(model)
    lora_config_enc = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=[
            "audio_tower.layers.*.self_attn.q_proj",
            "audio_tower.layers.*.self_attn.k_proj",
            "audio_tower.layers.*.self_attn.v_proj",
            "audio_tower.layers.*.self_attn.out_proj",
            "audio_tower.layers.*.fc1",
            "audio_tower.layers.*.fc2",
            "multi_modal_projector.linear_1",
            "multi_modal_projector.linear_2",
        ],
        task_type="FEATURE_EXTRACTION",
    )
    model_lora_enc = get_peft_model(model_lora_enc, lora_config_enc)
    lora_enc_total, lora_enc_trainable = count_params(model_lora_enc)
    del model_lora_enc

    # 5. Speaker-conditioned (FiLM): FiLM bank + x-vector model
    #    FiLM bank: 32 generators, each with MLP(256→512→2*1280) + gate(256→256→1)
    #    x-vector: SiAMResNet34
    import sys, os
    sys.path.insert(0, os.path.join("src", "finetune", "spk_cond_voxtral"))
    from film import FiLMBank
    film_bank = FiLMBank(num_layers=32, spk_dim=256, hidden_dim=512, out_dim=1280, use_gate=True, mode="per_layer")
    film_params = sum(p.numel() for p in film_bank.parameters())

    xvec_model = torch.jit.load("models/SiAMResNet34/samresnet34_w_features.jit", map_location="cpu")
    xvec_params = sum(p.numel() for p in xvec_model.parameters())

    spk_cond_trainable = film_params + xvec_params  # encoder & projector frozen

    # --- Print table ---
    techniques = [
        ("Full fine-tune",    total_all, full_trainable),
        ("Encoder fine-tune", total_all, enc_trainable),
        ("LoRA fine-tune",    lora_full_total, lora_full_trainable),
        ("Encoder LoRA*",     lora_enc_total, lora_enc_trainable),
        ("Speaker-cond (FiLM)", total_all + film_params + xvec_params, spk_cond_trainable),
    ]

    print(f"\n{'='*80}")
    print(f"{'Technique':<25} {'Total Params':>14} {'Trainable':>14} {'%':>8}")
    print(f"{'-'*80}")
    for name, total, trainable in techniques:
        pct = trainable / total * 100
        print(f"{name:<25} {fmt(total):>14} {fmt(trainable):>14} {pct:>7.2f}%")
    print(f"{'='*80}")

    print(f"\nComponent breakdown (Speaker-cond):")
    print(f"  FiLM bank (32 generators + gates): {fmt(film_params):>10}  ({film_params:,})")
    print(f"  SiAMResNet34 (x-vector):           {fmt(xvec_params):>10}  ({xvec_params:,})")
    print(f"\n* Encoder LoRA: wildcard pattern matched projector only ({fmt(lora_enc_trainable)} params)")


if __name__ == "__main__":
    main()
