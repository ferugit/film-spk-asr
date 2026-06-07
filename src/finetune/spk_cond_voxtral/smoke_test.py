"""
Smoke test for Speaker-Conditioned Voxtral.

Verifies:
  1. Model loads correctly
  2. Data collator produces correct outputs
  3. Forward pass runs without errors
  4. Backward pass produces gradients
  5. Only expected parameters are trainable

Usage:
    cd src/finetune/spk_cond_voxtral
    python smoke_test.py
"""

import sys
import os

# Ensure imports work when running from the script directory
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

from film import FiLMGenerator, FiLMBank
from model import SpeakerConditionedVoxtral
from data import SpkCondVoxtralDataCollator


def test_film_module():
    """Test FiLM generator produces correct shapes and near-identity init."""
    print("\n[1/5] Testing FiLM modules...")

    gen = FiLMGenerator(spk_dim=256, hidden_dim=512, out_dim=1280, use_gate=True)
    z = torch.randn(4, 256)
    gamma, beta, alpha = gen(z)

    assert gamma.shape == (4, 1280), f"gamma shape: {gamma.shape}"
    assert beta.shape == (4, 1280), f"beta shape: {beta.shape}"
    assert alpha.shape == (4, 1), f"alpha shape: {alpha.shape}"

    # Check near-identity initialization
    assert (gamma.mean() - 1.0).abs() < 0.1, f"gamma mean={gamma.mean():.3f} (expected ~1.0)"
    assert beta.abs().mean() < 0.1, f"beta mean={beta.abs().mean():.3f} (expected ~0.0)"
    assert alpha.mean() < 0.2, f"alpha mean={alpha.mean():.3f} (expected ~0.12)"

    print("  ✓ FiLMGenerator shapes correct")
    print(f"    γ mean={gamma.mean():.3f}, β mean={beta.mean():.3f}, α mean={alpha.mean():.3f}")

    # Test FiLM bank
    bank = FiLMBank(num_layers=32, spk_dim=256, hidden_dim=512, out_dim=1280, use_gate=True)
    h = torch.randn(4, 100, 1280)
    h_mod = bank.apply(0, h, z)
    assert h_mod.shape == h.shape, f"Shape mismatch: {h_mod.shape}"

    # Near-identity at init
    diff = (h_mod - h).abs().mean()
    print(f"  ✓ FiLMBank output shape correct, mean diff from identity: {diff:.4f}")

    n_params = sum(p.numel() for p in bank.parameters())
    print(f"  ✓ FiLMBank total params: {n_params:,}")

    return True


def test_xvector_model():
    """Test SiAMResNet34 JIT model loads and produces embeddings."""
    print("\n[2/5] Testing SiAMResNet34 x-vector model...")

    jit_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "models", "SiAMResNet34", "samresnet34_w_features.jit"
    )
    if not os.path.exists(jit_path):
        print(f"  ⚠ JIT model not found at {jit_path}, skipping")
        return False

    model = torch.jit.load(jit_path)
    model.eval()

    waveform = torch.randn(2, 16000 * 3)  # 3 seconds
    with torch.no_grad():
        emb = model(waveform)

    assert emb.shape == (2, 256), f"Embedding shape: {emb.shape}"
    print(f"  ✓ x-vector shape: {emb.shape}")

    # Test gradient flow
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    emb = model(waveform)
    emb.sum().backward()
    print("  ✓ Gradient flow through JIT model works")

    return True


def test_full_model():
    """Test the full SpeakerConditionedVoxtral model."""
    print("\n[3/5] Testing SpeakerConditionedVoxtral model...")

    model_checkpoint = "mistralai/Voxtral-Mini-3B-2507"
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
    jit_path = os.path.join(cache_dir, "SiAMResNet34", "samresnet34_w_features.jit")

    if not os.path.exists(jit_path):
        print(f"  ⚠ JIT model not found, skipping full model test")
        return False

    print("  Loading Voxtral (this takes a moment)...")
    voxtral = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )

    model = SpeakerConditionedVoxtral(
        voxtral_model=voxtral,
        xvector_model_path=jit_path,
        film_config={"hidden_dim": 512, "mode": "per_layer", "use_gate": True},
        train_xvector=True,
        train_encoder=False,
        train_projector=True,
    )

    print("  ✓ Model built successfully")
    return model, voxtral


def test_data_collator(processor, model_checkpoint):
    """Test the data collator produces correct batch format."""
    print("\n[4/5] Testing SpkCondVoxtralDataCollator...")

    collator = SpkCondVoxtralDataCollator(processor, model_checkpoint)

    # Create fake features
    features = [
        {
            "audio": {"array": np.random.randn(16000 * 2).astype(np.float32), "sampling_rate": 16000},
            "transcription": "Hola, esto es una prueba.",
            "dataset_source": "neurovoz",
            "speech_type": "PARKINSON",
        },
        {
            "audio": {"array": np.random.randn(16000 * 3).astype(np.float32), "sampling_rate": 16000},
            "transcription": "This is a test sentence.",
            "dataset_source": "torgo",
            "speech_type": "HC",
        },
    ]

    batch = collator(features)

    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  raw_waveforms shape: {batch['raw_waveforms'].shape}")
    print(f"  is_normative: {batch['is_normative']}")

    assert "raw_waveforms" in batch, "Missing raw_waveforms"
    assert "is_normative" in batch, "Missing is_normative"
    assert "input_features" in batch, "Missing input_features"
    assert batch["is_normative"][0] == False, "PARKINSON should not be normative"
    assert batch["is_normative"][1] == True, "HC should be normative"

    print("  ✓ Data collator produces correct outputs")
    return batch


def test_forward_backward(model, batch):
    """Test forward and backward pass."""
    print("\n[5/5] Testing forward and backward pass...")

    device = model.device

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    input_features = batch["input_features"].to(device)
    raw_waveforms = batch["raw_waveforms"].to(device)
    is_normative = batch["is_normative"].to(device)

    # Forward
    outputs = model(
        input_ids=input_ids,
        input_features=input_features,
        attention_mask=attention_mask,
        labels=labels,
        raw_waveforms=raw_waveforms,
        is_normative=is_normative,
    )

    loss = outputs.loss
    print(f"  Loss: {loss.item():.4f}")
    assert loss.requires_grad, "Loss should require grad"

    # Backward
    loss.backward()

    # Check gradients exist for trainable components
    film_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.film_bank.parameters()
    )
    xvec_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.xvector_model.parameters()
        if p.requires_grad
    )
    proj_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.voxtral.multi_modal_projector.parameters()
    )

    # Check decoder is frozen
    decoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.voxtral.language_model.parameters()
    )

    print(f"  FiLM bank has gradients:   {film_has_grad}")
    print(f"  x-vector has gradients:    {xvec_has_grad}")
    print(f"  Projector has gradients:   {proj_has_grad}")
    print(f"  Decoder has gradients:     {decoder_has_grad} (should be False)")

    assert film_has_grad, "FiLM bank should have gradients!"
    assert not decoder_has_grad, "Decoder should NOT have gradients!"

    print("  ✓ Forward/backward pass successful")
    return True


def main():
    print("=" * 60)
    print("SMOKE TEST: Speaker-Conditioned Voxtral")
    print("=" * 60)

    # 1) FiLM modules
    test_film_module()

    # 2) x-vector model
    test_xvector_model()

    # 3) Full model
    model_checkpoint = "mistralai/Voxtral-Mini-3B-2507"
    result = test_full_model()
    if result is False:
        print("\n⚠ Skipping full model tests (missing dependencies)")
        return

    model, voxtral = result

    # 4) Data collator
    processor = VoxtralProcessor.from_pretrained(model_checkpoint)
    batch = test_data_collator(processor, model_checkpoint)

    # 5) Forward/backward
    test_forward_backward(model, batch)

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
