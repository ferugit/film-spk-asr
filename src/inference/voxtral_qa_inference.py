#!/usr/bin/env python3
"""Run Voxtral inference on the pathological-speech QA benchmark (one sample at a time).

Supports three model types (auto-detected from the model directory):
  1. Full / encoder-only fine-tunes  →  standard from_pretrained
  2. LoRA adapters                   →  PeftModel + merge_and_unload
  3. Speaker-conditioned (FiLM)      →  SpeakerConditionedVoxtral wrapper
"""

import argparse
import json
import os
import re
import string
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from tqdm import tqdm


CHOICE_LETTERS = list(string.ascii_uppercase)  # A, B, C, D, ...

# Speech types considered normative for spk-cond model
NORMATIVE_SPEECH_TYPES = {"HC", "Unknown"}


def build_prompt(question: str, choices: list[str]) -> str:
    """Format the MCQA prompt with lettered choices."""
    options = "\n".join(
        f"({CHOICE_LETTERS[i]}) {c}" for i, c in enumerate(choices)
    )
    return (
        f"{question}\n{options}\n\n"
        "Answer with only one of the options in the exact format: (A) xxx. "
        "Do not add any other text."
    )


def detect_model_type(model_path: str) -> str:
    """Auto-detect model type from directory contents.

    Returns one of: 'lora', 'spk_cond', 'standard'.
    """
    p = Path(model_path)
    if not p.is_dir():
        return "standard"  # HuggingFace hub ID

    # Speaker-conditioned: has spk_cond_voxtral.pt
    if (p / "spk_cond_voxtral.pt").exists():
        return "spk_cond"

    # LoRA: has adapter_config.json at top level
    if (p / "adapter_config.json").exists():
        return "lora"

    # LoRA: only inside checkpoint-* subdirs (no top-level merge)
    checkpoints = sorted(p.glob("checkpoint-*/adapter_config.json"))
    if checkpoints:
        return "lora"

    return "standard"


def resolve_lora_adapter_path(model_path: str) -> str:
    """Return the path containing adapter_config.json.

    If the top-level dir has it, return model_path.
    Otherwise pick the highest-numbered checkpoint-* subdir.
    """
    p = Path(model_path)
    if (p / "adapter_config.json").exists():
        return model_path

    checkpoints = sorted(
        p.glob("checkpoint-*/adapter_config.json"),
        key=lambda cp: int(cp.parent.name.split("-")[-1]),
    )
    if checkpoints:
        return str(checkpoints[-1].parent)

    raise FileNotFoundError(
        f"No adapter_config.json found in {model_path} or its checkpoint-* subdirs"
    )


def load_model_and_processor(model_path: str, torch_dtype, base_model_id: str):
    """Load model + processor, handling standard / LoRA / spk-cond transparently.

    Returns (model, processor, model_type).
    """
    model_type = detect_model_type(model_path)
    print(f"Detected model type: {model_type}")

    if model_type == "standard":
        processor = AutoProcessor.from_pretrained(model_path)
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="cuda",
        )
        model.eval()
        return model, processor, model_type

    if model_type == "lora":
        from peft import PeftModel

        adapter_path = resolve_lora_adapter_path(model_path)
        # Read base_model_name_or_path from adapter config
        adapter_cfg = json.loads(
            (Path(adapter_path) / "adapter_config.json").read_text()
        )
        lora_base = adapter_cfg.get("base_model_name_or_path", base_model_id)
        print(f"  LoRA base model: {lora_base}")
        print(f"  Adapter path:    {adapter_path}")

        # Try loading processor from the finetuned dir first, fall back to base
        try:
            processor = AutoProcessor.from_pretrained(model_path)
        except Exception:
            processor = AutoProcessor.from_pretrained(lora_base)

        base_model = VoxtralForConditionalGeneration.from_pretrained(
            lora_base, torch_dtype=torch_dtype, device_map="cuda",
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
        model.eval()
        return model, processor, model_type

    if model_type == "spk_cond":
        # Import from the spk_cond_voxtral package
        spk_cond_dir = os.path.join(
            os.path.dirname(__file__), "..", "finetune", "spk_cond_voxtral"
        )
        sys.path.insert(0, os.path.abspath(spk_cond_dir))
        from eval import build_model_from_checkpoint

        try:
            processor = AutoProcessor.from_pretrained(model_path)
        except Exception:
            processor = AutoProcessor.from_pretrained(base_model_id)

        model = build_model_from_checkpoint(
            checkpoint_dir=model_path,
            base_model_id=base_model_id,
        )
        return model, processor, model_type

    raise ValueError(f"Unknown model type: {model_type}")


def parse_args():
    parser = argparse.ArgumentParser(description="Voxtral QA inference")
    parser.add_argument(
        "--model-path", type=str, default="mistralai/Voxtral-Mini-3B-2507",
        help="HuggingFace repo id or local path to the model",
    )
    parser.add_argument(
        "--base-model-id", type=str, default="mistralai/Voxtral-Mini-3B-2507",
        help="Base model ID (used for LoRA / spk-cond to load the base weights)",
    )
    parser.add_argument(
        "--questions", type=str,
        default="pathological-speech-questions/questions.json",
        help="Path to the questions JSON file",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to write the output JSON with model predictions",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="Maximum tokens to generate per answer",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--dry-run", type=int, nargs="?", const=3, default=None,
        metavar="N",
        help="Print the prompt for the first N questions (default 3) "
             "without loading the model, then exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Load questions
    questions_path = Path(args.questions)
    questions_dir = questions_path.parent
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    # ------- dry-run: show prompts & exit -------
    if args.dry_run is not None:
        n = min(args.dry_run, len(questions))
        for i, q in enumerate(questions[:n]):
            audio_path = str((questions_dir / q["audio_path"]).resolve())
            prompt_text = build_prompt(q["question"], q["choices"])
            print(f"\n{'='*60}")
            print(f"Question {i+1}/{n}  (id: {q['id']})")
            print(f"Audio : {audio_path}")
            print(f"Answer: {q['answer']}")
            print(f"{'─'*60}")
            print(prompt_text)
        print(f"\n{'='*60}")
        print(f"Dry-run complete – showed {n} of {len(questions)} prompts.")
        return

    # Load model & processor
    print(f"Loading model: {args.model_path}")
    model, processor, model_type = load_model_and_processor(
        args.model_path, torch_dtype, args.base_model_id,
    )

    # For spk_cond models, build a sample_id → is_normative lookup from TSVs
    normative_lookup: dict[str, bool] | None = None
    if model_type == "spk_cond":
        import pandas as pd
        normative_lookup = {}
        tsv_paths = [
            "data/combined_neurovoz_torgo_cv/test_torgo.tsv",
            "data/combined_neurovoz_torgo_cv/test_neurovoz.tsv",
        ]
        for tsv_path in tsv_paths:
            df = pd.read_csv(tsv_path, sep="\t")
            for _, row in df.iterrows():
                normative_lookup[row["sample_id"]] = row["speech_type"] in NORMATIVE_SPEECH_TYPES
        print(f"Loaded speech-type info for {len(normative_lookup)} samples "
              f"({sum(normative_lookup.values())} normative, "
              f"{sum(not v for v in normative_lookup.values())} pathological)")

    print(f"Running inference on {len(questions)} questions (model_type={model_type}) ...")

    results = []
    for q in tqdm(questions, desc="Inference"):
        # Resolve audio path relative to questions.json directory
        audio_path = str((questions_dir / q["audio_path"]).resolve())

        prompt_text = build_prompt(q["question"], q["choices"])

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(model.device, dtype=torch_dtype)

        with torch.no_grad():
            if model_type == "spk_cond":
                # Speaker-conditioned model needs raw waveform + is_normative
                wav_data, _sr = sf.read(audio_path, dtype="float32")
                if wav_data.ndim > 1:
                    wav_data = wav_data.mean(axis=1)
                raw_waveforms = torch.from_numpy(wav_data).unsqueeze(0).to(model.device)

                # Look up speech type from TSV data
                sample_id = Path(audio_path).stem
                is_norm = normative_lookup.get(sample_id, False)
                is_normative = torch.tensor([is_norm], dtype=torch.bool).to(model.device)

                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    input_features=inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    raw_waveforms=raw_waveforms,
                    is_normative=is_normative,
                    max_new_tokens=args.max_new_tokens,
                )
                # spk_cond generate() calls language_model.generate() with
                # inputs_embeds (no input_ids), so it returns only the NEW
                # tokens — no need to slice off the prompt.
                decoded = processor.batch_decode(
                    outputs, skip_special_tokens=True
                )[0].strip()
            else:
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                decoded = processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                )[0].strip()

        entry = dict(q)
        entry["model_output"] = decoded
        results.append(entry)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_path} ({len(results)} entries)")


if __name__ == "__main__":
    main()
