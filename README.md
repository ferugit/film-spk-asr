# FiLM-Based Speaker Conditioning of a SpeechLLM for Pathological Speech Recognition

Speaker conditioning via Feature-wise Linear Modulation (FiLM) for pathological
speech recognition. FiLM injects x-vector-derived speaker information into each
transformer layer of a frozen Voxtral-Mini encoder — only the FiLM bank and
x-vector extractor are trained (~1.6% of parameters). Benchmarked against
standard and parameter-efficient fine-tuning on Spanish and English pathological
speech (NeuroVoz, TORGO).

Accepted in **Odyssey 2026**: The Speaker and Language Recognition Workshop.

> Fernando López, Santosh Kesiraju, Jordi Luque.
> *FiLM-Based Speaker Conditioning of a SpeechLLM for Pathological Speech Recognition.*
> Telefónica Innovación Digital · Universidad Autónoma de Madrid · Brno University of Technology.
> arXiv: [2606.06211](https://arxiv.org/abs/2606.06211)

## Adaptation strategies

| Strategy     | What is trained                                  | Config |
|--------------|--------------------------------------------------|--------|
| **FFT**      | Full fine-tune (encoder + connector + decoder)   | `src/finetune/voxtral/config/full_finetuning.yaml` |
| **F-LoRA**   | LoRA on all attention + FF + connector           | `src/finetune/voxtral/config/lora.yaml` |
| **EFT**      | Encoder + connector fine-tune                    | `src/finetune/voxtral/config/encoder_only.yaml` |
| **E-LoRA**   | LoRA on encoder + connector only                 | `src/finetune/voxtral/config/encoder_lora.yaml` |
| **Spk-Cond** | FiLM bank + SiAMResNet34 x-vector extractor (proposed) | `src/finetune/spk_cond_voxtral/config/film_conditioning_lang_balanced.yaml` |

## Installation

Requires Python >= 3.12 and PyTorch with CUDA 12.8.

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Required assets

- **Base model**: `mistralai/Voxtral-Mini-3B-2507` — `bash scripts/download_voxtral_model.sh`
- **x-vector extractor** (Spk-Cond only): SiAMResNet34 TorchScript model at
  `models/SiAMResNet34/samresnet34_w_features.jit`
  (from the [WeSpeaker](https://github.com/wenet-e2c/wespeaker) toolkit, pretrained on
  VoxBlink2 + VoxCeleb2).

## Data

The experiments use **NeuroVoz** (Spanish, Parkinson's), **TORGO** (English, dysarthric),
and **CommonVoice ES** (normative Spanish, for balancing).

```bash
# 1. Prepare individual datasets
bash src/scripts/prepare_neurovoz.sh
bash src/scripts/prepare_torgo.sh
bash scripts/download_commonvoice_es.sh   # needs MDC_API_KEY

# 2. Merge into combined HF datasets
python src/scripts/create_combined_hf_dataset.py
python src/scripts/merge_cv_neurovoz_torgo.py
```

Splits are 70/10/20 stratified by speaker (no speaker leakage).

## Usage

All commands are run from the repository root.

1. **Train** — baselines or Spk-Cond (proposed):

```bash
# Baselines (pick a config: FFT / F-LoRA / EFT / E-LoRA)
python src/finetune/voxtral/finetuning.py \
    --config_file src/finetune/voxtral/config/full_finetuning.yaml

# Spk-Cond
python src/finetune/spk_cond_voxtral/finetuning.py \
    --config_file src/finetune/spk_cond_voxtral/config/film_conditioning_lang_balanced.yaml
```

2. **Evaluate** — WER on test sets:

```bash
bash scripts/evaluate_voxtral.sh           # baselines
bash scripts/evaluate_spk_cond_voxtral.sh  # Spk-Cond
```

3. **Post-processing** — hallucination cleanup:

```bash
bash scripts/text_postprocess.sh
```

4. **MCQA evaluation** — speaker sex/age question answering:

```bash
python src/scripts/generate_questions.py
bash scripts/evaluate_voxtral_qa.sh mistralai/Voxtral-Mini-3B-2507
python src/scripts/compare_qa_results.py
```

## Citation

```bibtex
@inproceedings{lopez2025film,
  title     = {FiLM-Based Speaker Conditioning of a SpeechLLM for Pathological Speech Recognition},
  author    = {L\'opez, Fernando and Kesiraju, Santosh and Luque, Jordi},
  booktitle = {Odyssey 2026: The Speaker and Language Recognition Workshop},
  year      = {2026},
  eprint    = {2606.06211},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2606.06211}
}
```

## License

MIT License. See [LICENSE](LICENSE).
