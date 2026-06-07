"""
Data loading and collation for Speaker-Conditioned Voxtral training.

Extends the original VoxtralDataCollator to additionally provide:
  - raw_waveforms  : padded raw audio for on-the-fly x-vector extraction
  - is_normative   : boolean mask (True for HC / Unknown speech types)

Also handles the three-language setup: NeuroVoz (es), TORGO (en), CommonVoice (es).
"""

import torch
import numpy as np
from datasets import load_from_disk, Audio


# Speech types considered normative (no pathological conditioning)
NORMATIVE_SPEECH_TYPES = {"HC", "Unknown"}


class SpkCondVoxtralDataCollator:
    """
    Data collator for Speaker-Conditioned Voxtral STT training.

    In addition to the standard Voxtral collation (input_ids, attention_mask,
    labels, input_features), this collator also produces:
      - raw_waveforms : (B, T_max) padded raw audio waveforms
      - is_normative  : (B,) boolean tensor
    """

    def __init__(self, processor, model_id):
        self.processor = processor
        self.model_id = model_id
        self.pad_id = processor.tokenizer.pad_token_id

    @staticmethod
    def pad_to(seq, fill, L):
        """Pad a list-like sequence to length L with fill value."""
        return seq + [fill] * (L - len(seq))

    def __call__(self, features):
        """
        Parameters
        ----------
        features : list[dict]
            Each dict must have:
              - "audio": {"array": np.ndarray, "sampling_rate": int}
              - "transcription": str
              - "dataset_source": str ("neurovoz", "torgo", "commonvoice_es")
              - "speech_type": str ("HC", "DYSARTHRIC", "PARKINSON", "Unknown")
        """
        texts = [f["transcription"] for f in features]
        audios = [f["audio"]["array"] for f in features]

        # ---- Language detection ----
        languages = []
        for f in features:
            source = f.get("dataset_source", "")
            if source in ("neurovoz", "commonvoice_es"):
                languages.append("es")
            else:  # torgo, etc.
                languages.append("en")

        # ---- Normative mask ----
        is_normative = torch.tensor(
            [f.get("speech_type", "Unknown") in NORMATIVE_SPEECH_TYPES for f in features],
            dtype=torch.bool,
        )

        # ---- Raw waveforms (for SiAMResNet34) ----
        # Pad to max length in the batch
        raw_arrays = [np.asarray(a, dtype=np.float32) for a in audios]
        max_wav_len = max(len(a) for a in raw_arrays)
        padded_wavs = np.zeros((len(raw_arrays), max_wav_len), dtype=np.float32)
        for i, a in enumerate(raw_arrays):
            padded_wavs[i, :len(a)] = a
        raw_waveforms = torch.from_numpy(padded_wavs)  # (B, T_max)

        # ---- Standard Voxtral prompt processing ----
        # 1) Build PROMPT:  [AUDIO]...[AUDIO] <transcribe>
        prompt = self.processor.apply_transcription_request(
            language=languages,
            model_id=self.model_id if hasattr(self, "model_id") else None,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )

        # Separate prompt tensors from passthrough (e.g., input_features)
        passthrough = {k: v for k, v in prompt.items()
                       if k not in ("input_ids", "attention_mask")}

        prompt_ids = prompt["input_ids"]        # (B, Lp)
        prompt_attn = prompt["attention_mask"]  # (B, Lp)
        B = prompt_ids.size(0)

        tok = self.processor.tokenizer

        # 2) Tokenize transcriptions
        text_tok = tok(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors=None,
        )
        text_ids_list = text_tok["input_ids"]

        # 3) Concatenate:  [PROMPT] + [TEXT] + [EOS]
        input_ids, attention_mask, labels = [], [], []
        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            ids = p_ids + t_ids + [tok.eos_token_id]
            attn = p_att + [1] * (len(t_ids) + 1)
            lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # 4) Pad to max length in batch
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        input_ids = [self.pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [self.pad_to(x, 0, max_len) for x in attention_mask]
        labels = [self.pad_to(x, -100, max_len) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            # --- Speaker conditioning extras ---
            "raw_waveforms": raw_waveforms,
            "is_normative": is_normative,
        }

        # 5) Pass-through processor outputs (e.g., input_features for mel specs)
        for k, v in passthrough.items():
            batch[k] = v

        return batch


def load_and_prepare_dataset(dataset_path):
    """
    Load the combined NeuroVoz + TORGO + CommonVoice dataset.

    Returns
    -------
    train_dataset, eval_dataset
    """
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # Ensure 16 kHz sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Dataset columns: {train_dataset.column_names}")

    # Print speech-type distribution
    from collections import Counter
    st = Counter(train_dataset["speech_type"])
    ds = Counter(train_dataset["dataset_source"])
    print(f"Speech types: {dict(st)}")
    print(f"Dataset sources: {dict(ds)}")

    return train_dataset, eval_dataset
