import torch
from datasets import load_from_disk, Audio


class VoxtralDataCollator:
    """Data collator for Voxtral STT training - processes audio and text."""
    
    def __init__(self, processor, model_id):
        self.processor = processor
        self.model_id = model_id
        self.pad_id = processor.tokenizer.pad_token_id

    @staticmethod
    def pad_to(seq, fill, L):
        """Pad sequence to length L with fill value."""
        return seq + [fill] * (L - len(seq))

    def __call__(self, features):
        """
        Each feature should have:
          - "audio": raw audio (whatever your processor expects)
          - "transcription":  transcription string
          - "dataset_source": either "neurovoz" (Spanish) or "torgo" (English)
        """
        texts  = [f["transcription"] for f in features]
        audios = [f["audio"]["array"] for f in features]
        
        # Determine language based on dataset source
        # NeuroVoz = Spanish, TORGO = English
        languages = []
        for f in features:
            if "dataset_source" in f:
                lang = "es" if f["dataset_source"] == "neurovoz" else "en"
            else:
                # Fallback: assume Spanish if not specified
                lang = "es"
            languages.append(lang)

        # 1) Build the PROMPT part: [AUDIO]…[AUDIO] <transcribe>
        # NOTE: Two languages: Spanish (NeuroVoz) & English (TORGO)
        # Voxtral supports multilingual batches, so pass language for each sample
        prompt = self.processor.apply_transcription_request(
            language=languages,  # Pass list of languages for each sample
            model_id=self.model_id if hasattr(self, "model_id") else None,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )
        
        # prompt["input_ids"]: shape [B, L_prompt]
        # keep any extra fields (e.g., audio features) to pass through to the model
        passthrough = {k: v for k, v in prompt.items()
                       if k not in ("input_ids", "attention_mask")}

        prompt_ids = prompt["input_ids"]           # [B, Lp]
        prompt_attn = prompt["attention_mask"]     # [B, Lp]
        B = prompt_ids.size(0) # Batch size

        # Get shortcut to tokenizer
        tok = self.processor.tokenizer

        # 2) Tokenize transcriptions WITHOUT padding; we'll pad after concatenation
        text_tok = tok(
            texts,
            add_special_tokens=False,
            padding=False, # no padding here
            truncation=True,
            max_length=256,
            return_tensors=None,
        )
        text_ids_list = text_tok["input_ids"]

        # 3) Concatenate: input_ids = [PROMPT] + [TEXT]
        input_ids, attention_mask, labels = [], [], []
        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            ids  = p_ids + t_ids + [tok.eos_token_id]
            attn = p_att + [1] * (len(t_ids) + 1) 
            
            # labels: mask prompt tokens, learn only on text tokens
            # -100 is the ignore index in PyTorch CrossEntropyLoss
            lab  = [-100] * len(p_ids) + t_ids + [tok.eos_token_id] 

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # 4) Pad to max length in batch

        # Get pad token id
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        # Get max length
        max_len = max(len(x) for x in input_ids)

        # Pad sequences
        input_ids      = [self.pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [self.pad_to(x, 0,      max_len) for x in attention_mask]
        labels         = [self.pad_to(x, -100,   max_len) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # 5) Include processor outputs needed by the model (e.g., audio features)
        for k, v in passthrough.items():
            batch[k] = v

        return batch


def load_and_prepare_dataset(dataset_path="../../../data/combined_neurovoz_torgo"):
    """Load and prepare combined NeuroVoz + TORGO dataset for training."""
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Cast audio to 16kHz (required for Voxtral) - should already be 16kHz but ensure
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Use the predefined splits
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    # Optional: select subset for testing (comment out for full training)
    # train_dataset = train_dataset.select(range(100))
    # eval_dataset = eval_dataset.select(range(50))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Dataset columns: {train_dataset.column_names}")
    
    return train_dataset, eval_dataset
