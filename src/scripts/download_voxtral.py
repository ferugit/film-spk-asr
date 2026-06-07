import argparse

import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor




if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description="Download and cache Hugging Face model.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--cache_dir', type=str, default='models/', help='Directory to cache the model')
    args = parser.parse_args()

    model_name = args.model_name
    cache_dir = args.cache_dir

    device = "auto" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map=device
    )

