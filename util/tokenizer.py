import os

from transformers import AutoTokenizer

TOKENIZERS = {
    "smollm": AutoTokenizer.from_pretrained(
        "/root/mixture_of_recursions/hf_models/SmolLM-135M",
        local_files_only=True
    ),
    "smollm2": AutoTokenizer.from_pretrained(
        "/root/mixture_of_recursions/hf_models/SmolLM2-135M",
        local_files_only=True
    ),
}


def load_tokenizer_from_config(cfg):
    tokenizer = TOKENIZERS[cfg.tokenizer]
    if tokenizer.pad_token is None:
        if cfg.tokenizer in ["smollm", "smollm2"]:
            # '<|endoftext|>'
            tokenizer.pad_token_id = 0
        else:
            raise ValueError(f"Tokenizer {cfg.tokenizer} does not have a pad token, please specify one in the config")
    return tokenizer