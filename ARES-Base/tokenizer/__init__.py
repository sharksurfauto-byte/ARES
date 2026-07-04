from .tokenizer import BaseTokenizer
from .bpe import GPT2BPETokenizer

__all__ = ["BaseTokenizer", "GPT2BPETokenizer", "get_tokenizer"]

def get_tokenizer(tokenizer_type: str = "gpt2-bpe", **kwargs) -> BaseTokenizer:
    """
    Factory function to instantiate tokenizers based on configuration.
    """
    if tokenizer_type.lower() in ["gpt2", "gpt2-bpe", "bpe"]:
        return GPT2BPETokenizer(**kwargs)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")