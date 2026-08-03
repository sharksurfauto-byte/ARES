# ARES-Base/ares_datasets/wikitext.py
from typing import Iterator, Optional
from tokenizer.tokenizer import BaseTokenizer
from .base_dataset import BaseTextDataset

class WikiTextDataset(BaseTextDataset):
    """
    WikiText-103-v1 streaming dataset wrapper for out-of-distribution transfer probing.
    """
    def __init__(
            self,
            tokenizer: BaseTokenizer,
            max_seq_length: int = 1024,
            split: str = "validation",
            cache_dir: Optional[str] = "data/cache",
            max_examples: Optional[int] = None,
            hf_dataset_name: str = "wikitext"
    ):
        self.hf_dataset_name = hf_dataset_name
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split=split,
            cache_dir=cache_dir,
            max_examples=max_examples
        )

    def _get_text_iterator(self) -> Iterator[str]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the 'datasets' library: pip install datasets")
            
        print(f"[WikiTextDataset] Loading '{self.hf_dataset_name}' (wikitext-103-v1, split {self.split})...")

        # Load dataset with streaming enabled to prevent massive local downloads
        dataset = load_dataset(
            self.hf_dataset_name,
            "wikitext-103-v1",
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=True
        )

        for sample in dataset:
            text = sample.get("text", "")
            # Filter out section headers and empty spaces typical in WikiText formatting
            if text.strip() and not (text.strip().startswith("=") and text.strip().endswith("=")):
                yield text