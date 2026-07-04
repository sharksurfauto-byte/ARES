# ARES-Base/datasets/openwebtext.py
from typing import Iterator, Optional
from tokenizer.tokenizer import BaseTokenizer
from datasets.base_dataset import BaseTextDataset

class OpenWebTextDataset(BaseTextDataset):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        max_seq_length: int = 1024,
        split: str = "train",
        cache_dir: Optional[str] = "data/cache",
        max_examples: Optional[int] = None,
        hf_dataset_name: str = "openwebtext"
    ):
        self.hf_dataset_name = hf_dataset_name
        # OpenWebText only provides a 'train' split by default; we handle validation splitting manually if needed
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

        print(f"[OpenWebTextDataset] Loading '{self.hf_dataset_name}'...")
        
        # Note: For massive datasets like OpenWebText, streaming is critical for startup speed
        dataset = load_dataset(
            self.hf_dataset_name, 
            split="train", 
            cache_dir=self.cache_dir,
            streaming=True
        )
        
        # Simple deterministic routing to create a validation split if requested
        for i, sample in enumerate(dataset):
            text = sample.get("text", "")
            if not text:
                continue
                
            if self.split == "val" and i % 100 == 0:
                yield text
            elif self.split == "train" and i % 100 != 0:
                yield text