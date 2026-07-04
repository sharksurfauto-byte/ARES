from typing import Iterator, Optional
from tokenizer.tokenizer import BaseTokenizer
from datasets.base_dataset import BaseTextDataset

class TinyStoriesDataset(BaseTextDataset):
    def __init__(
            self,
            tokenizer: BaseTokenizer,
            max_seq_length: int = 1024,
            split: str = "train",
            cache_dir: Optional[str] = "data/cache",
            max_examples: Optional[int] = None,
            hf_dataset_name: str = "roneneldan/TinyStories"
    ):
        self.hf_dataset_name=hf_dataset_name
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split=split,
            cache_dir=cache_dir,
            max_examples=max_examples
        )

    def _get_text_iterator(self)->Iterator[str]:
        #streams docs from the HF hub or local machine
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the 'datasets' library: pip install datasets")
            
        print(f"[TinyStoriesDataset] Loading '{self.hf_dataset_name}' ({self.split} split)...")

        #load dataset with streaming enables to prevent exessive RAM usage
        dataset=load_dataset(
            self.hf_dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=True
        )

        for sample in dataset:
            text=sample.get("text","")
            if text:
                yield text