from typing import Optional
from torch.utils.data import Dataset
from tokenizer.tokenizer import BaseTokenizer
from datasets.base_dataset import BaseTextDataset
from datasets.tinystories import TinyStoriesDataset
from datasets.openwebtext import OpenWebTextDataset

def get_dataset(
    dataset_name: str,
    tokenizer: BaseTokenizer,
    max_seq_length: int = 1024,
    split: str = "train",
    cache_dir: Optional[str] = "data/cache",
    max_examples: Optional[int] = None,
    **kwargs
) -> BaseTextDataset:
    #this is the base fucniton to instantiate ARES datasets dynamically from YAML configuration

    name_clean=dataset_name.lower().strip()

    if name_clean in ["tinystories", "tiny_stories", "tiny-stories"]:
        return TinyStoriesDataset(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split=split,
            cache_dir=cache_dir,
            max_examples=max_examples,
            **kwargs
        )
    elif name_clean in ["openwebtext", "webtext", "owt"]:
        return OpenWebTextDataset(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split=split,
            cache_dir=cache_dir,
            max_examples=max_examples,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported dataset name: '{dataset_name}'. Available: 'tinystories', 'openwebtext'")