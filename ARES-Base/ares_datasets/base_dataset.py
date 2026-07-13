#this is an abstract pytorch dataset that implements standard tokenization, caching, and seq block packing
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Iterator
from abc import ABC, abstractmethod
from tokenizer.tokenizer import BaseTokenizer

class BaseTextDataset(Dataset, ABC):
    """
    this is gonna be the abstract base class for all ARES text datasets. this handles tokenization, concatenation and block packing of text streams to maximize gpu util during causal language modeling
    """
    def __init__(
            self,
            tokenizer:BaseTokenizer,
            max_seq_length:int=1024,
            split:str='train',
            cache_dir:Optional[str]="data/cache",
            max_examples:Optional[int]=None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        self.cache_dir = cache_dir
        self.max_examples = max_examples

        self.input_ids: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

        self._prepare_dataset() #triggers dataset loading and block packing 

    @abstractmethod
    def _get_text_iterator(self)->Iterator[str]:
        pass

    def _prepare_dataset(self)->None:
        """
        Reads raw texts from the iterator, tokenizes them, concatenates them with EOS tokens,
        and slices them into uniform blocks of size max_seq_length.
        """
        print(f"[{self.__class__.__name__}] Preparing '{self.split}' dataset (Block Size: {self.max_seq_length})...")

        buffer: List[int] = []
        doc_count = 0
        block_count = 0

        for text in self._get_text_iterator():
            if not text or not text.strip():
                continue

            #Encode text without adding special tokens automatically, we append EOS manually
            tokens=self.tokenizer.encode(text, add_special_tokens=False)
            tokens.append(self.tokenizer.eos_token_id)
            buffer.extend(tokens)
            doc_count+=1

            #while we have enough tokens to form a complete train block
            while len(buffer)>=self.max_seq_length:
                #slice exactly max_seq_len tokens
                block_tokens=buffer[:self.max_seq_length]
                buffer=buffer[self.max_seq_length:]

                #convert to pytorch tensors
                tensor_block=torch.tensor(block_tokens, dtype=torch.long)
                self.input_ids.append(tensor_block)

                self.labels.append(tensor_block.clone())
                block_count+=1

                if self.max_examples and block_count>=self.max_examples:
                    break

            if self.max_examples and block_count>=self.max_examples:
                break
        
        print(f"[{self.__class__.__name__}] Processed {doc_count} documents into {len(self.input_ids)} uniform sequences.")

    def __len__(self) -> int:
        """Returns the total number of uniform sequence blocks available."""
        return len(self.input_ids)
    
    def __getitem__(self, idx:int)-> Dict[str, torch.Tensor]:
        #returs a single training dict formatted for ARESTrainer
        return {
            "input_ids":self.input_ids[idx],
            "labels":self.labels[idx]
        }