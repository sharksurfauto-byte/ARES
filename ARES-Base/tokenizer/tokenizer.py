# This defines the contract that all future tokenizer implementations in ARES must follow.

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any

class BaseTokenizer(ABC):
    """
    Abstract base class for all ARES tokenizers.
    Enforces a consistent interface across different tokenization algorithms.
    """
    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass