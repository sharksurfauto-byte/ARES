from typing import List, Optional
from tokenizer.tokenizer import BaseTokenizer

class GPT2BPETokenizer(BaseTokenizer):
    """
    GPT-2 Byte Pair Encoding (BPE) Tokenizer.
    Supports either 'tiktoken' (preferred/fastest) or Hugging Face 'transformers' as backends.
    """
    def __init__(self, model_name: str = "gpt2", use_tiktoken: bool = True):
        super().__init__()
        self.model_name = model_name
        self.use_tiktoken = use_tiktoken
        self._backend = self._initialize_backend()
        
    def _initialize_backend(self):
        if self.use_tiktoken:
            try:
                import tiktoken
                # GPT-2 uses the 'r50k_base' encoding
                return tiktoken.get_encoding("r50k_base")
            except ImportError:
                print("[GPT2BPETokenizer] 'tiktoken' not found. Falling back to 'transformers'...")      
                self.use_tiktoken = False

        if not self.use_tiktoken:
            try:
                from transformers import GPT2TokenizerFast
                tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
                # GPT-2 does not have an explicit pad token by default; map it to EOS
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer
            except ImportError:
                raise ImportError(
                    "Please install either 'tiktoken' (pip install tiktoken) or "
                    "'transformers' (pip install transformers) to use the GPT2BPETokenizer."
                )

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        """
        if self.use_tiktoken:
            return self._backend.encode(text, allowed_special="all" if add_special_tokens else set())    
        else:
            return self._backend.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        if self.use_tiktoken:
            if skip_special_tokens:
                special_ids = {self.eos_token_id, self.pad_token_id}
                token_ids = [tid for tid in token_ids if tid not in special_ids]
            return self._backend.decode(token_ids)
        else:
            return self._backend.decode(token_ids, skip_special_tokens=skip_special_tokens)
            
    @property
    def vocab_size(self) -> int:
        return 50257
        
    @property
    def eos_token_id(self) -> int:
        return 50256
        
    @property
    def pad_token_id(self) -> int:
        return 50256