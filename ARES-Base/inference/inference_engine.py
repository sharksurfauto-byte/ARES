import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union
from tokenizer.tokenizer import BaseTokenizer
from inference.generate import generate_sequence

"""
Unified research interface for ARES-Base text generation.
Connects the model, tokenizer, and sampler into a clean end-to-end pipeline.
"""

class ARESInferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        do_sample: bool = True,
        use_cache: bool = True
    ) -> Union[str, List[str]]:
        
        is_single_prompt = isinstance(prompt, str)
        prompts = [prompt] if is_single_prompt else prompt

        encoded_inputs = [self.tokenizer.encode(p) for p in prompts]

        # Pad sequences to match the longest prompt in the batch
        max_len = max(len(seq) for seq in encoded_inputs)
        padded_ids = []
        for seq in encoded_inputs:
            pad_length = max_len - len(seq)
            padded_ids.append([self.tokenizer.pad_token_id] * pad_length + seq)

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)

        # autoregressive genrations
        output_ids = generate_sequence(
            model=self.model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=use_cache
        )

        decoded_outputs = []
        for seq in output_ids.tolist():
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            decoded_outputs.append(text)

        return decoded_outputs[0] if is_single_prompt else decoded_outputs