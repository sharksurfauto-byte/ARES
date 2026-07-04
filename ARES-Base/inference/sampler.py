import torch
import torch.nn.functional as F
from typing import Optional

class ARESSampler:
    """
    A modular decoding strategy engine handling Greedy, Temperature Scaling,
    Top-K, and Top-p (Nucleus) sampling.
    """
    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Samples a token index from the input logits.

        Args:
            logits (torch.Tensor): Logits of the next token, shape (batch_size, vocab_size).
            temperature (float): Controls randomness. Lower values make the output more
                deterministic; higher values increase diversity. Default is 1.0.
            top_k (int, optional): Keep only the top-k highest probability tokens.
            top_p (float, optional): Keep only the cumulative top-p probability tokens.
            do_sample (bool): If False, defaults to deterministic greedy decoding (argmax).

        Returns:
            torch.Tensor: Sampled token IDs of shape (batch_size, 1).
        """
        # Deterministic greedy decoding
        if not do_sample or temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        
        # Temperature scaling
        logits = logits / max(temperature, 1e-5)

        # Top-K filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_val, _ = torch.topk(logits, k=top_k, dim=-1)
            min_top_k_val = kth_val[:, -1, None]
            # Mask out all logits below the kth threshold
            logits = torch.where(logits < min_top_k_val, torch.full_like(logits, float("-inf")), logits)
        
        # Top-P (Nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token that exceeds top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            # Scatter the sorted mask back to original unsorted positions
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        # Convert filtered logits to probabilities and sample via multinomial distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token