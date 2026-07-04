import torch
import torch.nn as nn
from typing import Optional
from inference.sampler import ARESSampler

@torch.no_grad()
def generate_sequence(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    do_sample: bool = True,
    eos_token_id: Optional[int] = 50256,
    pad_token_id: Optional[int] = 50256,
    use_cache: bool = True
) -> torch.Tensor:
    
    model.eval()
    batch_size, _ = input_ids.size()
    device = input_ids.device

    # Track completion status for each sequence in the batch (1 = generating, 0 = completed/EOS hit)
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

    # Initialize cache storage for autoregressive generation
    past_key_values = None
    current_input_ids = input_ids

    for step in range(max_new_tokens):
        # 1. Determine active sequence length (prompt length + generated tokens)
        past_len = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        
        # 2. Prevent IndexOutOfBounds error if sequence exceeds maximum positional embeddings
        if past_len + 1 > model.config.max_position_embeddings:
            print(f"[generate_sequence] Warning: Reached max_position_embeddings limit ({model.config.max_position_embeddings}). Stopping early.")
            break

        # 3. Slice inputs to last token if using cache
        if use_cache and past_key_values is not None:
            model_input_ids = current_input_ids[:, -1:]
        else:
            model_input_ids = current_input_ids

        # 4. Forward pass
        logits, _, past_key_values = model(
            input_ids=model_input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values
        )

        # 5. Extract logits of the next/last token and sample
        next_token_logits = logits[:, -1, :]
        next_tokens = ARESSampler.sample(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )

        # 6. If a sequence in the batch has reached EOS, replace subsequent tokens with PAD
        if eos_token_id is not None and pad_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + \
                          pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))
                
        current_input_ids = torch.cat([current_input_ids, next_tokens], dim=-1)

        # 7. Update sequence completion status
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens.squeeze(-1) != eos_token_id).long())

        # 8. Stop generation immediately if all sequences in the batch have reached EOS
        if unfinished_sequences.max() == 0:
            break
    
    return current_input_ids