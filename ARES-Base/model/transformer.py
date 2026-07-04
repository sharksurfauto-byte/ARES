import torch
import torch.nn as nn
from model.config import ARESConfig
from typing import Optional, Tuple, List
from model.block import TransformerBlock
from model.layers.embeddings import TokenEmbedding
from model.layers.positional_embedding import PositionalEmbedding

class TransformerStack(nn.Module):
    def __init__(self, config: ARESConfig):
        super().__init__()
        self.config = config
        
        # 1. Embedding Layers
        self.wte = TokenEmbedding(config)
        self.wpe = PositionalEmbedding(config)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # 2. Decoder Blocks Stack
        self.h = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        #final layernorm
        self.ln_f=nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        #returns Tuple of final hidden states and optional updated KV cache list.
        B,T = input_ids.size()

        #handle past len
        past_len=0
        if past_key_values is not None:
            past_len=past_key_values[0][0].size(-2)

        if position_ids is None:
            device=input_ids.device
            position_ids=torch.arange(
                past_len,
                past_len+T,
                dtype=torch.long,
                device=device
            ).unsqueeze(0) # Shape (1, T), will broadcast to (B, T)
        #combine token and pos embed: E=E_token+ E_pos
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)

        #pass through transformer block stack
        presents=[] if use_cache else None
        
        for i, block in enumerate(self.h):
            layer_past=past_key_values[i] if past_key_values is not None else None
            hidden_states, present=block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                layer_past=layer_past
            )

            if use_cache:
                presents.append(present)  #type: ignore

        hidden_states=self.ln_f(hidden_states)

        return hidden_states