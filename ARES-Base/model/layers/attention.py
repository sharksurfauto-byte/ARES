import torch
import torch.nn as nn
from typing import Optional, Tuple
from model.config import ARESConfig
import math

class CausalSelfAttention(nn.Module):
    #this uses the GPT-2 MHA
    def __init__(self, config:ARESConfig):
        super().__init__()
        self.config = config
        self.embed_dim=config.hidden_size
        self.num_heads=config.num_attention_heads
        self.head_dim=self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}"
            )
        
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask registered as a buffer so it is not updated during backpropagation
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
            .view(1, 1, config.max_position_embeddings, config.max_position_embeddings)
        )

    def forward(self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None)->Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        #returns tuple containing the attn output and optionally the key/value cache

        batch_size, seq_length, _ = hidden_states.size() #(B,T,C)
        #1. project to Q,K,V
        qkv=self.c_attn(hidden_states)
        q,k,v=qkv.split(self.embed_dim, dim=-1)

        #reshape and transpose for MHA
        # B,T,C -> (B, num_heads,T, head_dim)
        def split_heads(tensor:torch.Tensor)->torch.Tensor:
            return tensor.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        
        q = split_heads(q)
        k=split_heads(k)
        v=split_heads(v)

        if layer_past is not None:
            past_key,past_value=layer_past
            k=torch.cat((past_key, k), dim=2)
            v=torch.cat((past_value, v), dim=2)

        present=(k,v) if use_cache else None

        if hasattr(self.config, "use_sdpa") and self.config.use_sdpa and attention_mask is None:
            # we use pytorch's scaled_dot_product_attention. it has inbuilt causal masking
            is_causal=(layer_past is None)
            attn_output=torch.nn.functional.scaled_dot_product_attention(
                q,k,v,
                attn_mask=None,
                dropout_p=self.config.attn_pdrop if self.training else 0.0,
                is_causal=is_causal
            )
        
        else: #manual attn
            #compute attn scores
            attn_weights=torch.matmul(q,k.transpose(-1,-2))
            attn_weights=attn_weights / math.sqrt(self.head_dim)

            #apply causal mask
            query_length,key_length = q.size(-2),k.size(-2)
            #slice the causal mask to the current seq len
            causal_mask=self.bias[:,:,key_length-query_length:key_length, :key_length] #type: ignore
            #mask future tokens
            attn_weights.masked_fill(causal_mask==0, float('-inf'))

            #apply external attn mask
            if attention_mask is not None:
                attn_weights=attn_weights+attention_mask

            #  Compute Softmax and Apply Dropout
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            #  Compute Attention Output
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim) #recombine heads (B, num_heads, T, head_dim) -> (B, T, C)

        #final proj
        attn_output=self.c_proj(attn_output)
        attn_output=self.resid_dropout(attn_output)

        return attn_output, present
