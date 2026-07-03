import torch
import torch.nn as nn
from typing import Optional, Tuple
from model.config import ARESConfig
from model.layers.attention import CausalSelfAttention
from model.layers.feedforward import MLP

class TransformerBlock(nn.Module):
    """A single gpt-2 decoder block contanining Pre-layernorm, causal attn and an MLP connected via residual pathways"""
    def __init__(self,config:ARESConfig):
        super().__init__()
        self.config=config

        #pre layernorm for attn sub layer
        self.ln1=nn.LayerNorm(config.hidden_size,eps=config.layer_norm_epsilon)
        self.attn=CausalSelfAttention(config)

        #pre layernorm for MLP
        self.ln2=nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp=MLP(config)

    def forward(self,
                hidden_states:torch.Tensor,
                attention_mask:Optional[torch.Tensor]=None,
                use_cache:bool=False,
                layer_past:Optional[Tuple[torch.Tensor, torch.Tensor]]=None
                )->Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states (torch.Tensor): Shape (batch_size, seq_length, hidden_size)
            attention_mask (torch.Tensor, optional): Pre-formatted additive mask.
            use_cache (bool): Whether to return key/value cache.
            layer_past (tuple, optional): Cached key/values from previous generation steps.
            
        Returns:
            Tuple containing the block output and optional key/value cache
        """
        #causal self attn
        residual=hidden_states
        #apply pre layernorm
        hidden_states=self.ln1(hidden_states)
        #compute attn
        attn_outputs,present=self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            layer_past=layer_past
        )

        #addd residual connection
        hidden_states=residual+attn_outputs

        #FFN (MLP)
        residual=hidden_states
        hidden_states=self.ln2(hidden_states)
        mlp_outputs=self.mlp(hidden_states)
        hidden_states=residual+mlp_outputs

        return hidden_states,present
