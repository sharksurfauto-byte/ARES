import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from model.config import ARESConfig
from model.transformer import TransformerStack

class ARESBaseModel(nn.Module):
    def __init__(self, config:ARESConfig):
        super().__init__()
        self.config = config

        #core transformer decoder backbone
        self.transformer=TransformerStack(config)

        #language mopdelling head
        self.lm_head=nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        #sharing weights btw input embeds and lm head (WEIGHT TYING)
        self.lm_head.weight=self.transformer.wte.embed.weight

        self.apply(self._init_weights)

        # Applying special scaled initialization to residual proj weights
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                std = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)       
                torch.nn.init.normal_(param, mean=0.0, std=std)

    def _init_weights(self, module:nn.Module)->None:
        """
        Applies GPT-2 standard initialization schemes:
        - Linear and Embedding layers: Normal distribution N(0, initializer_range)
        - Biases and LayerNorm parameters: Initialized to 0 and 1 respectively
        - Residual projections (c_proj): Scaled by 1 / sqrt(2 * num_layers)
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        

    def forward(
            self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        #this returns a tuple containing :
        #1. logits: (B,T,vocab_size)
        #2. loss: cross entropy loss
        #3. past key vals

        # pass seq thorugh transformer backbone
        hidden_states, presents=self.transformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values
        )

        #project final hidden states to vocab logits
        logits=self.lm_head(hidden_states)

        #compute languafe modelling loss
        loss=None
        if labels is not None:
            shift_logits=logits[...,:-1,:].contiguous()
            shift_labels=labels[...,1:].contiguous()

            loss_fct=nn.CrossEntropyLoss()
            loss=loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return logits, loss, presents
    
    @classmethod
    def from_pretrained(cls, model_name:str="gpt2")->"ARESBaseModel":
        raise NotImplementedError(
            "Pre-trained checkpoint loader and Conv1D weight transposition pending implementation in training/checkpoint.py."
        )