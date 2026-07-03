import torch
import torch.nn as nn
from model.config import ARESConfig

class TokenEmbedding(nn.Module):
    # token embed layer that maps vocab to dence vectors

    def __init__(self,config:ARESConfig):
        super().__init__()
        self.embed=nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids:torch.Tensor)->torch.Tensor:
        # I/P: (B,T)
        #O/P: (B,T,C)
        return self.embed(input_ids)