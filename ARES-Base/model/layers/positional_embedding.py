import torch
import torch.nn as nn
from model.config import ARESConfig
from embeddings import TokenEmbedding

class PositionalEmbedding(nn.Module):
    def __init__(self, config:ARESConfig):
        super().__init__()
        self.embed=nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, position_ids:torch.Tensor)->torch.Tensor:
        #O/P: B,T,C
        return self.embed(position_ids)
    
