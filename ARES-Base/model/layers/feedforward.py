import torch
import torch.nn as nn
from model.config import ARESConfig

class MLP(nn.Module):
    """
    GPT-2 ffn (MLP) module
    projects hidden states to 4x hidden dim, applies GELU and proj it back down
    """

    def __init__(self, config:ARESConfig):
        super().__init__()
        self.config=config
        intermediate_size = 4 * config.hidden_size
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states:torch.Tensor)-> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states=self.act(hidden_states)
        hidden_states=self.c_proj(hidden_states)
        hidden_states=self.dropout(hidden_states)

        return hidden_states