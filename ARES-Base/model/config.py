from dataclasses import dataclass
from typing import Optional

@dataclass
class ARESConfig:
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    use_cache: bool = True  # for inference efficiency
    use_sdpa: bool = True