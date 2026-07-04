# ARES-Base/model/config.py
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import yaml

@dataclass
class ARESConfig:
    """
    Configuration dataclass for ARES-Base (GPT-2 architecture).
    All architectural hyperparameters should be routed through this class.
    """
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

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ARESConfig":
        """Loads ARESConfig from a YAML configuration file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return cls(
            vocab_size=data["dimensions"]["vocab_size"],
            max_position_embeddings=data["dimensions"]["max_position_embeddings"],
            hidden_size=data["dimensions"]["hidden_size"],
            num_hidden_layers=data["dimensions"]["num_hidden_layers"],
            num_attention_heads=data["dimensions"]["num_attention_heads"],
            layer_norm_epsilon=data["initialization"]["layer_norm_epsilon"],
            initializer_range=data["initialization"]["initializer_range"],
            embd_pdrop=data["dropout"]["embd_pdrop"],
            resid_pdrop=data["dropout"]["resid_pdrop"],
            attn_pdrop=data["dropout"]["attn_pdrop"],
            use_cache=data["execution"]["use_cache"],
            use_sdpa=data["execution"]["use_sdpa"],
        )