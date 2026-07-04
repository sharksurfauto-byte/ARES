import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union
from model.config import ARESConfig
from model.gpt import ARESBaseModel

class CheckpointManager:
    """
    Handles saving and loading of ARES checkpoints, as well as importing
    official pre-trained GPT-2 weights with Conv1D weight transposition.
    """
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        global_step: int,
        filepath: Union[str, Path],
        config: Optional[ARESConfig] = None
    ) -> None:
        """Saves a complete training checkpoint to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": config.__dict__ if config else None
        }
        
        torch.save(checkpoint_state, filepath)
        print(f"[CheckpointManager] Checkpoint successfully saved to: {filepath}")

    @staticmethod
    def load_checkpoint(
        filepath: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Loads a locally trained ARES checkpoint and restores states."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")
            
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        print(f"[CheckpointManager] Loaded checkpoint from step {checkpoint.get('global_step', 0)}")
        return {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0)
        }

    @classmethod
    def load_pretrained_gpt2(cls, model: ARESBaseModel, model_name: str = "gpt2") -> ARESBaseModel:
        """
        Downloads official OpenAI GPT-2 weights via Hugging Face Transformers,
        maps keys to ARES modular architecture, and performs Conv1D transposition.
        """
        try:
            from transformers import GPT2LMHeadModel
        except ImportError:
            raise ImportError(
                "Loading pre-trained GPT-2 weights requires 'transformers'. "
                "Please run: pip install transformers"
            )

        print(f"[CheckpointManager] Fetching official '{model_name}' weights from Hugging Face...")
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        hf_state_dict = hf_model.state_dict()
        
        # Get target ARES model state dict for validation
        ares_state_dict = model.state_dict()
        
        # Key translation mapping between HF GPT-2 and ARES modular architecture
        key_mapping = cls._generate_key_mapping(model.config.num_hidden_layers)
        
        # Tensor names that require transposition due to OpenAI's Conv1D legacy layout
        transposed_weights = {
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight"
        }

        new_state_dict = {}
        matched_keys = 0

        for hf_key, hf_tensor in hf_state_dict.items():
            # Handle standard naming differences
            ares_key = hf_key
            for hf_pattern, ares_pattern in key_mapping.items():
                if hf_pattern in hf_key:
                    ares_key = hf_key.replace(hf_pattern, ares_pattern)
                    break
            
            # Skip lm_head if it's tied to token embeddings in our architecture
            if ares_key == "lm_head.weight" and "lm_head.weight" not in ares_state_dict:
                continue

            if ares_key in ares_state_dict:
                # -------------------------------------------------------------
                # The Critical Nuance: Transpose Conv1D linear weights
                # -------------------------------------------------------------
                requires_transposition = any(pattern in ares_key for pattern in transposed_weights)
                
                if requires_transposition:
                    # Transpose from [in_features, out_features] to [out_features, in_features]
                    tensor_to_load = hf_tensor.t().contiguous()
                else:
                    tensor_to_load = hf_tensor

                # Verify shape compatibility before assignment
                expected_shape = ares_state_dict[ares_key].shape
                if tensor_to_load.shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for {ares_key}: expected {expected_shape}, got {tensor_to_load.shape}"
                    )
                
                new_state_dict[ares_key] = tensor_to_load
                matched_keys += 1
            else:
                print(f"[CheckpointManager] Warning: HF key '{hf_key}' (mapped to '{ares_key}') not found in ARES model.")

        # Load mapped and transposed weights into our ARES model
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        # Verify weight tying integrity
        if model.lm_head.weight is not model.transformer.wte.embed.weight:
            model.lm_head.weight = model.transformer.wte.embed.weight

        print(f"[CheckpointManager] Successfully imported {matched_keys} weight tensors from '{model_name}'.")
        if missing_keys:
            print(f"[CheckpointManager] Note: Missing keys (expected for tied weights): {missing_keys}")
            
        return model

    @staticmethod
    def _generate_key_mapping(num_layers: int) -> Dict[str, str]:
        """Generates string replacement rules from HF architecture to ARES modular architecture."""
        mapping = {
            "transformer.wte.weight": "transformer.wte.embed.weight",
            "transformer.wpe.weight": "transformer.wpe.embed.weight",
            ".ln_1.": ".ln1.",
            ".ln_2.": ".ln2.",
        }
        return mapping