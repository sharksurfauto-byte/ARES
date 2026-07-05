# ARES-Base/models/registry.py
import json
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
from model.config import ARESConfig
from model.gpt import ARESBaseModel

class ModelRegistry:
    """
    Centralized model catalog and artifact provenance registry for ARES.
    Manages metadata, parameter counts, dataset lineage, and weight loading.
    """
    def __init__(self, registry_path: Union[str, Path] = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        if not self.registry_path.exists():
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump({"models": {}}, f, indent=2)

    def _load_catalog(self) -> Dict[str, Any]:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_catalog(self, catalog: Dict[str, Any]) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

    def register_model(
        self,
        model_id: str,
        architecture: str,
        model: torch.nn.Module,
        weights_path: Union[str, Path],
        config_path: Union[str, Path],
        training_dataset: str,
        total_tokens_trained: int,
        val_perplexity: float,
        zero_shot_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Registers a newly trained or fine-tuned model into the catalog.
        Calculates exact parameter count automatically from the PyTorch module.
        """
        catalog = self._load_catalog()
        
        if model_id in catalog["models"]:
            print(f"[ModelRegistry] Warning: Overwriting existing metadata for model_id '{model_id}'")

        # Automatically compute trainable parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        entry = {
            "model_id": model_id,
            "architecture": architecture,
            "parameter_count": param_count,
            "training_dataset": training_dataset,
            "total_tokens_trained": total_tokens_trained,
            "val_perplexity": round(val_perplexity, 4),
            "zero_shot_metrics": zero_shot_metrics or {},
            "tags": tags or ["baseline"],
            "creation_date": datetime.now().isoformat(),
            "weights_path": str(Path(weights_path).resolve()),
            "config_path": str(Path(config_path).resolve()),
            "notes": notes
        }

        catalog["models"][model_id] = entry
        self._save_catalog(catalog)
        print(f"[ModelRegistry] Successfully registered '{model_id}' ({param_count:,} params | PPL: {val_perplexity:.2f})")
        return entry

    def get_metadata(self, model_id: str) -> Dict[str, Any]:
        """Retrieves the full metadata dictionary for a specific model ID."""
        catalog = self._load_catalog()
        if model_id not in catalog["models"]:
            raise KeyError(f"Model ID '{model_id}' not found in registry: {self.registry_path}")
        return catalog["models"][model_id]

    def query_models(
        self,
        tag: Optional[str] = None,
        max_perplexity: Optional[float] = None,
        architecture: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filters the model registry programmatically.
        Essential for automated MoE routing or ablation comparison tables.
        """
        catalog = self._load_catalog()
        results = []

        for meta in catalog["models"].values():
            if tag and tag not in meta.get("tags", []):
                continue
            if architecture and meta.get("architecture") != architecture:
                continue
            if max_perplexity is not None and meta.get("val_perplexity", float("inf")) > max_perplexity:
                continue
            results.append(meta)

        # Sort by validation perplexity ascending (best models first)
        results.sort(key=lambda x: x.get("val_perplexity", float("inf")))
        return results

    def load_model(self, model_id: str, device: str = "cpu") -> ARESBaseModel:
        """
        Reads registry metadata, instantiates the model cleanly from its snapshotted config,
        and loads the checkpoint weights into memory.
        """
        meta = self.get_metadata(model_id)
        
        config_path = meta["config_path"]
        weights_path = meta["weights_path"]

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Registered config file missing: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Registered weights file missing: {weights_path}")

        print(f"[ModelRegistry] Loading '{model_id}' onto {device.upper()}...")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        # Flatten the nested YAML structure to match ARESConfig parameters
        flat_config = {}
        if config_data:
            for section in ["dimensions", "dropout", "initialization", "execution"]:
                if section in config_data:
                    flat_config.update(config_data[section])
                    
        config = ARESConfig(**flat_config)
        model = ARESBaseModel(config)
        
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"[ModelRegistry] '{model_id}' ready for inference/evaluation.")
        return model