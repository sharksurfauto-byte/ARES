import sys
from pathlib import Path
import torch

sys.path.append("/kaggle/working/ARES/ARES-Base")

from model.config import ARESConfig
from model.gpt import ARESBaseModel
from models.registry import ModelRegistry

#paths for restored run
run_dir = "/kaggle/working/ARES/ARES-Base/experiments/runs/exp_001_baseline_run"
config_path = f"{run_dir}/configs/model.yaml"
weights_path = f"{run_dir}/checkpoints/checkpoint_epoch_1.pt"

#init model configs and weights
config=ARESConfig.from_yaml(config_path)
model=ARESBaseModel(config)

#register it in registry.json
registry = ModelRegistry(registry_path="/kaggle/working/ARES/ARES-Base/models/registry.json")
registry.register_model(
    model_id="ARES-Base-v1.0-tinystories-baseline_run",
    architecture="gpt2-decoder-only",
    model=model,
    weights_path=weights_path,
    config_path=config_path,
    training_dataset="tinystories",
    total_tokens_trained=125 * 4 * 1024 * 4,
    val_perplexity=6.88,
    notes="Restored from Kaggle baseline run"
)