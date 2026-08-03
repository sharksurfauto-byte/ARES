# ARES-Base/scripts/collect_probe_data.py
"""
This script handles the data collection. It runs validation data through the frozen baseline model, 
intercepts hidden states at specific layer depths using the hook registry, extracts output heuristics, 
and computes binary token correctness targets.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tokenizer import get_tokenizer
from ares_datasets import get_dataset
from models.registry import ModelRegistry
from utils.hooks import HookRegistry

def parse_args():
    parser = argparse.ArgumentParser(description="Collect internal activation datasets for ARES Failure Prediction")
    parser.add_argument("--model-id", type=str, required=True, help="Model ID registered in ModelRegistry")
    parser.add_argument("--registry-path", type=str, default="models/registry.json", help="Path to registry.json")
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "openwebtext", "wikitext"])
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to collect from")
    parser.add_argument("--max-examples", type=int, default=1000, help="Maximum number of sequences to process")
    parser.add_argument("--max-tokens", type=int, default=500000, help="Threshold to stop collecting tokens to prevent OOM")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for forward pass")
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", help="Comma-separated layer indices to extract activations from")
    parser.add_argument("--output-dir", type=str, default="data/probe_data", help="Directory to save collected tensors")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    parser.add_argument("--untrained", action="store_true", help="If set, initializes model weights randomly (Control Task)")
    return parser.parse_args()

def main():
    args = parse_args()
    target_layers = [int(x.strip()) for x in args.layers.split(",")]

    print("="*70)
    print(f" ARES Activation Collector | Target Layers: {target_layers} | Device: {args.device.upper()}")
    if args.untrained:
        print(" [CONTROL EXPERIMENT] Running with untrained (randomly initialized) weights!")
    print("="*70)

    # 1. Load model from registry or initialize randomly for control
    registry = ModelRegistry(registry_path=args.registry_path)
    if args.untrained:
        print(f"\n[1/3] Initializing model '{args.model_id}' with random (untrained) weights...")
        meta = registry.registry.get(args.model_id)
        if not meta:
            raise ValueError(f"Model ID '{args.model_id}' not found in registry. We need its config path.")
        
        from model.config import ARESConfig
        from model.gpt import ARESBaseModel
        config = ARESConfig.from_yaml(meta["config_path"])
        model = ARESBaseModel(config).to(args.device)
    else:
        print(f"\n[1/3] Loading pre-trained model '{args.model_id}' from registry...")
        model = registry.load_model(args.model_id, device=args.device)
    
    model.eval()

    # 2. Init tokenizer and dataset
    tokenizer = get_tokenizer("gpt2-bpe")
    max_seq_len = model.config.max_position_embeddings

    print(f"\n[1/3] Loading dataset '{args.dataset}' split '{args.split}'...")
    dataset = get_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        split=args.split,
        max_examples=args.max_examples
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(args.device == 'cuda')
    )

    # 3. Setup hooks to intercept layer outputs
    hooks = HookRegistry()

    # Store activations: layer idx -> list of tensors
    captured_activations: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}

    def save_activation_callback(payload):
        l_idx = payload.get("layer_idx")
        if l_idx in target_layers:
            # Capture block output and detach from computation graph to save memory
            captured_activations[l_idx].append(payload["block_output"].detach().cpu())

    hooks.register("after_block", save_activation_callback)

    # Data collection accumulators
    layer_tensors: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}
    collected_heuristics = {
        "max_prob": [],
        "entropy": [],
        "margin": [],
        "token_nll": [],
        "position": []
    }
    collected_labels = []
    collected_token_ids = []

    total_tokens_collected = 0
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction="none")

    print(f"\n[2/3] Processing sequences and capturing activations...")

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            inputs_ids = batch["input_ids"].to(args.device)
            B, T = inputs_ids.shape

            # Reset temp hook logs
            for layer in target_layers:
                captured_activations[layer].clear()

            # Forward pass (triggers block hooks)
            logits, _, _ = model(inputs_ids, hooks=hooks)

            # Check if all target layers are captured
            missing_layers = [l for l in target_layers if not captured_activations[l]]
            if missing_layers:
                print(f"[Warning] Step {step}: Missing activations for layers {missing_layers}. Skipping batch.")
                continue

            # Targets are shift-right next tokens
            shift_targets = inputs_ids[:, 1:].cpu()
            shift_logits = logits[:, :-1, :].cpu()

            probs = torch.softmax(shift_logits, dim=-1)
            max_prob, preds = torch.max(probs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
            
            top2_probs, _ = torch.topk(probs, k=2, dim=-1)
            margin = top2_probs[:, :, 0] - top2_probs[:, :, 1]
            
            token_nll = cross_entropy_fn(
                shift_logits.reshape(-1, model.config.vocab_size),
                shift_targets.reshape(-1)
            ).reshape(B, T - 1)

            positions = torch.arange(1, T, dtype=torch.float32).unsqueeze(0).expand(B, -1)

            # Binary Correctness: 1 if model predicted target token, else 0
            correctness = (preds == shift_targets).long()

            # Accumulate data
            collected_labels.append(correctness.reshape(-1))
            collected_token_ids.append(shift_targets.reshape(-1))
            collected_heuristics["max_prob"].append(max_prob.reshape(-1))
            collected_heuristics["entropy"].append(entropy.reshape(-1))
            collected_heuristics["margin"].append(margin.reshape(-1))
            collected_heuristics["token_nll"].append(token_nll.reshape(-1))
            collected_heuristics["position"].append(positions.reshape(-1))

            # Store hidden states
            for layer in target_layers:
                # Hook shape: (B, T, H)
                block_out = captured_activations[layer][0]
                shift_block_out = block_out[:, :-1, :]  # align with targets
                layer_tensors[layer].append(shift_block_out.reshape(-1, model.config.hidden_size))

            total_tokens_collected += correctness.numel()

            if (step + 1) % 20 == 0 or (step + 1) == len(dataloader):
                print(f"  Processed {step + 1}/{len(dataloader)} batches ({total_tokens_collected:,} tokens)...")

            if total_tokens_collected >= args.max_tokens:
                print(f"--> Reached max token threshold ({args.max_tokens:,}). Stopping collection.")
                break

    # 4. Save collected data to disk
    print(f"\n[3/3] Saving collected dataset tensors to disk...")
    
    # We append a specific suffix to directories for control groups
    folder_name = args.dataset
    if args.untrained:
        folder_name += "_untrained"
        
    out_path = Path(args.output_dir) / folder_name
    out_path.mkdir(parents=True, exist_ok=True)

    Y = torch.cat(collected_labels, dim=0)
    torch.save(Y, out_path / "labels.pt")
    
    token_ids_tensor = torch.cat(collected_token_ids, dim=0)
    torch.save(token_ids_tensor, out_path / "token_ids.pt")

    heuristics_dict = {
        key: torch.cat(collected_heuristics[key], dim=0) for key in collected_heuristics
    }
    torch.save(heuristics_dict, out_path / "heuristics.pt")

    for layer in target_layers:
        X_layer = torch.cat(layer_tensors[layer], dim=0)
        x_file = out_path / f"X_layer_{layer}.pt"
        torch.save(X_layer, x_file)
        print(f"  - Saved Layer {layer:02d} activations: {x_file} ({X_layer.shape})")

    # Logging summary statistics
    correct_count = (Y == 1).sum().item()
    incorrect_count = (Y == 0).sum().item()
    accuracy = (correct_count / Y.numel()) * 100 if Y.numel() > 0 else 0
    print(f"\n[SUCCESS] Final Dataset Generated:")
    print(f"  - Output Folder: {out_path}")
    print(f"  - Total Tokens: {Y.numel():,}")
    print(f"  - Model Accuracy: {accuracy:.2f}% (Correct: {correct_count:,} | Failures: {incorrect_count:,})")

if __name__ == "__main__":
    main()