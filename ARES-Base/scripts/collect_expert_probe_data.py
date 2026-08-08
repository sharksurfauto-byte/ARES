# ARES-Base/scripts/collect_expert_probe_data.py
r"""
Target File Location in Repository:
--> ARES-Base/scripts/collect_expert_probe_data.py

ARES-Orion Step 2 Dataset Collector (Phase 3)
==============================================
Collects pre-expert hidden state activations (h_t) and evaluates each token 
against all K=4 individual expert sub-networks to generate expert-specific 
correctness labels:

    Y_{e, t} = 1  if Expert e correctly predicts token x_{t+1}
    Y_{e, t} = 0  if Expert e fails (incorrect prediction)

Supports Dual Kaggle T4 Multi-GPU acceleration.

Author: ARES AI Research Team
Date: August 2026
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tokenizer import get_tokenizer
from ares_datasets import get_dataset
from models.registry import ModelRegistry
from utils.hooks import HookRegistry
from ares_moe.moe_layer import ARESMoELayer, prepare_model_for_multi_gpu

def parse_args():
    parser = argparse.ArgumentParser(description="Collect Expert-Specific Probe Datasets for ARES-Orion")
    parser.add_argument("--model-id", type=str, required=True, help="Model ID registered in ModelRegistry")
    parser.add_argument("--registry-path", type=str, default="models/registry.json", help="Path to registry.json")
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "openwebtext", "wikitext"])
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to collect from")
    parser.add_argument("--max-examples", type=int, default=150, help="Maximum number of sequences to process")
    parser.add_argument("--max-tokens", type=int, default=150000, help="Threshold to stop collecting tokens")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for forward pass")
    parser.add_argument("--moe-layers", type=str, default="4,8,11", help="MoE insertion layer indices")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of expert sub-networks")
    parser.add_argument("--output-dir", type=str, default="data/expert_probe_data", help="Directory to save tensors")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    return parser.parse_args()

def main():
    args = parse_args()
    moe_layers = [int(x.strip()) for x in args.moe_layers.split(",")]
    K = args.num_experts

    print("=" * 80)
    print(f" ARES-Orion Step 2: Expert Failure Dataset Collector")
    print(f" MoE Layers: {moe_layers} | Experts: {K} | Device: {args.device.upper()}")
    print("=" * 80)

    # 1. Load Pre-trained Base Model
    registry = ModelRegistry(registry_path=args.registry_path)
    print(f"\n[1/4] Loading pre-trained base model '{args.model_id}'...")
    model = registry.load_model(args.model_id, device=args.device)
    model.eval()

    # Configure Multi-GPU execution across Dual Kaggle T4 GPUs
    model, device = prepare_model_for_multi_gpu(model, args.device)

    # 2. Init Tokenizer & Streaming Dataset
    tokenizer = get_tokenizer("gpt2-bpe")
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    max_seq_len = raw_model.config.max_position_embeddings

    print(f"\n[2/4] Streaming dataset '{args.dataset}' split '{args.split}'...")
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
        pin_memory=(args.device.startswith("cuda"))
    )

    # 3. Setup HookRegistry to intercept hidden states
    hooks = HookRegistry()
    captured_activations: Dict[int, torch.Tensor] = {}

    def save_activation_callback(payload):
        l_idx = payload.get("layer_idx")
        if l_idx in moe_layers:
            captured_activations[l_idx] = payload["block_output"].detach()

    hooks.register("after_block", save_activation_callback)

    # 4. Instantiate MoE Layer Engine to evaluate expert sub-networks
    print(f"\n[3/4] Initializing MoE Layer Pool ({K} experts per MoE block)...")
    sample_moe_layer = ARESMoELayer(
        num_experts=K,
        top_k=1,
        hidden_size=raw_model.config.hidden_size,
        intermediate_size=getattr(raw_model.config, "intermediate_size", raw_model.config.hidden_size * 4)
    ).to(device)
    sample_moe_layer.eval()

    # Accumulators per MoE layer
    collected_hidden_states: Dict[int, List[torch.Tensor]] = {l: [] for l in moe_layers}
    collected_expert_labels: Dict[int, Dict[int, List[torch.Tensor]]] = {
        l: {e: [] for e in range(K)} for l in moe_layers
    }

    total_tokens_collected = 0

    print(f"\n[4/4] Processing sequences & evaluating expert-specific failure vectors...")

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            B, T = input_ids.shape
            
            if T < 2:
                continue
            shift_targets_gpu = input_ids[:, 1:] # Target tokens (B, T-1) on GPU

            # Clear previous batch activations & run DataParallel model forward pass across Dual T4 GPUs
            captured_activations.clear()
            logits, _, _ = model(input_ids, hooks=hooks)

            for layer_idx in moe_layers:
                if layer_idx not in captured_activations:
                    continue

                # Hidden state at MoE layer input (before FFN)
                layer_h = captured_activations[layer_idx][:, :-1, :] # (B, T-1, H)
                H_dim = layer_h.shape[-1]
                flat_h = layer_h.reshape(-1, H_dim) # (N, H)
                collected_hidden_states[layer_idx].append(flat_h.cpu())

                # Evaluate each Expert sub-network on flat_h directly on GPU
                for e_idx in range(K):
                    expert = sample_moe_layer.experts[e_idx]
                    
                    # Forward pass through Expert e FFN block
                    expert_ffn_out = expert(layer_h) # (B, T-1, H)
                    
                    # Compute token logits using base model LM Head directly on GPU
                    expert_logits = raw_model.lm_head(expert_ffn_out) # (B, T-1, Vocab) on GPU
                    expert_preds = torch.argmax(expert_logits, dim=-1) # (B, T-1) on GPU
                    
                    # Expert Correctness Label: 1 if Expert e correctly predicts target token, 0 if fail
                    expert_correct = (expert_preds == shift_targets_gpu).long().reshape(-1).cpu() # (N,)
                    collected_expert_labels[layer_idx][e_idx].append(expert_correct)

            total_tokens_collected += (T - 1) * B

            if (step + 1) % 20 == 0 or (step + 1) == len(dataloader):
                print(f"  Processed {step + 1}/{len(dataloader)} batches ({total_tokens_collected:,} tokens)...")

            if total_tokens_collected >= args.max_tokens:
                print(f"--> Reached max token threshold ({args.max_tokens:,}). Stopping collection.")
                break

    # 5. Save Tensors to Disk
    print(f"\n[SAVE] Exporting expert probe tensors to disk...")
    out_path = Path(args.output_dir) / args.dataset
    out_path.mkdir(parents=True, exist_ok=True)

    for layer_idx in moe_layers:
        if not collected_hidden_states[layer_idx]:
            continue
        # Save shared hidden states (h_t)
        X_h = torch.cat(collected_hidden_states[layer_idx], dim=0)
        torch.save(X_h, out_path / f"X_moe_layer_{layer_idx}.pt")
        
        print(f"\n  MoE Layer {layer_idx:02d} | Shared Hidden States: {X_h.shape}")
        
        # Save Expert-Specific Labels
        for e_idx in range(K):
            Y_e = torch.cat(collected_expert_labels[layer_idx][e_idx], dim=0)
            torch.save(Y_e, out_path / f"Y_layer_{layer_idx}_expert_{e_idx}.pt")
            acc_e = (Y_e == 1).float().mean() * 100.0
            print(f"    - Expert {e_idx} Target Accuracy: {acc_e:.2f}% (Saved Y_layer_{layer_idx}_expert_{e_idx}.pt)")

    print(f"\n[SUCCESS] Step 2 Dataset Generation Complete! Saved at '{out_path}'.")

if __name__ == "__main__":
    main()