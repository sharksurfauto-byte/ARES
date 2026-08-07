# ARES-Base/scripts/collect_probe_data.py
"""
ARES Dual-Granularity Activation Collector (GRM + LRM)

Collects two levels of hidden state activations for failure prediction:
1. GRM (Global Reliability Module): Prompt-level activation (last token of prompt) -> predicts overall sequence success (R_global).
2. LRM (Local Reliability Module): Autoregressive decoding activations -> predicts next-token correctness (R_local).
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
    parser = argparse.ArgumentParser(description="Collect GRM & LRM activation datasets for ARES Failure Prediction")
    parser.add_argument("--model-id", type=str, required=True, help="Model ID registered in ModelRegistry")
    parser.add_argument("--registry-path", type=str, default="models/registry.json", help="Path to registry.json")
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "openwebtext", "wikitext"])
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to collect from")
    parser.add_argument("--max-examples", type=int, default=150, help="Maximum number of sequences to process")
    parser.add_argument("--max-tokens", type=int, default=150000, help="Threshold to stop collecting tokens")
    parser.add_argument("--prompt-length", type=int, default=256, help="Token length designated as prompt context")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for forward pass")
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", help="Layer indices to extract activations from")
    parser.add_argument("--output-dir", type=str, default="data/probe_data", help="Directory to save collected tensors")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    parser.add_argument("--untrained", action="store_true", help="If set, initializes model weights randomly (Control Task)")
    return parser.parse_args()

def main():
    args = parse_args()
    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    P = args.prompt_length

    print("="*75)
    print(f" ARES Dual Collector (GRM + LRM) | Layers: {target_layers} | Device: {args.device.upper()}")
    if args.untrained:
        print(" [CONTROL EXPERIMENT] Running with untrained (randomly initialized) weights!")
    print("="*75)

    # 1. Load model from registry or initialize randomly for control
    registry = ModelRegistry(registry_path=args.registry_path)
    if args.untrained:
        print(f"\n[1/3] Initializing model '{args.model_id}' with random (untrained) weights...")
        meta = registry.registry.get(args.model_id)
        if not meta:
            raise ValueError(f"Model ID '{args.model_id}' not found in registry.")
        
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
    captured_activations: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}

    def save_activation_callback(payload):
        l_idx = payload.get("layer_idx")
        if l_idx in target_layers:
            captured_activations[l_idx].append(payload["block_output"].detach().cpu())

    hooks.register("after_block", save_activation_callback)

    # Accumulators
    lrm_layer_tensors: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}
    grm_layer_tensors: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}
    
    collected_lrm_labels = []
    collected_grm_labels = []
    collected_token_ids = []
    collected_heuristics = {
        "max_prob": [],
        "entropy": [],
        "margin": [],
        "token_nll": []
    }

    total_tokens_collected = 0
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction="none")

    print(f"\n[2/3] Processing sequences & extracting GRM + LRM activations...")

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            inputs_ids = batch["input_ids"].to(args.device)
            B, T = inputs_ids.shape
            
            if T <= P:
                continue

            for layer in target_layers:
                captured_activations[layer].clear()

            # Forward pass
            logits, _, _ = model(inputs_ids, hooks=hooks)

            missing_layers = [l for l in target_layers if not captured_activations[l]]
            if missing_layers:
                continue

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

            # Per-token correctness: 1 if correct token, 0 if failure
            correctness = (preds == shift_targets).long()
            
            # --- LRM (Local Decoding Step Activations) ---
            # Decoding steps: P-1 to T-2 (aligns with target tokens P to T-1)
            lrm_targets = correctness[:, P-1:]
            collected_lrm_labels.append(lrm_targets.reshape(-1))
            collected_token_ids.append(shift_targets[:, P-1:].reshape(-1))
            
            collected_heuristics["max_prob"].append(max_prob[:, P-1:].reshape(-1))
            collected_heuristics["entropy"].append(entropy[:, P-1:].reshape(-1))
            collected_heuristics["margin"].append(margin[:, P-1:].reshape(-1))
            collected_heuristics["token_nll"].append(token_nll[:, P-1:].reshape(-1))

            # --- GRM (Global Prompt-Level Activations) ---
            # Sequence success label: 1 if >70% tokens generated correctly in generation block, else 0
            gen_accuracy = correctness[:, P-1:].float().mean(dim=-1)
            grm_labels = (gen_accuracy >= 0.70).long()
            collected_grm_labels.append(grm_labels)

            # Extract layer hidden states
            for layer in target_layers:
                block_out = captured_activations[layer][0] # Shape: (B, T, H)
                
                # GRM: Hidden state at last prompt token (index P-1)
                grm_act = block_out[:, P-1, :] # Shape: (B, H)
                grm_layer_tensors[layer].append(grm_act)
                
                # LRM: Hidden states during generation (indices P-1 to T-2)
                lrm_act = block_out[:, P-1:-1, :] # Shape: (B, T-P, H)
                lrm_layer_tensors[layer].append(lrm_act.reshape(-1, model.config.hidden_size))

            total_tokens_collected += lrm_targets.numel()

            if (step + 1) % 20 == 0 or (step + 1) == len(dataloader):
                print(f"  Processed {step + 1}/{len(dataloader)} batches ({total_tokens_collected:,} LRM tokens)...")

            if total_tokens_collected >= args.max_tokens:
                print(f"--> Reached max token threshold ({args.max_tokens:,}). Stopping collection.")
                break

    # 4. Save collected data to disk
    print(f"\n[3/3] Saving GRM + LRM dataset tensors to disk...")
    folder_name = args.dataset
    if args.untrained:
        folder_name += "_untrained"
        
    out_path = Path(args.output_dir) / folder_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Save Labels
    Y_lrm = torch.cat(collected_lrm_labels, dim=0)
    Y_grm = torch.cat(collected_grm_labels, dim=0)
    torch.save(Y_lrm, out_path / "labels_lrm.pt")
    torch.save(Y_grm, out_path / "labels_grm.pt")
    
    # Legacy fallback alias for existing scripts
    torch.save(Y_lrm, out_path / "labels.pt")

    token_ids_tensor = torch.cat(collected_token_ids, dim=0)
    torch.save(token_ids_tensor, out_path / "token_ids.pt")

    heuristics_dict = {
        key: torch.cat(collected_heuristics[key], dim=0) for key in collected_heuristics
    }
    torch.save(heuristics_dict, out_path / "heuristics.pt")

    # Save layer tensors
    for layer in target_layers:
        # Save LRM (Local)
        X_lrm = torch.cat(lrm_layer_tensors[layer], dim=0)
        torch.save(X_lrm, out_path / f"X_lrm_layer_{layer}.pt")
        torch.save(X_lrm, out_path / f"X_layer_{layer}.pt") # fallback alias
        
        # Save GRM (Global)
        X_grm = torch.cat(grm_layer_tensors[layer], dim=0)
        torch.save(X_grm, out_path / f"X_grm_layer_{layer}.pt")
        
        print(f"  - Saved Layer {layer:02d}: GRM {X_grm.shape} | LRM {X_lrm.shape}")

    print(f"\n[SUCCESS] Final Dataset Generated at '{out_path}':")
    print(f"  - Total Sequences (GRM): {Y_grm.numel():,} | Total Tokens (LRM): {Y_lrm.numel():,}")
    print(f"  - GRM Prompt Success Rate: {(Y_grm == 1).float().mean() * 100:.2f}%")
    print(f"  - LRM Token Accuracy: {(Y_lrm == 1).float().mean() * 100:.2f}%")

if __name__ == "__main__":
    main()