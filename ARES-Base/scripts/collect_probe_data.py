"""
This script handles the data collection. It runs validation data through the frozen baseline model, intercepts hidden states at specific layer   
depths using the hook registry, extracts output heuristics, and computes binary token correctness targets.
"""
#### Explanation of Key Design Details:

# • Target Layers: It hooks layers 3, 6, 9, and 11 to observe how failure prediction performance evolves through model depth.
# • VRAM and Disk Guard (--max-tokens): 1,000 TinyStories validation sequences contain ~2 million tokens. Collecting 768-dimensional floats for    
# 4 layers would take 6.1 GB of RAM/disk. The --max-tokens flag limits collection to a safe number (default 500,000 tokens, which takes about 1.   
# 5 GB).
# • Correctness Labeling: It compares preds (argmax of logits) against the target token at position t + 1.
# • Heuristics Logging: It logs conventional uncertainty metrics (max_probability, entropy, margin, token_nll, token_position) for comparison.

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
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "openwebtext"])
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to collect from")
    parser.add_argument("--max-examples", type=int, default=1000, help="Maximum number of sequences to process")
    parser.add_argument("--max-tokens", type=int, default=500000, help="Threshold to stop collecting tokens to prevent OOM")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for forward pass")
    parser.add_argument("--layers", type=str, default="3,6,9,11", help="Comma-separated layer indices to extract activations from")
    parser.add_argument("--output-dir", type=str, default="data/probe_data", help="Directory to save collected tensors")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    return parser.parse_args()

def main():
    args=parse_args()
    target_layers=[int(x.strip()) for x in args.layers.split(",")]

    print("="*70)
    print(f" ARES Activation Collector | Target Layers: {target_layers} | Device: {args.device.upper()}")
    print("="*70)

    #1. load pretrained model from registry
    registry=ModelRegistry(registry_path=args.registry_path)
    model=registry.load_model(args.model_id, device=args.device)
    model.eval()

    #2. Init tokenizer and dataset
    tokenizer=get_tokenizer("gpt2-bpe")
    max_seq_len=model.config.max_position_embeddings

    print(f"\n[1/3] Loading dataset split '{args.split}'...")
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
        pin_memory=(args.device=='cuda')
    )

    #3. setup hooks to intercept layer outputs
    hooks=HookRegistry()

    #store activation: layer idx -> list of tensors
    captured_activations:Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}

    def save_activation_callback(payload):
        l_idx=payload.get("layer_idx")
        if l_idx in target_layers:
            #capture teh block output tensor and detach it form the graph
            captured_activations[l_idx].append(payload["block_output"].detach().cpu())

    hooks.register("after_block", save_activation_callback)

    #data collection accum
    layer_tensors:Dict[int,List[torch.Tensor]]= {layer:[] for layer in target_layers}
    collected_heuristics={
        "max_prob": [],
        "entropy": [],
        "margin": [],
        "token_nll": [],
        "position": []
    }
    collected_labels=[]

    total_tokens_collected=0
    cross_entropy_fn=torch.nn.CrossEntropyLoss(reduction="none")

    print(f"\n[2/3] Processing sequences and capturing activations...")

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            inputs_ids=batch["input_ids"].to(args.device)
            B,T=inputs_ids.shape

            #reset temp hook logs
            for layer in target_layers:
                captured_activations[layer].clear()

            #forward pass (this triggers the callbacks)
            logits,_,_ = model(inputs_ids, hooks=hooks)

            #check if all target layers are captures
            missing_layers=[l for l in target_layers if not captured_activations[l]]
            if missing_layers:
                print(f"[Warning] Step {step}: Missing activations for layers {missing_layers}. Skipping batch.")
                continue

            #target tokens
            shift_targets=inputs_ids[:,1:].cpu()
            #pred logtis
            shift_logits=logits[:,:-1,:].cpu()

            probs=torch.softmax(shift_logits, dim=-1)
            max_prob, preds=torch.max(probs, dim=-1)
            entropy=-torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
            #prob margin
            top2_probs,_=torch.topk(probs, k=2, dim=-1)
            margin=top2_probs[:,:,0]-top2_probs[:,:,1]
            token_nll=cross_entropy_fn(
                shift_logits.reshape(-1,model.config.vocab_size),
                shift_targets.reshape(-1)
            ).reshape(B,T-1)

            #position indices
            positions=torch.arange(1,T,dtype=torch.float32).unsqueeze(0).expand(B,-1)

            # --- Target Labels (1 if base model predicted correct token, else 0) ---
            correctness = (preds == shift_targets).long()

            #Accum data
            collected_labels.append(correctness.reshape(-1))
            collected_heuristics["max_prob"].append(max_prob.reshape(-1))
            collected_heuristics["entropy"].append(entropy.reshape(-1))
            collected_heuristics["margin"].append(margin.reshape(-1))
            collected_heuristics["token_nll"].append(token_nll.reshape(-1))
            collected_heuristics["position"].append(positions.reshape(-1))

            #store hidden states for each target layer
            for layer in target_layers:
                #shape form hooks: (B,T,H)
                block_out=captured_activations[layer][0]
                #shift to align with target tokens
                shift_block_out=block_out[:,:-1,:]
                layer_tensors[layer].append(shift_block_out.reshape(-1,model.config.hidden_size))

            total_tokens_collected+=correctness.numel()

            if (step+1)%20==0 or (step+1) ==len(dataloader):
                print(f"  Processed {step + 1}/{len(dataloader)} batches ({total_tokens_collected:,} tokens)...")

            #mem limit check
            if total_tokens_collected>=args.max_tokens:
                print(f"--> Reached max token threshold ({args.max_tokens:,}). Stopping collection.")
                break
    # concatenate and save collected data
    print(f"\n[3/3] Saving collected dataset tensors to disk...")
    out_path=Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    Y=torch.cat(collected_labels, dim=0)
    torch.save(Y,out_path / "labels.pt")

    #concat and save hueristics
    heuristics_dict={
        key: torch.cat(collected_heuristics[key],dim=0) for key in collected_heuristics
    }
    torch.save(heuristics_dict, out_path/"heuristics.pt")

    #concat and save hidden states layer by layer
    for layer in target_layers:
        X_layer=torch.cat(layer_tensors[layer], dim=0)
        x_file=out_path/f"X_layer_{layer}.pt"
        torch.save(X_layer, x_file)
        print(f"  - Saved Layer {layer:02d} activations: {x_file} ({X_layer.shape})")

    #log summary stats
    correct_count = (Y == 1).sum().item()
    incorrect_count = (Y == 0).sum().item()
    accuracy = (correct_count / Y.numel()) * 100 if Y.numel() > 0 else 0
    print(f"\n[SUCCESS] Final Dataset Generated:")
    print(f"  - Total Tokens: {Y.numel():,}")
    print(f"  - Base Model Accuracy: {accuracy:.2f}% (Correct: {correct_count:,} | Failures: {incorrect_count:,})")

if __name__=="__main__":
    main()