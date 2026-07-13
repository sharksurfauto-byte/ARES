import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.config import ARESConfig
from model.gpt import ARESBaseModel
from tokenizer import get_tokenizer
from ares_datasets import get_dataset
from training.trainer import ARESTrainer
from training.checkpoint import CheckpointManager
from experiments.manager import ExperimentManager
from models.registry import ModelRegistry
from utils.hooks import HookRegistry

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARES-Base v1.0 Pre-Training Script")
    
    # Configuration pathways
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model architecture YAML")
    parser.add_argument("--train-config", type=str, default="configs/training.yaml", help="Path to training parameters YAML")
    
    # Dataset & Execution overrides
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "openwebtext"], help="Target dataset")
    parser.add_argument("--exp-name", type=str, default="baseline_run", help="Experiment identifier name")
    parser.add_argument("--notes", type=str, default="", help="Hypothesis or notes for the experiment audit trail")
    parser.add_argument("--load-pretrained", action="store_true", help="Initialize with OpenAI GPT-2 weights instead of from scratch")
    
    # Hardware & Performance scaling
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device micro batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for asynchronous data fetching")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target execution device")

    return parser.parse_args()

def load_yaml_config(filepath:str)-> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open (filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def configure_optimizer(model:torch.nn.Module, weight_decay:float, learning_rate:float)->torch.optim.AdamW:
    decay_params=[]
    no_decay_params=[]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim()>=2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups=[
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

def main():
    args=parse_args()
    print("=" * 70)
    print(f" ARES-Base v1.0 Pre-Training Pipeline | Target Device: {args.device.upper()}")
    print("=" * 70)

    #1. init immutable exp manager
    exp_manager=ExperimentManager(base_dir="experiments/runs", exp_name=args.exp_name)
    exp_dir=exp_manager.init_experiment(
        config_paths=[args.model_config, args.train_config],
        notes=args.notes or f"Training ARES-Base on '{args.dataset}' dataset."
    )

    #2. load config schemas
    print("\n[1/7] Loading configuration schemas...")
    model_config=ARESConfig.from_yaml(args.model_config)
    train_yaml=load_yaml_config(args.train_config)

    #apply command_line overrides
    max_seq_len = model_config.max_position_embeddings
    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = train_yaml.get("optimizer", {}).get("weight_decay", 0.1)
    grad_accum_steps = train_yaml.get("training", {}).get("gradient_accumulation_steps", 4)
    max_grad_norm = train_yaml.get("training", {}).get("max_grad_norm", 1.0)

    #3. instantiate modular tokenizer
    print(f"[2/7] Initializing BPE Tokenizer (Vocab Size: {model_config.vocab_size:,})...")
    tokenizer = get_tokenizer("gpt2-bpe")

    #4. prepare streaming dataset & dataloaders
    print(f"[3/7] Setting up '{args.dataset}' dataset pipeline (Block Size: {max_seq_len})...")
    train_dataset = get_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        split="train"
    )
    
    val_dataset = get_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        split="validation"
    )

    #enable pinned mem for accelerated CUDA asynchronous DMA transfers
    use_pin_memory=(args.device=='cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    ) if len(val_dataset) > 0 else None

    #5. build GPT2 architecture & research hook registry
    print("[4/7] Constructing ARES-Base Model and Hook Registry...")
    hooks=HookRegistry()
    model=ARESBaseModel(model_config)

    if args.load_pretrained:
        print("--> Importing official OpenAI pre-trained weights (handling Conv1D transposition)...")
        model = CheckpointManager.load_pretrained_gpt2(model, model_name="gpt2")
    else:
        print("--> Weights randomly initialized from standard normal distribution N(0, 0.02).")

    #6. configure optimizer and learing rate scheduler
    print("[5/7] Configuring AdamW Optimizer with Weight Decay filtering...")
    optimizer = configure_optimizer(model, weight_decay=weight_decay, learning_rate=learning_rate)

    # Cosine annealing scheduler with linear warmup
    total_steps = (len(train_loader) // grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * 0.05)  # 5% warmup

    def lr_lambda(current_step:int)->float:
        if current_step<warmup_steps:
            return float(current_step)/float(max(1,warmup_steps))
        progress=float(current_step-warmup_steps)/float(max(1, total_steps-warmup_steps))
        return max(0.1,0.5*(1.0 + math.cos(math.pi * progress)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)

    #7. init trainer and exec training
    print("[6/7] Initializing ARESTrainer Orchestration Engine...")
    trainer = ARESTrainer(
        model=model,
        config=model_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        output_dir=exp_dir / "checkpoints",
        log_interval=train_yaml.get("logging", {}).get("log_interval", 10),
        eval_interval=train_yaml.get("logging", {}).get("eval_interval", 500),
        save_interval=train_yaml.get("logging", {}).get("save_interval", 1000),
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=grad_accum_steps
    )

    print("\n[7/7] Kicking off pre-training loop...")
    history = trainer.train(num_epochs=args.epochs)

    #8. post training registration and audit loggin
    print("\n" + "=" * 70)
    print(" Training Completed Successfully. Registering Artifacts...")
    print("=" * 70)

    best_val_loss = min(history["val_loss"]) if history["val_loss"] else history["train_loss"][-1]
    final_ppl = math.exp(best_val_loss)

    # Register baseline into central model registry
    registry = ModelRegistry()
    registry.register_model(
        model_id=f"ARES-Base-v1.0-{args.dataset}-{args.exp_name}",
        architecture="gpt2-decoder-only",
        model=model,
        weights_path=exp_dir / "checkpoints" / "best_model.pt",
        config_path=exp_dir / "configs" / "model.yaml",
        training_dataset=args.dataset,
        total_tokens_trained=total_steps * batch_size * max_seq_len * grad_accum_steps,
        val_perplexity=final_ppl,
        tags=["baseline", args.dataset, "frozen" if not args.load_pretrained else "finetuned"],
        notes=args.notes
    )

    print(f"\n[SUCCESS] Final Perplexity: {final_ppl:.2f} | Audit trail secured at: {exp_dir}")

if __name__ == "__main__":
    main()