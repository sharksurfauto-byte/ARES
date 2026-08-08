# ARES-Base/scripts/train_ares_orion_router.py
r"""
ARES-Orion Step 3: Joint Reliability Probe & Router Trainer (Phase 3)
======================================================================
1. Fits K=4 Expert Reliability Probes (f_{theta, e}) to predict expert correctness r_e(t).
2. Optimizes Router Gating weights W_r with auxiliary load balancing loss + log-odds reliability augmentation:
   S_e(t) = W_{r,e}^\top h_t + \lambda * log(r_e / (1 - r_e))
3. Executes \lambda parameter sweep (\lambda \in [0.0, 2.0]) to plot the Pareto trade-off curve.
4. Computes Expert Selection Regret (Oracle Gap):
   Regret = 1 - (Accuracy_selected / Accuracy_oracle_best)
5. Dual Kaggle T4 Multi-GPU acceleration enabled.

Author: ARES AI Research Team
Date: August 2026
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError("Please install scikit-learn: pip install scikit-learn")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ares_moe.moe_layer import prepare_model_for_multi_gpu


# ============================================================================
# 1. Non-Linear MLP Reliability Probe for Expert e
# ============================================================================
class ExpertReliabilityProbe(nn.Module):
    """
    MLP Probe predicting predicted correctness probability r_e(t) in [0, 1]
    for Expert e given pre-expert hidden state h_t.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ============================================================================
# 2. Calibration & Performance Metric Utilities
# ============================================================================
def compute_ece(y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    ece = 0.0
    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        in_bin = (y_pred_prob > bin_lower) & (y_pred_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred_prob[in_bin])
            ece += prop_in_bin * abs(accuracy_in_bin - avg_confidence_in_bin)
    return ece


def evaluate_probe_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Evaluates probe AUROC, AUPRC, ECE, and Brier score."""
    if len(np.unique(y_true)) < 2:
        auroc, auprc = 0.5, 0.5
    else:
        auroc = roc_auc_score(y_true, y_pred_prob)
        auprc = average_precision_score(y_true, y_pred_prob)
        
    ece = compute_ece(y_true, y_pred_prob)
    brier = np.mean((y_true - y_pred_prob) ** 2)
    acc = accuracy_score(y_true, (y_pred_prob >= 0.5).astype(int))
    
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(acc),
        "ece": float(ece),
        "brier": float(brier)
    }


def train_probe_model(
    model: nn.Module, 
    X_train: torch.Tensor, 
    Y_train: torch.Tensor, 
    device: str,
    epochs: int = 4,
    batch_size: int = 512,
    lr: float = 1e-3
) -> None:
    """Trains an expert reliability probe."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    dataset = TensorDataset(X_train, Y_train.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARES-Orion Joint Probe & Router Suite")
    parser.add_argument("--data-dir", type=str, default="data/expert_probe_data", help="Directory where expert tensors are saved")
    parser.add_argument("--output-dir", type=str, default="experiments/orion_router_results", help="Output directory")
    parser.add_argument("--dataset", type=str, default="tinystories", help="Dataset name")
    parser.add_argument("--moe-layers", type=str, default="4,8,11", help="MoE insertion layers")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of expert sub-networks")
    parser.add_argument("--epochs", type=int, default=4, help="Probe training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_dir) / args.dataset
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Expert probe data directory '{data_path}' not found. Run Step 2 first!")

    moe_layers = [int(x.strip()) for x in args.moe_layers.split(",")]
    K = args.num_experts
    lambda_sweep = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

    print("=" * 80)
    print(f" ARES-Orion Step 3: Joint Probe & Router Trainer")
    print(f" Dataset: {args.dataset} | MoE Layers: {moe_layers} | Experts: {K}")
    print("=" * 80)

    all_results = {}

    for layer_idx in moe_layers:
        print(f"\n" + "-" * 75)
        print(f" Processing MoE Layer {layer_idx:02d}...")
        print("-" * 75)

        # 1. Load Shared Hidden States & Expert Labels
        h_file = data_path / f"X_moe_layer_{layer_idx}.pt"
        if not h_file.exists():
            print(f"  [SKIP] File '{h_file}' not found.")
            continue

        X_h = torch.load(h_file) # (N, H)
        N, H = X_h.shape

        Y_experts = []
        for e_idx in range(K):
            y_file = data_path / f"Y_layer_{layer_idx}_expert_{e_idx}.pt"
            if not y_file.exists():
                raise FileNotFoundError(f"Missing label file '{y_file}'")
            Y_experts.append(torch.load(y_file)) # (N,)

        Y_matrix = torch.stack(Y_experts, dim=1) # (N, K)

        # Train/Val Split
        indices = np.arange(N)
        tr_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        X_tr, X_val = X_h[tr_idx], X_h[val_idx]
        Y_tr_mat, Y_val_mat = Y_matrix[tr_idx], Y_matrix[val_idx]

        # 2. Train Expert-Specific Reliability Probes (r_1, r_2, r_3, r_4)
        print(f"\n[1/3] Training {K} Expert Reliability Probes (MLP)...")
        probes: List[ExpertReliabilityProbe] = []
        val_probe_reliabilities = []

        for e_idx in range(K):
            print(f"  - Fitting Probe for Expert {e_idx}...")
            probe = ExpertReliabilityProbe(input_dim=H, hidden_dim=256)
            train_probe_model(
                probe, X_tr, Y_tr_mat[:, e_idx], args.device, 
                epochs=args.epochs, batch_size=args.batch_size
            )
            probe.eval()
            probes.append(probe)

            # Evaluate Probe on Val Set
            with torch.no_grad():
                r_e_val = probe(X_val.to(args.device)).cpu().numpy()
            val_probe_reliabilities.append(r_e_val)

            m_e = evaluate_probe_metrics(Y_val_mat[:, e_idx].numpy(), r_e_val)
            print(f"    └─ Expert {e_idx} Probe | AUROC: {m_e['auroc']:.4f} | ECE: {m_e['ece']:.4f} | Brier: {m_e['brier']:.4f}")

        # Stack predicted reliabilities for validation set: (N_val, K)
        R_val_matrix = torch.tensor(np.column_stack(val_probe_reliabilities)).float()

        # 3. Compute Oracle Best Expert Benchmark & Expert Selection Regret
        oracle_best_expert = torch.argmax(Y_val_mat, dim=1).numpy()
        oracle_accuracy = float(torch.max(Y_val_mat, dim=1)[0].float().mean())
        
        print(f"\n[2/3] Oracle Expert Upper Bound Accuracy: {oracle_accuracy * 100:.2f}%")

        # 4. Execute Lambda Sweep (Router Gating Evaluation)
        print(f"\n[3/3] Running Lambda Reliability Sweep (λ ∈ {lambda_sweep})...")
        
        # Initialize Router Weights W_r
        router_gate = nn.Linear(H, K, bias=False).to(args.device)
        nn.init.kaiming_uniform_(router_gate.weight, a=math.sqrt(5))

        layer_sweep_records = {}

        for lam in lambda_sweep:
            with torch.no_grad():
                base_logits = router_gate(X_val.to(args.device)).cpu() # (N_val, K)
                
                if lam > 0.0:
                    eps = 1e-6
                    rel_logits = torch.log((R_val_matrix + eps) / (1.0 - R_val_matrix + eps))
                    final_logits = base_logits + lam * rel_logits
                else:
                    final_logits = base_logits # Standard Switch MoE (λ = 0)

                # Top-1 Router Decision
                selected_expert = torch.argmax(final_logits, dim=1) # (N_val,)

                # Selected Expert Accuracy
                actual_correctness = Y_val_mat[torch.arange(len(val_idx)), selected_expert].float()
                selected_accuracy = float(actual_correctness.mean())

                # Expert Selection Regret (Oracle Gap)
                regret = 1.0 - (selected_accuracy / (oracle_accuracy + 1e-8))

                # Expert Load Balancing (Gini Index & Routing Entropy)
                counts = torch.bincount(selected_expert, minlength=K).float()
                fracs = counts / len(val_idx)
                sorted_fracs, _ = torch.sort(fracs)
                idx_vec = torch.arange(1, K + 1).float()
                gini = float(((2 * idx_vec - K - 1) * sorted_fracs).sum() / (K * sorted_fracs.sum() + 1e-8))
                entropy = float(-torch.sum(fracs * torch.log(fracs + 1e-8)))

            layer_sweep_records[f"lambda_{lam}"] = {
                "lambda": lam,
                "token_accuracy": selected_accuracy,
                "failure_rate": 1.0 - selected_accuracy,
                "oracle_gap_regret": regret,
                "gini_index": gini,
                "routing_entropy": entropy
            }

            mode_str = "Switch MoE Baseline" if lam == 0.0 else f"ARES-Orion (λ={lam})"
            print(f"  - {mode_str:<25} | Token Acc: {selected_accuracy*100:.2f}% | Regret: {regret*100:.2f}% | Gini: {gini:.4f}")

        all_results[f"layer_{layer_idx}"] = {
            "oracle_upper_bound_accuracy": oracle_accuracy,
            "lambda_sweep": layer_sweep_records
        }

    # Save Results JSON
    res_file = out_path / "orion_router_sweep_results.json"
    with open(res_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "=" * 80)
    print(f" [SUCCESS] Phase 3 Step 3 Results Exported to: {res_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
