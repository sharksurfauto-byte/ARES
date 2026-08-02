# ARES-Base/scripts/train_probe.py
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError("Please install scikit-learn: pip install scikit-learn")

sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Linear Probe Definition ---
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)

# --- MLP Probe Definition ---
class MLPProbe(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)

# --- Calibration Evaluation (ECE) ---
def compute_ece(y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) for binary classification."""
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

# --- Selective Prediction Curve Evaluator ---
def compute_selective_accuracy(y_base_correct: np.ndarray, y_pred_failure_prob: np.ndarray, coverages: list) -> Dict[float, float]:
    """Computes accuracy on retained predictions (coverage) sorting by predicted failure probability."""
    sorted_indices = np.argsort(y_pred_failure_prob)
    total_tokens = len(y_base_correct)
    
    results = {}
    for coverage in coverages:
        n_retain = int(total_tokens * coverage)
        if n_retain == 0:
            results[coverage] = 1.0
            continue
            
        retained_indices = sorted_indices[:n_retain]
        retained_correctness = y_base_correct[retained_indices]
        results[coverage] = np.mean(retained_correctness)
    return results

def evaluate_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Computes standard classification, calibration, and likelihood metrics."""
    if len(np.unique(y_true)) < 2:
        auroc, auprc = 0.5, 0.5
    else:
        auroc = roc_auc_score(y_true, y_pred_prob)
        auprc = average_precision_score(y_true, y_pred_prob)
        
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    ece = compute_ece(y_true, y_pred_prob)
    brier = np.mean((y_true - y_pred_prob) ** 2)
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "ece": ece,
        "brier": brier
    }

def train_probe_model(
    model: nn.Module, 
    X_train: torch.Tensor, 
    Y_train: torch.Tensor, 
    X_val: torch.Tensor, 
    device: str,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3
) -> np.ndarray:
    """Trains a probe using BCELoss and Adam."""
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
            
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val.to(device)).cpu().numpy()
    return val_preds

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate uncertainty heuristics and train reliability probes")
    parser.add_argument("--data-dir", type=str, default="data/probe_data", help="Directory where collected data is saved")
    parser.add_argument("--output-dir", type=str, default="experiments/probe_results", help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for neural probes")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--layers", type=str, default="3,6,9,11", help="Comma-separated layers to train probes on")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load labels and heuristics
    print("[1/4] Loading labels and heuristics...")
    Y_correct = torch.load(data_path / "labels.pt").numpy()
    heuristics = torch.load(data_path / "heuristics.pt")
    
    # Target: 1 represents base model FAILURE, 0 represents SUCCESS
    Y_failure = 1 - Y_correct
    
    # Train/Val split index mapping (80% train, 20% validation)
    indices = np.arange(len(Y_failure))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    y_val_failure = Y_failure[val_idx]
    y_val_correct = Y_correct[val_idx]

    # 2. Evaluate Conventional Heuristics
    print("\n[2/4] Evaluating Conventional Heuristics...")
    heuristic_results = {}
    
    # Max probability (lower confidence = higher failure chance)
    max_prob_val = heuristics["max_prob"].numpy()[val_idx]
    heuristic_results["Max Probability"] = evaluate_metrics(y_val_failure, 1.0 - max_prob_val)
    
    # Entropy (higher entropy = higher failure chance)
    entropy_val = heuristics["entropy"].numpy()[val_idx]
    heuristic_results["Predictive Entropy"] = evaluate_metrics(y_val_failure, entropy_val / max(1.0, entropy_val.max()))
    
    # Probability Margin (lower margin = higher failure chance)
    margin_val = heuristics["margin"].numpy()[val_idx]
    heuristic_results["Prob Margin"] = evaluate_metrics(y_val_failure, 1.0 - margin_val)

    # 3. Train Hidden Representation Probes
    print("\n[3/4] Training Probes on Hidden States...")
    layers = [int(x.strip()) for x in args.layers.split(",")]
    probe_results = {}
    
    val_predictions = {
        "Max Probability": 1.0 - max_prob_val,
        "Predictive Entropy": entropy_val
    }

    for layer in layers:
        print(f"Processing Layer {layer}...")
        x_file = data_path / f"X_layer_{layer}.pt"
        if not x_file.exists():
            print(f"  [Warning] File {x_file} not found. Skipping layer.")
            continue
            
        X = torch.load(x_file)
        
        # Split features
        X_train, X_val = X[train_idx], X[val_idx]
        
        # Train Linear Probe
        print(f"  - Fitting Linear Probe...")
        lin_probe = LinearProbe(input_dim=X.shape[1])
        lin_preds = train_probe_model(
            lin_probe, X_train, torch.tensor(Y_failure[train_idx]), 
            X_val, args.device, epochs=args.epochs, batch_size=args.batch_size
        )
        probe_results[f"Layer {layer} Linear"] = evaluate_metrics(y_val_failure, lin_preds)
        val_predictions[f"Layer {layer} Linear"] = lin_preds

        # Train MLP Probe
        print(f"  - Fitting MLP Probe...")
        mlp_probe = MLPProbe(input_dim=X.shape[1])
        mlp_preds = train_probe_model(
            mlp_probe, X_train, torch.tensor(Y_failure[train_idx]), 
            X_val, args.device, epochs=args.epochs, batch_size=args.batch_size
        )
        probe_results[f"Layer {layer} MLP"] = evaluate_metrics(y_val_failure, mlp_preds)
        val_predictions[f"Layer {layer} MLP"] = mlp_preds

    # 4. Display Metric Summaries
    print("\n" + "=" * 80)
    print(f" {'Method':<25} | {'AUROC':<8} | {'AUPRC':<8} | {'F1':<8} | {'ECE':<8} | {'Brier':<8}")
    print("=" * 80)
    
    # Print heuristics
    for name, r in heuristic_results.items():
        print(f" {name:<25} | {r['auroc']:.4f}   | {r['auprc']:.4f}   | {r['f1']:.4f}   | {r['ece']:.4f}   | {r['brier']:.4f}")
    print("-" * 80)
    
    # Print neural probes
    for name, r in probe_results.items():
        print(f" {name:<25} | {r['auroc']:.4f}   | {r['auprc']:.4f}   | {r['f1']:.4f}   | {r['ece']:.4f}   | {r['brier']:.4f}")
    print("=" * 80)

    # 5. Compute Selective Prediction Accuracy (Accuracy vs. Coverage)
    print("\n[4/4] Computing Selective Prediction Metrics...")
    coverages = [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]
    
    print("\nAccuracy on Retained Predictions (by Coverage):")
    print(f" {'Method':<25} | " + " | ".join([f"{int(c*100):>3}%" for c in coverages]))
    print("-" * 80)
    
    for name, preds_prob in val_predictions.items():
        sel_acc = compute_selective_accuracy(y_val_correct, preds_prob, coverages)
        acc_str = " | ".join([f"{sel_acc[c]*100:>5.2f}%" for c in coverages])
        print(f" {name:<25} | {acc_str}")
    print("=" * 80)

if __name__ == "__main__":
    main()
