# ARES-Base/scripts/train_probe.py
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
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

# --- Expected Calibration Error (ECE) ---
def compute_ece(y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10) -> float:
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
    if len(np.unique(y_true)) < 2:
        auroc, auprc = 0.5, 0.5
    else:
        auroc = roc_auc_score(y_true, y_pred_prob)
        auprc = average_precision_score(y_true, y_pred_prob)
        
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    acc = accuracy_score(y_true, y_pred_bin)
    ece = compute_ece(y_true, y_pred_prob)
    brier = np.mean((y_true - y_pred_prob) ** 2)
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "accuracy": acc,
        "ece": ece,
        "brier": brier
    }

def train_probe_model(
    model: nn.Module, 
    X_train: torch.Tensor, 
    Y_train: torch.Tensor, 
    device: str,
    epochs: int = 3,
    batch_size: int = 512,
    lr: float = 1e-3
) -> None:
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

def evaluate_probe_model(model: nn.Module, X: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(X.to(device)).cpu().numpy()
    return preds

# --- Hewitt-Liang Control Task Mapping ---
def construct_control_labels(token_ids: np.ndarray, y_real: np.ndarray, seed: int) -> np.ndarray:
    """
    Constructs a control task by mapping each token ID (vocabulary word) 
    to a random binary label, preserving the overall label distribution.
    """
    np.random.seed(seed)
    unique_tokens = np.unique(token_ids)
    
    # Probability of label 1 in the true task
    p_one = np.mean(y_real)
    
    # Assign a fixed random label to each unique token
    token_to_label = {
        token: np.random.choice([0, 1], p=[1 - p_one, p_one]) for token in unique_tokens
    }
    
    # Construct the control label array
    y_control = np.array([token_to_label[token] for token in token_ids])
    return y_control

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate uncertainty heuristics and train reliability probes")
    parser.add_argument("--data-dir", type=str, default="data/probe_data", help="Directory where collected data is saved")
    parser.add_argument("--output-dir", type=str, default="experiments/probe_results", help="Directory to save outputs")
    parser.add_argument("--id-dataset", type=str, default="tinystories", help="In-distribution dataset folder name")
    parser.add_argument("--ood-datasets", type=str, default="wikitext,openwebtext", help="Comma-separated OOD dataset folder names")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs for neural probes")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", help="Layers to train probes on")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of random seeds for replication")
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load ID Dataset (e.g. TinyStories)
    id_dir = data_path / args.id_dataset
    if not id_dir.exists():
        raise FileNotFoundError(f"In-distribution data directory '{id_dir}' does not exist. Run collection script first.")
        
    print(f"[1/5] Loading in-distribution '{args.id_dataset}'...")
    Y_id_correct = torch.load(id_dir / "labels.pt").numpy()
    token_ids_id = torch.load(id_dir / "token_ids.pt").numpy()
    heuristics_id = torch.load(id_dir / "heuristics.pt")
    
    # Failure label is 1, success is 0
    Y_id_failure = 1 - Y_id_correct
    
    # Prepare OOD Datasets (if they exist)
    ood_names = [x.strip() for x in args.ood_datasets.split(",")]
    ood_data = {}
    for name in ood_names:
        ood_dir = data_path / name
        if ood_dir.exists():
            print(f" Found OOD dataset directory '{name}'. Preparing zero-shot evaluation.")
            Y_ood_correct = torch.load(ood_dir / "labels.pt").numpy()
            heuristics_ood = torch.load(ood_dir / "heuristics.pt")
            ood_data[name] = {
                "Y_correct": Y_ood_correct,
                "Y_failure": 1 - Y_ood_correct,
                "heuristics": heuristics_ood,
                "dir": ood_dir
            }

    # 2. Evaluate Conventional Heuristics on ID Split
    print("\n[2/5] Evaluating Conventional Heuristics (In-Distribution Validation)...")
    indices = np.arange(len(Y_id_failure))
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42) # fixed eval split for heuristics
    
    y_val_failure = Y_id_failure[val_idx]
    max_prob_val = heuristics_id["max_prob"].numpy()[val_idx]
    entropy_val = heuristics_id["entropy"].numpy()[val_idx]
    margin_val = heuristics_id["margin"].numpy()[val_idx]
    
    heuristics_results = {
        "Max Probability": evaluate_metrics(y_val_failure, 1.0 - max_prob_val),
        "Predictive Entropy": evaluate_metrics(y_val_failure, entropy_val / max(1.0, entropy_val.max())),
        "Prob Margin": evaluate_metrics(y_val_failure, 1.0 - margin_val)
    }

    # 3. Layer Probing Sweep with Replication Seeds
    print(f"\n[3/5] Starting Layer Probing Sweep ({args.num_seeds} replication seeds)...")
    layers = [int(x.strip()) for x in args.layers.split(",")]
    seeds = [42 + i for i in range(args.num_seeds)]
    
    # Dictionary to store accumulated scores for stats reporting
    # Structure: layer_name -> metric -> list of values
    performance_records = {}

    for layer in layers:
        x_file = id_dir / f"X_layer_{layer}.pt"
        if not x_file.exists():
            print(f" [Warning] Layer {layer} activation file {x_file} not found. Skipping.")
            continue
            
        print(f" Processing Layer {layer:02d}...")
        X = torch.load(x_file)
        
        # Load OOD Layer activations (if available)
        ood_layer_X = {}
        for name, data in ood_data.items():
            ood_file = data["dir"] / f"X_layer_{layer}.pt"
            if ood_file.exists():
                ood_layer_X[name] = torch.load(ood_file)
        
        # Loop over replication seeds
        for seed in seeds:
            # ID Split (80/20)
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train_fail, y_val_fail = Y_id_failure[train_idx], Y_id_failure[val_idx]
            
            # Hewitt-Liang Control Labels
            y_train_control = construct_control_labels(token_ids_id[train_idx], y_train_fail, seed=seed)
            y_val_control = construct_control_labels(token_ids_id[val_idx], y_val_fail, seed=seed)
            
            # Setup architectures
            models = {
                "Linear": LinearProbe(input_dim=X.shape[1]),
                "MLP": MLPProbe(input_dim=X.shape[1])
            }
            
            for m_name, probe in models.items():
                # A. Train Probe on Real Task
                train_probe_model(probe, X_train, torch.tensor(y_train_fail), args.device, epochs=args.epochs, batch_size=args.batch_size)
                
                # Evaluate ID Validation
                preds_val = evaluate_probe_model(probe, X_val, args.device)
                metrics_id = evaluate_metrics(y_val_fail, preds_val)
                
                # B. Zero-Shot OOD Generalization
                ood_metrics_eval = {}
                for ood_name, ood_dict in ood_data.items():
                    if ood_name in ood_layer_X:
                        # No retraining of the probe! Evaluating frozen on OOD features
                        preds_ood = evaluate_probe_model(probe, ood_layer_X[ood_name], args.device)
                        metrics_ood = evaluate_metrics(ood_dict["Y_failure"], preds_ood)
                        ood_metrics_eval[ood_name] = metrics_ood
                
                # C. Hewitt-Liang Control Task
                control_probe = LinearProbe(input_dim=X.shape[1]) if m_name == "Linear" else MLPProbe(input_dim=X.shape[1])
                train_probe_model(control_probe, X_train, torch.tensor(y_train_control), args.device, epochs=args.epochs, batch_size=args.batch_size)
                preds_control = evaluate_probe_model(control_probe, X_val, args.device)
                metrics_control = evaluate_metrics(y_val_control, preds_control)
                
                # Calculate Selectivity
                selectivity = metrics_id["accuracy"] - metrics_control["accuracy"]
                
                # Record metrics
                rec_keys = [("ID", metrics_id)]
                for ood_name, m_dict in ood_metrics_eval.items():
                    rec_keys.append((f"OOD_{ood_name}", m_dict))
                rec_keys.append(("Control", metrics_control))
                
                for key_prefix, metrics_dict in rec_keys:
                    rec_name = f"Layer_{layer:02d}_{m_name}_{key_prefix}"
                    if rec_name not in performance_records:
                        performance_records[rec_name] = {m: [] for m in ["auroc", "auprc", "ece", "accuracy", "brier"]}
                        performance_records[rec_name]["selectivity"] = []
                    
                    for m in ["auroc", "auprc", "ece", "accuracy", "brier"]:
                        performance_records[rec_name][m].append(metrics_dict[m])
                    performance_records[rec_name]["selectivity"].append(selectivity)

    # 4. Display Stats-Backed Metric Summaries
    print("\n" + "=" * 110)
    print(f" {'Method/Layer':<30} | {'AUROC':<15} | {'AUPRC':<15} | {'ECE':<15} | {'Brier':<15} | {'Selectivity':<12}")
    print("=" * 110)
    
    # Print heuristics
    for name, r in heuristics_results.items():
        print(f" {name:<30} | {r['auroc']:.4f}        | {r['auprc']:.4f}        | {r['ece']:.4f}        | {r['brier']:.4f}        | N/A")
    print("-" * 110)
    
    # Print Probes with mean ± std
    summary_data = {}
    for layer in layers:
        for m_name in ["Linear", "MLP"]:
            # Print ID validation
            id_key = f"Layer_{layer:02d}_{m_name}_ID"
            if id_key in performance_records:
                r_id = performance_records[id_key]
                print(f" L{layer:02d} {m_name:<25} | "
                      f"{np.mean(r_id['auroc']):.4f}±{np.std(r_id['auroc']):.3f} | "
                      f"{np.mean(r_id['auprc']):.4f}±{np.std(r_id['auprc']):.3f} | "
                      f"{np.mean(r_id['ece']):.4f}±{np.std(r_id['ece']):.3f} | "
                      f"{np.mean(r_id['brier']):.4f}±{np.std(r_id['brier']):.3f} | "
                      f"{np.mean(r_id['selectivity'])*100:.1f}%")
                
                # Print OOD validation zero-shot scores
                for ood_name in ood_data:
                    ood_key = f"Layer_{layer:02d}_{m_name}_OOD_{ood_name}"
                    if ood_key in performance_records:
                        r_ood = performance_records[ood_key]
                        print(f"   -> OOD [{ood_name:<16}] | "
                              f"{np.mean(r_ood['auroc']):.4f}±{np.std(r_ood['auroc']):.3f} | "
                              f"{np.mean(r_ood['auprc']):.4f}±{np.std(r_ood['auprc']):.3f} | "
                              f"{np.mean(r_ood['ece']):.4f}±{np.std(r_ood['ece']):.3f} | "
                              f"{np.mean(r_ood['brier']):.4f}±{np.std(r_ood['brier']):.3f} | "
                              f"N/A")
                
                # Print Control Gating (Selectivity Check)
                control_key = f"Layer_{layer:02d}_{m_name}_Control"
                if control_key in performance_records:
                    r_ctrl = performance_records[control_key]
                    print(f"   -> Control [Hewitt-Liang]   | "
                          f"{np.mean(r_ctrl['auroc']):.4f}±{np.std(r_ctrl['auroc']):.3f} | "
                          f"{np.mean(r_ctrl['auprc']):.4f}±{np.std(r_ctrl['auprc']):.3f} | "
                          f"{np.mean(r_ctrl['ece']):.4f}±{np.std(r_ctrl['ece']):.3f} | "
                          f"{np.mean(r_ctrl['brier']):.4f}±{np.std(r_ctrl['brier']):.3f} | "
                          f"N/A")
                print("-" * 110)

    # 5. Export results to JSON
    export_dict = {}
    for k, v in performance_records.items():
        export_dict[k] = {m: [float(val) for val in v[m]] for m in v}
    
    # Save results file
    res_file = out_path / "phase_2_5_metrics.json"
    with open(res_file, "w") as f:
        json.dump(export_dict, f, indent=4)
    print(f"\n[SUCCESS] Phase 2.5 Metrics exported to: {res_file}")

if __name__ == "__main__":
    main()