# ARES-Base/scripts/train_probe.py
"""
ARES Probing Suite (GRM + LRM)

Trains and evaluates both:
1. GRM (Global Reliability Module): Prompt-level failure probes (R_global)
2. LRM (Local Reliability Module): Token-level failure probes (R_local)

Evaluates zero-shot OOD transfer (WikiText, OpenWebText), Hewitt-Liang Selectivity, 
and reports mean ± std across replication seeds.
"""

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
    np.random.seed(seed)
    unique_tokens = np.unique(token_ids)
    p_one = np.mean(y_real)
    token_to_label = {
        token: np.random.choice([0, 1], p=[1 - p_one, p_one]) for token in unique_tokens
    }
    return np.array([token_to_label[token] for token in token_ids])

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GRM and LRM probes on ID and OOD splits")
    parser.add_argument("--data-dir", type=str, default="data/probe_data", help="Directory where collected data is saved")
    parser.add_argument("--output-dir", type=str, default="experiments/probe_results", help="Directory to save outputs")
    parser.add_argument("--id-dataset", type=str, default="tinystories", help="In-distribution dataset folder name")
    parser.add_argument("--ood-datasets", type=str, default="wikitext,openwebtext", help="Comma-separated OOD dataset names")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
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
    
    id_dir = data_path / args.id_dataset
    if not id_dir.exists():
        raise FileNotFoundError(f"In-distribution data directory '{id_dir}' does not exist.")
        
    print(f"[1/4] Loading in-distribution '{args.id_dataset}' (GRM + LRM)...")
    
    # Load LRM (Token-Level)
    Y_lrm_correct = torch.load(id_dir / "labels_lrm.pt" if (id_dir / "labels_lrm.pt").exists() else id_dir / "labels.pt").numpy()
    token_ids_id = torch.load(id_dir / "token_ids.pt").numpy()
    heuristics_id = torch.load(id_dir / "heuristics.pt")
    Y_lrm_failure = 1 - Y_lrm_correct

    # Load GRM (Prompt-Level) if present
    has_grm = (id_dir / "labels_grm.pt").exists()
    if has_grm:
        Y_grm_correct = torch.load(id_dir / "labels_grm.pt").numpy()
        Y_grm_failure = 1 - Y_grm_correct

    # Load OOD Datasets
    ood_names = [x.strip() for x in args.ood_datasets.split(",")]
    ood_data = {}
    for name in ood_names:
        ood_dir = data_path / name
        if ood_dir.exists():
            print(f" Found OOD dataset directory '{name}'. Preparing zero-shot evaluation.")
            Y_lrm_ood_correct = torch.load(ood_dir / "labels_lrm.pt" if (ood_dir / "labels_lrm.pt").exists() else ood_dir / "labels.pt").numpy()
            has_grm_ood = (ood_dir / "labels_grm.pt").exists()
            Y_grm_ood_correct = torch.load(ood_dir / "labels_grm.pt").numpy() if has_grm_ood else None
            
            ood_data[name] = {
                "Y_lrm_failure": 1 - Y_lrm_ood_correct,
                "Y_grm_failure": (1 - Y_grm_ood_correct) if has_grm_ood else None,
                "dir": ood_dir
            }

    # Evaluate Heuristics on LRM
    indices_lrm = np.arange(len(Y_lrm_failure))
    _, val_idx = train_test_split(indices_lrm, test_size=0.2, random_state=42)
    y_val_fail = Y_lrm_failure[val_idx]
    
    max_prob_val = heuristics_id["max_prob"].numpy()[val_idx]
    entropy_val = heuristics_id["entropy"].numpy()[val_idx]
    
    heuristics_results = {
        "Max Probability (LRM)": evaluate_metrics(y_val_fail, 1.0 - max_prob_val),
        "Predictive Entropy (LRM)": evaluate_metrics(y_val_fail, entropy_val / max(1.0, entropy_val.max()))
    }

    print(f"\n[2/4] Layer Probing Sweep ({args.num_seeds} seeds)...")
    layers = [int(x.strip()) for x in args.layers.split(",")]
    seeds = [42 + i for i in range(args.num_seeds)]
    performance_records = {}

    for layer in layers:
        # Check files
        lrm_file = id_dir / f"X_lrm_layer_{layer}.pt" if (id_dir / f"X_lrm_layer_{layer}.pt").exists() else id_dir / f"X_layer_{layer}.pt"
        if not lrm_file.exists():
            continue
            
        print(f" Processing Layer {layer:02d}...")
        X_lrm = torch.load(lrm_file)
        X_grm = torch.load(id_dir / f"X_grm_layer_{layer}.pt") if has_grm and (id_dir / f"X_grm_layer_{layer}.pt").exists() else None

        # OOD Activations
        ood_lrm_X = {}
        ood_grm_X = {}
        for o_name, o_dict in ood_data.items():
            f_lrm = o_dict["dir"] / f"X_lrm_layer_{layer}.pt" if (o_dict["dir"] / f"X_lrm_layer_{layer}.pt").exists() else o_dict["dir"] / f"X_layer_{layer}.pt"
            if f_lrm.exists():
                ood_lrm_X[o_name] = torch.load(f_lrm)
            f_grm = o_dict["dir"] / f"X_grm_layer_{layer}.pt"
            if f_grm.exists():
                ood_grm_X[o_name] = torch.load(f_grm)

        for seed in seeds:
            # --- LRM (Token-Level Probing) ---
            tr_idx, val_idx = train_test_split(indices_lrm, test_size=0.2, random_state=seed)
            X_tr, X_va = X_lrm[tr_idx], X_lrm[val_idx]
            y_tr_fail, y_va_fail = Y_lrm_failure[tr_idx], Y_lrm_failure[val_idx]
            y_tr_ctrl = construct_control_labels(token_ids_id[tr_idx], y_tr_fail, seed=seed)
            y_va_ctrl = construct_control_labels(token_ids_id[val_idx], y_va_fail, seed=seed)

            for m_name in ["Linear", "MLP"]:
                probe = LinearProbe(X_lrm.shape[1]) if m_name == "Linear" else MLPProbe(X_lrm.shape[1])
                train_probe_model(probe, X_tr, torch.tensor(y_tr_fail), args.device, epochs=args.epochs, batch_size=args.batch_size)
                
                p_val = evaluate_probe_model(probe, X_va, args.device)
                m_id = evaluate_metrics(y_va_fail, p_val)

                # Control Probe
                ctrl_probe = LinearProbe(X_lrm.shape[1]) if m_name == "Linear" else MLPProbe(X_lrm.shape[1])
                train_probe_model(ctrl_probe, X_tr, torch.tensor(y_tr_ctrl), args.device, epochs=args.epochs, batch_size=args.batch_size)
                p_ctrl = evaluate_probe_model(ctrl_probe, X_va, args.device)
                m_ctrl = evaluate_metrics(y_va_ctrl, p_ctrl)
                selectivity = m_id["accuracy"] - m_ctrl["accuracy"]

                rec_key = f"L{layer:02d}_{m_name}_LRM_ID"
                if rec_key not in performance_records:
                    performance_records[rec_key] = {met: [] for met in ["auroc", "auprc", "ece", "brier"]}
                    performance_records[rec_key]["selectivity"] = []
                for met in ["auroc", "auprc", "ece", "brier"]:
                    performance_records[rec_key][met].append(m_id[met])
                performance_records[rec_key]["selectivity"].append(selectivity)

                # OOD LRM Eval
                for o_name in ood_lrm_X:
                    p_ood = evaluate_probe_model(probe, ood_lrm_X[o_name], args.device)
                    m_ood = evaluate_metrics(ood_data[o_name]["Y_lrm_failure"], p_ood)
                    o_key = f"L{layer:02d}_{m_name}_LRM_OOD_{o_name}"
                    if o_key not in performance_records:
                        performance_records[o_key] = {met: [] for met in ["auroc", "auprc", "ece", "brier"]}
                    for met in ["auroc", "auprc", "ece", "brier"]:
                        performance_records[o_key][met].append(m_ood[met])

            # --- GRM (Prompt-Level Probing) ---
            if X_grm is not None:
                indices_grm = np.arange(len(Y_grm_failure))
                tr_g_idx, va_g_idx = train_test_split(indices_grm, test_size=0.2, random_state=seed)
                X_tr_g, X_va_g = X_grm[tr_g_idx], X_grm[va_g_idx]
                y_tr_g_fail, y_va_g_fail = Y_grm_failure[tr_g_idx], Y_grm_failure[va_g_idx]

                for m_name in ["Linear", "MLP"]:
                    g_probe = LinearProbe(X_grm.shape[1]) if m_name == "Linear" else MLPProbe(X_grm.shape[1])
                    train_probe_model(g_probe, X_tr_g, torch.tensor(y_tr_g_fail), args.device, epochs=args.epochs, batch_size=args.batch_size)
                    
                    p_va_g = evaluate_probe_model(g_probe, X_va_g, args.device)
                    m_g_id = evaluate_metrics(y_va_g_fail, p_va_g)
                    
                    rec_key = f"L{layer:02d}_{m_name}_GRM_ID"
                    if rec_key not in performance_records:
                        performance_records[rec_key] = {met: [] for met in ["auroc", "auprc", "ece", "brier"]}
                    for met in ["auroc", "auprc", "ece", "brier"]:
                        performance_records[rec_key][met].append(m_g_id[met])

                    # OOD GRM Eval
                    for o_name in ood_grm_X:
                        if ood_data[o_name]["Y_grm_failure"] is not None:
                            p_g_ood = evaluate_probe_model(g_probe, ood_grm_X[o_name], args.device)
                            m_g_ood = evaluate_metrics(ood_data[o_name]["Y_grm_failure"], p_g_ood)
                            og_key = f"L{layer:02d}_{m_name}_GRM_OOD_{o_name}"
                            if og_key not in performance_records:
                                performance_records[og_key] = {met: [] for met in ["auroc", "auprc", "ece", "brier"]}
                            for met in ["auroc", "auprc", "ece", "brier"]:
                                performance_records[og_key][met].append(m_g_ood[met])

    # Display Metrics Table
    print("\n" + "=" * 115)
    print(f" {'Method / Granularity':<32} | {'AUROC':<15} | {'AUPRC':<15} | {'ECE':<15} | {'Brier':<15} | {'Selectivity':<10}")
    print("=" * 115)
    
    for name, r in heuristics_results.items():
        print(f" {name:<32} | {r['auroc']:.4f}        | {r['auprc']:.4f}        | {r['ece']:.4f}        | {r['brier']:.4f}        | N/A")
    print("-" * 115)

    for layer in layers:
        for m_name in ["Linear", "MLP"]:
            # LRM
            k_lrm = f"L{layer:02d}_{m_name}_LRM_ID"
            if k_lrm in performance_records:
                r = performance_records[k_lrm]
                sel_str = f"{np.mean(r['selectivity'])*100:.1f}%" if "selectivity" in r else "N/A"
                print(f" L{layer:02d} {m_name} LRM (Token ID)          | "
                      f"{np.mean(r['auroc']):.4f}±{np.std(r['auroc']):.3f} | "
                      f"{np.mean(r['auprc']):.4f}±{np.std(r['auprc']):.3f} | "
                      f"{np.mean(r['ece']):.4f}±{np.std(r['ece']):.3f} | "
                      f"{np.mean(r['brier']):.4f}±{np.std(r['brier']):.3f} | "
                      f"{sel_str}")
                
                # OOD LRM
                for o_name in ood_names:
                    k_ood = f"L{layer:02d}_{m_name}_LRM_OOD_{o_name}"
                    if k_ood in performance_records:
                        ro = performance_records[k_ood]
                        print(f"   -> LRM OOD [{o_name:<15}] | "
                              f"{np.mean(ro['auroc']):.4f}±{np.std(ro['auroc']):.3f} | "
                              f"{np.mean(ro['auprc']):.4f}±{np.std(ro['auprc']):.3f} | "
                              f"{np.mean(ro['ece']):.4f}±{np.std(ro['ece']):.3f} | "
                              f"{np.mean(ro['brier']):.4f}±{np.std(ro['brier']):.3f} | "
                              f"N/A")

            # GRM
            k_grm = f"L{layer:02d}_{m_name}_GRM_ID"
            if k_grm in performance_records:
                rg = performance_records[k_grm]
                print(f" L{layer:02d} {m_name} GRM (Prompt ID)         | "
                      f"{np.mean(rg['auroc']):.4f}±{np.std(rg['auroc']):.3f} | "
                      f"{np.mean(rg['auprc']):.4f}±{np.std(rg['auprc']):.3f} | "
                      f"{np.mean(rg['ece']):.4f}±{np.std(rg['ece']):.3f} | "
                      f"{np.mean(rg['brier']):.4f}±{np.std(rg['brier']):.3f} | "
                      f"N/A")
            print("-" * 115)

    # Save to JSON
    export_dict = {}
    for k, v in performance_records.items():
        export_dict[k] = {m: [float(val) for val in v[m]] for m in v}
    res_file = out_path / "phase_2_5_grm_lrm_metrics.json"
    with open(res_file, "w") as f:
        json.dump(export_dict, f, indent=4)
    print(f"\n[SUCCESS] Phase 2.5 Metrics exported to: {res_file}")

if __name__ == "__main__":
    main()