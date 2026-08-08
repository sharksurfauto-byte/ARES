"""
ARES-Orion MoE Module (Phase 3)
================================
This module implements:
1. ARESExpert: Individual FFN Expert sub-network (H -> D_ff -> H).
2. ARESMoELayer: Sparse Mixture-of-Experts layer with 3-Way Router Ablation:
   - Mode 0 ('baseline_switch'): Standard dot-product gating (Switch MoE style).
   - Mode 1 ('semantic_centroid'): Cosine similarity with domain centroids.
   - Mode 2 ('ares_orion'): Calibrated Reliability Probe Gating (W_r h_t + lambda * logit(r_e)).
3. Multi-GPU Utility: Automatic Dual Kaggle T4 GPU utilization via DataParallel / Device placement.
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

#1. Single FFN Expert Sub-network
class ARESExpert(nn.Module):
    # a single FFN expert block
    # x (B,T,H) -> Linear (H, D_ff) -> GELU -> Linear(D_ff, H) -> Dropout ->Output (B,T,H)

    def __init__(self, hidden_size:int=768, intermediate_size:int=3072, dropout:float=0.1):
        super().__init__()
        self.fc1=nn.Linear(hidden_size, intermediate_size)
        self.act=nn.GELU()
        self.fc2=nn.Linear(intermediate_size, hidden_size)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))

#2. Sparse MOE layer
class ARESMoELayer(nn.Module):
    """
    Its a sparse MoE layer containing K paraller FFN experts and a router gating function

    suppports 3 controlled gating modes
    1. baseline_switch : standard switch transformer dot-prod scaling
    2. semantic_centroid : domain centroid similarity gating
    3. ares_orion : probe-calibrated reliability gating
    """
    def __init__(
        self, 
        num_experts: int = 4, 
        top_k: int = 1, 
        hidden_size: int = 768, 
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        lambda_rel: float = 1.0,
        router_mode: str = "baseline_switch"
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.lambda_rel = lambda_rel
        self.router_mode = router_mode

        #expert pool (K parallel experts)
        self.experts=nn.ModuleList([
            ARESExpert(hidden_size, intermediate_size, dropout)
            for _ in range(num_experts)
        ])

        #router gating proj
        self.gate=nn.Linear(hidden_size, num_experts, bias=False)

        #domain centroids for mode 1 (semantic affinity)
        self.register_buffer("expert_centroids", torch.zeros(num_experts, hidden_size))

        #load balancing loss coeff
        self.aux_loss_coef=0.01

    def compute_auxiliary_loss(self, router_probes:torch.Tensor, expert_indices:torch.Tensor)->torch.Tensor:
        """
        Computes Switch Transformer load-balancing auxiliary loss.
        L_aux = K * sum_{e=1}^K (f_e * P_e)
        Where:
            f_e = fraction of tokens routed to expert e
            P_e = mean router probability assigned to expert e
        """
        N_tokens=router_probes.shape[0]
        if N_tokens==0:
            return torch.Tensor(0.0, device=router_probes.device)

        #fraction of tokens assigned to each expert (f_e)
        one_hot=F.one_hot(expert_indices, num_classes=self.num_experts).float()
        f_e=one_hot.mean(dim=0)

        #mean gating prob assigned to each expert (P_e)
        P_e=router_probes.mean(dim=0)

        aux_loss=self.num_experts*torch.sum(f_e * P_e) * self.aux_loss_coef
        return aux_loss

    def forward(self, hidden_states:torch.Tensor, probe_reliabilities:Optional[torch.Tensor] =None)->Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Forward pass for the MoE layer.
        
        Args:
            hidden_states: Tensor of shape (B, T, H)
            probe_reliabilities: Optional Tensor of shape (B, T, K) containing predicted 
                                 correctness probabilities r_e(t) in [0, 1] for each expert.
        
        Returns:
            output_states: Tensor of shape (B, T, H)
            aux_loss: Scalar auxiliary load-balancing loss
            metrics_dict: Dictionary of diagnostic metrics (routing entropy, Gini coefficient)
        """
        B,T,H=hidden_states.shape
        flat_hidden=hidden_states.reshape(-1,H) #(N,H) where N=B*T
        N=flat_hidden.shape[0]

        # 1. Compute Base Gating Logits
        if self.router_mode == "semantic_centroid":
            # Mode 1: Cosine similarity with expert centroids
            norm_hidden = F.normalize(flat_hidden, p=2, dim=-1)
            norm_centroids = F.normalize(self.expert_centroids, p=2, dim=-1)
            base_logits = torch.matmul(norm_hidden, norm_centroids.T) # (N, K)
        else:
            # Mode 0 & Mode 2: Linear projection W_r * h_t
            base_logits = self.gate(flat_hidden) # (N, K)

        # 2. Apply Probe Reliability Augmentation (ARES-Orion Mode 2)
        if self.router_mode=="ares_orion" and probe_reliabilities is not None:
            flat_rel = probe_reliabilities.reshape(-1,self.num_experts)#(N,K)
            eps=1e-6
            #log odds reliability logit : log(r_e / (1-r_e))
            rel_logits=torch.log((flat_rel+eps)/(1.0-flat_rel+eps))

            #combine semantic logit + lambda*reliability logit
            router_logits=base_logits+self.lambda_rel*rel_logits
        else:
            router_logits=base_logits

        # Router Softmax Probabilities
        router_probs = F.softmax(router_logits, dim=-1) # (N, K)

        #3. Top-k expert selection
        topk_weights, topk_indices=torch.topk(router_probs,self.top_k, dim=-1) # N,topk

        if self.top_k==1:
            topk_weights=topk_weights.squeeze(-1) # (N,)
            selected_experts=topk_indices.squeeze(-1) #(N,)
        else:
            #normalize topk weights so they sum up to 1
            topk_weights=topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            selected_experts=topk_indices[:,0] #primary expoert for aux loss

        # 4. Sparse Routing & Expert Computation
        output_flat = torch.zeros_like(flat_hidden)

        if self.top_k == 1:
            # Efficient grouped execution per expert
            for e_idx in range(self.num_experts):
                mask = (selected_experts == e_idx)
                if mask.any():
                    expert_tokens = flat_hidden[mask] # (N_e, H)
                    expert_out = self.experts[e_idx](expert_tokens) # (N_e, H)
        
                    # Weight by routing probability
                    gating_scale = topk_weights[mask].unsqueeze(-1) # (N_e, 1)
                    output_flat[mask] = expert_out * gating_scale

        else:
            # Multi-expert top-k accumulation
            for k in range(self.top_k):
                k_expert_idx = topk_indices[:, k]
                k_weight = topk_weights[:, k].unsqueeze(-1)
                
                for e_idx in range(self.num_experts):
                    mask = (k_expert_idx == e_idx)
                    if mask.any():
                        expert_tokens = flat_hidden[mask]
                        expert_out = self.experts[e_idx](expert_tokens)
                        output_flat[mask] += expert_out * k_weight[mask]

        # Reshape output back to (B, T, H)
        output_states = output_flat.reshape(B, T, H)

        # 5. Compute Auxiliary Loss & Diagnostic Metrics
        aux_loss = self.compute_auxiliary_loss(router_probs, selected_experts)

        # Diagnostic Metrics (Gini Coefficient & Routing Entropy)
        with torch.no_grad():
            expert_counts = torch.bincount(selected_experts, minlength=self.num_experts).float()
            expert_fractions = expert_counts / N
        
            # Gini Coefficient (0 = perfect balance, 1 = total collapse to 1 expert)
            sorted_fracs, _ = torch.sort(expert_fractions)
            index = torch.arange(1, self.num_experts + 1, device=hidden_states.device).float()
            gini = ((2 * index - self.num_experts - 1) * sorted_fracs).sum() / (self.num_experts * sorted_fracs.sum() + 1e-8)
        
            # Routing Entropy
            entropy = -torch.sum(expert_fractions * torch.log(expert_fractions + 1e-8))
        
        metrics_dict = {
            "gini_index": float(gini.cpu()),
            "routing_entropy": float(entropy.cpu()),
            "aux_loss": float(aux_loss.cpu())
        }
        
        return output_states, aux_loss, metrics_dict

#3. Dual Kaggle T4 Multi-GPU Setup Utility
def prepare_model_for_multi_gpu(model: nn.Module, device_str: str = "cuda") -> Tuple[nn.Module, str]:
    """
    Detects available GPUs and configures multi-GPU execution.
    On Kaggle T4 x 2, automatically uses torch.nn.DataParallel across both GPUs.
    
    Returns:
        model: Model wrapped in DataParallel if 2 GPUs present
        primary_device: Main device string ('cuda:0' or 'cpu')
    """
    if device_str.startswith("cuda") and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[Multi-GPU Engine] Detected {gpu_count} CUDA Device(s):")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  - GPU {i}: {gpu_name} ({vram_gb:.2f} GB VRAM)")

        if gpu_count > 1:
            print(f"\n--> Activating Dual-GPU DataParallel across all {gpu_count} GPUs!")
            model = model.to("cuda:0")
            model = nn.DataParallel(model)
            primary_device = "cuda:0"
        else:
            print("\n--> Single GPU detected. Moving model to CUDA.")
            model = model.to("cuda:0")
            primary_device = "cuda:0"
    else:
        print("[Multi-GPU Engine] No CUDA available. Running on CPU.")
        model = model.to("cpu")
        primary_device = "cpu"

    return model, primary_device

# 4. Verification Test Execution
if __name__ == "__main__":
    print("=" * 80)
    print(" ARES-Orion Step 1 Verification Test (Sparse MoE Layer & Dual-GPU Utilities)")
    print("=" * 80)

    # 1. Instantiate Sparse MoE Layer
    moe_layer = ARESMoELayer(
        num_experts=4,
        top_k=1,
        hidden_size=768,
        intermediate_size=3072,
        lambda_rel=1.0,
        router_mode="ares_orion"
    )

    # 2. Test Multi-GPU Wrapper
    moe_layer, device = prepare_model_for_multi_gpu(moe_layer)

    # 3. Create Synthetic Input Tensors
    B, T, H, K = 4, 128, 768, 4
    dummy_hidden = torch.randn(B, T, H).to(device)
    dummy_reliability = torch.sigmoid(torch.randn(B, T, K)).to(device) # Predicted r_e(t) in [0, 1]

    # 4. Run Forward Pass
    out_states, aux_loss, metrics = moe_layer(dummy_hidden, probe_reliabilities=dummy_reliability)

    print("\n[VERIFICATION RESULTS]:")
    print(f"  - Input Hidden Shape:       {dummy_hidden.shape}")
    print(f"  - Output Hidden Shape:      {out_states.shape}")
    print(f"  - Auxiliary Loss:           {aux_loss.item():.6f}")
    print(f"  - Routing Gini Index:       {metrics['gini_index']:.4f} (0=balanced, 1=collapsed)")
    print(f"  - Routing Entropy:          {metrics['routing_entropy']:.4f}")
    print("\n[SUCCESS] Step 1 Module Verification Complete!")
