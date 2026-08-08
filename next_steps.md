# ARES-Orion (Phase 3): Locked Technical Specification & Execution Roadmap

This document outlines the locked, implementation-ready specification for **ARES-Orion (Phase 3)** incorporating all 12 scientific refinements.

---

## 1. Central Research Question

> **Central Hypothesis**:
> *"Can calibrated predictions of expert-specific token reliability improve sparse MoE routing beyond conventional representation-based gating, while preserving computational efficiency and load balance?"*

```text
                        ┌──────────────────────────────┐
                        │         User Prompt          │
                        └──────────────┬───────────────┘
                                       │
  =====================================▼=====================================
  GLOBAL RELIABILITY MODULE (GRM)  [Macro Policy Layer]
  ===========================================================================
                                       │
                         Global Reliability Score (R_global)
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
            High Reliability (R_global ≥ τ)   Low Reliability (R_global < τ)
            (Fast Path / Base Compute)        (Activate Orion MoE Pipeline)
                                                        │
  ======================================================▼====================
  LOCAL RELIABILITY MODULE (LRM / Orion) [Micro Token Routing]
  ===========================================================================
                                                        │
                                                 Decoding Step t
                                                        │
                                            Token Hidden State h_t
                                                        │
                         ┌──────────────────────────────┴──────────────────────────────┐
                         ▼                                                             ▼
             [ Baseline Router W_r ]                                       [ LRM Reliability Probing ]
           (Feature-Space Alignment)                                       h_t ──► Probe 1 ──► r_1 = P(correct | E_1)
                         │                                                 h_t ──► Probe 2 ──► r_2 = P(correct | E_2)
                         │                                                 h_t ──► Probe 3 ──► r_3 = P(correct | E_3)
                         │                                                 h_t ──► Probe 4 ──► r_4 = P(correct | E_4)
                         │                                                             │
                         └──────────────────────────────┬──────────────────────────────┘
                                                        ▼
                                           [ Calibrated Gating Score ]
                                     S_e(t) = W_{r,e} h_t + λ log(r_e / (1 - r_e))
                                                        │
                                                        ▼
                                              Selected Top-1 Expert
                                                        │
                                                    Next Token
```

---

## 2. Experimental Controls & Causal Design

To ensure zero reviewer objections regarding capacity scaling or expert quality variance:

1.  **Identical Expert Initialization**: All $K=4$ experts are initialized from the same frozen ARES-Base checkpoint.
2.  **Identical Training Budget**: Each expert is trained for the exact same number of tokens, steps, optimizer parameters, and learning rate schedule on its domain partition subset.
3.  **Frozen Expert Pool**: During router evaluation, **the expert pool is strictly frozen**. Performance differences between routers are 100% causally attributable to **gating intelligence**.

---

## 3. The 3-Way Controlled Router Ablation Suite

```text
                              [ Frozen Expert Pool (K=4) ]
                                           │
             ┌─────────────────────────────┼─────────────────────────────┐
             ▼                             ▼                             ▼
       [ Baseline A ]                [ Baseline B ]                 [ Proposed ]
     Standard Switch MoE          Semantic Affinity MoE        ARES-Orion Reliability MoE
   (Dot-product gating)           (Domain centroid gating)     (Calibrated probe gating)
```

1.  **Baseline A (Standard Switch MoE)**: $S_e(t) = \mathbf{W}_{r,e}^\top h_t^{(l)}$. Standard token-choice routing.
2.  **Baseline B (Semantic Affinity MoE)**: $S_e(t) = \text{CosineSimilarity}(h_t^{(l)}, \mathbf{C}_e)$, where $\mathbf{C}_e$ is Expert $e$'s domain centroid.
3.  **Proposed (ARES-Orion MoE)**:
    $$S_e(t) = \mathbf{W}_{r,e}^\top h_t^{(l)} + \lambda \cdot \log\left(\frac{r_e(t) + \epsilon}{1 - r_e(t) + \epsilon}\right)$$
    where $r_e(t) = P(\text{Expert } e \text{ predicts target token correctly} \mid h_t^{(l)})$.

---

## 4. Evaluation Suite & Metrics

### Primary Model Metrics
*   **Token Prediction Failure Rate**: Fraction of generated tokens where model prediction fails.
*   **Perplexity (PPL)**: Language modeling quality on ID (`tinystories`) and OOD (`wikitext`, `openwebtext`).
*   **Expected Calibration Error (ECE) & Brier Score**: Routing decision calibration.
*   **Expert Selection Regret (Oracle Gap)**:
    $$\text{Regret} = 1 - \frac{\text{Accuracy}_{\text{selected expert}}}{\text{Accuracy}_{\text{oracle best expert}}}$$

### System & Compute Metrics
*   **Inference Latency & Throughput**: Milliseconds per token, tokens per second.
*   **Total FLOPs per Sequence**: Computational cost per generated sequence.
*   **Routing Gini Index & Entropy**: Measure of expert load balancing.

---

## 5. Implementation Roadmap (4 Execution Steps)

```text
[ Step 1: Expert Pool Construction ] ──► [ Step 2: Expert Failure Data Extraction ]
                                                              │
                                                              ▼
[ Step 4: Full Pareto Benchmark ]  ◄─── [ Step 3: Probe & Router Training ]
```

### Step 1: Expert Pool Construction (`ares_moe/`)
*   Build `ARESMoELayer` module supporting $K=4$ parallel FFN experts inserted at Layers 4, 8, and 11.
*   Fine-tune Expert 1–4 from ARES-Base on domain splits with identical hyperparameter budgets.

### Step 2: Expert Failure Dataset Collection
*   Pass sequences through each individual Expert $e$ to collect expert-specific activation states $h_t$ and binary token correctness labels $Y_{e, \text{correct}}$.

### Step 3: Reliability Probe & Router Joint Training
*   Train 4 MLP probes $f_{\theta, e}(h_t)$ to predict $r_e(t)$.
*   Train gating router weights $\mathbf{W}_r$ with auxiliary load-balancing loss + reliability logit augmentation.
*   Run $\lambda$ parameter sweep across $\lambda \in \{0.0, 0.1, 0.25, 0.5, 1.0, 2.0\}$.

### Step 4: Full System Evaluation
*   Evaluate Baseline A, Baseline B, and ARES-Orion across the metrics suite.
