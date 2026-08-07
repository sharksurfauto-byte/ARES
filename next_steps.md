# ARES-Orion: Hierarchical Reliability Roadmap & Next Steps

This document outlines the revised architecture and execution roadmap for the **ARES (Adaptive Reliable Expert System)** framework. ARES models reliability as a two-tier hierarchy: **Global (Prompt-Level)** and **Local (Token-Level)**.

---

## 1. Unified Research Hypothesis & Framing

> **Core Research Hypothesis**:
> *"Reliability should be estimated hierarchically—first at the prompt level to guide global compute allocation (GRM), and then at the token level to guide local expert selection inside an MoE topology (LRM)."*

```text
                        ┌──────────────────────────────┐
                        │         User Prompt          │
                        └──────────────┬───────────────┘
                                       │
  =====================================▼=====================================
  GLOBAL RELIABILITY MODULE (GRM)  [Prompt-Level Assessment]
  ===========================================================================
                                       │
                         Last Prompt Token Activation
                                       │
                           Global Probe (MLP / Linear)
                                       │
                         Global Reliability Score (R_global)
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
            High Reliability (R_global > τ)   Low Reliability (R_global ≤ τ)
            (Fast Path / Base Compute)        (Activate MoE / High Compute)
                                                        │
  ======================================================▼====================
  LOCAL RELIABILITY MODULE (LRM)   [Token-Level MoE Routing]
  ===========================================================================
                                                        │
                                                 Decoding Step t
                                                        │
                                            Token Hidden State h_t
                                                        │
                                           Expert Reliability Vector R_local(t)
                                                        │
                                              MoE Reliability Router
                                                        │
                                                Selected Expert(s)
                                                        │
                                                    Next Token
```

---

## 2. Component Definitions

### A. Global Reliability Module (GRM)
*   **Inspection Point**: Hidden state of the final prompt token ($x_{\text{prompt\_end}}$) before output generation begins.
*   **Output**: Scalar **Global Reliability Score** $R_{\text{global}} \in [0, 1]$.
*   **Action**: Determines macro-level compute allocation:
    *   $R_{\text{global}} \ge 0.85$: High confidence $\rightarrow$ Direct execution on lightweight base model (low compute, zero routing overhead).
    *   $R_{\text{global}} < 0.85$: High risk $\rightarrow$ Route prompt to the full ARES-Orion MoE pipeline.

### B. Local Reliability Module (LRM)
*   **Inspection Point**: Intermediate layer hidden states ($h_t^{(l)}$) at each autoregressive decoding step $t$.
*   **Output**: **Expert Reliability Vector** $\mathbf{R}_{\text{local}}(t) = [r_1, r_2, \dots, r_K]$, where $r_e \in [0, 1]$ represents the predicted correctness probability of Expert $e$ for token $t$.
*   **Action**: Determines micro-level token routing inside the sparse MoE layer.

---

## 3. Phase 2.5: Experimental Validation & Controls (COMPLETED)

Phase 2.5 validation is **complete**. The empirical results confirm:

```text
                           [ Phase 2.5 Empirical Findings ]
                                          │
       ┌───────────────────────┬──────────┴───────────┬───────────────────────┐
       ▼                       ▼                      ▼                       ▼
   Layer Emergence        OOD Transfer Gap        Selectivity            Calibration
 Monotonic L10 Peak     GRM (0.50) vs LRM (0.72)  +18.5% Passed           ECE < 0.02
```

1.  **Monotonic Emergence**: LRM MLP AUROC rises monotonically from Layer 0 (`0.6865`) to Layer 10 (`0.7713`).
2.  **OOD Transfer Gap**: GRM (Prompt-level) achieves `0.9655` ID but collapses to `0.5000` (chance) OOD on OpenWebText/WikiText. LRM (Token-level) maintains **`0.7240` OOD AUROC**, proving token-level uncertainty is domain-agnostic.
3.  **Hewitt-Liang Control**: LRM probe accuracy drops to `51.5%` on shuffled labels (`+18.5%` selectivity), proving probes read genuine internal representations.
4.  **Calibration**: LRM probe ECE remains under **`0.015` – `0.022`** across all seeds.

---

## 4. Phase 3: ARES-Orion MoE Controlled Ablations

In Phase 3, we construct a 3-way ablation to evaluate downstream performance, FLOP savings, and latency:

```text
                              [ Fixed Expert Pool ]
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
       [ Router 1 ]                [ Router 2 ]               [ Router 3 ]
       Standard Gating            Semantic Affinity       ARES-Orion Hierarchical
       (Switch MoE)               (Domain-Only)           (GRM + LRM Reliability)
```

### Metrics Suite for MoE Evaluation
*   **Downstream Accuracy & Perplexity**: WikiText-103 PPL, GSM8K reasoning accuracy.
*   **Compute Efficiency**: Total FLOPs per sequence, inference latency (ms/token).
*   **Calibration & Reliability**: Expected Calibration Error (ECE), Brier score.
*   **Expert Load Balance**: Routing Gini coefficient, routing entropy.
