
# **Low-Rank Decomposition for Large Language Models (LLMs)**

**Final Clean Version (2022–2026)**

---

## **1. Weight Compression (SVD-Based)**

| Paper (Title)                                                                  | Year | Venue              | Category                 | Code                                                                                         |
| ------------------------------------------------------------------------------ | ---- | ------------------ | ------------------------ | -------------------------------------------------------------------------------------------- |
| **GF-SVD**: Global Knowledge-Infused Singular Value Decomposition of LLMs      | 2026 | Information Fusion | Weight Compression (SVD) | —                                                                                            |
| **SVD-LLM**: Truncation-aware Singular Value Decomposition for LLM Compression | 2025 | ICLR               | Weight Compression (SVD) | [https://github.com/AIoT-MLSys-Lab/SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM)       |
| **SVD-LLM V2**: Optimizing Singular Value Truncation for LLM Compression       | 2025 | NAACL              | Weight Compression (SVD) | [https://github.com/AIoT-MLSys-Lab/SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM)       |
| **Dobi-SVD**: Differentiable SVD for LLM Compression                           | 2025 | ICLR               | Weight Compression (SVD) | —                                                                                            |
| **FLAR-SVD**: Fast and Latency-Aware SVD for Model Compression                 | 2025 | CVPR Workshop      | Weight Compression (SVD) | —                                                                                            |
| **NSVD**: Nested Activation-Aware Decomposition for LLM Compression            | 2025 | arXiv              | Weight Compression (SVD) | —                                                                                            |
| **ResSVD**: Residual Compensated SVD for LLM Compression                       | 2025 | arXiv              | Weight Compression (SVD) | —                                                                                            |
| **DipSVD**: Dual-importance Protected SVD                                      | 2025 | arXiv              | Weight Compression (SVD) | —                                                                                            |
| **Spectral Pruning** for Neural Network Compression                            | 2023 | NeurIPS            | Weight Compression (SVD) | —                                                                                            |
| **ASVD**: Activation-aware SVD for Compressing LLMs                            | 2023 | arXiv              | Weight Compression (SVD) | [https://github.com/chenhaotian1997/ASVD4LLMA](https://github.com/chenhaotian1997/ASVD4LLMA) |

---

## **2. Variable-Rank (Adaptive Rank Selection)**

| Paper (Title)                                                     | Year | Venue | Category                 | Code                                                                                     |
| ----------------------------------------------------------------- | ---- | ----- | ------------------------ | ---------------------------------------------------------------------------------------- |
| **AdaSVD**: Adaptive Singular Value Decomposition for LLMs        | 2025 | arXiv | Variable-Rank            | [https://github.com/ZHITENGLI/AdaSVD](https://github.com/ZHITENGLI/AdaSVD)               |
| **ARA**: Adaptive Rank Allocation for Efficient LLM Compression   | 2025 | arXiv | Variable-Rank            | —                                                                                        |
| **GF-Rank**: Globally Optimized Rank Selection via Fermi Function | 2025 | arXiv | Variable-Rank            | —                                                                                        |
| **Layer-wise Dynamic Rank** for Compressing LLMs                  | 2025 | arXiv | Variable-Rank            | —                                                                                        |
| **Learning to Prune Low-Rank Components** in Large Models         | 2025 | arXiv | Variable-Rank            | —                                                                                        |
| **Any-Precision Low-Rank Compression** for LLMs                   | 2025 | arXiv | Variable-Rank            | —                                                                                        |
| **LASER**: Layer-Selective Rank Reduction                         | 2024 | ICLR  | Variable-Rank / Analysis | [https://github.com/pratyushasharma/laser](https://github.com/pratyushasharma/laser)     |
| **LLRC**: Low-Rank Compression via Differentiable Rank Selection  | 2024 | ICLR  | Variable-Rank            | —                                                                                        |
| **RankMe**: Assessing the Rank of Neural Representations          | 2024 | ICLR  | Variable-Rank / Analysis | [https://github.com/facebookresearch/rankme](https://github.com/facebookresearch/rankme) |
| **SoftLMs**: Efficient Adaptive Low-Rank Approximation            | 2024 | arXiv | Variable-Rank            | —                                                                                        |
| **Dynamic Low-Rank Compression** of Transformers                  | 2024 | arXiv | Variable-Rank            | —                                                                                        |
| **Adaptive Rank Selections** for Low-Rank Approximation           | 2024 | NAACL | Variable-Rank            | —                                                                                        |

---

## **3. Weight Compression (General Low-Rank & Tensor Decomposition)**

| Paper (Title)                                                             | Year | Venue          | Category                      | Code                                                                           |
| ------------------------------------------------------------------------- | ---- | -------------- | ----------------------------- | ------------------------------------------------------------------------------ |
| **PiFa**: Pivoting Factorization for Efficient LLM Inference              | 2025 | ICML           | Low-Rank Factorization        | —                                                                              |
| **Saten**: Sparse Augmented Tensor Networks for Post-Training Compression | 2025 | EMNLP Findings | Tensor Decomposition (TT)     | —                                                                              |
| **TensorLLM**: Tensorising Multi-Head Attention for Enhanced Reasoning    | 2025 | IEEE Access    | Tensor Decomposition (Tucker) | —                                                                              |
| **Rank-Aware Low-Rank Decomposition** for Efficient LLMs                  | 2025 | arXiv          | Weight Compression (Low-Rank) | —                                                                              |
| **CALDERA**: Low-Rank and Low-Precision Decomposition                     | 2024 | NeurIPS        | Low-Rank + Quant              | [https://github.com/IST-DASLab/caldera](https://github.com/IST-DASLab/caldera) |
| **CompactifAI**: Quantum-Inspired Tensor Network Compression for LLMs     | 2024 | arXiv          | Tensor Decomposition          | —                                                                              |
| **T-MAC**: CPU Inference with Low-Bit Tensor Compression                  | 2024 | arXiv          | Tensor + Quant                | [https://github.com/microsoft/T-MAC](https://github.com/microsoft/T-MAC)       |
| **LQER**: Low-Rank Quantization Error Reconstruction                      | 2024 | arXiv          | Low-Rank + Quant              | —                                                                              |
| **Adaptive Feature-based Low-Rank Compression** via Bayesian Optimization | 2024 | EMNLP Findings | Low-Rank + Quant              | —
| **A Token is Worth over 1,000 Tokens**: Efficient Knowledge Distillation through Low-Rank Clone | 2025 | NeurIPS (Spotlight) |  Low-Rank + Distillation | https://github.com/CURRENTF/LowRankClone |
| **Lillama**: Large Language Models Compression via Low-Rank Feature Distillation | 2024 | arXiv | Low-Rank + Distillation | — |                                                                           |
| **Training-Free Low-Rank Decomposition** for LLM Compression              | 2024 | arXiv          | Low-Rank                      | —                                                                              |
| **TensorGPT**: Tensor-Train Decomposition for LLM Embeddings              | 2023 | arXiv          | Tensor Decomposition          | [https://github.com/idiap/TensorGPT](https://github.com/idiap/TensorGPT)       |
| **PowerSGD++**: Practical Low-Rank Gradient Compression                   | 2023 | ICML           | Low-Rank                      | [https://github.com/epfml/powersgd](https://github.com/epfml/powersgd)         |

---

## **4. Low-Rank + Sparsity (Hybrid Methods)**

| Paper (Title)                                                                   | Year | Venue          | Category                    | Code                                                                                         |
| ------------------------------------------------------------------------------- | ---- | -------------- | --------------------------- | -------------------------------------------------------------------------------------------- |
| **SLiM**: One-shot Quantization and Sparsity with Low-rank Approximation        | 2025 | ICML           | Low-Rank + Quant + Sparsity | —                                                                                            |
| **Adaptive Low-Rank and Sparsity Co-Design** for LLM Compression                | 2025 | arXiv          | Low-Rank + Sparsity         | —                                                                                            |
| **Hybrid Compression of LLMs** via Low-Rank and Structured Sparsity             | 2025 | arXiv          | Low-Rank + Sparsity         | —                                                                                            |
| **Large Language Model Compression** with Global Rank and Sparsity Optimization | 2025 | arXiv          | Low-Rank + Sparsity         | —                                                                                            |
| **1+1 > 2**: Synergistic Sparse and Low-Rank Compression                        | 2025 | EMNLP Findings | Low-Rank + Sparsity         | —                                                                                            |
| **LS-PRISM**: Layer-Selective Pruning via Low-Rank Approximation                | 2025 | Neurocomputing | Low-Rank + Sparsity         | —                                                                                            |
| **Scatterbrain**: Unifying Sparse and Low-rank Attention                        | 2024 | NeurIPS        | Low-Rank + Sparsity         | [https://github.com/HazyResearch/scatterbrain](https://github.com/HazyResearch/scatterbrain) |
| **SoLA**: Soft Activation Sparsity and Low-Rank Decomposition                   | 2024 | AAAI           | Low-Rank + Sparsity         | [https://github.com/Wisshin/SoLA](https://github.com/Wisshin/SoLA)                           |
| **LORAP**: Low-Rank + Sparse Decomposition for Transformers                     | 2024 | arXiv          | Low-Rank + Sparsity         | —                                                                                            |
| **LoSparse**: Structured Compression via Low-Rank and Sparse Approximation      | 2023 | arXiv          | Low-Rank + Sparsity         | [https://github.com/LoSparse/LoSparse](https://github.com/LoSparse/LoSparse)                 |

---

## **5. KV-Cache Compression (Attention & Cache)**

| Paper (Title)                                                       | Year | Venue          | Category                     | Code                                                                                       |
| ------------------------------------------------------------------- | ---- | -------------- | ---------------------------- | ------------------------------------------------------------------------------------------ |
| **Palu**: KV-Cache Compression with Low-Rank Projection             | 2025 | ICLR           | KV-Cache (Low-Rank)          | [https://github.com/shadowpa0327/Palu](https://github.com/shadowpa0327/Palu)               |
| **ShadowKV**: KV Cache in Shadows for Long-Context Inference        | 2025 | ICML           | KV-Cache (Low-Rank + Quant)  | [https://github.com/ByteDance-Seed/ShadowKV](https://github.com/ByteDance-Seed/ShadowKV)   |
| **SALS**: Sparse Attention in Latent Space for KV Cache Compression | 2025 | arXiv          | KV-Cache (Low-Rank + Sparse) | —                                                                                          |
| **SVDq**: 1.25-bit KV Cache Compression                             | 2025 | arXiv          | KV-Cache (SVD + Quant)       | —                                                                                          |
| **KQ-SVD**: KV Cache Compression with Guarantees                    | 2025 | arXiv          | KV-Cache (SVD + Quant)       | —                                                                                          |
| **xKV**: Cross-Layer SVD for KV-Cache Compression                   | 2025 | arXiv          | KV-Cache (SVD)               | [https://github.com/abdelfattah-lab/xKV](https://github.com/abdelfattah-lab/xKV)           |
| **ReCalKV**: Low-Rank KV Cache via Head Reordering                  | 2025 | arXiv          | KV-Cache (Low-Rank)          | [https://github.com/XIANGLONGYAN/ReCalKV](https://github.com/XIANGLONGYAN/ReCalKV)         |
| **EliteKV**: RoPE-Aware Low-Rank KV Cache Compression               | 2025 | arXiv          | KV-Cache (Low-Rank + Quant)  | [https://github.com/CiaranZhou/EliteKV](https://github.com/CiaranZhou/EliteKV)             |
| **ZipCache**: KV Cache Quantization with Low-Rank Correction        | 2024 | arXiv          | KV-Cache (Low-Rank + Quant)  | —                                                                                          |
| **SnapKV**: KV Cache Reduction via Token Clustering                 | 2024 | arXiv          | KV-Cache (Rank-aware)        | —                                                                                          |
| **EigenAttention**: Attention in Low-Rank Space                     | 2024 | EMNLP Findings | KV-Cache (Low-Rank)          | [https://github.com/UtkarshSaxena1/EigenAttn](https://github.com/UtkarshSaxena1/EigenAttn) |
| **LoRC**: Low-Rank Content for KV Cache Compression                 | 2024 | arXiv          | KV-Cache (Low-Rank)          | —                                                                                          |

---



## **6. Optimization, Training & Analysis**

| Paper (Title)                                                             | Year | Venue   | Category                | Code                                                                                                 |
| ------------------------------------------------------------------------- | ---- | ------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
—                                                                                                    |
| **Low-Rank Training Does Not Mean Low-Rank Weights**                      | 2024 | NeurIPS | Analysis                | —                                                                                                    |
| **SVDFormer**: Lightweight Transformer via SVD                            | 2023 | arXiv   | Architecture            | —                                                                                                    |
| **Compressing Transformers**: Features Are Low-Rank, but Weights Are Not! | 2023 | arXiv   | Analysis                | [https://github.com/HazyResearch/feature-low-rank](https://github.com/HazyResearch/feature-low-rank) |
| **Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning** | 2021 | ACL     | Analysis (Foundational) | —                                                                                                    |

---
