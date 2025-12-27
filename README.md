# **Low-Rank Decomposition for Large Language Model Compression: A Survey**


---

## **1. Metric-Aware SVD**

| Paper                                                                                                                                    | Method        | Year | Venue | Code                                                 |
|------------------------------------------------------------------------------------------------------------------------------------------|---------------|------|-------|------------------------------------------------------|
| [SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression](https://arxiv.org/abs/2403.07378)          | SVD-LLM       | 2025 | ICLR  | [Link](https://github.com/AIoT-MLSys-Lab/SVD-LLM)    |
| [Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression](https://arxiv.org/abs/2410.03765)                    | Basis Sharing | 2025 | ICLR  | [Link](https://github.com/TUDa-HWAI/Basis_Sharing)   |
| [Large Language Model Compression via the Nested Activation-Aware Decomposition](https://arxiv.org/abs/2503.17101)                       | NSVD          | 2025 | arXiv | -                                                    |
| [ResSVD: Residual Compensated SVD for Large Language Model Compression](https://arxiv.org/abs/2505.20112)                                | ResSVD        | 2025 | arXiv | -                                                    |
| [DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights](https://arxiv.org/abs/2501.18596) | DeltaLLM | 2025 | arXiv | - |
| [Numerical Optimizations for Weighted Low-rank Estimation on Language Model](https://arxiv.org/abs/2211.09718) | TFWSVD | 2022 | EMNLP Findings | - |
| [Language model compression with weighted low-rank factorizations](https://arxiv.org/abs/2207.00112)                                     | FWSVD         | 2022 | arXiv | -                                                    |


---

## **2. Adaptive Rank Allocation**

| Paper                                                                                                                                                                                                                                                           | Method     | Year | Venue          | Code                                                             |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|------|----------------|------------------------------------------------------------------|
| [SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression](https://arxiv.org/abs/2503.12340)                                                                                                                                       | SVD-LLM V2 | 2025 | NAACL          | [Link](https://github.com/AIoT-MLSys-Lab/SVD-LLM)                |
| [Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives](https://arxiv.org/abs/2502.02723)                                                                                                                                                  | Dobi-SVD   | 2025 | ICLR           | [Link](https://github.com/wangqinsi1/Dobi-SVD)                   |
| [FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inference](https://arxiv.org/abs/2510.09332) | FLRC | 2025 | EMNLP  | - |
| [FLAR-SVD: Fast and Latency-Aware Singular Value Decomposition for Model Compression](https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Thoma_FLAR-SVD_Fast_and_Latency-Aware_Singular_Value_Decomposition_for_Model_Compression_CVPRW_2025_paper.pdf) | FLAR-SVD   | 2025 | CVPR Workshop  | [Link](https://github.com/MoritzTho/FLAR-SVD)                    |
| [Globally optimized SVD compression of LLMs via Fermi-function-based rank selection and gauge fixing](https://arxiv.org/abs/2512.03062) | FermiGrad + PivGa | 2025 | arXiv | - |
| [FLAT-LLM: Fine-grained Low-rank Activation Space Transformation for Large Language Model Compression](https://arxiv.org/abs/2505.23966)                                                                                                                        | FLAT-LLM   | 2025 | arXiv          | [Link](https://github.com/TTTTTTris/FLAT-LLM)                    |
| [DipSVD: Dual-importance Protected SVD for Efficient LLM Compression](https://arxiv.org/abs/2506.20353)                                  | DipSVD        | 2025 | arXiv | -        |
| [CPSVD: Enhancing Large Language Model Compression via Column-Preserving Singular Value Decomposition](https://arxiv.org/abs/2510.19385) | CPSVD         | 2025 | arXiv | [Link](https://github.com/Pupu792/CPSVD)             |
| [AdaSVD: Adaptive Singular Value Decomposition for Large Language Models](https://arxiv.org/abs/2502.01403)                                                                                                                                                     | AdaSVD     | 2025 | arXiv          | [Link](https://github.com/ZHITENGLI/AdaSVD)                      |
| [ARA: Adaptive Rank Allocation for Efficient Large Language Model SVD Compression](https://arxiv.org/abs/2510.19389)                                                                                                                                            | ARA        | 2025 | arXiv          | [Link](https://github.com/Pupu792/ARA)                           |
| [Layer-wise dynamic rank for compressing large language models](https://arxiv.org/abs/2509.25622)                                                                                                                                                               | D-Rank     | 2025 | arXiv          | -                                                                |
| [Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM](https://arxiv.org/abs/2510.05544) | PGSVD | 2025 | arXiv | - |
| [CALR: Corrective Adaptive Low-Rank Decomposition for Efficient Large Language Model Layer Compression](https://arxiv.org/abs/2508.16680)                                                                                                                       | CALR       | 2024 | ICLR           | -                                                                |
| [Basis Selection: Low-Rank Decomposition of Pretrained Large Language Models for Target Applications](https://arxiv.org/abs/2405.15877)                                                                                                                         | Basel      | 2024 | arXiv          | -                                                                |
| [Low-Rank Compression of Language Models Via Differentiable Rank Selection](https://openreview.net/forum?id=960Ny6IjEr)                                                                                                                                         | LLRC       | 2024 | Openreview     | -                                                                |
| [SoftLMs: Efficient Adaptive Low-Rank Approximation of Language Models using Soft-Thresholding Mechanism](https://arxiv.org/abs/2411.10543)                                                                                                                     | SoftLMs    | 2024 | arXiv          | -                                                                |
| [Adaptive Rank Selections for Low-Rank Approximation of Language Models](https://aclanthology.org/2024.naacl-long.13/)                                                                                                                                          | ARS        | 2024 | NAACL          | [Link](https://github.com/sidhantls/adaptive-rank-selection-svd) |
| [Adaptive Feature-based Low-Rank Compression of Large Language Models via Bayesian Optimization](https://arxiv.org/abs/2405.10616)                                                                                                                              | Bolaco     | 2024 | EMNLP findings | [Link](https://github.com/Dereck0602/Bolaco)                     |
| [Dynamic Low-rank Estimation for Transformer-based Language Models](https://aclanthology.org/2023.findings-emnlp.621/)                                                                                                                                          | RankDyna   | 2023 | ACL Findings   | -                                                                |
| [ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models](https://arxiv.org/abs/2312.05821)            | ASVD          | 2023 | arXiv | [Link](https://github.com/chenhaotian1997/ASVD4LLMA) |
---

## **3. Structured and Tensor Factorization**
| Paper                                                                                                                                                                     | Method      | Year | Venue          | Code                                                             |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|------|----------------|------------------------------------------------------------------|
| [Pivoting Factorization: A Compact Meta Low-Rank Representation of Sparsity for Efficient Inference in Large Language Models](https://openreview.net/forum?id=5OLRHkzTYk) | PiFa        | 2025 | ICML           | [Link](https://github.com/zhaoyang-zheng/pivoting_factorization) |
| [ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Transformations](https://arxiv.org/abs/2506.02818)     | ProcrustesGPT        | 2025 | arXiv   | [Link](https://github.com/GrishKate/ProcrustesGPT)       |
| [MoDeGPT: Modular Decomposition for Large Language Model Compression](https://arxiv.org/abs/2408.09632)                                                                   | MoDeGPT     | 2025 | ICLR           | -                                                                |
| [Saten: Sparse Augmented Tensor Networks for Post-Training Compression of Large Language Models](https://arxiv.org/abs/2505.14871)                                        | Saten       | 2025 | EMNLP Findings | -                                                                |
| [TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs](https://arxiv.org/abs/2501.15674)                                            | TensorLLM   | 2025 | IJCNN          | [Link](https://github.com/guyuxuan9/TensorLLM)                   |
| [CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks](https://arxiv.org/abs/2401.14109)                                      | CompactifAI | 2024 | arXiv          | -                                                                |
| [TensorGPT: Efficient Compression of Large Language Models based on Tensor-Train Decomposition](https://arxiv.org/abs/2307.00526)                                         | TensorGPT   | 2023 | arXiv          | -                                                                |

---

## **4. Synergistic Low-Rank Frameworks**


| Paper                                                                                                                                                                                                              | Method       | Year | Venue          | Code                                             |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|------|----------------|--------------------------------------------------|
| [SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping](https://arxiv.org/abs/2512.13494) | SkipCat | 2026 | AAAI | - |
| [SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression](https://arxiv.org/abs/2410.09615)                                                                                | SLiM         | 2025 | ICML           | [Link](https://github.com/Paramathic/slim)       |
| [FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models](https://arxiv.org/abs/2508.01506) | FlashSVD | 2025 | arXiv | - |
| [HASSLE-free: A unified Framework for Sparse plus Low-Rank Matrix Decomposition for LLMs](https://arxiv.org/abs/2502.00899)                                                                                        | HASSLE-free  | 2025 | arXiv          | -                                                |
| [UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs](https://arxiv.org/abs/2512.03383)                                                                                                    | UniQL        | 2025 | arXiv          | -                                                |
| [Large Language Model Compression with Global Rank and Sparsity Optimization](https://arxiv.org/abs/2505.03801)                                                                                                    | CAP          | 2025 | arXiv          | -                                                |
| [1+1>2: A Synergistic Sparse and Low-Rank Compression Method for Large Language Models](https://arxiv.org/abs/2510.26446)                                                                                          | SSLC         | 2025 | EMNLP Findings | -                                                |
| [LS-PRISM: A layer-selective pruning method via low-rank approximation and sparsification for efficient large language model compression](https://www.sciencedirect.com/science/article/abs/pii/S0893608025007907) | LS-PRISM     | 2025 | Neurocomputing | -                                                |
| [MLoRQ: Bridging Low-Rank and Quantization for Transformer Compression](https://arxiv.org/abs/2507.09616) | MLoRQ | 2025 | arXiv | - |
| [Low-Rank Prehab: Preparing Neural Networks for SVD Compression](https://arxiv.org/abs/2512.01980) | Prehab | 2025 | arXiv | - |
| [SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compression](https://ojs.aaai.org/index.php/AAAI/article/view/33923)                                                | SoLA         | 2025 | AAAI           | [Link](https://github.com/Wisshin/SoLA)          |
| [Compressing Large Language Models using Low Rank and Low Precision Decomposition](https://arxiv.org/abs/2405.18886)                                                                                               | CALDERA      | 2024 | NeurIPS        | -                                                |
| [LoRAP: Transformer Sub-Layers Deserve Differentiated Structured Compression for Large Language Models](https://arxiv.org/abs/2404.09695)                                                                          | LORAP        | 2024 | ICML           | [Link](https://github.com/lihuang258/LoRAP)      |
| [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/abs/2306.11222)                                                                           | LoSparse     | 2023 | ICML           | [Link](https://github.com/yxli2123/LoSparse)     |
| [LQER: Low-Rank Quantization Error Reconstruction for LLMs](https://arxiv.org/abs/2402.02446)                                                                                                                      | LQER         | 2024 | ICML           | -                                                |
| [A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](https://arxiv.org/abs/2505.12781)                                                                                    | LRC          | 2025 | NeurIPS        | [Link](https://github.com/CURRENTF/LowRankClone) |
| [Efficient One-shot Compression via Low-Rank Local Feature Distillation](https://aclanthology.org/2025.naacl-long.291/)                                                                                            | Lillama      | 2025 | NAACL          | [Link](https://github.com/yaya-sy/lillama)       |
| [Matrix Compression via Randomized Low Rank and Low Precision Factorization](https://arxiv.org/abs/2310.11028) | LRLP | 2023 | NeurIPS | - |
| [Scatterbrain: Unifying Sparse and Low-rank Attention Approximation](https://arxiv.org/abs/2110.15343)                                                                                                             | Scatterbrain | 2021 | NeurIPS        | -                                                |


---

## **5.Runtime KV-Cache Approximation**

| Paper                                                                                                                                 | Method         | Year | Venue          | Code                                                |
|---------------------------------------------------------------------------------------------------------------------------------------|----------------|------|----------------|-----------------------------------------------------|
| [Palu: Compressing KV-Cache with Low-Rank Projection](https://arxiv.org/abs/2407.21118)                                               | Palu           | 2025 | ICLR           | [Link](https://github.com/shadowpa0327/Palu)        |
| [ShadowKV: KV Cache in Shadows for Long-Context Inference](https://arxiv.org/abs/2410.21465)                                          | ShadowKV       | 2025 | ICML           | [Link](https://github.com/ByteDance-Seed/ShadowKV)  |
| [SALS: Sparse Attention in Latent Space for KV Cache Compression](https://arxiv.org/abs/2510.24273)                                   | SALS           | 2025 | arXiv          | -                                                   |
| [SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention](https://arxiv.org/abs/2502.15304)                                   | SVDq           | 2025 | arXiv          | -                                                   |
| [KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity](https://arxiv.org/abs/2512.05916)                   | KQ-SVD         | 2025 | arXiv          | -                                                   |
| [xKV: Cross-Layer SVD for KV-Cache Compression](https://arxiv.org/abs/2503.18893)                                                     | xKV            | 2025 | arXiv          | [Link](https://github.com/abdelfattah-lab/xKV)      |
| [ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration](https://arxiv.org/abs/2505.24357)                | ReCalKV        | 2025 | arXiv          | [Link](https://github.com/XIANGLONGYAN/ReCalKV)     |
| [EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection](https://arxiv.org/abs/2503.01586) | EliteKV        | 2025 | arXiv          | [Link](https://github.com/CiaranZhou/EliteKV)       |
| [Eigen Attention: Attention in Low-Rank Space for KV Cache Compression](https://arxiv.org/abs/2408.05646)                             | EigenAttention | 2024 | EMNLP Findings | [Link](https://github.com/UtkarshSaxena1/EigenAttn) |
| [LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy](https://arxiv.org/abs/2410.03111)              | LoRC           | 2024 | arXiv          | -                                                   |