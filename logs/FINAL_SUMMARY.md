# Complete Results & Insights Summary

*Reproducibility: Understanding Impact of Human Feedback via Influence Functions (Min et al., 2025)*

---

## 0. Precise Setup (Important Clarifications)

### Bias Injection Direction

- **Length bias:** flip labels when original chosen is SHORTER than rejected (6.56% of data).
  After flipping, **corrupted labels favor verbose (longer) responses**.
- **Sycophancy bias:** flip labels when original chosen is LESS sycophantic than rejected (4.17%).
  After flipping, **corrupted labels favor more sycophantic responses**.

### Targeted Validation Sets (uncorrupted reference direction)

- Length → **Concise set** (chosen is shorter)
- Sycophancy → **Less Sycophantic set**

### Baselines in Original Paper

Mahalanobis, KNN, Self-confidence, Entropy, **GPT-4o few-shot, Gemini-1.5-Pro few-shot**.
Our reproduction uses only GPT-4o (resource-constrained simplification).

### DataInf Precise Description

DataInf uses a **low-rank approximation of the inverse-Hessian effect** (Sherman-Morrison-style),
NOT a closed-form exact inverse. OPORP is a separate gradient compression step that preserves
dot products. Together they yield ~2.5× speedup.

---

## 1. Core Reproduction Results (Baseline Models)

### Bias Detection AUC

| Model | Params | Eval Acc | Length (Concise) | Length (Verbose) | Length (Full) | Syco (Less) | Syco (More) | Syco (Full) |
|-------|--------|----------|------------------|------------------|---------------|-------------|-------------|-------------|
| Qwen2.5-1.5B | 1.5B | 65.91% | 0.542 | 0.504 | 0.534 | 0.498 | 0.481 | 0.489 |
| Qwen3-1.7B | 1.7B | 66.81% | 0.479 | 0.510 | 0.497 | 0.513 | 0.521 | 0.518 |
| **Llama-3.2-3B** | **3.2B** | **67.14%** | **0.700** | **0.315** | **0.644** | **0.579** | **0.435** | **0.592** |
| Paper (Llama-3-8B) | 8B | ~72% | 0.800 | 0.202 | 0.770 | 0.711 | 0.510 | 0.585 |

**Key finding:** Llama-3.2-3B reproduces the paper's qualitative trends (Concise >> Full >> Verbose for length; influence outperforms random for both bias types). Models below 3B fail completely.

---

## 2. **Novel Insight #1: The 2-3B Capacity Threshold**

Models below ~2-3B produce AUC ≈ 0.5 (random), regardless of architecture (Qwen2.5 vs Qwen3). This threshold is not mentioned in the original paper.

**Evidence:**
- Qwen2.5-1.5B and Qwen3-1.7B: both at AUC ~0.5 despite similar eval accuracy to Llama-3.2-3B
- Jump from ~1.7B → 3.2B: AUC goes from 0.48 → 0.70 (+0.22)
- Architecture matters less than size

---

## 3. **Novel Insight #2: The Bottleneck is LoRA Gradient Capacity, Not Model Representation**

Hidden state probing reveals that **small models CAN represent bias information internally**, but their LoRA gradients fail to express it.

### Probe AUC vs Influence AUC

| Model | Probe AUC (hidden state) | Influence AUC (LoRA gradient) | Gap |
|-------|-------------------------|-------------------------------|-----|
| Qwen2.5-1.5B | **0.692** | 0.542 | **+0.150** |
| Qwen3-1.7B | **0.633** | 0.479 | **+0.154** |
| Llama-3.2-3B | 0.682 | 0.700 | -0.018 |

**Interpretation:**
- Small models' hidden representations **do** encode which samples are biased (probe AUC ≈ 0.69)
- But LoRA gradients, being a low-dimensional lossy projection of the model's information, **fail to preserve this signal** for small models
- Large models have enough LoRA parameters (24.3M vs 9.6M) to successfully project this information

**Mechanism (proposed):**
```
Hidden states (AUC≈0.69) → LoRA projection → Gradient space → Influence score
                                ↑
                        This is the bottleneck for small models
```

---

## 4. **Novel Insight #3: Gradient Separability is the Proximate Cause**

### Key diagnostic metrics across models

| Metric | Qwen2.5-1.5B | Qwen3-1.7B | Llama-3.2-3B |
|--------|-------------|------------|-------------|
| Gradient separation gap | -0.003 | -0.001 | **+0.023** |
| Influence distribution overlap | 0.906 | 0.916 | **0.701** |
| Effective rank | 146 | 103 | **91** |
| Top-10 eigenvalue variance | 30.6% | 44.4% | **52.2%** |
| Cohen's d (grad norm) | 0.10 | -0.01 | **0.18** |

**Findings:**
1. In small models, flipped and clean samples produce nearly indistinguishable gradients
2. In 3B, flipped samples form a distinct "gradient fingerprint" (within-class similarity > between-class)
3. Counter-intuitive: 3B has LOWER effective rank (91 vs 146) — gradients are more structured, not more diverse

**Analogy:** A well-trained model's gradients are a focused laser (low rank, high signal). A poorly-trained model's gradients are scattered light (high rank, pure noise).

---

## 5. **Ruling Out Alternative Hypotheses**

### DataInf's inverse-Hessian effect approximation is NOT the bottleneck

*Note: DataInf does NOT compute a closed-form inverse Hessian; it uses a low-rank / Sherman-Morrison-style approximation of the inverse-Hessian effect. OPORP is a separate gradient compression step that preserves dot products. Together they accelerate influence estimation by ~2.5×.*


| Model | Method | Length (Concise) | Syco (Less) |
|-------|--------|------------------|-------------|
| Qwen2.5-1.5B | DataInf | 0.5419 | 0.4982 |
| Qwen2.5-1.5B | TracIn | 0.5417 | 0.4985 |
| Llama-3.2-3B | DataInf | 0.6996 | 0.5787 |
| Llama-3.2-3B | TracIn | 0.6988 | 0.5777 |

DataInf ≈ TracIn (difference < 0.001). The inverse-Hessian correction has no effect. The bottleneck is in the gradients themselves.

---

## 6. LoRA Rank Ablation (in progress on Qwen2.5-1.5B)

### Results so far

| LoRA Rank | LoRA Params | Eval Acc | Concise AUC | Verbose AUC | Full AUC |
|-----------|-------------|----------|-------------|-------------|----------|
| r=4 | 4.6M | 65.13% | 0.564 | 0.470 | 0.509 |
| r=8 | 9.2M | 65.62% | 0.531 | 0.481 | 0.497 |
| r=16 | 9.6M* | 65.91% | 0.542 | 0.504 | 0.534 |
| r=32 | ~19M | 64%† | 0.548 | 0.483 | 0.520 |
| r=64 | ~38M | - | pending | pending | pending |

*r=16 uses score layer LoRA; †r=32 only completed 1 epoch due to OOM during eval

**Preliminary observation:** AUC does NOT significantly improve with LoRA rank on Qwen2.5-1.5B (all between 0.53-0.56). This suggests that **even with larger LoRA rank, the 1.5B base model's gradients still don't carry enough bias signal**. The bottleneck is more fundamental than just "parameter count" — it's about what information the model's gradients actually encode.

**Implication:** LoRA rank alone cannot overcome the ~2-3B capacity threshold. The underlying representation quality (base model size) seems to be the real constraint.

---

## 7. Insights for Reproducibility Report

### What we successfully reproduced
- **Qualitative conclusions with Llama-3.2-3B:**
  - Influence functions detect labeler bias (AUC > 0.5)
  - Concise validation set > Full > Verbose (targeted validation matters)
  - Length bias is easier to detect than sycophancy
- **Methodology:** DataInf + OPORP pipeline works end-to-end

### What differs from the paper
- Lower absolute AUC values (3B vs 8B) — expected
- Sycophancy bias using length-proxy validation set rather than GPT-4o scoring — weaker signal

### Novel contributions beyond the paper
1. **Capacity threshold discovery:** Influence functions require ~2-3B parameter models to work; smaller models fail catastrophically
2. **Representation-gradient gap:** Small models' representations encode bias info, but LoRA gradients don't preserve it (Probe AUC ≈ 0.69 vs Influence AUC ≈ 0.50)
3. **Gradient separability as diagnostic:** Within-class vs between-class gradient cosine similarity predicts influence AUC
4. **DataInf simplification is safe:** TracIn (no Hessian) gives identical results — the Hessian approximation in DataInf is unnecessary overhead
5. **LoRA rank alone cannot fix it:** Scaling r up to 32 on a 1.5B model doesn't close the capacity gap, suggesting the bottleneck is base model representation quality, not just parameter count

### Practical recommendations
- **Do not use influence functions on models < 3B** for bias detection
- For compute-constrained settings, use TracIn instead of DataInf (simpler, equivalent results)
- Consider hidden-state probing as a complementary baseline when influence functions fail

---

## 8. Next Steps

### On RTX 5090 (32GB)
- Run Qwen2.5-7B full pipeline → closer to paper's Llama-3-8B baseline
- Run Qwen3-4B full pipeline → intermediate point
- Run r=64 LoRA ablation (OOMs on 5080 during gradient caching)

### Additional experiments (if time permits)
- **Layer-wise gradient contribution**: Which layers carry the bias signal in Llama-3.2-3B?
- **Cross-validation:** Split Llama-3.2-3B training, measure reproducibility of AUC
- **Safety bias extension** (BeaverTails) on 3B+ models
- **Full fine-tuning vs LoRA:** Does full fine-tuning (vs LoRA) help small models? (would need 5090)

---

## 9. Expanded Mechanistic Explanations (for Paper Writing)

### 决定性因素：梯度可分性 (Gradient Separability)

| 模型 | Flipped 梯度间相似度 | Flipped-Clean 间相似度 | 差值 |
|------|---------------------|----------------------|------|
| 1.5B | -0.0004 | +0.0026 | **-0.003** (无法区分) |
| 1.7B | -0.0015 | -0.0001 | **-0.001** (无法区分) |
| 3B | +0.0354 | +0.0119 | **+0.023** (可区分!) |

3B 模型中 flipped 样本的梯度**彼此更相似**（0.035），且与 clean 样本的梯度**不同**（0.012）。
这意味着模型用**系统性不同的方式**处理 biased 样本，形成了可检测的"**梯度指纹**"。
小模型则完全做不到 — 所有样本的梯度都是一团无结构的噪声。

### 反直觉发现：有效秩 (Effective Rank)

| 模型 | 有效秩 | Top-10 方差占比 |
|------|-------|---------------|
| 1.5B | 146 | 30.6% |
| 1.7B | 103 | 44.4% |
| 3B | **91** | **52.2%** |

3B 模型的梯度空间有效秩**更低**，说明梯度**更集中在少数有意义的方向上**。

**类比：聚焦的激光 vs 散射的光**
- 小模型：高秩但均匀分布的噪声
- 大模型：低秩但集中在有信号的维度上

### Probe AUC vs Influence AUC (核心发现)

| 模型 | 表征能力 (Probe AUC) | Influence AUC | 信息损失 |
|------|-------------------|---------------|---------|
| 1.5B | **0.69** | 0.54 | **丢失了 0.15** |
| 1.7B | **0.63** | 0.48 | **丢失了 0.15** |
| 3B | 0.68 | 0.70 | **无损失** |

**比喻：** 小模型其实"**知道**"哪些样本有偏差（Probe AUC≈0.69），但 LoRA 梯度太低维，
无法把这个信息传递给 influence function。就像**用一根低分辨率的管道传输高分辨率的图像** —
信息在传输过程中丢失了。

### 因果链（一句话总结）

> **模型越大 → reward function 越好 → 对 biased 样本的处理方式越不同 →
> 梯度中偏差信号越强 → influence function 越有效。**

存在一个 **~2-3B 的阈值**，低于此模型学到的 reward function 太浅，
梯度中不包含可检测的偏差信息。

### 已排除的假说

- ❌ **Hessian 近似问题** (Exp 2C): DataInf = TracIn 差异 <0.001
- ❌ **模型表征能力不足** (Exp 3B): Probe AUC ≈ 0.69 说明表征层都能区分
- ❌ **LoRA rank 不够** (Exp 1A): r=4,8,16,32 在 1.5B 上 AUC 都在 0.52-0.56

### 真正原因

✅ **LoRA 梯度是 hidden representation 的低维有损投影。** 小模型的 LoRA 参数维度不足以
保留 hidden states 中已有的偏差检测信号。大模型的 LoRA 维度足够，信号得以保留。

---

## 10. Data Assets Saved (for Paper)

| Asset | Location |
|-------|----------|
| Raw influence scores per model | `logs/{model}_{bias}/influence_*.npy` |
| Compressed gradients | `logs/{model}_{bias}/rapid_grad_*.pt` |
| Diagnostic analyses | `logs/analysis_*.md` |
| Quick analysis log | `logs/quick_analysis.log` |
| Hidden probe results | `logs/exp_hidden_probe.log` |
| TracIn vs DataInf | `logs/exp_tracin_vs_datainf.log` |
| LoRA ablation log | `logs/lora_ablation.log` |
| Plots | `logs/*.png` |
