# 1.5B Pipeline Analysis Report

## 1. Results Summary

### Length Bias Detection
| Validation Set | AUC | Paper Target (Llama-3-8B) |
|---------------|-----|---------------------------|
| **Concise** | 0.5419 | 0.800 |
| Verbose | 0.5038 | 0.202 |
| Full | 0.5344 | 0.770 |

### Sycophancy Bias Detection
| Validation Set | AUC | Paper Target (Llama-3-8B) |
|---------------|-----|---------------------------|
| **Less Sycophantic** | 0.4982 | 0.711 |
| More Sycophantic | 0.4805 | 0.510 |
| Full | 0.4888 | 0.585 |

**结论：** AUC ≈ 0.5，基本等于随机分类器。Influence function 未能有效区分 flipped 和 clean 样本。

---

## 2. Training Quality

| Metric | Length Bias | Sycophancy Bias | Paper (Llama-3-8B) |
|--------|-----------|-----------------|---------------------|
| Best Eval Accuracy | 65.91% | 60.26% | ~72% |
| Final Eval Loss | 0.6094 | 0.6634 | — |
| Training Epochs | 4 | 4 | 4 |

**观察：**
- Length bias 模型训练正常，accuracy 从 ~49% → 65.9%，有明显提升
- Sycophancy bias 模型训练较弱，accuracy 仅 ~60%，且 epoch 2-4 没有提升
- 两个模型的 eval accuracy 都低于原论文的 72%，符合 1.5B 模型容量较小的预期

---

## 3. Gradient & Influence Score Diagnostics

### Gradient Statistics
| Metric | Length | Sycophancy |
|--------|--------|------------|
| Grad mean | 0.000004 | -0.000175 |
| Grad std | 0.419 | 0.766 |
| Grad norm (mean ± std) | 76.2 ± 75.8 | 143.8 ± 133.8 |
| Non-zero fraction | 55.04% | 55.04% |

**观察：** 梯度本身是非零的、有方差的，OPORP 压缩正常工作。

### Influence Score Separation
| Dataset | Flipped Mean | Clean Mean | Separation Ratio |
|---------|-------------|------------|-----------------|
| Length (Concise) | -209.8 | -840.6 | **0.105** |
| Sycophancy (Less Syco) | -427.8 | -376.7 | **0.012** |

**关键发现：**
- Separation ratio 极低（< 0.2），说明 flipped 和 clean 样本的 influence score 分布几乎完全重叠
- Length bias 有微弱的分离信号（0.105），与 AUC=0.54 一致
- Sycophancy 几乎没有分离（0.012），与 AUC≈0.50 一致

---

## 4. Root Cause Analysis

### 原因 1：模型容量不足（主要原因）
- **1.5B vs 8B**：原论文使用 Llama-3-8B（42M LoRA 参数），我们使用 Qwen2.5-1.5B（18.5M LoRA 参数）
- 较小的模型学到的 reward function 较弱（accuracy 65% vs 72%），导致梯度中包含的偏差信号不够强
- Influence function 依赖于梯度的信息量——模型越强，梯度携带的关于数据质量的信号越强

### 原因 2：TRL 数据处理不一致（潜在问题）
- TRL RewardTrainer 在训练时自行 tokenize 数据（从 `chosen`/`rejected` 文本列）
- 梯度缓存使用的是我们自行 tokenize 的 `_tokenized` 版本
- 两者的 tokenization 结果可能略有差异（如 EOS token 处理、padding 方式不同）
- 数据集大小一致（14879/6121 和 15000/1071），但 token 级别可能不完全对齐

### 原因 3：Sycophancy 验证集质量（次要原因）
- 预构建数据集的 test split 没有 `answer_gpt` 列
- 使用长度作为 sycophancy 的代理划分验证集，不够准确
- Less/More Sycophantic 子集可能未真正捕捉 sycophancy 特征

---

## 5. 结论

1.5B 的实验**成功验证了整个 pipeline 的可运行性**：
- ✅ 数据下载、tokenization、过滤
- ✅ LoRA reward model 训练 + checkpoint 保存
- ✅ 4-bit 量化加载 + 逐样本梯度计算
- ✅ OPORP 压缩（42M → 65K dims）
- ✅ DataInf influence score 计算
- ✅ AUC/ROC 评估

但 **AUC ≈ 0.5 不具备偏差检测能力**，需要 7B 模型来获得有意义的结果。

---

## 6. Next Steps

### 立即行动：在 RTX 5090 上跑 7B
1. 将 `dataset/` 和代码复制到 5090 机器
2. 使用 `Qwen2.5-7B-Instruct` 重新训练 + 梯度缓存 + influence 计算
3. 预期：eval accuracy ~68-72%，AUC 应接近 0.7-0.8

### 可选优化（如果 7B AUC 仍不理想）：
1. **统一 tokenization**：让梯度缓存使用 TRL 处理后的数据，而非自行 tokenize 的版本
2. **增加 LoRA rank**：从 r=16 增加到 r=32，增加可训练参数
3. **调整 DataInf lambda**：当前 lambda = 0.1 * mean(grad²) / n_train，可以尝试其他系数
4. **对比 DataInf-only vs OPORP+DataInf**：验证压缩是否损失了关键信息
