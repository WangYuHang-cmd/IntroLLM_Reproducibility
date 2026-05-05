# Deep Investigation Results

## Experiment 2C: TracIn vs DataInf — 排除 Hessian 假说

| Model | Method | Length (Concise) | Sycophancy (Less) |
|-------|--------|-----------------|-------------------|
| Qwen2.5-1.5B | DataInf | 0.5419 | 0.4982 |
| Qwen2.5-1.5B | TracIn | 0.5417 | 0.4985 |
| Qwen3-1.7B | DataInf | 0.4791 | 0.5126 |
| Qwen3-1.7B | TracIn | 0.4789 | 0.5124 |
| Llama-3.2-3B | DataInf | 0.6996 | 0.5787 |
| Llama-3.2-3B | TracIn | 0.6988 | 0.5777 |

**结论：DataInf ≈ TracIn，差异 < 0.001。**

**Hessian 逆近似不是问题所在。** 有无 inverse Hessian 修正几乎不影响结果。
问题纯粹出在梯度的点积本身 — 小模型的梯度不携带偏差信息。

---

## Experiment 3B: Hidden State Probing — **关键发现**

| Model | Probe AUC | Influence AUC | Gap |
|-------|-----------|---------------|-----|
| Qwen2.5-1.5B | **0.6922** | 0.5419 | **+0.150** |
| Qwen3-1.7B | **0.6328** | 0.4791 | **+0.154** |
| Llama-3.2-3B | 0.6816 | **0.6996** | -0.018 |

### 这告诉我们什么？

**小模型的内部表征实际上 CAN 区分 biased 样本！**

- Qwen2.5-1.5B 的 hidden states 上训练一个线性分类器就能达到 AUC=0.69
- 但 influence function 只有 AUC=0.54
- **Gap = 0.15 — 信息存在于模型中，但 LoRA 梯度没有捕获到**

**大模型则没有这个 gap：**

- Llama-3.2-3B 的 Probe AUC (0.68) ≈ Influence AUC (0.70)
- LoRA 梯度成功传递了表征中的偏差信息

### 机制解释

LoRA 梯度是模型完整信息的一个 **有损投影**。

```
Full model info (hidden states) --[LoRA projection]--> Gradient space --[DataInf]--> Influence score
         AUC ≈ 0.69                                      AUC ≈ 0.54 (1.5B)
                                                          AUC ≈ 0.70 (3B)
```

- 小模型的 LoRA 参数少 (9.6M) → 梯度空间维度低 → 投影损失大 → 偏差信号被噪声淹没
- 大模型的 LoRA 参数多 (24.3M) → 梯度空间维度高 → 投影损失小 → 偏差信号保留

**这直接预测：增大小模型的 LoRA rank 应该能缩小这个 gap。**

---

## 综合结论

问题不在于：
- ~~DataInf 的 Hessian 近似~~ (排除 by Exp 2C)
- ~~模型的表征能力~~ (排除 by Exp 3B: probe AUC ≈ 0.69 for all models)

问题在于：
- **LoRA 梯度的信息容量不足** — 小模型的 LoRA 参数太少，梯度空间维度太低，
  无法保留 hidden states 中已有的偏差检测信号

---

## Next Step: LoRA Rank Ablation

**最优先实验：** 在 Qwen2.5-1.5B 上变化 LoRA rank (r=4,8,16,32,64)

预期：
- 如果 r=64 时 AUC 从 0.54 提升到 ~0.65+，证实瓶颈在梯度维度
- 如果 AUC 不变，则瓶颈在模型训练质量（但 probe 结果暗示不太可能）
