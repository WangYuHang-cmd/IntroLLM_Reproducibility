# RTX 5080 Complete Results (3 Models)

## Training Eval Accuracy

| Model | Params | Length Bias | Sycophancy Bias |
|-------|--------|------------|-----------------|
| Qwen2.5-1.5B | 1.5B | 65.91% | 60.26% |
| Qwen3-1.7B | 1.7B | 66.81% | 63.06% |
| Llama-3.2-3B | 3.2B | **67.14%** | **65.21%** |
| Paper (Llama-3-8B) | 8B | ~72% | ~72% |

## Influence Function AUC

| Model | Length (Concise) | Length (Verbose) | Length (Full) | Syco (Less) | Syco (More) | Syco (Full) |
|-------|-----------------|-----------------|---------------|-------------|-------------|-------------|
| Qwen2.5-1.5B | 0.542 | 0.504 | 0.534 | 0.498 | 0.481 | 0.489 |
| Qwen3-1.7B | 0.479 | 0.510 | 0.497 | 0.513 | 0.521 | 0.518 |
| **Llama-3.2-3B** | **0.700** | **0.315** | **0.644** | **0.579** | **0.435** | **0.592** |
| Paper (Llama-3-8B) | 0.800 | 0.202 | 0.770 | 0.711 | 0.510 | 0.585 |

## Key Findings

### 1. Model Size Threshold
- **1.5B/1.7B 模型 (AUC ≈ 0.5):** 完全无法检测偏差，等于随机分类器
- **3B 模型 (AUC = 0.70):** 出现明显的偏差检测能力，Length bias 的 Concise AUC 达到 0.70
- **存在一个 ~2-3B 的 "capacity threshold"**，低于此阈值 influence function 无效

### 2. Llama-3.2-3B 结果与论文趋势一致
- **Length bias:** Concise AUC=0.70 (论文 0.80)，Verbose AUC=0.31 (论文 0.20)
  - 两者趋势完全一致：targeted validation set (Concise) >> 反向 validation set (Verbose)
  - Verbose AUC < 0.5 是正确的——用错误的验证集会导致反向检测
- **Sycophancy bias:** Less Sycophantic AUC=0.58 (论文 0.71), More Sycophantic AUC=0.43 (论文 0.51)
  - 趋势一致但信号较弱，可能因为 sycophancy 验证集使用了长度代理而非真实标注

### 3. 模型架构的影响
- Qwen2.5-1.5B vs Qwen3-1.7B：参数量接近但结果相似（都 ≈ 0.5），说明 Qwen2.5 和 Qwen3 在这个任务上没有本质差异
- Llama-3.2-3B 的跳跃性提升说明 **参数量是决定性因素**，而非架构

### 4. 对 7B 实验的预期
- 从 3B→8B 的 AUC 提升 (0.70→0.80) 大约 +0.10
- 预计 Qwen2.5-7B 的 Length Concise AUC 应在 **0.72-0.78** 之间
- 如果低于 0.65，需要排查 tokenization 对齐问题
