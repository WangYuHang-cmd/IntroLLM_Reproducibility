---
base_model: Qwen/Qwen3-4B
library_name: transformers
model_name: Qwen3-4B_length
tags:
- generated_from_trainer
- reward-trainer
- trl
licence: license
---

# Model Card for Qwen3-4B_length

This model is a fine-tuned version of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

text = "The capital of France is Paris."
rewarder = pipeline(model="None", device="cuda")
output = rewarder(text)[0]
print(output["score"])
```

## Training procedure





This model was trained with Reward.

### Framework versions

- TRL: 1.1.0
- Transformers: 5.5.4
- Pytorch: 2.11.0+cu128
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```