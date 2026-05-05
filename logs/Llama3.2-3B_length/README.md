---
base_model: meta-llama/Llama-3.2-3B-Instruct
library_name: peft
model_name: Llama3.2-3B_length
tags:
- base_model:adapter:meta-llama/Llama-3.2-3B-Instruct
- lora
- reward-trainer
- transformers
- trl
licence: license
---

# Model Card for Llama3.2-3B_length

This model is a fine-tuned version of [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
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

- PEFT 0.19.0
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