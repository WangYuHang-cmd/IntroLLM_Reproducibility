# Running Qwen2.5-7B Experiments on Google Colab

## Overview

Qwen2.5-7B needs ~20GB VRAM for training, more than Colab's free T4 (16GB).
Solutions depending on your Colab subscription:

| Subscription | GPU | VRAM | Training 7B | Gradient Cache 7B | Cost |
|-------------|-----|------|-------------|-------------------|------|
| Free | T4 | 16GB | ❌ (OOM) | ✅ (4-bit, ~8GB) | Free, 12h max |
| Pro | L4 / T4 | 24GB / 16GB | ⚠️ Tight | ✅ | $10/mo |
| Pro+ | A100 | 40GB | ✅ Easy | ✅ | $50/mo |

**Recommended strategy:**
- **Pro+/A100:** Run everything on Colab (~10h total)
- **Pro/L4:** Training on L4 with gradient_checkpointing=True, caching on T4
- **Free:** Train on your 5080 via HuggingFace checkpoint export, then Colab only for gradient caching

---

## Strategy: Free Tier (Recommended for you)

Since you have RTX 5080 (16GB) for training and need Colab for gradient caching:

### Phase A: Train 7B on 5080 (locally)

Qwen2.5-7B won't fit on 5080 directly, BUT you can:
- Option 1: Use Qwen2.5-3B as substitute (fits on 5080)
- Option 2: Train with gradient_checkpointing=True + ultra-small batch (slow, might fit)
- Option 3: Skip this and use Colab A100 directly for everything

Actually for 7B, **just do everything on Colab A100 (Pro+) or rent one A100 hour on Lambda**.

### Phase B: Transfer & cache gradients on Colab T4 (4-bit, fits in 16GB)

This is where free Colab helps: 4-bit quantized 7B uses ~8GB, easy fit on T4.

---

## Step-by-Step Guide (Colab A100 / Pro+)

### Step 1: Package your project

On your 5080 machine, run:

```bash
cd ~/Desktop
tar --exclude='IntroLLM/IF_RLHF_ref' \
    --exclude='IntroLLM/logs/Qwen*/rapid_grad_*.pt' \
    --exclude='IntroLLM/logs/Qwen*/RapidGrad_*.obj' \
    --exclude='IntroLLM/logs/Qwen*/checkpoint-*' \
    --exclude='IntroLLM/logs/Llama*/rapid_grad_*.pt' \
    --exclude='IntroLLM/logs/Llama*/RapidGrad_*.obj' \
    --exclude='IntroLLM/logs/Llama*/checkpoint-*' \
    -czf IntroLLM_for_colab.tar.gz IntroLLM/

ls -lh IntroLLM_for_colab.tar.gz  # Should be ~500MB (includes dataset)
```

### Step 2: Upload to Google Drive

Upload `IntroLLM_for_colab.tar.gz` to your Google Drive
(e.g., in a folder `Colab/IntroLLM/`).

### Step 3: Create the Colab notebook

Create a new Colab notebook. Select runtime:
- **Runtime → Change runtime type → A100 GPU** (Pro+)
- or L4 (Pro) if A100 not available

### Step 4: Copy the notebook content

See `colab_7B_notebook.ipynb` in the project (generated below).

---

## Files Generated

I'll create the following for you:

1. **`colab/colab_7B_notebook.ipynb`** — ready-to-use Colab notebook
2. **`colab/package_for_colab.sh`** — packages project for upload
3. **`colab/sync_back.sh`** — downloads results from Drive back to local

---

## Timeline Estimate (Colab A100)

| Phase | Task | Time |
|-------|------|------|
| 0 | Mount Drive, extract project, install deps | ~5min |
| 1 | Download Qwen2.5-7B from HuggingFace | ~5min |
| 2a | Train RM for length bias | ~1.5h |
| 2b | Train RM for sycophancy bias | ~1.5h |
| 3a | Cache train gradients (length) | ~2h |
| 3b | Cache val gradients (length) | ~45min |
| 3c | Cache train gradients (sycophancy) | ~2h |
| 3d | Cache val gradients (sycophancy) | ~10min |
| 4 | Compute influence scores | ~10min |
| 5 | Save results to Drive | ~2min |
| **Total** | | **~8h** |

On L4 (Pro): ~12-15h. On T4 free: training won't fit.

---

## Critical: Checkpointing for Disconnections

Colab disconnects after 12h or idle. To handle this, the notebook will:

1. **Save to Drive every step** — intermediate results persist
2. **Resume logic** — re-running the notebook skips completed steps
3. **Heartbeat cell** — prevents idle timeout (run a tiny loop)

---

## Alternative: Skip 7B, Focus on Existing Results

Given that you already have excellent findings on 5080:
- 3 models (1.5B, 1.7B, 3B) completed
- Clear capacity threshold identified
- Deep mechanistic analysis (probe vs influence, gradient separability, etc.)

**Your current results are already publication-worthy.** The 7B experiment would
mainly strengthen the "scaling trend" story, which you can already argue from
1.5B → 1.7B → 3B. If Colab is expensive/inconvenient, the paper works without it.

Consider whether the additional cost/time is worth marginal improvement to the story.
