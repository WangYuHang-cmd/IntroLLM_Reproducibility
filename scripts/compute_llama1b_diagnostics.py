"""
Compute all missing diagnostic metrics for Llama-3.2-1B from cached gradients.
Outputs numbers to paste into generate_paper_figures.py.

Computes:
  - Gradient separability (fig3): within_flipped, within_clean, between, separation
  - Effective rank & variance (fig4): effective_rank, var_top10
  - Cohen's d (fig2): cohen_d
  - Distribution overlap (fig8): dist_overlap
  - TracIn AUC (fig5): tracin_auc_concise, tracin_auc_full
"""

import numpy as np
import torch
import sys
sys.path.insert(0, ".")
from src.influence.datainf import rapid_datainf, rapid_tracin, get_roc_auc

np.random.seed(42)

MODEL_DIR_LENGTH = "logs/Llama3.2-1B_length"
MODEL_DIR_SYCO   = "logs/Llama3.2-1B_sycophancy"
DATA_DIR_LENGTH  = "dataset/length_dataset"
DATA_DIR_SYCO    = "dataset/sycophancy_dataset"

# ── Load cached gradients ──────────────────────────────────────────────────
print("Loading cached gradients...")
train_grads_l = torch.load(f"{MODEL_DIR_LENGTH}/rapid_grad_train.pt", weights_only=False)[65536]
val_grads_l   = torch.load(f"{MODEL_DIR_LENGTH}/rapid_grad_val.pt",   weights_only=False)[65536]
flipped_l     = np.load(f"{DATA_DIR_LENGTH}/flipped_indices.npy")
concise_idx   = np.load(f"{DATA_DIR_LENGTH}/concise_indices.npy").tolist()
verbose_idx   = np.load(f"{DATA_DIR_LENGTH}/verbose_indices.npy").tolist()
print(f"  train: {len(train_grads_l)}, val: {len(val_grads_l)}, flipped: {len(flipped_l)}")

G_train = torch.stack(train_grads_l).float()
flipped_set = set(flipped_l.tolist())

# ── 1. Gradient Separability (fig3) ───────────────────────────────────────
print("\n[1/4] Gradient separability...")
n_sample = min(500, len(G_train))
idx = np.random.choice(len(G_train), n_sample, replace=False)
G_s = G_train[idx]
labels = np.array([1 if i in flipped_set else 0 for i in idx])

norms = torch.norm(G_s, dim=1, keepdim=True)
normed = G_s / (norms + 1e-10)
cos_sim = torch.mm(normed, normed.t()).numpy()

fm = labels == 1
cm = labels == 0
within_f_mat = cos_sim[np.ix_(fm, fm)].copy(); np.fill_diagonal(within_f_mat, np.nan)
within_c_mat = cos_sim[np.ix_(cm, cm)].copy(); np.fill_diagonal(within_c_mat, np.nan)
between_mat  = cos_sim[np.ix_(fm, cm)]

within_flipped = float(np.nanmean(within_f_mat))
within_clean   = float(np.nanmean(within_c_mat))
between        = float(np.nanmean(between_mat))
separation     = within_flipped - between

print(f"  within_flipped: {within_flipped:.6f}")
print(f"  within_clean:   {within_clean:.6f}")
print(f"  between:        {between:.6f}")
print(f"  separation:     {separation:.6f}")

# ── 2. Effective Rank & Variance (fig4) ───────────────────────────────────
print("\n[2/4] Effective rank...")
n_sv = min(1000, len(G_train))
idx2 = np.random.choice(len(G_train), n_sv, replace=False)
G_sv = G_train[idx2]
GGT = G_sv @ G_sv.t()
eigenvalues = torch.linalg.eigvalsh(GGT).flip(0)
eigenvalues = eigenvalues[eigenvalues > 0]
p = eigenvalues / eigenvalues.sum()
entropy = -(p * torch.log(p + 1e-10)).sum()
effective_rank = torch.exp(entropy).item()
cumvar = torch.cumsum(eigenvalues, 0) / eigenvalues.sum()
var_top10 = cumvar[min(9, len(cumvar)-1)].item()

print(f"  effective_rank: {effective_rank:.1f}")
print(f"  var_top10:      {var_top10:.4f}")

# ── 3. Cohen's d  ─────────────────────────────────────────────────────────
print("\n[3/4] Cohen's d (gradient norm)...")
norms_all = torch.norm(G_train, dim=1)
f_norms = norms_all[[i for i in range(len(norms_all)) if i in flipped_set]]
c_norms = norms_all[[i for i in range(len(norms_all)) if i not in flipped_set]]
pooled_std = torch.sqrt((f_norms.std()**2 + c_norms.std()**2) / 2)
cohen_d = ((f_norms.mean() - c_norms.mean()) / pooled_std).item()
print(f"  cohen_d: {cohen_d:.4f}")

# ── 4. Distribution overlap (from influence_concise.npy) ──────────────────
print("\n[4/4] Distribution overlap...")
import json
from pathlib import Path
inf_file = f"{MODEL_DIR_LENGTH}/influence_concise.npy"
influence = np.load(inf_file)
inf_f = [influence[i] for i in range(len(influence)) if i in flipped_set]
inf_c = [influence[i] for i in range(len(influence)) if i not in flipped_set]
lo = np.percentile(np.concatenate([inf_f, inf_c]), 1)
hi = np.percentile(np.concatenate([inf_f, inf_c]), 99)
bins = np.linspace(lo, hi, 50)
hf, _ = np.histogram(inf_f, bins=bins, density=True)
hc, _ = np.histogram(inf_c, bins=bins, density=True)
hf_n = hf / (hf.sum() + 1e-10)
hc_n = hc / (hc.sum() + 1e-10)
dist_overlap = float(np.sum(np.minimum(hf_n, hc_n)))
print(f"  dist_overlap: {dist_overlap:.4f}")

# ── 5. TracIn AUC (fig5) ──────────────────────────────────────────────────
print("\n[5/5] TracIn AUC...")
influence_tracin_concise = rapid_tracin(train_grads_l, val_grads_l, concise_idx)
auc_tracin_concise, _, _ = get_roc_auc(influence_tracin_concise, flipped_l)
influence_tracin_full = rapid_tracin(train_grads_l, val_grads_l, list(range(len(val_grads_l))))
auc_tracin_full, _, _  = get_roc_auc(influence_tracin_full, flipped_l)
print(f"  TracIn Concise AUC: {auc_tracin_concise:.4f}")
print(f"  TracIn Full AUC:    {auc_tracin_full:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY — paste into generate_paper_figures.py Llama-3.2-1B entry")
print("="*60)
print(f'            "grad_within_flipped": {within_flipped:.6f},')
print(f'            "grad_within_clean":   {within_clean:.6f},')
print(f'            "grad_between":        {between:.6f},')
print(f'            "grad_separation":     {separation:.6f},')
print(f'            "effective_rank": {effective_rank:.1f},')
print(f'            "var_top10": {var_top10:.4f},')
print(f'            "dist_overlap": {dist_overlap:.4f},')
print(f'            "cohen_d": {cohen_d:.4f},')
print()
print(f"TracIn Concise AUC: {auc_tracin_concise:.4f}  (DataInf: 0.6918)")
print(f"TracIn Full AUC:    {auc_tracin_full:.4f}  (DataInf: 0.4249)")
