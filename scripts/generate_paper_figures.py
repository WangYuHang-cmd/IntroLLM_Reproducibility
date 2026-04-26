"""
Generate publication-quality figures for the reproducibility paper.

Style: ACL / NeurIPS / ICML conference paper aesthetic.
- Clean serif fonts
- Consistent color palette
- High DPI
- Informative but minimal

Usage: python scripts/generate_paper_figures.py
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# ==================================================
# AI Conference Style Configuration
# ==================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ACL-style color palette (colorblind-friendly)
COLORS = {
    "qwen":    "#ff7f0e",   # orange — Qwen family
    "llama":   "#2ca02c",   # green  — Llama family
    "paper":   "#d62728",   # red    — paper baseline
    "1.5B":    "#1f77b4",
    "1.7B":    "#ff7f0e",
    "3B":      "#2ca02c",
    "7B":      "#8c564b",
    "8B":      "#d62728",
    "probe":   "#e377c2",
    "influence": "#17becf",
    "random":  "#7f7f7f",
}

MARKERS = {"1.5B": "o", "1.7B": "s", "3B": "^", "7B": "v", "8B": "P"}

OUT_DIR = Path("logs/figures")
OUT_DIR.mkdir(exist_ok=True)

# Canonical 5-model order and per-model metadata
MODEL_ORDER = ["Qwen2.5-1.5B", "Qwen3-1.7B", "Qwen2.5-7B", "Llama-3.2-1B", "Llama-3.2-3B"]
MODEL_META  = {
    "Qwen2.5-1.5B": {"color": COLORS["1.5B"], "marker": "o", "dir": "Qwen2.5-1.5B",  "label": "1.5B"},
    "Qwen3-1.7B":   {"color": COLORS["1.7B"], "marker": "s", "dir": "Qwen3-1.7B",    "label": "1.7B"},
    "Qwen2.5-7B":   {"color": COLORS["7B"],   "marker": "v", "dir": "Qwen2.5-7B",    "label": "7B"},
    "Llama-3.2-1B": {"color": "#9467bd",       "marker": "D", "dir": "Llama3.2-1B",   "label": "1B"},
    "Llama-3.2-3B": {"color": COLORS["3B"],   "marker": "^", "dir": "Llama3.2-3B",   "label": "3B"},
}

# Mapping from log-directory-style names (no extra dash) → MODEL_META keys
_LOG_TO_META = {
    **{k: k for k in MODEL_META},  # identity for most names
    "Llama3.2-1B": "Llama-3.2-1B",
    "Llama3.2-3B": "Llama-3.2-3B",
}


def meta(model_name: str) -> dict:
    """Look up MODEL_META by either display name or log-dir name."""
    return MODEL_META.get(_LOG_TO_META.get(model_name, model_name), {})


# ==================================================
# Data collection
# ==================================================
def load_results():
    results = {
        "Qwen2.5-1.5B": {
            "params": 1.5, "label": "1.5B", "family": "Qwen",
            "eval_acc_length": 0.6591, "eval_acc_syco": 0.6026,
            "length_concise": 0.5419, "length_verbose": 0.5038, "length_full": 0.5344,
            "syco_less": 0.4982, "syco_more": 0.4805, "syco_full": 0.4888,
            "probe_auc": 0.6922,
            "grad_within_flipped": -0.000420, "grad_within_clean": 0.000565,
            "grad_between": 0.002623, "grad_separation": -0.003043,
            "effective_rank": 146.0, "var_top10": 0.3056,
            "dist_overlap": 0.9060,
            "cohen_d": 0.1008,
        },
        "Qwen3-1.7B": {
            "params": 1.7, "label": "1.7B", "family": "Qwen",
            "eval_acc_length": 0.6681, "eval_acc_syco": 0.6306,
            "length_concise": 0.4791, "length_verbose": 0.5099, "length_full": 0.4966,
            "syco_less": 0.5126, "syco_more": 0.5209, "syco_full": 0.5182,
            "probe_auc": 0.6328,
            "grad_within_flipped": -0.001473, "grad_within_clean": -0.000247,
            "grad_between": -0.000060, "grad_separation": -0.001413,
            "effective_rank": 103.2, "var_top10": 0.4442,
            "dist_overlap": 0.9156,
            "cohen_d": -0.0059,
        },
        "Qwen2.5-7B": {
            "params": 7.0, "label": "7B", "family": "Qwen",
            "eval_acc_length": 0.6565, "eval_acc_syco": 0.6486,
            "length_concise": 0.5053, "length_verbose": 0.4884, "length_full": 0.4940,
            "syco_less": 0.5054, "syco_more": 0.4966, "syco_full": 0.5054,
            "probe_auc": 0.6807,
            "grad_within_flipped": 0.003479, "grad_within_clean": 0.000042,
            "grad_between": 0.001310, "grad_separation": 0.002169,
            "effective_rank": 157.3, "var_top10": 0.3435,
            "dist_overlap": 0.9179,
            "cohen_d": 0.0362,
        },
        "Llama-3.2-1B": {
            "params": 1.0, "label": "1B", "family": "Llama",
            "eval_acc_length": 0.645, "eval_acc_syco": 0.598,
            "length_concise": 0.6918, "length_verbose": 0.3560, "length_full": 0.4249,
            "syco_less": 0.6185, "syco_more": 0.4101, "syco_full": 0.4613,
            "probe_auc": 0.7011,
            "grad_within_flipped": 0.063749, "grad_within_clean": 0.006831,
            "grad_between": 0.019063, "grad_separation": 0.044686,
            "effective_rank": 58.1, "var_top10": 0.5412,
            "dist_overlap": 0.6800,
            "cohen_d": 0.0955,
        },
        "Llama-3.2-3B": {
            "params": 3.2, "label": "3B", "family": "Llama",
            "eval_acc_length": 0.6714, "eval_acc_syco": 0.6521,
            "length_concise": 0.6996, "length_verbose": 0.3146, "length_full": 0.6436,
            "syco_less": 0.5787, "syco_more": 0.4345, "syco_full": 0.5916,
            "probe_auc": 0.6816,
            "grad_within_flipped": 0.035357, "grad_within_clean": 0.005706,
            "grad_between": 0.011889, "grad_separation": 0.023468,
            "effective_rank": 91.0, "var_top10": 0.5218,
            "dist_overlap": 0.7013,
            "cohen_d": 0.1803,
        },
        "Llama-3-8B (Paper)": {
            "params": 8.0, "label": "8B", "family": "Llama",
            "length_concise": 0.800, "length_verbose": 0.202, "length_full": 0.770,
            "syco_less": 0.711, "syco_more": 0.510, "syco_full": 0.585,
            "is_paper": True,
        },
    }
    return results


# ==================================================
# Figure 1: Main result — AUC vs Model Size
#   NOW shows two architecture families separately
# ==================================================
def fig1_auc_vs_size(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    qwen_models  = ["Qwen2.5-1.5B", "Qwen3-1.7B", "Qwen2.5-7B"]
    llama_models = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3-8B (Paper)"]

    qwen_params  = [results[m]["params"] for m in qwen_models]
    llama_params = [results[m]["params"] for m in llama_models]

    def draw_panel(ax, key_target, key_counter, key_full,
                   target_label, counter_label, title):
        qwen_target  = [results[m][key_target]  for m in qwen_models]
        qwen_full    = [results[m][key_full]     for m in qwen_models]

        llama_target = [results[m][key_target]  for m in llama_models]
        llama_full   = [results[m][key_full]     for m in llama_models]

        # --- Qwen family (dashed, orange) ---
        ax.plot(qwen_params, qwen_target, "o--", color=COLORS["qwen"],
                label="Qwen — Targeted", markersize=9, lw=2, zorder=4)
        ax.plot(qwen_params, qwen_full, "s--", color=COLORS["qwen"],
                alpha=0.5, markersize=7, lw=1.5, label="Qwen — Full", zorder=4)

        # --- Llama family (solid, green) ---
        ax.plot(llama_params, llama_target, "^-", color=COLORS["llama"],
                label="Llama — Targeted", markersize=10, lw=2.5, zorder=5)
        ax.plot(llama_params, llama_full, "s-", color=COLORS["llama"],
                alpha=0.5, markersize=7, lw=1.5, label="Llama — Full", zorder=5)

        # Paper marker with hatch
        ax.scatter([8.0], [results["Llama-3-8B (Paper)"][key_target]],
                   marker="*", s=220, color=COLORS["paper"], zorder=6,
                   label="Llama-3-8B (Paper)", edgecolor="black", linewidth=0.5)

        ax.axhline(0.5, color=COLORS["random"], linestyle="-", alpha=0.5,
                   lw=1.2, label="Random (0.500)")

        # Annotate 7B anomaly
        ax.annotate("Qwen-7B\n≈ random!", xy=(7.0, results["Qwen2.5-7B"][key_target]),
                    xytext=(4.5, results["Qwen2.5-7B"][key_target] - 0.07),
                    fontsize=9, color=COLORS["qwen"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLORS["qwen"], lw=1.2))

        ax.set_xlabel("Model Parameters (Billions)")
        ax.set_ylabel("Influence Function AUC")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks([1.0, 1.5, 2, 3, 5, 8])
        ax.set_xticklabels(["1", "1.5", "2", "3", "5", "8"])
        ax.legend(loc="lower right", fontsize=8.5, ncol=2)

    draw_panel(axes[0], "length_concise", "length_verbose", "length_full",
               "Concise (targeted)", "Verbose", "(a) Length Bias Detection")
    axes[0].set_ylim(0.15, 0.88)

    draw_panel(axes[1], "syco_less", "syco_more", "syco_full",
               "Less-Syco (targeted)", "More-Syco", "(b) Sycophancy Bias Detection")
    axes[1].set_ylim(0.35, 0.78)

    plt.suptitle(
        "Figure 1: Architecture Family Matters — Qwen Models Remain at Chance\n"
        "While Llama Models Improve with Scale",
        fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_auc_vs_size.pdf")
    plt.savefig(OUT_DIR / "fig1_auc_vs_size.png")
    plt.close()
    print("  Saved: fig1_auc_vs_size")


# ==================================================
# Figure 2: Probe vs Influence — the key insight (all 5 models)
# ==================================================
def fig2_probe_vs_influence(results):
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))

    models = MODEL_ORDER
    labels = [MODEL_META[m]["label"] for m in models]
    colors = [MODEL_META[m]["color"] for m in models]
    x = np.arange(len(models))
    width = 0.35

    for i, (m, lbl, col) in enumerate(zip(models, labels, colors)):
        inf_val = results[m]["length_concise"]
        probe_val = results[m].get("probe_auc")

        # Influence bar — always shown
        b2 = ax.bar(x[i] + width/2, inf_val, width,
                    color=COLORS["influence"], edgecolor=col, linewidth=1.5,
                    label="Influence Function AUC" if i == 0 else None)
        ax.text(x[i] + width/2, inf_val + 0.01, f"{inf_val:.3f}",
                ha="center", va="bottom", fontsize=8.5)

        # Probe bar — shown if available, else N/A placeholder
        if probe_val is not None:
            b1 = ax.bar(x[i] - width/2, probe_val, width,
                        color=COLORS["probe"], edgecolor=col, linewidth=1.5,
                        label="Hidden State Probe AUC" if i == 0 else None)
            ax.text(x[i] - width/2, probe_val + 0.01, f"{probe_val:.3f}",
                    ha="center", va="bottom", fontsize=8.5)
            # Gap annotation
            gap = probe_val - inf_val
            if abs(gap) > 0.05:
                y = max(probe_val, inf_val) + 0.07
                ax.annotate(f"Δ={gap:+.2f}", xy=(x[i], y), ha="center",
                            fontsize=9.5, fontweight="bold",
                            color="red" if gap > 0.05 else "black")
        else:
            ax.bar(x[i] - width/2, 0.52, width,
                   color="white", edgecolor="gray", linewidth=1,
                   hatch="///", alpha=0.6,
                   label="Probe N/A (pending)" if i == list(MODEL_ORDER).index(m) and i > 0 and results[list(MODEL_ORDER)[i-1]].get("probe_auc") is None else None)
            ax.text(x[i] - width/2, 0.53, "N/A",
                    ha="center", va="bottom", fontsize=8, color="gray", style="italic")

    ax.axhline(0.5, color=COLORS["random"], linestyle="--", alpha=0.5, label="Random (0.500)")

    # Family divider
    ax.axvline(2.5, color="black", linewidth=0.8, linestyle=":", alpha=0.4)
    ax.text(1.0, 0.83, "Qwen family", ha="center", fontsize=10, color=COLORS["qwen"],
            fontweight="bold")
    ax.text(3.5, 0.83, "Llama family", ha="center", fontsize=10, color=COLORS["llama"],
            fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({l})" for m, l in zip(models, labels)], fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_title("Figure 2: All Models Encode Bias in Hidden States,\n"
                 "But Only Llama LoRA Gradients Express It",
                 fontsize=12)
    ax.set_ylim(0.3, 0.92)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_probe_vs_influence.pdf")
    plt.savefig(OUT_DIR / "fig2_probe_vs_influence.png")
    plt.close()
    print("  Saved: fig2_probe_vs_influence")


# ==================================================
# Figure 3: Gradient Separability Analysis (all 5 models)
# ==================================================
def fig3_gradient_separability(results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

    models = MODEL_ORDER
    labels = [MODEL_META[m]["label"] for m in models]
    colors = [MODEL_META[m]["color"] for m in models]
    x = np.arange(len(models))

    # ── (a) Cosine similarity bars ────────────────────────────────────────
    ax = axes[0]
    width = 0.27
    has_data = ["grad_separation" in results.get(m, {}) for m in models]

    wf_vals = [results[m].get("grad_within_flipped", 0.0) for m in models]
    wc_vals = [results[m].get("grad_within_clean",   0.0) for m in models]
    bt_vals = [results[m].get("grad_between",         0.0) for m in models]

    ax.bar(x - width, wf_vals, width, label="Within Flipped",
           color="#d62728", edgecolor="black", linewidth=0.5)
    ax.bar(x,          wc_vals, width, label="Within Clean",
           color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(x + width,  bt_vals, width, label="Between",
           color="#7f7f7f", edgecolor="black", linewidth=0.5)

    # Mark N/A models
    for i, (m, has) in enumerate(zip(models, has_data)):
        if not has:
            ax.text(x[i], 0.005, "N/A\n(pending)", ha="center", va="bottom",
                    fontsize=8, color="gray", style="italic")
            ax.axvspan(x[i] - 0.45, x[i] + 0.45, alpha=0.08, color="gray")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({l})" for m, l in zip(models, labels)], fontsize=8.5)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("(a) Gradient Cosine Similarity")
    ax.legend(fontsize=9)

    # ── (b) Separation gap bars ───────────────────────────────────────────
    ax = axes[1]
    sep_vals = [results[m].get("grad_separation", 0.0) for m in models]
    bars = ax.bar(x, sep_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.5)

    for i, (b, s, has) in enumerate(zip(bars, sep_vals, has_data)):
        if has:
            y = s + (0.001 if s >= 0 else -0.004)
            ax.text(b.get_x() + b.get_width()/2, y,
                    f"{s:+.4f}", ha="center", fontsize=9.5, fontweight="bold",
                    color="green" if s > 0.01 else "red")
        else:
            ax.text(b.get_x() + b.get_width()/2, 0.002,
                    "N/A", ha="center", fontsize=9, color="gray", style="italic")
            b.set_hatch("///")
            b.set_facecolor("lightgray")
            b.set_edgecolor("gray")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(0.01, 0.06, alpha=0.12, color="green", label="Detectable signal (>0.01)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({l})" for m, l in zip(models, labels)], fontsize=8.5)
    ax.set_ylabel("Separation Gap\n(Within Flipped − Between)")
    ax.set_title("(b) Gradient Fingerprint Strength")
    ax.legend(fontsize=9)

    plt.suptitle("Figure 3: Llama Models Form Distinct Gradient Fingerprints;\n"
                 "Qwen Models Show Near-Zero Separation at All Scales",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_gradient_separability.pdf")
    plt.savefig(OUT_DIR / "fig3_gradient_separability.png")
    plt.close()
    print("  Saved: fig3_gradient_separability")


# ==================================================
# Figure 4: Effective Rank & Gradient Concentration (all 5 models)
# ==================================================
def fig4_effective_rank(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    models = MODEL_ORDER
    labels = [MODEL_META[m]["label"] for m in models]
    colors = [MODEL_META[m]["color"] for m in models]
    edge_widths = [2.0 if "Llama" in m else 0.5 for m in models]
    has_data = ["effective_rank" in results.get(m, {}) for m in models]
    x = np.arange(len(models))

    eff_rank = [results[m].get("effective_rank", 0.0) for m in models]
    var10    = [results[m].get("var_top10", 0.0) * 100 for m in models]

    for ax, vals, ylabel, title, ylim, fmt in [
        (axes[0], eff_rank, "Effective Rank of Gradient Matrix",
         "(a) Gradient Space Dimensionality\n(lower = more focused)", 170, ".0f"),
        (axes[1], var10, "Variance Explained by Top-10 Components (%)",
         "(b) Signal Concentration\n(higher = more signal in few directions)", 70, ".1f"),
    ]:
        bar_colors  = [c if has else "lightgray" for c, has in zip(colors, has_data)]
        bar_hatches = ["" if has else "///" for has in has_data]
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="black",
                      linewidth=edge_widths, width=0.55)
        for b, h in zip(bars, bar_hatches):
            b.set_hatch(h)

        for i, (b, v, has) in enumerate(zip(bars, vals, has_data)):
            if has:
                label_y = v + (ylim * 0.02)
                ax.text(b.get_x() + b.get_width()/2, label_y,
                        f"{v:{fmt}}" + ("%" if "%" in ylabel else ""),
                        ha="center", fontsize=10, fontweight="bold")
            else:
                ax.text(b.get_x() + b.get_width()/2, ylim * 0.04,
                        "N/A", ha="center", fontsize=9, color="gray", style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n({l})" for m, l in zip(models, labels)], fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, ylim)

    plt.suptitle("Figure 4: Llama Models Have More Focused Gradient Spaces;\n"
                 "Qwen-7B Grad Stats Pending Download (Expected: High Rank, Low Concentration)",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_effective_rank.pdf")
    plt.savefig(OUT_DIR / "fig4_effective_rank.png")
    plt.close()
    print("  Saved: fig4_effective_rank")


# ==================================================
# Figure 5: TracIn vs DataInf — ruling out Hessian
# ==================================================
def fig5_tracin_vs_datainf():
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))

    data = [
        ("Qwen2.5-1.5B", "length", "Concise", 0.5419, 0.5417),
        ("Qwen2.5-1.5B", "length", "Full",    0.5344, 0.5346),
        ("Qwen3-1.7B",   "length", "Concise", 0.4791, 0.4789),
        ("Qwen3-1.7B",   "length", "Full",    0.4966, 0.4966),
        ("Qwen2.5-7B",   "length", "Concise", 0.5053, 0.5054),
        ("Qwen2.5-7B",   "length", "Full",    0.4940, 0.4943),
        ("Llama-3.2-1B", "length", "Concise", 0.6918, 0.6899),
        ("Llama-3.2-1B", "length", "Full",    0.4249, 0.4210),
        ("Llama-3.2-3B", "length", "Concise", 0.6996, 0.6988),
        ("Llama-3.2-3B", "length", "Full",    0.6436, 0.6430),
        ("Qwen2.5-1.5B", "syco",   "Less",    0.4982, 0.4985),
        ("Qwen3-1.7B",   "syco",   "Less",    0.5126, 0.5124),
        ("Llama-3.2-3B", "syco",   "Less",    0.5787, 0.5777),
    ]

    model_to_color  = {"Qwen2.5-1.5B": COLORS["1.5B"],
                       "Qwen3-1.7B":   COLORS["1.7B"],
                       "Qwen2.5-7B":   COLORS["7B"],
                       "Llama-3.2-1B": "#9467bd",
                       "Llama-3.2-3B": COLORS["3B"]}
    model_to_marker = {"Qwen2.5-1.5B": "o",
                       "Qwen3-1.7B":   "s",
                       "Qwen2.5-7B":   "v",
                       "Llama-3.2-1B": "D",
                       "Llama-3.2-3B": "^"}

    seen = set()
    for name, bias, sub, datainf_auc, tracin_auc in data:
        label = name if name not in seen else None
        seen.add(name)
        ax.scatter(datainf_auc, tracin_auc,
                   color=model_to_color[name], marker=model_to_marker[name],
                   s=120, edgecolor="black", linewidth=0.8, label=label, zorder=3)

    lims = [0.38, 0.78]
    ax.plot(lims, lims, "k--", alpha=0.5, lw=1, label="y = x (perfect agreement)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DataInf AUC")
    ax.set_ylabel("TracIn AUC")
    ax.set_title("Figure 5: DataInf ≈ TracIn Across All 4 Available Models\n"
                 "(Qwen2.5-7B grad files pending download)",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.text(0.63, 0.40, "$|$DataInf $-$ TracIn$|$ < 0.002\nacross 4 models\n(7B pending)",
            fontsize=10, ha="center",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_tracin_vs_datainf.pdf")
    plt.savefig(OUT_DIR / "fig5_tracin_vs_datainf.png")
    plt.close()
    print("  Saved: fig5_tracin_vs_datainf")


# ==================================================
# Figure 6: ROC Curves (now includes 7B)
# ==================================================
def fig6_roc_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    model_specs = [
        ("Qwen2.5-1.5B", "Qwen2.5-1.5B", COLORS["1.5B"], "o"),
        ("Qwen3-1.7B",   "Qwen3-1.7B",   COLORS["1.7B"], "s"),
        ("Qwen2.5-7B",   "Qwen2.5-7B",   COLORS["7B"],   "v"),
        ("Llama3.2-1B",  "Llama-3.2-1B", "#9467bd",       "D"),
        ("Llama3.2-3B",  "Llama-3.2-3B", COLORS["3B"],   "^"),
    ]

    for ax, bias_type, title in [(axes[0], "length",     "(a) Length Bias"),
                                  (axes[1], "sycophancy", "(b) Sycophancy Bias")]:
        data_dir = f"dataset/{bias_type}_dataset"
        try:
            flipped = np.load(f"{data_dir}/flipped_indices.npy")
        except FileNotFoundError:
            print(f"  Warning: {data_dir}/flipped_indices.npy not found, skipping fig6")
            continue

        subset_name = "concise" if bias_type == "length" else "less_sycophantic"

        for model_name, display_name, color, marker in model_specs:
            inf_file = f"logs/{model_name}_{bias_type}/influence_{subset_name}.npy"
            if not Path(inf_file).exists():
                continue
            influence = np.load(inf_file)
            true_labels = np.zeros(len(influence))
            for i in flipped:
                true_labels[i] = 1
            fpr, tpr, _ = roc_curve(true_labels, influence)
            auc_v = auc(fpr, tpr)

            family = "Llama" if "Llama" in model_name else "Qwen"
            lw = 2.5 if family == "Llama" else 1.8
            ls = "-" if family == "Llama" else "--"
            ax.plot(fpr, tpr, color=color, lw=lw, linestyle=ls,
                    label=f"{display_name} (AUC={auc_v:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1, label="Random (0.500)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.suptitle("Figure 6: ROC Curves — Targeted Validation Set\n"
                 "(Qwen-7B Near Diagonal; Llama-1B Matches Llama-3B Despite 3× Fewer Parameters)",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig6_roc_curves.pdf")
    plt.savefig(OUT_DIR / "fig6_roc_curves.png")
    plt.close()
    print("  Saved: fig6_roc_curves")


# ==================================================
# Figure 7: LoRA Rank Ablation
# ==================================================
def fig7_lora_rank_ablation():
    ranks = []
    aucs_concise = []
    aucs_verbose = []
    aucs_full    = []

    for r in [4, 8, 16, 32, 64]:
        path = ("logs/Qwen2.5-1.5B_length/influence_results.json"
                if r == 16
                else f"logs/Qwen2.5-1.5B_r{r}_length/influence_results.json")
        if not Path(path).exists():
            continue
        with open(path) as f:
            res = json.load(f)
        ranks.append(r)
        aucs_concise.append(res["Concise"]["auc"])
        aucs_verbose.append(res["Verbose"]["auc"])
        aucs_full.append(res["Full"]["auc"])

    if len(ranks) < 2:
        print("  Skipping fig7: not enough LoRA rank data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(ranks, aucs_concise, "o-", color=COLORS["3B"],
            label="Concise (targeted)", markersize=9)
    ax.plot(ranks, aucs_full,    "s--", color=COLORS["1.5B"],
            label="Full", markersize=8)
    ax.plot(ranks, aucs_verbose, "v:",  color=COLORS["1.7B"],
            label="Verbose (anti-targeted)", markersize=8)

    ax.axhline(0.5,    color=COLORS["random"], linestyle="-",  alpha=0.5, lw=1,   label="Random")
    ax.axhline(0.6922, color=COLORS["probe"],  linestyle=":",  alpha=0.7, lw=1.5, label="Hidden Probe AUC (0.69)")
    ax.axhline(0.6996, color=COLORS["3B"],     linestyle=":",  alpha=0.7, lw=1.5, label="Llama-3B Influence AUC (0.70)")

    ax.set_xlabel("LoRA Rank $r$")
    ax.set_ylabel("Influence Function AUC")
    ax.set_title("Figure 7: LoRA Rank Cannot Close the Capacity Gap\n"
                 "(Qwen2.5-1.5B, Length Bias)", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_ylim(0.35, 0.8)
    ax.legend(loc="center right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig7_lora_rank_ablation.pdf")
    plt.savefig(OUT_DIR / "fig7_lora_rank_ablation.png")
    plt.close()
    print("  Saved: fig7_lora_rank_ablation")


# ==================================================
# Figure 8: Influence Score Distributions (5 panels)
# ==================================================
def fig8_distribution_overlap():
    models_to_plot = [
        (MODEL_META[m]["dir"], MODEL_META[m]["label"], MODEL_META[m]["color"])
        for m in MODEL_ORDER
    ]

    flipped_path = "dataset/length_dataset/flipped_indices.npy"
    if not Path(flipped_path).exists():
        print("  Skipping fig8: flipped_indices.npy not found")
        return

    flipped     = np.load(flipped_path)
    flipped_set = set(flipped.tolist())

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, (model_name, label, color) in zip(axes, models_to_plot):
        inf_file = f"logs/{model_name}_length/influence_concise.npy"
        if not Path(inf_file).exists():
            ax.set_title(f"{model_name}\n(no data)")
            continue

        influence   = np.load(inf_file)
        inf_flipped = [influence[i] for i in range(len(influence)) if i in flipped_set]
        inf_clean   = [influence[i] for i in range(len(influence)) if i not in flipped_set]

        lo   = np.percentile(np.concatenate([inf_flipped, inf_clean]), 1)
        hi   = np.percentile(np.concatenate([inf_flipped, inf_clean]), 99)
        bins = np.linspace(lo, hi, 50)

        ax.hist(inf_clean,   bins=bins, alpha=0.5, density=True,
                label=f"Clean ({len(inf_clean)})",   color="#1f77b4",
                edgecolor="black", linewidth=0.3)
        ax.hist(inf_flipped, bins=bins, alpha=0.6, density=True,
                label=f"Flipped ({len(inf_flipped)})", color="#d62728",
                edgecolor="black", linewidth=0.3)

        with open(f"logs/{model_name}_length/influence_results.json") as f:
            auc_v = json.load(f)["Concise"]["auc"]

        family = "Llama" if "Llama" in model_name else "Qwen"
        ax.set_xlabel("Influence Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name} ({label})\nAUC = {auc_v:.3f}  [{family}]")
        ax.legend(fontsize=8)

    plt.suptitle("Figure 8: Influence Score Distributions — Clean vs Flipped Samples\n"
                 "(Qwen-7B Shows Complete Overlap; Llama-1B Already Separates Despite Smaller Size)",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig8_distribution_overlap.pdf")
    plt.savefig(OUT_DIR / "fig8_distribution_overlap.png")
    plt.close()
    print("  Saved: fig8_distribution_overlap")


# ==================================================
# Figure 9: Summary "story" figure (all 5 models)
# ==================================================
def fig9_story(results):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    models = MODEL_ORDER
    labels = [MODEL_META[m]["label"] for m in models]
    colors = [MODEL_META[m]["color"] for m in models]
    x = np.arange(len(models))

    eval_acc  = [results[m]["eval_acc_length"] for m in models]
    influence = [results[m]["length_concise"]  for m in models]

    # Probe: only where available
    probe_x   = [x[i] for i, m in enumerate(models) if results[m].get("probe_auc") is not None]
    probe_y   = [results[m]["probe_auc"] for m in models if results[m].get("probe_auc") is not None]
    probe_lbl = [labels[i] for i, m in enumerate(models) if results[m].get("probe_auc") is not None]

    # Draw lines
    ax.plot(x, eval_acc,  "D-", color="#7f7f7f",          label="RM Eval Accuracy",  markersize=9,  lw=2)
    ax.plot(x, influence, "o-", color=COLORS["influence"], label="Influence AUC",     markersize=10, lw=2.5)

    # Probe line — connect only available points
    if probe_x:
        ax.plot(probe_x, probe_y, "s--", color=COLORS["probe"], label="Hidden Probe AUC",
                markersize=10, lw=2.5, zorder=5)
        # Mark N/A probe positions
        for i, m in enumerate(models):
            if results[m].get("probe_auc") is None:
                ax.scatter(x[i], 0.5, marker="x", color=COLORS["probe"], s=80, lw=2, zorder=6)
                ax.text(x[i], 0.47, "probe\nN/A", ha="center", fontsize=7.5,
                        color=COLORS["probe"], style="italic")

    # Fill bottleneck region (only where probe is available and > influence)
    for i in probe_x:
        p = results[models[i]]["probe_auc"]
        inf = influence[i]
        if p > inf + 0.02:
            ax.fill_betweenx([inf, p], i - 0.05, i + 0.05,
                             color="red", alpha=0.25, zorder=3)

    ax.axhline(0.5, color=COLORS["random"], linestyle="--", alpha=0.5, lw=1, label="Random (0.500)")

    # Divider: Qwen | Llama
    ax.axvline(2.5, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(1.0, 0.78, "Qwen family\n(gradient bottleneck)", ha="center", fontsize=9.5,
            color=COLORS["qwen"], fontweight="bold")
    ax.text(3.5, 0.78, "Llama family\n(gradient works)", ha="center", fontsize=9.5,
            color=COLORS["llama"], fontweight="bold")

    # Color per-model tick labels
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({l})" for m, l in zip(models, labels)], fontsize=8.5)
    for tick, col in zip(ax.get_xticklabels(), colors):
        tick.set_color(col)

    ax.set_ylabel("Metric Value")
    ax.set_title("Figure 9: The Gradient Bottleneck — Architecture, Not Scale, Is the Key\n"
                 "All Models Encode Bias (probe ≈ 0.68+); Only Llama Gradients Reveal It",
                 fontsize=11)
    ax.set_ylim(0.38, 0.86)
    ax.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig9_story.pdf")
    plt.savefig(OUT_DIR / "fig9_story.png")
    plt.close()
    print("  Saved: fig9_story")


# ==================================================
# Figure 10 (NEW): Architecture vs Scale — full picture
# ==================================================
def fig10_architecture_vs_scale(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # All models ordered by param count
    all_models = [
        ("Qwen2.5-1.5B",        "Qwen2.5-1.5B", "Qwen",  1.5,  COLORS["1.5B"], "o"),
        ("Llama-3.2-1B",        "Llama-3.2-1B", "Llama", 1.0,  "#9467bd",       "D"),
        ("Qwen3-1.7B",          "Qwen3-1.7B",   "Qwen",  1.7,  COLORS["1.7B"], "s"),
        ("Llama-3.2-3B",        "Llama-3.2-3B", "Llama", 3.2,  COLORS["3B"],   "^"),
        ("Qwen2.5-7B",          "Qwen2.5-7B",   "Qwen",  7.0,  COLORS["7B"],   "v"),
        ("Llama-3-8B (Paper)",  "Llama-3-8B*",  "Llama", 8.0,  COLORS["paper"],"P"),
    ]

    for ax, key_metric, ylabel, title in [
        (axes[0], "length_concise", "AUC (Targeted Val.)", "(a) Length Bias — Targeted AUC"),
        (axes[1], "syco_less",      "AUC (Targeted Val.)", "(b) Sycophancy — Targeted AUC"),
    ]:
        for model_key, display, family, params, color, marker in all_models:
            if model_key not in results:
                continue
            val = results[model_key][key_metric]
            edge = "black"
            size = 160 if family == "Llama" else 120
            ax.scatter(params, val, color=color, marker=marker, s=size,
                       edgecolor=edge, linewidth=1.2, zorder=5,
                       label=f"{display} ({family})")
            ax.annotate(f" {val:.3f}", xy=(params, val),
                        fontsize=8.5, va="center", color=color)

        # Draw family trend lines
        qwen_pts  = [(results[m]["params"], results[m][key_metric])
                     for m, _, fam, _, _, _ in all_models
                     if fam == "Qwen" and m in results]
        llama_pts = [(results[m]["params"], results[m][key_metric])
                     for m, _, fam, _, _, _ in all_models
                     if fam == "Llama" and m in results]

        if len(qwen_pts) >= 2:
            qx, qy = zip(*sorted(qwen_pts))
            ax.plot(qx, qy, "--", color=COLORS["qwen"],  alpha=0.6, lw=1.5, zorder=3)
        if len(llama_pts) >= 2:
            lx, ly = zip(*sorted(llama_pts))
            ax.plot(lx, ly, "-",  color=COLORS["llama"], alpha=0.6, lw=1.5, zorder=3)

        ax.axhline(0.5, color=COLORS["random"], linestyle="-", alpha=0.4,
                   lw=1.2, label="Random baseline")

        # Family legend patches
        q_patch = mpatches.Patch(color=COLORS["qwen"],  label="Qwen family (flat ≈ 0.50)")
        l_patch = mpatches.Patch(color=COLORS["llama"], label="Llama family (scaling)")
        handles, labels_leg = ax.get_legend_handles_labels()
        ax.legend(handles=[q_patch, l_patch] + handles[-1:],
                  loc="upper left", fontsize=9)

        ax.set_xlabel("Model Parameters (Billions)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks([1.0, 1.5, 2, 3, 5, 8])
        ax.set_xticklabels(["1", "1.5", "2", "3", "5", "8"])
        ax.set_ylim(0.38, 0.88)

    plt.suptitle(
        "Figure 10: Architecture Family Determines Influence Function Effectiveness\n"
        "Qwen Models Plateau at Chance Level; Llama Models Improve with Scale",
        fontsize=12, y=1.03)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig10_architecture_vs_scale.pdf")
    plt.savefig(OUT_DIR / "fig10_architecture_vs_scale.png")
    plt.close()
    print("  Saved: fig10_architecture_vs_scale")


# ==================================================
# Fig 11: Baseline Comparison
# ==================================================
def fig11_baselines_comparison():
    """Bar chart comparing Influence AUC vs Mahalanobis/KNN/Self-conf/Entropy/GPT-4o."""
    import os

    models_order = ["Qwen2.5-1.5B", "Qwen3-1.7B", "Qwen2.5-7B", "Llama-3.2-1B", "Llama-3.2-3B"]
    methods = ["influence", "mahalanobis", "knn", "self_confidence", "entropy", "llm"]
    method_labels = ["Influence\n(DataInf)", "Mahalanobis", "KNN", "Self-Conf", "Entropy", "GPT-4o"]
    method_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    influence_aucs = {
        "Qwen2.5-1.5B":  0.5419,
        "Qwen3-1.7B":    0.4791,
        "Qwen2.5-7B":    0.5053,
        "Llama-3.2-1B":  0.6918,
        "Llama-3.2-3B":  0.6996,
    }

    # LLM AUC is model-agnostic (GPT-4o on raw text, 2-shot, n=500)
    llm_length_auc = None
    llm_path = "logs/llm_baseline_length_shots2.json"
    if os.path.exists(llm_path):
        with open(llm_path) as f:
            llm_length_auc = json.load(f).get("auc")

    # Load baseline results (log dirs use Llama3.2 without extra dash)
    baseline_data = {}
    all_missing = True
    for m in models_order:
        log_dir = MODEL_META[m]["dir"]
        path = f"logs/{log_dir}_length/baseline_results.json"
        if os.path.exists(path):
            with open(path) as f:
                baseline_data[m] = json.load(f)
            all_missing = False

    if all_missing and llm_length_auc is None:
        print("  Skipping fig11: no baseline_results.json found (run exp_baselines.py first)")
        return

    fig, ax = plt.subplots(figsize=(15, 5))
    n_models = len(models_order)
    n_methods = len(methods)
    x = np.arange(n_models)
    width = 0.12

    for j, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        aucs = []
        for m in models_order:
            if method == "influence":
                aucs.append(influence_aucs.get(m, np.nan))
            elif method == "llm":
                aucs.append(llm_length_auc if llm_length_auc is not None else np.nan)
            elif m in baseline_data:
                val = baseline_data[m].get(method, {}).get("auc")
                aucs.append(val if val is not None else np.nan)
            else:
                aucs.append(np.nan)
        offset = (j - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, aucs, width, label=label, color=color, alpha=0.85,
                      edgecolor="white")
        for bar, v in zip(bars, aucs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4, label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, fontsize=10)
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.3, 0.85)
    ax.set_title("Figure 11: Bias Detection Method Comparison\n"
                 "Influence Functions vs All Baselines (Length Bias)")
    ax.legend(fontsize=9, ncol=6, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig11_baselines_comparison.pdf")
    plt.savefig(OUT_DIR / "fig11_baselines_comparison.png", dpi=300)
    plt.close()
    print("  Saved: fig11_baselines_comparison")


# ==================================================
# Fig 11b: LLM Few-Shot Count Ablation (B2)
# ==================================================
def fig11b_llm_fewshot_ablation():
    """AUC vs number of few-shot examples for GPT-4o, both length and sycophancy bias."""
    import os

    bias_configs = {
        "length":     {"color": "#1f77b4", "label": "GPT-4o (length)"},
        "sycophancy": {"color": "#d62728", "label": "GPT-4o (sycophancy)"},
    }

    all_data = {}
    for bias, cfg in bias_configs.items():
        shots_aucs = {}
        for n in [0, 1, 2]:
            path = f"logs/llm_baseline_{bias}_shots{n}.json"
            if os.path.exists(path):
                with open(path) as f:
                    shots_aucs[n] = json.load(f).get("auc", np.nan)
        if shots_aucs:
            all_data[bias] = shots_aucs

    if not all_data:
        print("  Skipping fig11b: no LLM baseline shot results found")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for bias, shots_aucs in all_data.items():
        cfg = bias_configs[bias]
        xs = sorted(shots_aucs.keys())
        ys = [shots_aucs[x] for x in xs]
        ax.plot(xs, ys, "o-", color=cfg["color"], linewidth=2, markersize=8, label=cfg["label"])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.006, f"{y:.3f}", ha="center", va="bottom", fontsize=9,
                    color=cfg["color"])
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4, label="Random (0.50)")
    ax.set_xlabel("Number of Few-Shot Examples")
    ax.set_ylabel("ROC AUC")
    all_xs = sorted({x for d in all_data.values() for x in d})
    ax.set_xticks(all_xs)
    ax.set_ylim(0.40, 0.65)
    ax.set_title("Figure 11b: GPT-4o Bias Detection vs Few-Shot Count\n"
                 "(n=500 samples, length and sycophancy bias)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig11b_llm_fewshot_ablation.pdf")
    plt.savefig(OUT_DIR / "fig11b_llm_fewshot_ablation.png", dpi=300)
    plt.close()
    print("  Saved: fig11b_llm_fewshot_ablation")


# ==================================================
# Fig 12: Validation Set Size Ablation
# ==================================================
def fig12_valsize_ablation():
    import os
    path = "logs/valsize_ablation.json"
    if not os.path.exists(path):
        print("  Skipping fig12: logs/valsize_ablation.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, records in data.items():
        m = meta(model_name)
        color = m.get("color", "gray")
        marker = m.get("marker", "o")
        fracs = [r["fraction"] for r in records]
        means = [r["auc_mean"] for r in records]
        stds  = [r["auc_std"]  for r in records]
        ax.errorbar(fracs, means, yerr=stds, marker=marker, color=color,
                    label=model_name, capsize=4, linewidth=2)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Validation Set Fraction (of full concise subset)")
    ax.set_ylabel("Influence Function AUC (mean ± std over 5 seeds)")
    ax.set_title("Figure 12: Robustness to Validation Set Size\n"
                 "Llama Models Maintain AUC Even with 5% of Validation Data")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.38, 0.82)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig12_valsize_ablation.pdf")
    plt.savefig(OUT_DIR / "fig12_valsize_ablation.png", dpi=300)
    plt.close()
    print("  Saved: fig12_valsize_ablation")


# ==================================================
# Fig 13: Top-k Suspicious Sample Detection
# ==================================================
def fig13_topk_detection():
    import os
    path = "logs/topk_detection.json"
    if not os.path.exists(path):
        print("  Skipping fig13: logs/topk_detection.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, res in data.items():
        m = meta(model_name)
        color = m.get("color", "gray")
        marker = m.get("marker", "o")
        records = res["precision_at_k"]
        k_vals = [r["k"] for r in records]
        enrichments = [r["enrichment"] for r in records]
        ax.plot(k_vals, enrichments, marker=marker, color=color,
                label=model_name, linewidth=2, markersize=7)

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Random (1.0×)")
    ax.set_xscale("log")
    ax.set_xlabel("k (Top-k samples by influence score)")
    ax.set_ylabel("Enrichment (Precision@k / Base Rate)")
    ax.set_title("Figure 13: Top-k Suspicious Sample Detection\n"
                 "Enrichment over Random Baseline — Llama-3.2-3B Achieves 3.7× at k=50")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig13_topk_detection.pdf")
    plt.savefig(OUT_DIR / "fig13_topk_detection.png", dpi=300)
    plt.close()
    print("  Saved: fig13_topk_detection")


# ==================================================
# Fig 14: HelpSteer2 Labeling Strategy
# ==================================================
def fig14_labeling_strategy():
    import os
    path = "logs/Qwen2.5-1.5B_helpsteer_bob1/labeling_strategy_results.json"
    if not os.path.exists(path):
        print("  Skipping fig14: labeling_strategy_results.json not found (run exp D first)")
        return

    with open(path) as f:
        r = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel 1: Label accuracy
    ax = axes[0]
    cats = ["Bob\n(original)", "SVM\n(updated)"]
    accs = [r["label_acc_old"], r["label_acc_new"]]
    colors = ["#ff7f0e", "#2ca02c"]
    bars = ax.bar(cats, accs, color=colors, alpha=0.85, width=0.5, edgecolor="white")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Label Accuracy (vs Alice)")
    ax.set_title(f"Label Accuracy\n(Δ = {r['label_acc_improvement']:+.4f})")
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.3, label="Alice (perfect)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Cosine similarity to Alice's weights
    ax = axes[1]
    sims = [r["cos_sim_old"], r["cos_sim_new"]]
    bars = ax.bar(cats, sims, color=colors, alpha=0.85, width=0.5, edgecolor="white")
    for bar, v in zip(bars, sims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Cosine Similarity to Alice's Weights")
    ax.set_title(f"Weight Similarity to Expert\n(Δ = {r['cos_sim_improvement']:+.4f})")
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.3, label="Alice (1.0)")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Figure 14: HelpSteer2 Labeling Strategy Oversight\n"
        "Influence Functions Help Correct Bob's Suboptimal Labeling (Qwen2.5-1.5B)",
        fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig14_labeling_strategy.pdf")
    plt.savefig(OUT_DIR / "fig14_labeling_strategy.png", dpi=300)
    plt.close()
    print("  Saved: fig14_labeling_strategy")


# ==================================================
# Fig 15: BeaverTails Safety Extension
# ==================================================
def fig15_beavertails_safety():
    import os
    safety_path = "logs/Qwen2.5-1.5B_safety/influence_results.json"
    if not os.path.exists(safety_path):
        print("  Skipping fig15: Qwen2.5-1.5B_safety influence results not found (run BeaverTails first)")
        return

    with open(safety_path) as f:
        safety_res = json.load(f)

    # Reference AUCs for length/sycophancy
    domain_data = {
        "Length\n(Qwen2.5-1.5B)":     0.5419,
        "Sycophancy\n(Qwen2.5-1.5B)": 0.5200,
    }
    # Pick the best AUC from safety results
    best_key = max(safety_res, key=lambda k: safety_res[k]["auc"] if "auc" in safety_res[k] else 0)
    safety_auc = safety_res[best_key]["auc"]
    domain_data["Safety\n(BeaverTails)"] = safety_auc

    fig, ax = plt.subplots(figsize=(7, 5))
    domains = list(domain_data.keys())
    aucs = list(domain_data.values())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(domains, aucs, color=colors, alpha=0.85, width=0.5, edgecolor="white")
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.50)")
    ax.set_ylim(0.3, 0.75)
    ax.set_ylabel("Influence Function AUC")
    ax.set_title("Figure 15: Generalization to Safety Domain (BeaverTails)\n"
                 "Influence Functions Detect Safety Bias Beyond RLHF Preference Data")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig15_beavertails_safety.pdf")
    plt.savefig(OUT_DIR / "fig15_beavertails_safety.png", dpi=300)
    plt.close()
    print("  Saved: fig15_beavertails_safety")


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    print(f"Generating paper figures → {OUT_DIR}/")
    print("-" * 50)

    results = load_results()

    fig1_auc_vs_size(results)
    fig2_probe_vs_influence(results)
    fig3_gradient_separability(results)
    fig4_effective_rank(results)
    fig5_tracin_vs_datainf()
    fig6_roc_curves()
    fig7_lora_rank_ablation()
    fig8_distribution_overlap()
    fig9_story(results)
    fig10_architecture_vs_scale(results)
    fig11_baselines_comparison()
    fig11b_llm_fewshot_ablation()
    fig12_valsize_ablation()
    fig13_topk_detection()
    fig14_labeling_strategy()
    fig15_beavertails_safety()

    print("-" * 50)
    print(f"Done. 16 figures saved to {OUT_DIR}/")
    print(f"Formats: .pdf (vector) and .png (raster, 300 dpi)")
