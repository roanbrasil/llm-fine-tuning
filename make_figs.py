import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
INK, QUIET, EDGE = "#1f2328", "#57606a", "#8c959f"
BLUE, PINK, GREEN, ORANGE = "#2f6fb3", "#d6336c", "#2b8a3e", "#e8590c"


def box(ax, x, y, w, h, text, color=EDGE, fill="white", fs=11, bold=False, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=color, lw=1.6, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, weight="bold" if bold else "normal")


def arrow(ax, a, b, color=EDGE, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=1.5, linestyle=ls))


# ---------- Fig 1: two independent choices ----------
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
ax.text(0.3, 5.85, "Six methods, two independent choices", fontsize=17, weight="bold", color=INK)
ax.text(0.3, 5.45, "Pick one from each column — e.g. DPO trained with QLoRA", fontsize=12, color=QUIET)

ax.text(2.6, 4.85, "1 · HOW to change the weights", fontsize=13, weight="bold", color=BLUE, ha="center")
ax.text(8.4, 4.85, "2 · WHAT to teach", fontsize=13, weight="bold", color=PINK, ha="center")
left = [("Full fine-tuning", "all weights · >100 GB (7B)"),
        ("LoRA", "frozen base + low-rank adapters · ~20 GB"),
        ("QLoRA", "4-bit base + LoRA adapters · ~8 GB")]
right = [("Instruction tuning (SFT)", "prompt → response pairs"),
         ("RLHF", "human rankings → reward model → PPO"),
         ("DPO", "chosen vs rejected pairs, no reward model")]
for i, (t, s) in enumerate(left):
    y = 3.75 - i * 1.35
    ax.add_patch(FancyBboxPatch((0.3, y), 4.6, 0.95, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#eaf2fb", ec=BLUE, lw=1.6))
    ax.text(0.55, y + 0.62, t, fontsize=12.5, weight="bold", color=INK)
    ax.text(0.55, y + 0.25, s, fontsize=10.5, color=QUIET)
for i, (t, s) in enumerate(right):
    y = 3.75 - i * 1.35
    ax.add_patch(FancyBboxPatch((6.1, y), 4.6, 0.95, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#fdecf2", ec=PINK, lw=1.6))
    ax.text(6.35, y + 0.62, t, fontsize=12.5, weight="bold", color=INK)
    ax.text(6.35, y + 0.25, s, fontsize=10.5, color=QUIET)
ax.text(5.5, 4.95, "×", fontsize=30, ha="center", va="center", color=QUIET)
# recommended path
arrow(ax, (4.95, 1.45), (6.05, 1.45), color=GREEN)
arrow(ax, (4.95, 1.7), (6.05, 3.95), color=GREEN, ls="--")
ax.text(5.5, 0.35, "Common single-GPU path: SFT with QLoRA, then DPO with QLoRA",
        fontsize=11, color=GREEN, ha="center", weight="bold")
fig.savefig("figures/fig1-two-axes.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- Fig 2: LoRA ----------
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 11); ax.set_ylim(-0.6, 5); ax.axis("off")
ax.text(0.3, 4.65, "LoRA: W stays frozen, only two thin matrices learn", fontsize=16, weight="bold", color=INK)
ax.text(0.3, 4.27, "Example: d = 4096, r = 16  →  131k trainable parameters instead of 16.7M (≈0.8%)",
        fontsize=11.5, color=QUIET)
# input
box(ax, 0.3, 1.7, 1.2, 0.8, "x", fs=14, bold=True)
# frozen W
ax.add_patch(Rectangle((2.3, 1.0), 2.4, 2.4, fc="#eef1f4", ec=EDGE, lw=1.6, hatch="//"))
ax.text(3.5, 2.35, "W", fontsize=22, weight="bold", ha="center", color=INK)
ax.text(3.5, 1.85, "d × d · frozen", fontsize=11, ha="center", color=QUIET)
ax.text(3.5, 0.62, "pretrained weights", fontsize=10.5, ha="center", color=QUIET)
# A and B
ax.add_patch(Rectangle((5.6, 2.9), 1.9, 0.35, fc="#fde3d2", ec=ORANGE, lw=1.8))
ax.text(6.55, 3.07, "A  (r × d)", fontsize=10.5, ha="center", va="center", color=INK)
ax.add_patch(Rectangle((5.6, 0.25), 0.35, 1.6, fc="#fde3d2", ec=ORANGE, lw=1.8))
ax.text(6.15, 0.45, "B  (d × r)", fontsize=10.5, va="center", color=INK)
ax.text(5.6, 3.45, "trainable · B starts at 0", fontsize=10.5, ha="left", color=ORANGE, weight="bold")
# sum
ax.add_patch(plt.Circle((8.6, 2.1), 0.32, fc="white", ec=EDGE, lw=1.6))
ax.text(8.6, 2.1, "+", fontsize=18, ha="center", va="center", color=INK)
box(ax, 9.4, 1.7, 1.3, 0.8, "h", fs=14, bold=True)
arrow(ax, (1.5, 2.1), (2.3, 2.1))
ax.plot([1.9, 1.9, 5.2, 5.2], [2.1, 3.75, 3.75, 3.07], color=EDGE, lw=1.5)
arrow(ax, (5.2, 3.07), (5.6, 3.07))
arrow(ax, (6.55, 2.9), (5.85, 1.88), color=ORANGE)
arrow(ax, (4.7, 2.1), (8.28, 2.1))
arrow(ax, (5.95, 1.55), (8.38, 1.88), color=ORANGE)
arrow(ax, (8.92, 2.1), (9.4, 2.1))
ax.text(8.6, 1.4, "× α / r", fontsize=11, ha="center", color=ORANGE)
ax.text(5.5, -0.35, "h = W x + (α / r) · B A x      —      merge later as  W′ = W + (α / r) · B A",
        fontsize=12, ha="center", color=INK)
fig.savefig("figures/fig2-lora.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- Fig 3: VRAM ----------
methods = ["Full fine-tuning\n(bf16 + Adam)", "LoRA\n(bf16 base)", "QLoRA\n(4-bit base)"]
vram = [112, 20, 8]
colors = [PINK, BLUE, GREEN]
fig, ax = plt.subplots(figsize=(10, 4.6))
bars = ax.barh(methods[::-1], vram[::-1], color=colors[::-1], height=0.55)
for b, v in zip(bars, vram[::-1]):
    ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2,
            f"~{v} GB" + ("+" if v == 112 else ""), va="center", fontsize=12, color=INK, weight="bold")
for g, lbl in [(8, "8 GB"), (24, "24 GB"), (80, "80 GB")]:
    ax.axvline(g, color=EDGE, ls="--", lw=1)
    ax.text(g + 1, 2.45, lbl + " card", fontsize=9.5, color=QUIET, ha="left", va="bottom")
ax.set_xlim(0, 130); ax.set_ylim(-0.5, 2.7)
ax.set_xlabel("Approximate VRAM to fine-tune a 7B model (GB)\nRough figures; batch size, sequence length and gradient checkpointing change them.", color=QUIET, labelpad=10)
ax.set_title("QLoRA puts a 7B fine-tune on a consumer GPU", loc="left", fontsize=15,
             weight="bold", color=INK, pad=14)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(colors=INK, labelsize=11)
fig.savefig("figures/fig3-vram.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- Fig 4: RLHF vs DPO ----------
fig, ax = plt.subplots(figsize=(11, 6.4))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.4); ax.axis("off")
ax.text(0.3, 6.05, "RLHF trains a judge and runs RL; DPO optimizes the pairs directly",
        fontsize=15.5, weight="bold", color=INK)
# RLHF row
ax.text(0.3, 5.35, "RLHF", fontsize=13, weight="bold", color=PINK)
xs = [0.3, 2.45, 4.6, 6.75, 8.9]
labels = ["SFT model", "Generate\nanswers", "Humans\nrank them", "Train\nreward model", "Optimize\nwith PPO"]
for x, l in zip(xs, labels):
    box(ax, x, 3.95, 1.8, 1.05, l, color=PINK, fill="#fdecf2", fs=11)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + 1.8, 4.47), (b, 4.47))
# loop back
ax.add_patch(FancyArrowPatch((9.8, 3.95), (3.35, 3.95), connectionstyle="arc3,rad=-0.12",
                             arrowstyle="-|>", mutation_scale=14, color=PINK, lw=1.4, ls="--"))
ax.text(6.6, 3.2, "repeat in rounds with the new policy", fontsize=10.5, color=PINK, ha="center")
ax.text(10.7, 5.35, "in memory: policy · reference · reward · value  (4)",
        fontsize=10, color=QUIET, ha="right")
# DPO row
ax.text(0.3, 2.25, "DPO", fontsize=13, weight="bold", color=BLUE)
xs2 = [0.3, 3.0, 5.7, 8.4]
labels2 = ["SFT model", "Pairs:\nchosen vs rejected", "Preference loss\n(β-scaled log-ratios)", "Aligned model"]
for x, l in zip(xs2, labels2):
    box(ax, x, 0.85, 2.3, 1.05, l, color=BLUE, fill="#eaf2fb", fs=11)
for a, b in zip(xs2[:-1], xs2[1:]):
    arrow(ax, (a + 2.3, 1.37), (b, 1.37))
ax.text(10.7, 2.25, "in memory: policy · reference (cacheable)  (2)", fontsize=10, color=QUIET, ha="right")
ax.text(5.5, 0.3, "No reward model, no sampling loop — an ordinary supervised training run",
        fontsize=11, color=BLUE, ha="center", weight="bold")
fig.savefig("figures/fig4-rlhf-vs-dpo.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("ok")
