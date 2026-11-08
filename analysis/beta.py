"""Learned beta per layer for BSARec on LastFM (reproduces the paper's Fig. 7a).

BSARec scales the high-frequency component by beta = sqrt_beta**2
(src/model/bsarec.py, FrequencyLayer), one value per hidden dimension and layer.
Loads checkpoints on the CPU only; nothing is trained.

Reads src/output/BSARec_LastFM_s{seed}.pt (ours) and BSARec_LastFM_best.pt
(shipped by the upstream authors). Writes results/beta.csv and figures/beta.png,
and prints a summary (saved to results/beta_summary.txt).
"""
import re
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUTPUT, RESULTS, FIGURES, HPARAMS

SEEDS = re.search(r'SEEDS="([^"]+)"', HPARAMS.read_text()).group(1).split()
RUNS = {f"s{s}": OUTPUT / f"BSARec_LastFM_s{s}.pt" for s in SEEDS}
RUNS["authors"] = OUTPUT / "BSARec_LastFM_best.pt"
KEY = re.compile(r"item_encoder\.blocks\.(\d+)\.layer\.filter_layer\.sqrt_beta")

rows, values = [], {}
for run, path in RUNS.items():
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k, v in sd.items():
        m = KEY.fullmatch(k)
        if m:
            layer = int(m.group(1)) + 1                      # 1-based, as in the paper
            beta = (v.double() ** 2).flatten().numpy()       # one value per hidden dim
            values[run, layer] = beta
            rows.append({"run": run, "layer": layer, "mean": beta.mean(), "std": beta.std(),
                         "min": beta.min(), "median": np.median(beta), "max": beta.max()})
df = pd.DataFrame(rows).sort_values(["run", "layer"])
df.round(4).to_csv(RESULTS / "beta.csv", index=False)

ours = df[df.run != "authors"]
print("### beta per layer and run (beta = sqrt_beta^2, over 64 hidden dims)")
print(df.round(4).to_string(index=False))
print("\n### Ours: mean of per-run mean beta, ± std over seeds")
g = ours.groupby("layer")["mean"]
print((g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)).to_string())
print("\n### Layer 1 mean beta > layer 2 mean beta?")
for run, d in df.groupby("run"):
    m = d.set_index("layer")["mean"]
    print(f"- {run}: {m[1]:.4f} vs {m[2]:.4f} -> {'yes' if m[1] > m[2] else 'no'}")

# Figure: per layer, every hidden dim's beta (small dots) and each run's mean (large marker).
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
fig, ax = plt.subplots(figsize=(6, 3.6))
layers = sorted(df.layer.unique())
order = [f"s{s}" for s in SEEDS] + ["authors"]
width = 0.8 / len(order)
rng = np.random.default_rng(0)                                # fixed jitter: identical PNG every run
for i, run in enumerate(order):
    color, marker = (ORANGE, "D") if run == "authors" else (BLUE, "o")
    for L in layers:
        x = L + (i - (len(order) - 1) / 2) * width
        b = values[run, L]
        ax.scatter(x + rng.uniform(-width / 4, width / 4, len(b)), b, s=6, color=color,
                   alpha=0.35, linewidths=0)
        ax.scatter([x], [b.mean()], s=64, color=color, marker=marker, edgecolors="#fcfcfb",
                   linewidths=2, zorder=3)
        ax.annotate(run, (x, 0), xytext=(0, -14), textcoords="offset points",
                    xycoords=("data", "axes fraction"), ha="center", fontsize=7, color=MUTED)
ax.set_xticks(layers, [f"Layer {L}" for L in layers])
ax.tick_params(axis="x", pad=16, length=0, colors=INK)
ax.tick_params(axis="y", colors=MUTED, labelsize=8)
ax.set_ylabel("learned β (per hidden dim)", color=MUTED, fontsize=9)
ax.set_ylim(bottom=0)
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=7, label="ours: mean over dims, one per seed"),
           plt.Line2D([], [], marker="D", ls="", color=ORANGE, markersize=6, label="authors' checkpoint: mean over dims"),
           plt.Line2D([], [], marker="o", ls="", color=MUTED, alpha=0.5, markersize=3, label="one hidden dim")]
ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
ax.set_title("BSARec on LastFM: learned β by layer", fontsize=10, color=INK, loc="left")
plt.tight_layout()
plt.savefig(FIGURES / "beta.png", dpi=200, metadata={"Software": None})
