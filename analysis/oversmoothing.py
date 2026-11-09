"""Measure oversmoothing layer by layer in trained SASRec and BSARec models.

The paper's claim: self-attention acts as a low-pass filter, so position
representations within a sequence become more alike with depth (oversmoothing),
and BSARec's Fourier branch counters this. For every test sequence (the input
the model sees when predicting each user's last artist), we take the
representation at each layer and compute over the sequence's real (non-padding)
positions:

  cos_sim    mean pairwise cosine similarity between positions (1 = all identical)
  hf_share   share of energy outside the lowest DFT frequencies along the
             sequence (k = 0, 1 and its mirror; the same split BSARec uses for
             c = 3). Lower = smoother, i.e. more low-pass.
  top1_share sigma_1^2 / sum sigma^2 of the positions x hidden matrix
             (1 = rank-one collapse)

Layer 0 is the embedding output; layers 1-2 are the transformer/BSARec blocks.

The paper's Fig. 3 ("user sequence output embedding") can also be read as comparing
users rather than positions, so part "across users" measures each user's
last-position vector (the one the model ranks artists with) across the 1,090 users:
cos_sim = mean pairwise cosine between users, top1_share = sigma_1^2 / sum sigma^2
of the users x hidden matrix (hf_share does not apply). Part "across users centered"
repeats top1_share after subtracting the mean user vector, so a direction shared by
all users cannot count as collapse (centered cos_sim is ~0 by construction, so NaN).
Robustness check ("centered"): the same metrics after subtracting, per layer,
the mean vector over all real positions of all sequences, so a vector shared by
every position (e.g. a LayerNorm bias) cannot inflate the similarity.
Forward hooks also capture BSARec's two branches inside each layer (attention
vs Fourier) without modifying src/model/.

Models: SASRec, BSARecAlpha0 (SASRec architecture with BSARec's loss) and
BSARec, seeds from hparams.sh. Model settings are read from each run's log.
Inference only, on the CPU; nothing is trained.

Writes results/oversmoothing.csv, results/oversmoothing_groups.csv,
figures/oversmoothing.png; prints a summary (saved to results/oversmoothing.txt).
"""
import ast, re, sys
from argparse import Namespace
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT, DATA, OUTPUT, RESULTS, FIGURES, HPARAMS

sys.path.insert(0, str(ROOT / "src"))
from model import MODEL_DICT  # upstream model code, unmodified

SEEDS = re.search(r'SEEDS="([^"]+)"', HPARAMS.read_text()).group(1).split()
MODELS = ["SASRec", "BSARecAlpha0", "BSARec"]
LOW_BINS = 2   # BSARec keeps rfft bins < c // 2 + 1 = 2 as "low frequency" (c = 3)


def run_args(train_name):
    """Parse the Namespace(...) line a training log starts with (no eval)."""
    line = next(l for l in open(OUTPUT / f"{train_name}.log") if "Namespace(" in l)
    body = line[line.index("Namespace(") + 10: line.rindex(")")]
    pairs = re.findall(r"(\w+)=('(?:[^'\\]|\\.)*'|[^,]+)", body)
    return Namespace(**{k: ast.literal_eval(v.strip()) for k, v in pairs})


def load_model(train_name):
    args = run_args(train_name)
    model = MODEL_DICT[args.model_type.lower()](args=args)
    sd = torch.load(OUTPUT / f"{train_name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.eval(), args


def test_inputs(max_len):
    """Same inputs as the upstream test DataLoader: all but the last item, left-padded."""
    seqs = [list(map(int, l.split()))[1:] for l in open(DATA) if l.strip()]
    ids = [([0] * max_len + s[:-1])[-max_len:] for s in seqs]
    return seqs, torch.tensor(ids, dtype=torch.long)


def metrics(x):
    """x: (n_positions, hidden) for one sequence's real positions."""
    n = len(x)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    sim = xn @ xn.T
    cos = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    energy = (np.abs(np.fft.fft(x, axis=0, norm="ortho")) ** 2).sum(1)  # per frequency bin
    k = np.minimum(np.arange(n), n - np.arange(n))                       # distance from DC
    hf = energy[k >= LOW_BINS].sum() / energy.sum()
    sv = np.linalg.svd(x, compute_uv=False) ** 2
    return cos, hf, sv[0] / sv.sum()


def across_users(u, center=False):
    """u: (users, hidden), each user's last-position vector."""
    if center:
        u = u - u.mean(0)
    un = u / np.linalg.norm(u, axis=1, keepdims=True).clip(1e-12)
    n = len(u)
    sv = np.linalg.svd(u, compute_uv=False) ** 2
    cos = np.nan if center else ((un @ un.T).sum() - n) / (n * (n - 1))
    return {"cos_sim": cos, "hf_share": np.nan, "top1_share": sv[0] / sv.sum()}


def summarize(reps, n_real, center=False):
    """reps: (users, max_len, hidden) -> per-user metric arrays."""
    mu = np.concatenate([r[-n:] for r, n in zip(reps, n_real)]).mean(0) if center else 0
    out = np.array([metrics(r[-n:] - mu) for r, n in zip(reps, n_real)])
    return {"cos_sim": out[:, 0], "hf_share": out[:, 1], "top1_share": out[:, 2]}


seqs, input_ids = test_inputs(50)
n_real = (input_ids > 0).sum(1).numpy()
length = pd.Series([len(s) for s in seqs])
group = pd.qcut(length.rank(method="first"), 3, labels=["short", "medium", "long"])  # as in groups.py

rows, grows = [], []
for model_name in MODELS:
    for seed in SEEDS:
        model, args = load_model(f"{model_name}_LastFM_s{seed}")
        assert args.max_seq_length == input_ids.shape[1]
        branch = {}
        hooks = []
        for i, block in enumerate(model.item_encoder.blocks, 1):
            layer = block.layer
            att = layer.attention_layer if hasattr(layer, "attention_layer") else layer
            hooks.append(att.register_forward_hook(lambda m, inp, out, i=i: branch.__setitem__(("attention", i), out)))
            if hasattr(layer, "filter_layer"):
                hooks.append(layer.filter_layer.register_forward_hook(lambda m, inp, out, i=i: branch.__setitem__(("fourier", i), out)))
        with torch.no_grad():
            layers = model(input_ids, all_sequence_output=True)
        for h in hooks:
            h.remove()

        named = {("block", i): t for i, t in enumerate(layers)}
        if model_name == "BSARecAlpha0":                       # alpha = 0: Fourier output is multiplied by 0
            branch = {k: v for k, v in branch.items() if k[0] != "fourier"}
        named.update(branch)
        named.update({("block centered", i): t for i, t in enumerate(layers)})
        for (part, i), t in named.items():
            m = summarize(t.double().numpy(), n_real, center=(part == "block centered"))
            rows.append({"model": model_name, "seed": seed, "part": part, "layer": i,
                         **{k: v.mean() for k, v in m.items()}})
            if part == "block":
                for g in ["short", "medium", "long"]:
                    sel = (group == g).to_numpy()
                    grows.append({"model": model_name, "seed": seed, "layer": i, "history_length": g,
                                  **{k: v[sel].mean() for k, v in m.items()}})
        for i, t in enumerate(layers):
            for part, center in [("across users", False), ("across users centered", True)]:
                rows.append({"model": model_name, "seed": seed, "part": part, "layer": i,
                             **across_users(t[:, -1, :].double().numpy(), center)})

df, gdf = pd.DataFrame(rows), pd.DataFrame(grows)
df.round(4).to_csv(RESULTS / "oversmoothing.csv", index=False)
gdf.round(4).to_csv(RESULTS / "oversmoothing_groups.csv", index=False)

METRICS = ["cos_sim", "hf_share", "top1_share"]
pd.set_option("display.width", 200)


def mean_std(d, by, cols=METRICS):
    g = d.groupby(by)[cols]
    return g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)


blocks = df[df.part == "block"]
print(f"### Block outputs, mean over {len(seqs)} test sequences; mean ± std over seeds {SEEDS}")
print("(layer 0 = embeddings; cos_sim/top1_share higher and hf_share lower = smoother)")
print(mean_std(blocks, ["model", "layer"]).reindex(MODELS, level=0).to_string())

print("\n### Change from layer 0 to layer 2 (mean over seeds)")
m = blocks.groupby(["model", "layer"])[METRICS].mean()
print((m.xs(2, level="layer") - m.xs(0, level="layer")).reindex(MODELS).round(4).to_string())

print("\n### Robustness: block outputs after removing the vector shared by all positions (mean ± std over seeds)")
print(mean_std(df[df.part == "block centered"], ["model", "layer"]).reindex(MODELS, level=0).to_string())

print("\n### Inside each layer: attention branch vs Fourier branch (mean ± std over seeds)")
print(mean_std(df[df.part.isin(["attention", "fourier"])], ["model", "layer", "part"]).reindex(MODELS, level=0).to_string())

print("\n### Final layer (2) by history length (mean ± std over seeds)")
print(mean_std(gdf[gdf.layer == 2], ["history_length", "model"])
      .reindex(pd.MultiIndex.from_product([["short", "medium", "long"], MODELS])).to_string())

print("\n### Per-seed final-layer cos_sim: is BSARec smoother than SASRec?")
for part in ["block", "block centered"]:
    fin = df[(df.part == part) & (df.layer == 2)].pivot(index="seed", columns="model", values="cos_sim")[MODELS]
    print(f"{part}:"); print(fin.round(4).to_string())

print("\n### Across users: each user's last-position vector, the one used for ranking (mean ± std over seeds)")
print("(the other reading of the paper's Fig. 3; higher = users' representations more alike)")
across = df[df.part == "across users"]
print(mean_std(across, ["model", "layer"], ["cos_sim", "top1_share"]).reindex(MODELS, level=0).to_string())
print("per seed, layer 2 cos_sim:")
print(across[across.layer == 2].pivot(index="seed", columns="model", values="cos_sim")[MODELS].round(4).to_string())
print("Robustness: top1_share after removing the mean user vector (mean ± std over seeds); per seed, layer 2:")
centered = df[df.part == "across users centered"]
print(mean_std(centered, ["model", "layer"], ["top1_share"]).reindex(MODELS, level=0).to_string())
print(centered[centered.layer == 2].pivot(index="seed", columns="model", values="top1_share")[MODELS].round(4).to_string())

# Figure: two small multiples (cos_sim, hf_share) by layer; one line per model (seed mean),
# faint points per seed. Colour + marker shape per model; direct labels at the line ends.
COLORS = {"SASRec": "#2a78d6", "BSARecAlpha0": "#eb6834", "BSARec": "#1baf7a"}
MARKERS = {"SASRec": "o", "BSARecAlpha0": "s", "BSARec": "D"}
LABELS = {"SASRec": "SASRec", "BSARecAlpha0": "BSARec α=0", "BSARec": "BSARec"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
titles = {"cos_sim": "Cosine similarity between positions (↑ smoother)",
          "hf_share": "High-frequency energy share (↓ smoother)"}
for ax, met in zip(axes, ["cos_sim", "hf_share"]):
    for mod in MODELS:
        d = blocks[blocks.model == mod]
        mean = d.groupby("layer")[met].mean()
        ax.scatter(d.layer, d[met], color=COLORS[mod], marker=MARKERS[mod], s=14, alpha=0.35, linewidths=0)
        ax.plot(mean.index, mean.values, color=COLORS[mod], marker=MARKERS[mod], lw=2, ms=7,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=LABELS[mod])
    ax.set_xticks([0, 1, 2], ["embedding", "layer 1", "layer 2"])
    ax.set_title(titles[met], fontsize=9, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlim(-0.2, 2.75)
    ends = blocks[blocks.layer == 2].groupby("model")[met].mean().sort_values()
    ys = list(ends.values)
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for j in range(1, len(ys)):                               # nudge labels apart
        ys[j] = max(ys[j], ys[j - 1] + 0.07 * span)
    for (mod, _), y in zip(ends.items(), ys):
        ax.annotate(LABELS[mod], (2, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Oversmoothing by layer on LastFM test sequences (seed mean; faint points = seeds)",
             fontsize=10, color=INK, x=0.01, ha="left")
plt.tight_layout()
plt.savefig(FIGURES / "oversmoothing.png", dpi=200, metadata={"Software": None})
