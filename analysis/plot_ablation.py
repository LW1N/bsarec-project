import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import HPARAMS, RESULTS, FIGURES
hp = dict(l.strip().split("=", 1) for l in open(HPARAMS) if "=" in l)
df = pd.read_csv(RESULTS / "ablation.csv", dtype={"value": str})
df.loc[(df.param == "alpha") & (df.value == "default"), "value"] = hp["ALPHA"]
df.loc[(df.param == "c") & (df.value == "default"), "value"] = hp["C"]
df["value"] = df.value.astype(float)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for i, p in enumerate(["alpha", "c"]):
    d = df[df.param == p].sort_values("value")
    ax[i].plot(d.value, d["NDCG@10"], marker="o")
    ax[i].set_xlabel(p); ax[i].set_ylabel("NDCG@10")
    print(f"### NDCG@10 by {p}")
    print(d[["value", "HR@10", "NDCG@10"]].to_string(index=False), "\n")
plt.tight_layout()
plt.savefig(FIGURES / "ablation.png", dpi=200)
