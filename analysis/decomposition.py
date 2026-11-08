"""Split the BSARec-over-SASRec gain into a loss effect and a Fourier-branch effect.

BSARec at alpha = 0 (BSARecAlpha0) has SASRec's architecture but BSARec's
full-softmax cross-entropy loss, so per seed:
    loss effect   = BSARecAlpha0 - SASRec
    branch effect = BSARec - BSARecAlpha0
BSARecAlpha1 (alpha = 1) is the Fourier branch alone, without self-attention.

Reads results/results.csv; writes results/variants_table.csv and prints a
report (saved to results/decomposition.txt).
"""
import pandas as pd
from config import RESULTS

MODELS = ["SASRec", "BSARecAlpha0", "BSARec", "BSARecAlpha1"]
KEY = ["NDCG@10", "HR@10"]

df = pd.read_csv(RESULTS / "results.csv")
df = df[df.model.isin(MODELS)]
metrics = [c for c in df.columns if c not in ("model", "seed")]
seeds = sorted(set.intersection(*(set(df.seed[df.model == m]) for m in MODELS)))
df = df[df.seed.isin(seeds)]

g = df.groupby("model")[metrics]
table = (g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)).reindex(MODELS)
table.to_csv(RESULTS / "variants_table.csv")
print(f"### SASRec vs BSARec variants (test, mean ± std over seeds {seeds})")
print(table.to_string())

wide = df.pivot(index="seed", columns="model", values=KEY)
for k in KEY:
    w = wide[k]
    d = pd.DataFrame({"SASRec": w.SASRec, "BSARecAlpha0": w.BSARecAlpha0, "BSARec": w.BSARec,
                      "loss effect": w.BSARecAlpha0 - w.SASRec,
                      "branch effect": w.BSARec - w.BSARecAlpha0})
    d["loss share"] = d["loss effect"] / (d["loss effect"] + d["branch effect"])
    d.loc["mean"] = d.mean()
    d.loc["mean", "loss share"] = d.loc["mean", "loss effect"] / (d.loc["mean", "loss effect"] + d.loc["mean", "branch effect"])
    print(f"\n### {k} decomposition: SASRec -> BSARecAlpha0 (loss) -> BSARec (branch)")
    print(d.round(4).to_string())

print("\n### BSARecAlpha1 (Fourier branch only) minus SASRec, mean over seeds")
print((g.mean().loc["BSARecAlpha1"] - g.mean().loc["SASRec"]).round(4).to_string())
