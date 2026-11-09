"""Does the oversmoothing picture change with depth? Final-layer metrics at 2, 4 and 8 layers.

Reads results/oversmoothing{,_L4,_L8}.csv (written by oversmoothing.py --layers L) and
test NDCG@10 from results/results.csv (depth-sweep runs are named {Model}L{L}).
For each depth, the metrics are taken at that model's last block. Prints a summary
(saved to results/oversmoothing_depth.txt).
"""
import pandas as pd
from config import RESULTS

DEPTHS = [2, 4, 8]
MODELS = ["SASRec", "BSARecAlpha0", "BSARec"]
# (part in oversmoothing.csv, metric, label); higher = smoother except hf_share
MEASURES = [("block", "cos_sim", "within: cos_sim ↑"),
            ("block", "hf_share", "within: hf_share ↓"),
            ("block centered", "cos_sim", "within centered: cos_sim ↑"),
            ("across users", "cos_sim", "across users: cos_sim ↑"),
            ("across users centered", "top1_share", "across centered: top1_share ↑")]

frames = []
for L in DEPTHS:
    d = pd.read_csv(RESULTS / f"oversmoothing{'' if L == 2 else f'_L{L}'}.csv")
    frames.append(d[d.layer == L].assign(depth=L))
final = pd.concat(frames)

res = pd.read_csv(RESULTS / "results.csv")
acc = [{"depth": L, "model": m, "seed": row.seed, "label": "test NDCG@10", "value": row["NDCG@10"]}
       for L in DEPTHS for m in MODELS
       for _, row in res[res.model == (m if L == 2 else f"{m}L{L}")].iterrows()]

long = [final[final.part == part].assign(label=label, value=lambda x, m=met: x[m])[["depth", "model", "seed", "label", "value"]]
        for part, met, label in MEASURES]
long = pd.concat(long + [pd.DataFrame(acc)])
labels = ["test NDCG@10"] + [m[2] for m in MEASURES]
pd.set_option("display.width", 200)

g = long.groupby(["label", "depth", "model"]).value
table = (g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)).unstack("model")[MODELS]
print("### Final block, mean ± std over seeds (within = positions of a sequence; across = users' last-position vectors)")
print(table.reindex(labels, level=0).to_string())

print("\n### Per seed: BSARec minus SASRec / BSARec minus BSARecAlpha0 (same sign on every seed = consistent)")
wide = long.pivot_table(index=["label", "depth", "seed"], columns="model", values="value")
diff = pd.DataFrame({"BSARec - SASRec": wide.BSARec - wide.SASRec,
                     "BSARec - Alpha0": wide.BSARec - wide.BSARecAlpha0}).round(4)
print(diff.unstack("seed").reindex(labels, level=0).to_string())
