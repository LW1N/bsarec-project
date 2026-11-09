"""Per-user group analysis (history length, taste, test-artist popularity).

Reads analysis/predictions/{Model}_s{seed}_preds.npy and _answers.npy (top-20
predictions saved by a --do_eval run; see run_all.sh) and prints NDCG@10 per
group. Usage: python analysis/groups.py --seed 42 > analysis/results/groups_s42.txt
"""
import argparse, collections, numpy as np, pandas as pd
from config import DATA, PREDICTIONS
MODELS = ["GRU4Rec", "SASRec", "BSARec"]
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--seed", required=True, help="training seed of the checkpoints that made the predictions")
seed = parser.parse_args().seed

seqs = [list(map(int, l.split()))[1:] for l in open(DATA) if l.strip()]
cnt = collections.Counter(i for s in seqs for i in s[:-1])
head = [i for i, _ in cnt.most_common(int(0.2 * len(cnt)))]
u = pd.DataFrame({"length": [len(s) for s in seqs],
                  "mainstream": [np.mean([cnt[i] for i in s[:-1]]) for s in seqs],
                  "target_head": [s[-1] in set(head) for s in seqs]})
u["history_length"] = pd.qcut(u.length.rank(method="first"), 3, labels=["short", "medium", "long"])
u["taste"] = pd.qcut(u.mainstream.rank(method="first"), 3, labels=["niche", "mixed", "mainstream"])
u["test_artist"] = np.where(u.target_head, "popular (top 20%)", "long tail")

print("### Share of top-10 recs that are top-20% artists")
for m in MODELS:
    preds = np.load(PREDICTIONS / f"{m}_s{seed}_preds.npy")
    ans = np.load(PREDICTIONS / f"{m}_s{seed}_answers.npy").reshape(-1)
    assert len(preds) == len(u), "row count mismatch: check the patch"
    hit = preds[:, :10] == ans[:, None]
    rank = np.where(hit.any(1), hit.argmax(1) + 1, 0)
    u[m] = np.where(rank > 0, 1 / np.log2(rank + 1), 0.0)
    print(f"- {m}: {np.isin(preds[:, :10], head).mean():.1%}")
print("\n### Overall NDCG@10 (should match the logs)")
print(u[MODELS].mean().round(4).to_string())
for g in ["history_length", "taste", "test_artist"]:
    print(f"\n### NDCG@10 by {g}")
    print(u.groupby(g)[MODELS].mean().round(4).to_string())
