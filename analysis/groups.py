import collections, numpy as np, pandas as pd
from config import DATA, OUTPUT as OUT
MODELS = ["GRU4Rec", "SASRec", "BSARec"]

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
    preds = np.load(f"{OUT}/{m}_preds.npy")
    ans = np.load(f"{OUT}/{m}_answers.npy").reshape(-1)
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
