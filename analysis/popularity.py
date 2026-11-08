import json, math, collections
from config import DATA as PATH, RESULTS
KS = [5, 10, 20]

seqs = [list(map(int, l.split()))[1:] for l in open(PATH) if l.strip()]
all_items = sorted({i for s in seqs for i in s})

def rank_of(target, hist, ranking, max_k):
    r = 0
    for i in ranking:
        if i in hist:
            continue
        r += 1
        if i == target:
            return r
        if r >= max_k:
            return None
    return None

def evaluate(cut):  # cut=2 -> validation, cut=1 -> test
    cnt = collections.Counter(i for s in seqs for i in s[:-cut])
    ranking = sorted(all_items, key=lambda i: -cnt[i])
    res = {f"HR@{k}": 0.0 for k in KS} | {f"NDCG@{k}": 0.0 for k in KS}
    for s in seqs:
        r = rank_of(s[-cut], set(s[:-cut]), ranking, max(KS))
        for k in KS:
            if r is not None and r <= k:
                res[f"HR@{k}"] += 1
                res[f"NDCG@{k}"] += 1 / math.log2(r + 1)
    return {m: round(v / len(seqs), 4) for m, v in res.items()}

out = {"valid": evaluate(2), "test": evaluate(1)}
print(json.dumps(out, indent=2))
json.dump(out, open(RESULTS / "popularity.json", "w"), indent=2)
