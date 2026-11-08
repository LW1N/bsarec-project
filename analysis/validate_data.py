import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ROOT, DATA as PATH, FIGURES as OUT

seqs, bad = {}, 0
with open(PATH) as f:
    for n, line in enumerate(f, 1):
        try:
            ids = list(map(int, line.split()))
        except ValueError:
            bad += 1
            continue
        if ids[0] in seqs:
            print(f"duplicate user {ids[0]} on line {n}")
        seqs[ids[0]] = ids[1:]

lengths = np.array([len(s) for s in seqs.values()])
counts = collections.Counter(i for s in seqs.values() for i in s)
items = sorted(counts)
c = np.array([v for _, v in counts.most_common()])

print("users:", len(seqs), "(paper: 1090)")
print("artists:", len(counts), "(paper: 3646)")
print("interactions:", int(lengths.sum()), "(paper: 52551)")
print("unparseable lines:", bad)
print("artist id range:", items[0], "-", items[-1], "| missing ids:", items[-1] - items[0] + 1 - len(items))
print("seq length min/median/max:", lengths.min(), int(np.median(lengths)), lengths.max())
print("users with < 3 interactions (cannot split):", int((lengths < 3).sum()))
print("users longer than 50 (model sees last 50):", int((lengths > 50).sum()))
print("back-to-back repeats:", sum(a == b for s in seqs.values() for a, b in zip(s, s[1:])))
print("test targets already earlier in history:", sum(s[-1] in s[:-1] for s in seqs.values()))
print("top 1%% of artists' share of interactions: %.1f%%" % (100 * c[:max(1, len(c)//100)].sum() / c.sum()))
print("artists with < 5 interactions: %.1f%%" % (100 * (c < 5).mean()))

plt.figure(); plt.hist(lengths, bins=50)
plt.xlabel("interactions per user"); plt.ylabel("users"); plt.title("Sequence lengths")
plt.savefig(f"{OUT}/seq_lengths.png", dpi=200, bbox_inches="tight")
plt.figure(); plt.loglog(np.arange(1, len(c) + 1), c)
plt.xlabel("artist rank"); plt.ylabel("interactions"); plt.title("Artist popularity")
plt.savefig(f"{OUT}/artist_popularity.png", dpi=200, bbox_inches="tight")
print("figures saved to", OUT.relative_to(ROOT))
