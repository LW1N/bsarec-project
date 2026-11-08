import json, pandas as pd
from config import RESULTS as R
df = pd.read_csv(f"{R}/results.csv")
m = [c for c in df.columns if c not in ("model", "seed")]
mean, std = df.groupby("model")[m].mean(), df.groupby("model")[m].std()
table = mean.round(4).astype(str) + " ± " + std.round(4).astype(str)
pop = json.load(open(f"{R}/popularity.json"))["test"]
table.loc["Popularity"] = [str(pop[c]) for c in m]
# Paper's Table 2 order; models we have not run are skipped. Variants (BSARecAlpha*)
# are reported separately by decomposition.py.
ORDER = ["Popularity", "GRU4Rec", "Caser", "SASRec", "BERT4Rec", "FMLPRec", "DuoRec", "FEARec", "BSARec"]
table = table.reindex([m for m in ORDER if m in table.index])
table.to_csv(f"{R}/main_table.csv")
print("### Main results (test, mean ± std over seeds)")
print(table.to_string())
print("\n### BSARec vs SASRec (% change in mean)")
print(((mean.loc["BSARec"] / mean.loc["SASRec"] - 1) * 100).round(1).to_string())
paper = pd.read_csv(f"{R}/paper.csv").set_index("model")
idx = [i for i in paper.index if i in mean.index]
cols = [c for c in m if c in paper.columns]
print("\n### Ours (mean) minus paper")
print((mean.loc[idx, cols] - paper.loc[idx, cols]).round(4).to_string())
