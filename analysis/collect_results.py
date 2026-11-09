import os, re, glob, csv
from config import OUTPUT as OUT, RESULTS as RES
COLS = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]
PAT = re.compile(r"(HR|HIT|NDCG)@(\d+)\D{0,4}?([01]\.\d+)")

def last_metrics(path):
    """Metrics on the last metrics line after 'Test Score' (the test result), or None
    if the run has not reached testing yet (e.g. still training)."""
    found, tested = None, False
    for line in open(path, errors="ignore"):
        tested = tested or "Test Score" in line
        m = PAT.findall(line)
        if tested and len(m) >= 4:
            found = {("HR" if k == "HIT" else k) + "@" + n: float(v) for k, n, v in m}
    return found

main, abl = [], []
for path in sorted(glob.glob(f"{OUT}/*.log")):
    name = os.path.basename(path)[:-4]
    if not re.fullmatch(r"\w+?_LastFM_s\d+|abl_(alpha|c)[\d.]+", name):
        continue                                   # eval/predict/smoke-test/upstream logs
    met = last_metrics(path)
    if met is None:
        print("skipped (no test result yet):", name)
        continue
    row = [met.get(c, "") for c in COLS]
    m = re.fullmatch(r"(\w+?)_LastFM_s(\d+)", name)
    if m:
        main.append([m.group(1), m.group(2)] + row)
        if m.group(1) == "BSARec" and m.group(2) == "42":
            abl += [["alpha", "default"] + row, ["c", "default"] + row]
    m = re.fullmatch(r"abl_(alpha|c)([\d.]+)", name)
    if m:
        abl.append([m.group(1), m.group(2)] + row)

for fname, head, rows in [("results.csv", ["model", "seed"], main),
                          ("ablation.csv", ["param", "value"], abl)]:
    with open(f"{RES}/{fname}", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head + COLS)
        w.writerows(rows)
    print(f"{fname}: {len(rows)} rows")
