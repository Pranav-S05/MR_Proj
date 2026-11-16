"""python scripts/check_patch.py --variant first_w30_o15 --seq 07 --log results/final/runs/upstream/C4_w30_o15/07/run.log.gz"""
import argparse, gzip, re
import numpy as np
import lib
from exp_main import Scorer, load_variant
ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True); ap.add_argument("--seq", default="07"); ap.add_argument("--log", required=True)
a = ap.parse_args()
seqs, data, feats, y, ok = load_variant(a.variant)
ours = Scorer(data, feats, y, ok).corr(a.seq, "C4_fuse_depth+speed", [o for o in seqs if o != a.seq])
pat = re.compile(r"Seam scale calibration: log s ([+-][0-9.]+) -> ([+-][0-9.]+)")
up = np.array([float(m.group(2)) - float(m.group(1)) for m in map(pat.search, gzip.open(a.log, "rt", encoding="utf-8", errors="replace")) if m])
n = min(len(ours), len(up))
print("seams ours / upstream:", len(ours), len(up))
print("correlation %.3f, mean difference %+.4f" % (np.corrcoef(ours[:n], up[:n])[0, 1], (up[:n] - ours[:n]).mean()))
