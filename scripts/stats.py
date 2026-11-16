"""Paired statistics of a results CSV written by exp_main.py: per method against `none`, with loops and without.

  python scripts/stats.py results/first_w30_o15.csv [more.csv ...]
Reports: mean ATE, percent change of the mean, geometric-mean ratio, drives better / worse (0.5% band),
Wilcoxon signed-rank p and one-sample t p on log(ATE / ATE_none) (two-sided), and the worst ratio.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def main():
    for path in sys.argv[1:]:
        rows = defaultdict(dict)
        for r in csv.DictReader(open(path)):
            rows[r["method"]][r["seq"]] = (float(r["ate_chain"]), float(r["ate_loops"]))
        seqs = sorted(rows["none"])
        print(f"\n{path}: {len(seqs)} drives")
        for idx, title in ((1, "with loops"), (0, "chain")):
            print(f"  {title}")
            base = np.array([rows["none"][s][idx] for s in seqs])
            print(f"    {'method':22}{'mean':>8}{'change':>8}{'gmean':>7}{'b/w':>7}{'p_wilc':>8}{'p_t':>8}{'worst':>7}{'ci95':>14}")
            for m in rows:
                v = np.array([rows[m][s][idx] for s in seqs])
                lr = np.log(v / base)
                rng = np.random.default_rng(0)
                boot = rng.integers(0, len(seqs), (10000, len(seqs)))
                ch = v[boot].mean(1) / base[boot].mean(1) - 1
                lo, hi = np.percentile(ch, [2.5, 97.5])
                b, w = int((lr < np.log(0.995)).sum()), int((lr > np.log(1.005)).sum())
                if m == "none":
                    pw = pt = 1.0
                else:
                    pw = stats.wilcoxon(lr).pvalue if np.any(lr != 0) else 1.0
                    pt = stats.ttest_1samp(lr, 0).pvalue
                print(f"    {m:22}{v.mean():8.2f}{100 * (v.mean() / base.mean() - 1):+7.1f}%{np.exp(lr.mean()):7.3f}"
                      f"{b:>4}/{w:<2}{pw:8.3f}{pt:8.3f}{np.exp(lr.max()):7.2f}"
                      f"{f'[{100 * lo:+.0f}%,{100 * hi:+.0f}%]':>14}")


if __name__ == "__main__":
    main()
