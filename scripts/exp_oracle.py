"""Oracle component breakdown (PLAN.md X1): replace one component of every seam by its true value, score ATE.

  python scripts/exp_oracle.py --variant first_w30_o15
Components of the seam error label e = Log(Z^-1 A_i^-1 A_j): rotation 0-2, translation 3-5, log-scale 6.
Writes results/oracle_<variant>.csv.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

import lib
from mr_cpu import lie

PARTS = [("measured", []), ("true rotation", [0, 1, 2]), ("true translation", [3, 4, 5]), ("true scale", [6]),
         ("true rotation+scale", [0, 1, 2, 6]), ("true all", list(range(7)))]


def corrected(d, comps):
    k = d["seam"]
    e = d["err"][k].copy()
    keep = np.zeros(7, bool)
    keep[comps] = True
    e[:, ~keep] = 0
    e[~d["valid"][k]] = 0
    return d["table"]["z"][k] @ lie.exp(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="first_w30_o15")
    a = ap.parse_args()
    seqs = lib.available(a.variant)
    res = {}
    for s in seqs:
        d = lib.load(a.variant, s)
        for name, comps in PARTS:
            z = corrected(d, comps) if comps else None
            res[(name, s)] = (lib.solve(d, z, loops=False), lib.solve(d, z, loops=True))
    for idx, title in ((0, "chain, no loops"), (1, "with loops, equal weights")):
        print(f"\nATE {title} (m), variant {a.variant}")
        print(f"{'seam':22}" + "".join(f"{s:>8}" for s in seqs) + f"{'mean':>9}")
        for name, _ in PARTS:
            v = [res[(name, s)][idx] for s in seqs]
            print(f"{name:22}" + "".join(f"{x:8.2f}" for x in v) + f"{np.mean(v):9.2f}")
    lib.RES.mkdir(parents=True, exist_ok=True)
    with open(lib.RES / f"oracle_{a.variant}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "seq", "ate_chain", "ate_loops"])
        for (name, s), (c, l) in res.items():
            w.writerow([name, s, f"{c:.4f}", f"{l:.4f}"])


if __name__ == "__main__":
    main()
