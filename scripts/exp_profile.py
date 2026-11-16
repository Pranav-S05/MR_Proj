"""In-window scale profile and seam scale bias of a variant (needs ground truth; diagnosis only).

  python scripts/exp_profile.py --variant first_w30_o15 [--seqs 07 05]
Prints, per drive: the step-length profile (predicted / true step, by position in the window), the seam
log-scale error (mean, std, lag-1 autocorrelation), the total scale drift of the chain, and ATE.
"""
import argparse

import numpy as np

import lib

ap = argparse.ArgumentParser()
ap.add_argument("--variant", nargs="+", default=["mid"])
ap.add_argument("--seqs", nargs="*")
a = ap.parse_args()
for variant in a.variant:
    seqs = [s for s in lib.available(variant) if not a.seqs or s in a.seqs]
    print(f"\n=== {variant} ===")
    print("seq  wins | step profile at positions 0%, 10%, ... 100% of the window            | half ratio | seam logs err: mean"
          "    std   lag1 | drift x | loops | ATE chain / loops (equal) / loops (VGGT-Long residual)")
    tot = []
    for s in seqs:
        d = lib.load(variant, s)
        prof = lib.step_profile(d)
        e = d["err"][d["seam"], 6][d["valid"][d["seam"]]]
        lag = np.corrcoef(e[:-1], e[1:])[0, 1] if len(e) > 3 else np.nan
        if len(prof):
            m = np.exp(np.log(prof).mean(0))
            h = prof.shape[1] // 2
            ratio = np.exp(np.log(prof[:, :h]).mean() - np.log(prof[:, h:]).mean())
            pick = np.linspace(0, len(m) - 1, 11).round().astype(int)
            ptxt = " ".join(f"{m[k]:.3f}" for k in pick)
        else:
            ratio, ptxt = np.nan, "(no window with the car moving throughout)"
        ates = (lib.solve(d, loops=False), lib.solve(d), lib.solve(d, mode="upstream"))
        tot.append(ates)
        print(f"{s}  {len(prof):4d} | {ptxt} | {ratio:10.4f} | {e.mean():19.4f} {e.std():6.4f} {lag:6.2f} | "
              f"{np.exp(abs(e.sum())):7.1f} | {d['n_loops']:5d} | {ates[0]:7.2f} / {ates[1]:7.2f} / {ates[2]:7.2f}")
    if tot:
        t = np.mean(tot, 0)
        print(f"mean ATE over {len(tot)} drives: chain {t[0]:.2f}, loops (equal) {t[1]:.2f}, loops (VGGT-Long residual) {t[2]:.2f}")
