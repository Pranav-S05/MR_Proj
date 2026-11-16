"""Do edge weights start to matter once the scale bias is removed?

For every drive, everything is fitted on the OTHER drives (leave one drive out): the seam-scale correction and the
weighting schemes of the main project (uniform blocks, raw confidence, geometry, calibrated, plus the oracle that
uses the true error sizes). Solved with loops, no robust kernel; ATE in metres.

  python scripts/exp_weights.py --variant mid
"""
from __future__ import annotations

import argparse

import numpy as np

import lib
from mr_cpu import labels
from mr_cpu.weighting import Fitted
from exp_main import fit as fit_corr, pooled

SCHEMES = ("equal", "uniform", "raw_conf", "geometry", "calibrated", "oracle")


def corrected(d, corr):
    """Copy of drive d with seam scales corrected by exp(corr) and labels recomputed."""
    out = dict(d)
    t = {k: v.copy() for k, v in d["table"].items()}
    if corr is not None:
        t["z"][d["seam"]] = lib.rescale_seams(d, corr)
    out["table"] = t
    out["err"], out["valid"] = labels.edge_errors(d["gauges"], d["gauge_ok"], t["i"], t["j"], t["z"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mid")
    a = ap.parse_args()
    seqs = lib.available(a.variant)
    raw = {s: lib.load(a.variant, s) for s in seqs}
    y = {s: raw[s]["err"][raw[s]["seam"], 6] for s in seqs}
    ok = {s: raw[s]["valid"][raw[s]["seam"]] for s in seqs}
    feats = {s: lib.seam_features(raw[s]) for s in seqs}

    def c4(target, fitting):
        F, Y = pooled(fitting, feats, y, ok)
        return fit_corr("C4_fuse_depth+speed", F, Y)(feats[target])
    for cond in ("raw seams", "scale bias removed (constant from the other drives)",
                 "scale bias removed (drive's own mean: hindsight)", "true seam scale (hindsight)",
                 "scale calibrated with C4 (other drives)"):
        data = {}
        for s in seqs if not cond.startswith("scale calibrated") else []:
            if cond.startswith("raw"):
                data[s] = corrected(raw[s], None)
            elif "other drives" in cond:
                c = np.concatenate([y[o][ok[o]] for o in seqs if o != s]).mean()
                data[s] = corrected(raw[s], np.full(len(y[s]), c))
            elif "own mean" in cond:
                data[s] = corrected(raw[s], np.full(len(y[s]), y[s][ok[s]].mean()))
            else:
                data[s] = corrected(raw[s], np.where(ok[s], y[s], 0.0))
        print(f"\n=== {cond} ===")
        print(f"{'scheme':12}" + "".join(f"{s:>9}" for s in seqs) + f"{'mean':>9}")
        rows = {k: [] for k in SCHEMES}
        for s in seqs:
            others = [o for o in seqs if o != s]
            if cond.startswith("scale calibrated"):
                data = {s: corrected(raw[s], c4(s, others)),
                        **{o: corrected(raw[o], c4(o, [q for q in seqs if q not in (s, o)])) for o in others}}
            fit = Fitted.fit([data[o]["table"] for o in others], [data[o]["err"] for o in others],
                             [data[o]["valid"] for o in others])
            d = data[s]
            for scheme in SCHEMES:
                # the oracle is only defined on labelled edges; give the rest the uniform variances
                if scheme == "oracle":
                    info = fit.information("uniform", d["table"])
                    io = fit.information("oracle", d["table"], d["err"])
                    info[d["valid"]] = io[d["valid"]]
                else:
                    info = fit.information(scheme, d["table"])
                rows[scheme].append(lib.solve(d, info=info))
        for scheme in SCHEMES:
            print(f"{scheme:12}" + "".join(f"{x:9.2f}" for x in rows[scheme]) + f"{np.mean(rows[scheme]):9.2f}")


if __name__ == "__main__":
    main()
