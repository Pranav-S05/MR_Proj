"""Coefficients of the calibrated seam scale for the patched upstream VGGT-Long (PLAN.md X4).

  python scripts/fit_upstream_coeffs.py --variant first_w75_o30

For every KITTI drive 00-10, C1 and C3 are fitted on the OTHER drives of our run `variant` (same chunk setting) and
written to results/final/runs/upstream/coeffs_<variant>.json in the form read by phoenix/make_upstream_calib.py.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import lib
from exp_main import fit as fit_corr, iso_fit, load_variant, pooled, ridge_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    a = ap.parse_args()
    seqs, data, feats, y, ok = load_variant(a.variant)
    out = {"C4": {}, "C1": {}, "C3": {}, "fitted_on": {}}
    for s in lib.ALL:
        fitting = [o for o in seqs if o != s]
        F, Y = pooled(fitting, feats, y, ok)
        c = float(Y.mean())
        x = F["depth_minus_point"]
        g = np.isfinite(x)
        mu, sd, w = ridge_fit(x[g, None], Y[g])
        b = float(w[1] / sd[0])
        out["C1"][s] = {"a": c, "b": 0.0, "a0": c}
        out["C3"][s] = {"a": float(w[0] - b * mu[0]), "b": b, "a0": c}
        p3 = fit_corr("C3_fuse_depth", F, Y)
        m = iso_fit(F["speed"], Y - p3(F))
        xs, ys = (m.x, m.y) if not m.decreasing else (-m.x[::-1], m.y[::-1])
        keep = np.r_[True, (ys[1:-1] != ys[:-2]) | (ys[1:-1] != ys[2:]), True]
        out["C4"][s] = {**out["C3"][s], "iso_x": xs[keep].tolist(), "iso_y": ys[keep].tolist()}
        out["fitted_on"][s] = fitting
    p = lib.RUNS / "upstream" / f"coeffs_{a.variant}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    for s in lib.ALL:
        print(s, "C1 %+.4f" % out["C1"][s]["a"], "C3 a %+.4f b %+.3f" % (out["C3"][s]["a"], out["C3"][s]["b"]))
    print("wrote", p)


if __name__ == "__main__":
    main()
