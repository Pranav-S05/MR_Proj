"""Main experiment: VGGT-Long against VGGT-Long + calibrated seam scale (PLAN.md X1-X3).

  python scripts/exp_main.py --variant first_w30_o15
  python scripts/exp_main.py --variant first_w60_o30 --transfer-from first_w30_o15

Every correction c is added to the log of VGGT-Long's seam scale and is fitted on OTHER drives only (leave one drive
out). Methods:
  none        VGGT-Long's seam as is
  C1 const    the mean log-scale error of the fitting drives
  C2 iso_speed  isotonic curve of the error against travel per window / scene size (`speed`), fallback C1
  C3 fuse_depth ridge on (depth-ratio scale minus point-cloud scale): a calibrated fusion of two scale cues
  C3i iso_depth isotonic curve of the error against the same depth-minus-point feature
  C3p fuse_depth_pose  ridge on depth-minus-point and pose-minus-point
  C4 fuse_depth+speed  C3, then an isotonic curve of its remaining error against `speed`
  auto        nested selection: for each held-out drive, pick among C1-C4 by an inner leave-one-out on the
              fitting drives only (score: mean log ATE ratio to `none`, with loops). Honest estimate of "pick the best"
  oracle      the true log-scale error (bound)
With --transfer-from, every method is also fitted once on ALL drives of another variant and applied unchanged here.
Writes results/<variant>[_from_<src>].csv and prints the tables.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import lib
from mr_cpu.calibration import Isotonic

CANDIDATES = ["C1_const", "C2_iso_speed", "C3_fuse_depth", "C3i_iso_depth", "C3p_fuse_depth_pose", "C4_fuse_depth+speed"]
OUT = lib.RES


def ridge_fit(x, y, lam=1e-3):
    mu, sd = x.mean(0), x.std(0) + 1e-12
    a = np.c_[np.ones(len(x)), (x - mu) / sd]
    reg = lam * len(x) * np.eye(a.shape[1])
    reg[0, 0] = 0
    return mu, sd, np.linalg.solve(a.T @ a + reg, a.T @ y)


def ridge_predict(m, x):
    mu, sd, w = m
    return np.c_[np.ones(len(x)), (x - mu) / sd] @ w


def iso_fit(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    rho = np.corrcoef(np.argsort(np.argsort(x[ok])), y[ok])[0, 1]  # Spearman sign decides the direction
    return Isotonic.fit(x[ok], y[ok], decreasing=bool(rho < 0), floor=-np.inf)


def fit(method, F, Y):
    """F: dict feature -> array over fitting seams, Y: log-scale errors. Returns predictor(features) -> c."""
    const = float(Y.mean())

    def with_fallback(cols, pred):
        def p(f):
            x = np.c_[[f[c] for c in cols]].T
            fin = np.isfinite(x).all(1)
            out = np.full(len(x), const)
            if fin.any():
                out[fin] = pred(x[fin])
            return out
        return p

    def rows(cols):
        x = np.c_[[F[c] for c in cols]].T
        good = np.isfinite(x).all(1)
        return x[good], Y[good]

    if method == "C1_const":
        return lambda f: np.full(len(f["log_s_point"]), const)
    if method == "C2_iso_speed":
        x, y = rows(["speed"])
        m = iso_fit(x[:, 0], y)
        return with_fallback(["speed"], lambda x: m.predict(x[:, 0]))
    if method in ("C3_fuse_depth", "C3p_fuse_depth_pose"):
        cols = ["depth_minus_point"] + (["pose_minus_point"] if method == "C3p_fuse_depth_pose" else [])
        x, y = rows(cols)
        m = ridge_fit(x, y)
        return with_fallback(cols, lambda x: ridge_predict(m, x))
    if method == "C3i_iso_depth":
        x, y = rows(["depth_minus_point"])
        m = iso_fit(x[:, 0], y)
        return with_fallback(["depth_minus_point"], lambda x: m.predict(x[:, 0]))
    if method == "C4_fuse_depth+speed":
        p3 = fit("C3_fuse_depth", F, Y)
        r = Y - p3(F)
        x = F["speed"]
        m = iso_fit(x, r)
        def p(f):
            c = p3(f)
            fin = np.isfinite(f["speed"])
            c[fin] += m.predict(f["speed"][fin])
            return c
        return p
    raise KeyError(method)


def load_variant(variant):
    seqs = lib.available(variant)
    data = {s: lib.load(variant, s) for s in seqs}
    feats = {s: lib.seam_features(data[s]) for s in seqs}
    y = {s: data[s]["err"][data[s]["seam"], 6] for s in seqs}
    ok = {s: data[s]["valid"][data[s]["seam"]] for s in seqs}
    return seqs, data, feats, y, ok


def pooled(seqs, feats, y, ok):
    F = {k: np.concatenate([feats[s][k][ok[s]] for s in seqs]) for k in feats[seqs[0]]}
    return F, np.concatenate([y[s][ok[s]] for s in seqs])


class Scorer:
    """Caches ATE (chain, loops) per (drive, method, fitting set)."""

    def __init__(self, data, feats, y, ok):
        self.data, self.feats, self.y, self.ok = data, feats, y, ok
        self.cache = {}

    def corr(self, s, method, fitting):
        if method == "none":
            return np.zeros(len(self.y[s]))
        if method == "oracle":
            return np.where(self.ok[s], self.y[s], 0.0)
        F, Y = pooled(fitting, self.feats, self.y, self.ok)
        return fit(method, F, Y)(self.feats[s])

    def ate(self, s, method, fitting, loops_only=False):
        key = (s, method, tuple(fitting), loops_only)
        if key not in self.cache:
            c = self.corr(s, method, fitting)
            d = self.data[s]
            z = None if method == "none" else lib.rescale_seams(d, c)
            chain = math.nan if loops_only else lib.solve(d, z, loops=False)
            r = (self.y[s] - c)[self.ok[s]]
            self.cache[key] = (chain, lib.solve(d, z, loops=True), float(r.mean()), float(r.std()))
        return self.cache[key]


def choose(scorer, fitting):
    """Inner leave-one-out over the fitting drives; returns the candidate with the lowest mean log ATE ratio."""
    best, best_score = None, math.inf
    for m in CANDIDATES:
        sc = []
        for s in fitting:
            inner = [o for o in fitting if o != s]
            base = scorer.ate(s, "none", inner, loops_only=True)[1]
            sc.append(math.log(scorer.ate(s, m, inner, loops_only=True)[1] / base))
        if np.mean(sc) < best_score:
            best, best_score = m, float(np.mean(sc))
    return best


def sign_test(k, n):
    """Two-sided exact sign test p-value for k improvements out of n."""
    t = max(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(t, n + 1)) / 2 ** n)


def summarise(rows, methods, seqs, idx, title):
    print(f"\nATE {title} (m)")
    print(f"{'method':22}" + "".join(f"{s:>8}" for s in seqs) + f"{'mean':>9}{'gmean_ratio':>12}{'better':>8}{'worst':>8}{'p_sign':>8}")
    base = np.array([rows[("none", s)][idx] for s in seqs])
    for m in methods:
        v = np.array([rows[(m, s)][idx] for s in seqs])
        ratio = v / base
        k = int((ratio < 0.995).sum())
        n = int((np.abs(ratio - 1) >= 0.005).sum())
        print(f"{m:22}" + "".join(f"{x:8.2f}" for x in v) + f"{v.mean():9.2f}{np.exp(np.log(ratio).mean()):12.3f}"
              f"{k:>5}/{len(seqs):<2}{ratio.max():8.2f}{sign_test(k, n) if n else 1.0:8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="first_w30_o15")
    ap.add_argument("--transfer-from")
    ap.add_argument("--no-auto", action="store_true")
    a = ap.parse_args()
    seqs, data, feats, y, ok = load_variant(a.variant)
    sc = Scorer(data, feats, y, ok)
    methods = ["none"] + CANDIDATES + ([] if a.no_auto else ["auto"]) + ["oracle"]
    rows, picks = {}, {}
    for s in seqs:
        fitting = [o for o in seqs if o != s]
        for m in methods:
            if m == "auto":
                picks[s] = choose(sc, fitting)
                rows[(m, s)] = sc.ate(s, picks[s], fitting)
            else:
                rows[(m, s)] = sc.ate(s, m, fitting)
    tag = a.variant
    if a.transfer_from:
        s2, d2, f2, y2, ok2 = load_variant(a.transfer_from)
        F, Y = pooled(s2, f2, y2, ok2)
        for m in CANDIDATES:
            p = fit(m, F, Y)
            name = f"T:{m}"
            for s in seqs:
                c = p(feats[s])
                z = lib.rescale_seams(data[s], c)
                r = (y[s] - c)[ok[s]]
                rows[(name, s)] = (lib.solve(data[s], z, loops=False), lib.solve(data[s], z, loops=True),
                                   float(r.mean()), float(r.std()))
            methods.insert(-1, name)
        tag += f"_from_{a.transfer_from}"
    print(f"variant {a.variant}: {len(seqs)} drives {seqs}; loops per drive "
          + " ".join(f"{s}:{data[s]['n_loops']}" for s in seqs))
    print("\nseam log-scale error left (mean)")
    print(f"{'method':22}" + "".join(f"{s:>8}" for s in seqs))
    for m in methods:
        print(f"{m:22}" + "".join(f"{rows[(m, s)][2]:8.4f}" for s in seqs))
    summarise(rows, methods, seqs, 0, "chain, no loops")
    summarise(rows, methods, seqs, 1, "with loops, equal weights")
    if picks:
        print("\nauto picked: " + " ".join(f"{s}:{p}" for s, p in picks.items()))
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seq", "ate_chain", "ate_loops", "resid_mean", "resid_std", "auto_pick"])
        for m in methods:
            for s in seqs:
                w.writerow([m, s, *[f"{v:.4f}" for v in rows[(m, s)]], picks.get(s, "") if m == "auto" else ""])
    print(f"\nwrote {OUT / (tag + '.csv')}")


if __name__ == "__main__":
    main()
