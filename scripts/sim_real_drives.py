"""The one-loop simulation of sim_bias_vs_weights.py, fed with each real drive's measured seam errors (E2 add-on).

  python scripts/sim_real_drives.py --variant first_w30_o15
Per drive, from the true log-scale errors y_k of its seams: bias b = mean y; local noise sigma_k from a rolling RMS
(21 seams) of y - b; bias strength = N |b| / sqrt(sum sigma^2); clustering = max over m of |m/N - V_m/V| (0 = noisy
seams spread evenly). Then the simulation with that drive's sigma_k and b (one loop over the whole drive, 2000
noise draws): RMS accumulated error with perfect variance weights / with equal weights. Below 1 = weights help.
"""
import argparse

import numpy as np

import lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="first_w30_o15")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    print(f"{'seq':4}{'N':>5}{'bias':>9}{'strength':>10}{'cluster':>9}{'perfect/equal':>15}{'(no bias)':>11}")
    for s in lib.available(a.variant):
        d = lib.load(a.variant, s)
        k = d["seam"]
        y = d["err"][k, 6][d["valid"][k]]
        n, b = len(y), y.mean()
        w = min(21, n - 1 + n % 2)
        r2 = np.convolve((y - b) ** 2, np.ones(w) / w, mode="same")
        sig = np.sqrt(np.maximum(r2, 1e-8))
        v = np.cumsum(sig ** 2)
        clus = np.max(np.abs(np.arange(1, n + 1) / n - v / v[-1]))
        strength = n * abs(b) / np.sqrt(v[-1])
        out = []
        for bias in (b, 0.0):
            acc = np.zeros(2)
            for _ in range(2000):
                e = bias + sig * rng.standard_normal(n)
                for m, share in enumerate((np.full(n, 1 / n), sig ** 2 / v[-1])):
                    acc[m] += np.sqrt(np.mean(np.cumsum(e - share * e.sum()) ** 2))
            out.append(acc[1] / acc[0])
        print(f"{s:4}{n:5d}{b:+9.4f}{strength:10.2f}{clus:9.3f}{out[0]:15.3f}{out[1]:11.3f}")


if __name__ == "__main__":
    main()
