"""Calibration of error second moments: isotonic maps, reliability statistics, power laws.

Everything here is supervised post-hoc calibration: it is fitted on labelled fitting sequences
(05/07) and must be scored on other sequences. Targets are squared scalar errors (or squared
block norms divided by the block dimension); fitted values are second moments, which equal
variances only when the error is unbiased.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


def _pav(y, w):
    """Pooled adjacent violators for a non-decreasing fit (weighted)."""
    blocks = []  # [start, end, weight, mean]
    for k, (yy, ww) in enumerate(zip(y, w)):
        blocks.append([k, k, ww, yy])
        while len(blocks) > 1 and blocks[-2][3] > blocks[-1][3]:
            r, l = blocks.pop(), blocks.pop()
            tot = l[2] + r[2]
            blocks.append([l[0], r[1], tot, (l[2] * l[3] + r[2] * r[3]) / tot])
    out = np.empty(len(y))
    for a, b, _, m in blocks:
        out[a:b + 1] = m
    return out


@dataclass
class Isotonic:
    """Monotone map x -> E[y | x]; decreasing=True for confidences, False for residuals."""
    x: np.ndarray
    y: np.ndarray
    decreasing: bool
    floor: float

    @classmethod
    def fit(cls, x, y, decreasing=True, floor=1e-12):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 2:
            raise ValueError("Need at least two finite samples")
        sign = -1. if decreasing else 1.
        ux, inverse, counts = np.unique(sign * x, return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=y) / counts
        fitted = np.maximum(_pav(means, counts.astype(float)), floor)
        return cls(ux, fitted, decreasing, floor)

    def predict(self, x):
        sign = -1. if self.decreasing else 1.
        return np.maximum(np.interp(sign * np.asarray(x, float), self.x, self.y), self.floor)

    def to_dict(self):
        return {"x": self.x.tolist(), "y": self.y.tolist(), "decreasing": self.decreasing, "floor": self.floor}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["x"]), np.array(d["y"]), d["decreasing"], d["floor"])


# Kept for the original teaching lab / tests.
@dataclass
class MonotoneVariance:
    x: np.ndarray
    variance: np.ndarray
    floor: float

    @classmethod
    def fit(cls, confidence, squared_error, floor=1e-8):
        confidence, squared_error = np.asarray(confidence, float), np.asarray(squared_error, float)
        if confidence.ndim != 1 or confidence.shape != squared_error.shape or len(confidence) < 2:
            raise ValueError("Need at least two matching one-dimensional arrays")
        if not np.all(np.isfinite(confidence)) or not np.all(np.isfinite(squared_error)) or np.any(confidence <= 0) or np.any(squared_error < 0) or not np.isfinite(floor) or floor <= 0:
            raise ValueError("Confidence and floor must be positive; targets finite and nonnegative")
        iso = Isotonic.fit(confidence, squared_error, decreasing=True, floor=floor)
        return cls(iso.x, iso.y, floor)

    def predict(self, confidence):
        confidence = np.asarray(confidence, float)
        if np.any(confidence <= 0) or not np.all(np.isfinite(confidence)):
            raise ValueError("Confidence must be finite and positive")
        return np.maximum(np.interp(-confidence, self.x, self.variance), self.floor)


def scalar_scores(error, variance):
    error, variance = np.asarray(error, float), np.asarray(variance, float)
    if error.shape != variance.shape or error.size == 0 or np.any(variance <= 0) or not np.all(np.isfinite(error)) or not np.all(np.isfinite(variance)):
        raise ValueError("Need matching finite errors and positive variances")
    return {"mean_normalized_squared_error_target_1": float(np.mean(error**2 / variance)), "gaussian_nll": float(np.mean(.5 * (np.log(2 * np.pi * variance) + error**2 / variance))), "coverage_approximately_95_percent": float(np.mean(np.abs(error) <= 1.96 * np.sqrt(variance)))}


def equal_count_edges(x, n_bins=10):
    """Bin edges (fixed on fitting data) giving roughly equal counts."""
    q = np.quantile(np.asarray(x, float), np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    return np.unique(q)


def reliability(x, sq_err, pred_var, edges):
    """Per-bin of x: count, mean x, empirical RMS error, predicted RMS, ratio."""
    x, sq_err, pred_var = (np.asarray(a, float) for a in (x, sq_err, pred_var))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if not m.any():
            continue
        rows.append({"bin": k, "count": int(m.sum()), "mean_x": float(np.mean(x[m])),
                     "empirical_rms": float(np.sqrt(np.mean(sq_err[m]))),
                     "predicted_rms": float(np.sqrt(np.mean(pred_var[m]))),
                     "ratio_msq": float(np.mean(sq_err[m]) / np.mean(pred_var[m]))})
    return rows


def calibration_error(rows):
    """sum_b (n_b / N) |mean_b(e^2) / mean_b(predicted variance) - 1| (0 = calibrated)."""
    n = sum(r["count"] for r in rows)
    return float(sum(r["count"] / n * abs(r["ratio_msq"] - 1) for r in rows))


def gaussian_nll(sq_err, pred_var):
    sq_err, pred_var = np.asarray(sq_err, float), np.asarray(pred_var, float)
    return float(np.mean(.5 * (np.log(2 * np.pi * pred_var) + sq_err / pred_var)))


def coverage_curve(sq_err, pred_var, dof=1, levels=np.linspace(.05, .95, 19)):
    """Fraction of normalised errors inside the nominal chi-square(dof) quantile, per level."""
    from scipy.stats import chi2
    q = np.asarray(sq_err, float) / np.asarray(pred_var, float) * dof
    return [(float(p), float(np.mean(q <= chi2.ppf(p, dof)))) for p in levels]


def loglog_slope(conf, err, n_bins=20):
    """Slope b of log RMS(err | conf-bin) against log mean(conf). The loss-optimum idealisation of
    VGGT's confidence (C ~ alpha / E[e]) predicts b = -1. Returns (b, intercept)."""
    conf, err = np.asarray(conf, float), np.asarray(err, float)
    edges = equal_count_edges(conf, n_bins)
    rows = reliability(conf, err ** 2, np.ones_like(err), edges)
    lx = np.log([r["mean_x"] for r in rows])
    ly = np.log([r["empirical_rms"] for r in rows])
    b, a = np.polyfit(lx, ly, 1)
    return float(b), float(a)


def fit_power_law(conf, err, n_bins=20):
    """Two-constant power law sigma = K * conf^(-p) (Deng et al. 2026 form, with sigma_hat = alpha/C):
    fitted on binned RMS in log-log space. Returns (K, p)."""
    b, a = loglog_slope(conf, err, n_bins)
    return float(np.exp(a)), float(-b)


def shrunk_inflation(target, predicted, prior_count=5.0):
    """Multiplicative loop-edge inflation exp(n/(n+k) * mean log(target/pred)), shrunk to 1."""
    target, predicted = np.asarray(target, float), np.asarray(predicted, float)
    ok = (target > 0) & (predicted > 0)
    n = ok.sum()
    if n == 0:
        return 1.0
    return float(np.exp(n / (n + prior_count) * np.mean(np.log(target[ok] / predicted[ok]))))


def normal_quantile(p):
    return float(norm.ppf(p))
