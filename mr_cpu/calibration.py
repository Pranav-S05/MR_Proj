"""One-feature calibration of second moments, for independent fit/evaluation sets."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MonotoneVariance:
    x: np.ndarray
    variance: np.ndarray
    floor: float

    @classmethod
    def fit(cls, confidence, squared_error, floor=1e-8):
        """Fit decreasing variance versus confidence using pooled adjacent violators.

        Targets must be squared scalar errors, or squared block norm / block DoF.
        These estimate second moments; they are covariance only with zero bias.
        """
        confidence, squared_error = np.asarray(confidence, float), np.asarray(squared_error, float)
        if confidence.ndim != 1 or confidence.shape != squared_error.shape or len(confidence) < 2:
            raise ValueError("Need at least two matching one-dimensional arrays")
        if not np.all(np.isfinite(confidence)) or not np.all(np.isfinite(squared_error)) or np.any(confidence <= 0) or np.any(squared_error < 0) or not np.isfinite(floor) or floor <= 0:
            raise ValueError("Confidence and floor must be positive; targets finite and nonnegative")
        # Negating confidence turns a decreasing fit into an increasing fit.
        x, groups, count = np.unique(-confidence, return_inverse=True, return_counts=True)
        means = np.bincount(groups, weights=squared_error) / count
        blocks = []
        for i, (mean, n) in enumerate(zip(means, count)):
            blocks.append([i, i, float(n), float(mean)])
            while len(blocks) > 1 and blocks[-2][3] > blocks[-1][3]:
                right, left = blocks.pop(), blocks.pop()
                total = left[2] + right[2]
                blocks.append([left[0], right[1], total, (left[2] * left[3] + right[2] * right[3]) / total])
        fitted = np.empty(len(x))
        for start, end, _, mean in blocks:
            fitted[start:end + 1] = max(mean, floor)
        return cls(x, fitted, floor)

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
