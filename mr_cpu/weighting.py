"""Edge information matrices for the compared weighting schemes.

All matrices are block-diagonal in our residual order [rotation (3), upsilon (3), log-scale (1)]:
  Lambda = diag(1/v_rot I3, 1/(v_trans L^2) I3, 1/v_scale),
where v_* are per-coordinate second moments and L is the edge's characteristic length (median
predicted depth of node j's window, so translation variances are scale-free).

Schemes (labels follow the proposal):
  a  vggt_long   identity information with VGGT-Long's residual convention (reproduction anchor)
  b  uniform     one fitted variance per block, the same for every edge (pure DoF balancing)
  c  raw_conf    uniform blocks scaled by (mean confidence)^2, the loss-implied heuristic
  d  geometry    isotonic map from the alignment's relative residual (geometry only)
  e  calibrated  isotonic map from mean point confidence (ours)
  f  oracle      per-edge true squared errors (hindsight diagnostic, uses test GT, not a method)
  g  power_law   Deng et al. two-constant power law per pixel, summed over the edge's pixels
Every fitted quantity comes from the fitting sequences only. For the graph, all schemes except
(a) are multiplied by a constant, fixed on the fitting seams, that gives them the same mean
information trace as (b) ("same average weight"). The constants are not recomputed per sequence,
so kernel thresholds tuned on the fitting sequences keep their meaning on held-out ones. The
unscaled variances are what calibration metrics (NEES etc.) are computed on.
"""
from __future__ import annotations

import numpy as np

from .alignment import CONF_HIST_CENTRES
from .calibration import Isotonic, shrunk_inflation, fit_power_law
from .labels import block_targets

BLOCKS = ("rot", "trans", "scale")
SCHEMES = {"a": "vggt_long", "b": "uniform", "c": "raw_conf", "d": "geometry", "e": "calibrated",
           "f": "oracle", "g": "power_law"}
FLOORS = {"rot": (1e-4) ** 2, "trans": (1e-4) ** 2, "scale": (1e-5) ** 2}


def info_from_var(v, char_length):
    """(E,) variances per block -> (E,7,7) diagonal information."""
    n = len(char_length)
    d = np.empty((n, 7))
    d[:, :3] = 1 / v["rot"][:, None]
    d[:, 3:6] = 1 / (v["trans"] * char_length ** 2)[:, None]
    d[:, 6] = 1 / v["scale"]
    return np.einsum("ni,ij->nij", d, np.eye(7))


class Fitted:
    """Everything fitted on the fitting sequences, serialisable to JSON."""

    def __init__(self, d):
        self.d = d

    @classmethod
    def fit(cls, tables, errors, valid, pixel_conf=None, pixel_err=None, conf_feature="conf_mean",
            geo_feature="residual_rel"):
        """tables: list of edge tables; errors: list of (E,7); valid: list of bool masks."""
        t = {k: np.concatenate([tb[k] for tb in tables]) for k in tables[0]}
        e, ok = np.concatenate(errors), np.concatenate(valid)
        tg = block_targets(e, t["f_char_length"])
        seam, loop = (t["kind"] == 0) & ok, (t["kind"] == 1) & ok
        d = {"conf_feature": conf_feature, "geo_feature": geo_feature,
             "n_seams": int(seam.sum()), "n_loops": int(loop.sum())}
        d["uniform"] = {b: float(np.mean(tg[b][seam])) for b in BLOCKS}
        d["mean_conf"] = float(np.mean(t["f_" + conf_feature][seam]))
        d["models"], d["loop_inflation"] = {}, {}
        for name, feat, dec in (("calibrated", conf_feature, True), ("geometry", geo_feature, False)):
            x = t["f_" + feat]
            d["models"][name] = {b: Isotonic.fit(x[seam], tg[b][seam], decreasing=dec, floor=FLOORS[b]).to_dict()
                                 for b in BLOCKS}
            pred = {b: Isotonic.from_dict(d["models"][name][b]).predict(x[loop]) for b in BLOCKS}
            d["loop_inflation"][name] = {b: shrunk_inflation(tg[b][loop], pred[b]) for b in BLOCKS}
        d["loop_inflation"]["uniform"] = {b: shrunk_inflation(tg[b][loop], np.full(loop.sum(), d["uniform"][b]))
                                          for b in BLOCKS}
        if pixel_conf is not None and len(pixel_conf):
            k, p = fit_power_law(pixel_conf, pixel_err)
            d["power_law"] = {"K": k, "p": p}
        else:
            d["power_law"] = {"K": 1.0, "p": 1.0}
        # independent-pixel information is far too large (pixels share errors); its overall scale is
        # therefore fitted on the fitting seams, leaving only the relative weighting to the power law
        rel = _power_law_rel(t, d["power_law"]["p"])
        d["power_law"]["seam_scale"] = {b: float(np.mean(tg[b][seam] * rel[seam])) for b in BLOCKS}
        out = cls(d)
        # Graph rescaling constants, fixed on the fitting seams: each scheme gets the same mean
        # information trace as (b). Fixed constants keep kernel thresholds comparable across sequences.
        # Averages run over ALL fitting seams (no labels needed), except the oracle (labelled seams only).
        d["norm"] = {}
        for scheme in ("uniform", "raw_conf", "geometry", "calibrated", "power_law", "oracle"):
            use = seam if scheme == "oracle" else t["kind"] == 0
            sub = {k: v[use] for k, v in t.items()}
            ref = np.trace(info_from_var(out.variances("uniform", sub), sub["f_char_length"]), axis1=1, axis2=2).mean()
            info = info_from_var(out.variances(scheme, sub, e[use]), sub["f_char_length"])
            d["norm"][scheme] = float(ref / np.trace(info, axis1=1, axis2=2).mean())
        return out

    def variances(self, scheme, table, errors=None):
        """Per-block per-coordinate variances (E,) for a scheme, before any graph rescaling."""
        n = len(table["kind"])
        loop = table["kind"] == 1
        u = self.d["uniform"]
        if scheme in ("uniform", "raw_conf"):
            v = {b: np.full(n, u[b]) for b in BLOCKS}
            infl = self.d["loop_inflation"]["uniform"]
            v = {b: np.where(loop, v[b] * infl[b], v[b]) for b in BLOCKS}
            if scheme == "raw_conf":
                c = table["f_" + self.d["conf_feature"]] / self.d["mean_conf"]
                v = {b: v[b] / np.maximum(c, 1e-6) ** 2 for b in BLOCKS}
            return v
        if scheme in ("calibrated", "geometry"):
            feat = self.d["conf_feature"] if scheme == "calibrated" else self.d["geo_feature"]
            x = table["f_" + feat]
            infl = self.d["loop_inflation"][scheme]
            v = {}
            for b in BLOCKS:
                pred = Isotonic.from_dict(self.d["models"][scheme][b]).predict(x)
                v[b] = np.where(loop, pred * infl[b], pred)
            return v
        if scheme == "power_law":
            rel = _power_law_rel(table, self.d["power_law"]["p"])
            return {b: self.d["power_law"]["seam_scale"][b] / rel for b in BLOCKS}
        if scheme == "oracle":
            if errors is None:
                raise ValueError("Oracle needs the true edge errors")
            tg = block_targets(errors, table["f_char_length"])
            return {b: np.maximum(tg[b], FLOORS[b]) for b in BLOCKS}
        raise ValueError(scheme)

    def information(self, scheme, table, errors=None, normalise=True):
        if scheme == "vggt_long":
            return np.tile(np.eye(7), (len(table["kind"]), 1, 1))
        info = info_from_var(self.variances(scheme, table, errors), table["f_char_length"])
        return info * self.d["norm"][scheme] if normalise else info


def _power_law_rel(table, p):
    """Relative edge information from per-pixel sigma ~ C^-p and independent pixels:
    I = sum_px C^(2p). Loop edges chain two alignments: 1/I = 1/I_a + 1/I_b."""
    w = CONF_HIST_CENTRES ** (2 * p)
    i1 = table["conf_hist"] @ w
    i2 = table["conf_hist2"] @ w
    loop = table["kind"] == 1
    return np.where(loop, 1 / (1 / np.maximum(i1, 1e-300) + 1 / np.maximum(i2, 1e-300)), i1)
