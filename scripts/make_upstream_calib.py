"""Make external/VGGT-Long/vggt_long_calib.py: upstream vggt_long.py plus the calibrated seam scale (PLAN.md X4).

usage (repository root on Phoenix): python scripts/make_upstream_calib.py
The original vggt_long.py is not changed. The diff is written to scripts/vggt_long_calib.diff.
At run time the correction is read from the environment variable SEAM_CALIB (JSON):
  {"a": .., "b": .., "a0": ..}   log s += a + b * x,  x = median over shared frames of median log(depth1 / depth2)
                                  minus log s  (a0 when fewer than 4 frames have 30 valid pixels)
  {"a": c, "b": 0, "a0": c}      the constant correction (C1)
Unset SEAM_CALIB = unpatched behaviour. Scale is changed about window 2's overlap camera centroid, as in study/lib.py.
"""
import difflib
from pathlib import Path

VL = Path(__file__).resolve().parents[1] / "external" / "VGGT-Long"
src = (VL / "vggt_long.py").read_text()
anchor = '            print("Estimated Scale:", s)\n'
assert src.count(anchor) == 1, "anchor not unique"
patch = '''            _calib = os.environ.get("SEAM_CALIB")
            if _calib:  # calibrated seam scale (final report, Sec. VI)
                import json as _json
                _p = _json.loads(_calib)
                _d1, _d2 = chunk_data1['depth'][-self.overlap:], chunk_data2['depth'][:self.overlap]
                _c1, _c2 = chunk_data1['depth_conf'][-self.overlap:], chunk_data2['depth_conf'][:self.overlap]
                _thr = min(np.median(_c1), np.median(_c2)) * 0.1
                _meds = []
                for _k in range(len(_d1)):
                    _ok = (_c1[_k] > _thr) & (_c2[_k] > _thr) & (_d1[_k] > 0) & (_d2[_k] > 0)
                    if _ok.sum() >= 30:
                        _meds.append(np.median(np.log(_d1[_k][_ok].astype(np.float64) / _d2[_k][_ok])))
                _c = _p["a"] + _p["b"] * (np.median(_meds) - np.log(s)) if len(_meds) >= 4 else _p["a0"]
                _e1 = chunk_data1['extrinsic'][-self.overlap:].astype(np.float64)
                _e2 = chunk_data2['extrinsic'][:self.overlap].astype(np.float64)
                if "iso_x" in _p:
                    _si = np.linalg.norm(np.diff(_e1[:, :3, 3], axis=0), axis=1)
                    _sj = np.linalg.norm(np.diff(_e2[:, :3, 3], axis=0), axis=1)
                    _pthr = min(np.median(conf1), np.median(conf2)) * 0.1
                    _v = (conf1 > _pthr) & (conf2 > _pthr)
                    _dep = np.einsum('nhwk,nk->nhw', point_map2 - _e2[:, None, None, :3, 3], _e2[:, :3, 2])[_v]
                    _dep = _dep[np.isfinite(_dep) & (_dep > 0)]
                    _char = np.median(_dep) if len(_dep) else np.nan
                    if len(_si) >= 4 and _char > 0 and _si.sum() / _char / s > 0.3 and _si.min() > 0 and _sj.min() > 0:
                        _c += float(np.interp(np.log(_sj.sum() / _char), _p["iso_x"], _p["iso_y"]))
                _centre = _e2[:, :3, 3].mean(0)
                _s_new = s * np.exp(_c)
                t = t + (s - _s_new) * (R @ _centre)
                print("Seam scale calibration: log s %+.4f -> %+.4f" % (np.log(s), np.log(_s_new)))
                s = _s_new
'''
out = src.replace(anchor, patch + anchor)
(VL / "vggt_long_calib.py").write_text(out)
Path("scripts/vggt_long_calib.diff").write_text("".join(difflib.unified_diff(
    src.splitlines(True), out.splitlines(True), "vggt_long.py", "vggt_long_calib.py")))
print("written", VL / "vggt_long_calib.py")
