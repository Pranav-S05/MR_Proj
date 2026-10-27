"""Run frozen VGGT over overlapping KITTI windows (+ VGGT-Long loop windows) and write the cache.

Needs an NVIDIA GPU and VGGT-Long's environment (see README "GPU machine"). It imports the
model code from the pinned VGGT-Long clone, so the frontend is exactly the baseline's:
  external/VGGT-Long                (commit c160869d1d99c96bb227f414afb3bc68c29c9a76)
  external/VGGT-Long/weights/model.pt                    VGGT-1B
  external/VGGT-Long/weights/dino_salad.ckpt             SALAD place recognition
  external/VGGT-Long/weights/dinov2_vitb14_pretrain.pth  its DINOv2 backbone

Usage (from the project root):
  python -m mr_gpu.run_vggt memory-sweep --seq 07 --windows 8 16 24 32 48 64
  python -m mr_gpu.run_vggt run --seq 07 --window 32 --overlap 16 --max-frames 96 --smoke   # quick check
  python -m mr_gpu.run_vggt run --seq 07 --window 32 --overlap 16                          # full sequence
Output: cache/kitti/<seq>/ (format in mr_cpu/cache.py). Re-running resumes: finished windows are skipped.

Per-window self-checks (stored in manifest["checks"], summarised with --smoke):
  * depth head unprojected with the predicted cameras vs the point head (should agree closely);
  * the window's first camera is the identity (VGGT's convention);
  * the window's predicted cameras vs KITTI ground truth after one Sim(3): small residual means the
    camera-to-world convention is right. The same fit with the poses inverted is reported for contrast.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mr_cpu import kitti  # noqa: E402
from mr_cpu.cache import find_loop_pairs, loop_windows, save_npz, window_ranges  # noqa: E402
from mr_cpu.labels import fit_gauge  # noqa: E402


def setup_vggt_long(path):
    path = Path(path).resolve()
    if not (path / "base_models" / "vggt").exists():
        sys.exit(f"VGGT-Long not found at {path}. Clone it there (see README).")
    for p in (str(path), str(path / "base_models")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return path


def load_model(vl_root, weights, device):
    import torch
    from vggt.models.vggt import VGGT  # vendored copy inside VGGT-Long
    weights = Path(weights) if weights else vl_root / "weights" / "model.pt"
    if not weights.exists():
        sys.exit(f"VGGT weights not found: {weights}\n  wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt -O {weights}")
    model = VGGT()
    state = torch.load(weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        sys.exit(f"Checkpoint is missing {len(missing)} keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"note: {len(unexpected)} unexpected checkpoint keys ignored")
    model.eval().to(device)
    cls = {}

    def hook(_, __, out):  # DINOv2 patch embed returns a dict with the CLS token
        if isinstance(out, dict) and "x_norm_clstoken" in out:
            cls["tokens"] = out["x_norm_clstoken"].detach().float()
    model.aggregator.patch_embed.register_forward_hook(hook)
    return model, cls, weights


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def infer(model, cls_store, paths, device, dtype):
    """One VGGT forward pass. Returns numpy dict (full resolution) and timing/memory."""
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = load_and_preprocess_images([str(p) for p in paths]).to(device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        pred = model(images)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    w2c = np.tile(np.eye(4), (len(paths), 1, 1))
    w2c[:, :3, :4] = extr[0].float().cpu().numpy()
    out = {"w2c": w2c, "intrinsics": intr[0].float().cpu().numpy(),
           "points": pred["world_points"][0].float().cpu().numpy(),
           "points_conf": pred["world_points_conf"][0].float().cpu().numpy(),
           "depth": pred["depth"][0, ..., 0].float().cpu().numpy(),
           "depth_conf": pred["depth_conf"][0].float().cpu().numpy(),
           "cls": cls_store["tokens"].cpu().numpy() if "tokens" in cls_store else None,
           "hw": tuple(images.shape[-2:])}
    del pred, images
    torch.cuda.empty_cache()
    return out, elapsed, peak


def window_checks(out, gt_c2w):
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    c2w = np.linalg.inv(out["w2c"])
    unproj = unproject_depth_map_to_point_map(out["depth"][..., None], out["w2c"][:, :3, :4], out["intrinsics"])
    conf = out["points_conf"]
    good = conf > np.median(conf)
    rel_depth = np.einsum("nhwk,nk->nhw", out["points"] - c2w[:, None, None, :3, 3], c2w[:, :3, 2])
    diff = np.linalg.norm(unproj - out["points"], axis=-1)[good]
    checks = {"depth_vs_points_rel": float(np.median(diff) / max(np.median(np.abs(rel_depth[good])), 1e-9)),
              "first_cam_identity_dev": float(np.abs(c2w[0] - np.eye(4)).max())}
    if gt_c2w is not None:
        _, info = fit_gauge(c2w, gt_c2w, min_spread_m=0.5)
        _, info_inv = fit_gauge(out["w2c"], gt_c2w, min_spread_m=0.5)
        checks.update(gt_pos_rms_rel=info["position_rms_rel"], gt_rot_rms_deg=info["rotation_rms_deg"],
                      gt_pos_rms_rel_if_inverted=info_inv["position_rms_rel"], gt_spread_m=info["gt_spread_m"])
    return checks


def to_cache(out, frames, stride):
    s = slice(None, None, stride)
    arrays = dict(frame_ids=np.asarray(frames), c2w=np.linalg.inv(out["w2c"]), intrinsics=out["intrinsics"].astype(np.float64),
                  points=out["points"][:, s, s].astype(np.float32), points_conf=out["points_conf"][:, s, s].astype(np.float16),
                  depth=out["depth"][:, s, s].astype(np.float16), depth_conf=out["depth_conf"][:, s, s].astype(np.float16),
                  stride=np.array(stride))
    if out["cls"] is not None:
        arrays["cls"] = out["cls"].astype(np.float16)  # VGGT encoder (DINOv2) CLS token per frame
    return arrays


def assemble_cls(out_dir, n_ranges, n_frames):
    """Per-frame VGGT-encoder descriptor (first window that saw the frame), L2-normalised."""
    desc = None
    for k in range(n_ranges):
        with np.load(out_dir / "windows" / f"w{k:04d}.npz") as w:
            if "cls" not in w.files:
                return None
            if desc is None:
                desc = np.full((n_frames, w["cls"].shape[1]), np.nan, np.float32)
            f = w["frame_ids"]
            fill = np.isnan(desc[f, 0])
            desc[f[fill]] = w["cls"][fill]
    if desc is None or np.isnan(desc).any():
        return None
    return desc / np.linalg.norm(desc, axis=1, keepdims=True)


def salad_descriptors(vl_root, paths, device):
    """VGGT-Long's SALAD descriptors (same model, preprocessing and weights as its LoopDetector)."""
    from LoopModels.LoopModel import LoopDetector
    cfg = {"Weights": {"SALAD": str(vl_root / "weights" / "dino_salad.ckpt"),
                       "DNIO": str(vl_root / "weights" / "dinov2_vitb14_pretrain.pth")},
           "Loop": {"SALAD": {"image_size": [336, 336], "batch_size": 32, "similarity_threshold": 0.85,
                              "top_k": 5, "use_nms": True, "nms_threshold": 25}}}
    for k, p in cfg["Weights"].items():
        if not Path(p).exists():
            raise FileNotFoundError(f"{k} weights missing: {p}")
    # absolute paths: we chdir below, and LoopDetector replaces unreadable images by zeros without raising
    paths = [Path(p).resolve() for p in paths]
    cwd = os.getcwd()
    os.chdir(vl_root)  # its DINOv2 is loaded with torch.hub from the relative path ./LoopModels/dinov2
    try:
        det = LoopDetector(image_dir=str(paths[0].parent), output=os.devnull, config=cfg)
        det.load_model()
        det.image_paths = list(paths)
        desc = det.extract_descriptors().float().numpy()
    finally:
        os.chdir(cwd)
    del det
    return desc


def cmd_run(a):
    import torch
    vl_root = setup_vggt_long(a.vggt_long)
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    paths = kitti.image_paths(a.kitti, a.seq)
    if not paths:
        sys.exit(f"No images in {a.kitti}/sequences/{a.seq}/image_2")
    gt = kitti.gt_cam2_poses(a.kitti, a.seq) if (Path(a.kitti) / "poses" / f"{a.seq}.txt").exists() else None
    if gt is not None and len(gt) != len(paths):
        sys.exit(f"{len(paths)} images but {len(gt)} GT poses: incomplete download?")
    if a.max_frames:
        paths = paths[:a.max_frames]
        gt = None if gt is None else gt[:a.max_frames]
    n = len(paths)
    out_dir = Path(a.out or ROOT / "cache" / "kitti" / a.seq)
    ranges = window_ranges(n, a.window, a.overlap)
    man_path = out_dir / "manifest.json"
    if man_path.exists():
        old = json.loads(man_path.read_text())
        if (old["window"], old["overlap"], old["stride"], old["n_frames"]) != (a.window, a.overlap, a.stride, n):
            sys.exit(f"{out_dir} holds a different configuration; use another --out")
    model, cls_store, weights = load_model(vl_root, a.weights, device)
    orig_hw = kitti.image_size(a.kitti, a.seq)
    manifest = {"engine": "vggt", "sequence": a.seq, "n_frames": n, "window": a.window, "overlap": a.overlap,
                "stride": a.stride, "orig_hw": list(orig_hw), "model_hw": list(kitti.model_input_size(*orig_hw)),
                "windows": [list(r) for r in ranges], "vggt_long_commit": _git_commit(vl_root),
                "checkpoint": str(weights), "checkpoint_sha256": sha256(weights) if a.hash else None,
                "dtype": str(dtype), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "checks": {}, "timing": {}}
    times, peaks = [], []
    t_all = time.time()
    for k, (s, e) in enumerate(ranges):
        target = out_dir / "windows" / f"w{k:04d}.npz"
        if target.exists():
            continue
        out, dt, peak = infer(model, cls_store, paths[s:e], device, dtype)
        if tuple(out["hw"]) != tuple(manifest["model_hw"]):
            sys.exit(f"Model input {out['hw']} != expected {manifest['model_hw']}: preprocessing changed?")
        manifest["checks"][k] = window_checks(out, None if gt is None else gt[s:e])
        save_npz(target, **to_cache(out, np.arange(s, e), a.stride))
        times.append(dt)
        peaks.append(peak)
        c = manifest["checks"][k]
        print(f"window {k + 1}/{len(ranges)} frames {s}-{e - 1}: {dt:.2f}s, peak {peak:.2f} GB, "
              f"depth/point {c['depth_vs_points_rel']:.3f}"
              + (f", GT pos rms {100 * c['gt_pos_rms_rel']:.1f}% rot {c['gt_rot_rms_deg']:.2f} deg" if "gt_pos_rms_rel" in c else ""),
              flush=True)
    if times:
        manifest["timing"] = {"sec_per_window": float(np.mean(times)), "peak_gpu_mem_gb": float(np.max(peaks)),
                              "frames_per_sec": float(sum(e - s for s, e in ranges) / max(sum(times), 1e-9)),
                              "unique_frames_per_sec": float(n / max(time.time() - t_all, 1e-9)),
                              "note": "frames_per_sec counts overlap frames twice (forward time only)"}
    cls_desc = assemble_cls(out_dir, len(ranges), n)
    if cls_desc is not None:
        np.save(out_dir / "descriptors_vggt_cls.npy", cls_desc.astype(np.float16))
    _write_manifest(man_path, manifest)
    if a.smoke:
        _smoke_report(manifest)
    if a.no_loops:
        return
    # ---------------- loop closure (VGGT-Long: SALAD retrieval + 40-frame loop windows)
    if a.descriptor == "salad":
        dpath = out_dir / "descriptors.npy"
        if not dpath.exists():
            desc = salad_descriptors(vl_root, paths, device)
            np.save(dpath, desc.astype(np.float16))
        desc = np.load(dpath).astype(np.float32)
    else:
        desc = np.load(out_dir / "descriptors_vggt_cls.npy").astype(np.float32)
    pairs = find_loop_pairs(desc, threshold=a.loop_threshold, top_k=5, min_gap=10, nms=25)
    loops = loop_windows(ranges, pairs, half_window=10)
    print(f"{len(pairs)} loop pairs -> {len(loops)} loop windows ({a.descriptor}, threshold {a.loop_threshold})")
    for m, lw in enumerate(loops):
        target = out_dir / "loops" / f"l{m:04d}.npz"
        if target.exists():
            continue
        frames = np.r_[np.arange(*lw["range_a"]), np.arange(*lw["range_b"])]
        out, dt, peak = infer(model, cls_store, [paths[f] for f in frames], device, dtype)
        save_npz(target, **to_cache(out, frames, a.stride))
        print(f"loop window {m + 1}/{len(loops)}: frames {lw['range_a']} + {lw['range_b']} ({dt:.2f}s)", flush=True)
    manifest.update(loop_windows=loops, loop_pairs=[list(p) for p in pairs], descriptor=a.descriptor,
                    loop_threshold=a.loop_threshold)
    _write_manifest(man_path, manifest)
    print(f"done: {out_dir}")


def _write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text()) if path.exists() else {}
    old_checks = old.get("checks", {})
    old_checks.update({str(k): v for k, v in manifest["checks"].items()})
    merged = {**old, **manifest, "checks": old_checks}
    if not manifest.get("timing"):
        merged["timing"] = old.get("timing", {})
    path.write_text(json.dumps(merged, indent=1))


def _smoke_report(manifest):
    checks = list(manifest["checks"].values())
    tests = [("depth head agrees with point head (median diff < 10% of depth)",
              all(c["depth_vs_points_rel"] < .1 for c in checks)),
             ("first camera of each window is the identity", all(c["first_cam_identity_dev"] < 1e-3 for c in checks))]
    with_gt = [c for c in checks if "gt_pos_rms_rel" in c and c["gt_spread_m"] > 3]
    if with_gt:
        tests.append(("predicted cameras match GT after one Sim(3) (median rms < 5% of travel, rot < 1 deg)",
                      np.median([c["gt_pos_rms_rel"] for c in with_gt]) < .05 and np.median([c["gt_rot_rms_deg"] for c in with_gt]) < 1))
        tests.append(("treating poses as camera-to-world fits GT better than the inverse",
                      np.median([c["gt_pos_rms_rel"] for c in with_gt]) < np.median([c["gt_pos_rms_rel_if_inverted"] for c in with_gt])))
    print("\nSMOKE TEST")
    for name, ok in tests:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  timing: {manifest.get('timing')}")


def _git_commit(path):
    head = Path(path) / ".git" / "HEAD"
    if not head.exists():
        return None
    ref = head.read_text().strip()
    if ref.startswith("ref:"):
        p = Path(path) / ".git" / ref[5:]
        return p.read_text().strip() if p.exists() else ref
    return ref


def cmd_memory(a):
    import torch
    vl_root = setup_vggt_long(a.vggt_long)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model, cls_store, _ = load_model(vl_root, a.weights, "cuda")
    paths = kitti.image_paths(a.kitti, a.seq)
    rows = []
    for w in a.windows:
        try:
            infer(model, cls_store, paths[:w], "cuda", dtype)  # warm-up
            _, dt, peak = infer(model, cls_store, paths[:w], "cuda", dtype)
            rows.append({"window": w, "peak_gpu_mem_gb": round(peak, 3), "sec": round(dt, 3), "frames_per_sec": round(w / dt, 2)})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"window": w, "peak_gpu_mem_gb": "OOM", "sec": None, "frames_per_sec": None})
        print(rows[-1], flush=True)
    out = ROOT / "results" / "memory_sweep.csv"
    out.parent.mkdir(exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    with open(out, "w") as f:
        f.write("gpu,window,peak_gpu_mem_gb,sec,frames_per_sec\n")
        for r in rows:
            f.write(f"{gpu},{r['window']},{r['peak_gpu_mem_gb']},{r['sec']},{r['frames_per_sec']}\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "memory-sweep"):
        p = sub.add_parser(name)
        p.add_argument("--kitti", default=str(ROOT / "data" / "kitti"))
        p.add_argument("--seq", required=True)
        p.add_argument("--vggt-long", default=str(ROOT / "external" / "VGGT-Long"))
        p.add_argument("--weights", help="VGGT checkpoint (default <vggt-long>/weights/model.pt)")
        if name == "run":
            p.add_argument("--window", type=int, default=32)
            p.add_argument("--overlap", type=int, default=16)
            p.add_argument("--stride", type=int, default=1,
                           help="pixel subsampling of cached maps (1 = full model resolution, as VGGT-Long uses)")
            p.add_argument("--max-frames", type=int, help="only the first N frames (smoke tests)")
            p.add_argument("--out", help="cache directory (default cache/kitti/<seq>)")
            p.add_argument("--descriptor", choices=["salad", "vggt_cls"], default="salad")
            p.add_argument("--loop-threshold", type=float, default=0.85)
            p.add_argument("--no-loops", action="store_true")
            p.add_argument("--smoke", action="store_true", help="print PASS/FAIL convention checks")
            p.add_argument("--hash", action="store_true", help="record the checkpoint sha256 (slow)")
        else:
            p.add_argument("--windows", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
    a = ap.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        cmd_run(a) if a.cmd == "run" else cmd_memory(a)


if __name__ == "__main__":
    main()
