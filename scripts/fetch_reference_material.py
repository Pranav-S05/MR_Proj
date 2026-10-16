"""Download small public reference files, never model weights or image archives.

Remote Python is saved for reading only; this script never imports or executes it.
"""
from __future__ import annotations

import hashlib
import argparse
import io
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 12 * 1024 * 1024


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "MR-Project-reference-check"})
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Refusing a reference download larger than 12 MiB")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "reference_material"
    output.mkdir(exist_ok=True)
    manifest = {"purpose": "Read-only reference snapshots; not an installed baseline", "repositories": [], "downloads": [], "errors": []}
    if args.poses_only and (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest["errors"] = [e for e in manifest["errors"] if "data_odometry_poses.zip" not in e.get("url", "")]
        manifest["downloads"] = [e for e in manifest["downloads"] if "data_odometry_poses.zip" not in e.get("url", "")]
    repos = {
        "DengKaiCQ/VGGT-Long": ["README.md", "LICENSE.txt", "configs/base_config.yaml", "vggt_long.py", "loop_utils/sim3loop.py", "loop_utils/sim3utils.py", "base_models/base_model.py"],
        "facebookresearch/vggt": ["README.md", "vggt/heads/dpt_head.py", "vggt/heads/utils.py", "vggt/utils/geometry.py"],
        "jashshah999/vggt-factor-refinement": ["README.md", "src/uncertainty.py"],
    }
    for repo, files in ({} if args.poses_only else repos).items():
        try:
            commit = json.loads(get(f"https://api.github.com/repos/{repo}/commits/main"))
            sha = commit["sha"]
            manifest["repositories"].append({"repository": repo, "commit": sha, "commit_date": commit["commit"]["committer"]["date"]})
            for name in files:
                url = f"https://raw.githubusercontent.com/{repo}/{sha}/{name}"
                try:
                    payload = get(url)
                    target = output / repo.split("/")[1] / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    manifest["downloads"].append({"url": url, "path": str(target.relative_to(ROOT)), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
                except (OSError, ValueError) as exc:
                    manifest["errors"].append({"url": url, "error": str(exc)})
        except (OSError, ValueError, KeyError) as exc:
            manifest["errors"].append({"repository": repo, "error": str(exc)})
    # KITTI's public ground-truth-only archive: no images, no LiDAR, no weights.
    url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"
    try:
        payload = get(url)
        records = []
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for sequence in range(11):
                name = f"poses/{sequence:02d}.txt"
                matches = [n for n in archive.namelist() if n == name or n.endswith("/" + name)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one archive member for {name}, found {matches}")
                info = archive.getinfo(matches[0])
                if info.file_size > MAX_BYTES:
                    raise ValueError("Unexpected pose-file size")
                content = archive.read(info)
                target = ROOT / "data" / "kitti" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                records.append({"path": str(target.relative_to(ROOT)), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        manifest["downloads"].append({"url": url, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "files": records})
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        manifest["errors"].append({"url": url, "error": str(exc)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repositories": manifest["repositories"], "download_count": len(manifest["downloads"]), "errors": manifest["errors"]}, indent=2))
    if manifest["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
