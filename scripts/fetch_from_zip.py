"""Fetch the files of one KITTI drive out of a remote zip with HTTP range requests (only the needed bytes).

usage: python fetch_from_zip.py URL PREFIX OUT_DIR
  e.g. URL = https://datasets.rrcx.tk/KITTI/odometry/data_odometry_velodyne.zip,
       PREFIX = dataset/sequences/00/velodyne/
Every written file is checked: PNG images are decoded, Velodyne .bin files must hold whole 16-byte points.
"""
import io
import sys
import time
from pathlib import Path

from fetch_seq_from_zip import HttpFile  # same range-request reader
import zipfile


def check(name, data):
    if name.endswith(".png"):
        from PIL import Image
        Image.open(io.BytesIO(data)).load()
    elif name.endswith(".bin") and (len(data) == 0 or len(data) % 16):
        raise ValueError(f"{name}: {len(data)} bytes is not a whole number of points")


def main():
    url, prefix, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(HttpFile(url))
    members = sorted((m for m in z.infolist() if m.filename.startswith(prefix) and not m.is_dir()),
                     key=lambda m: m.header_offset)
    print(len(members), "members under", prefix, flush=True)
    t0, bad = time.time(), 0
    for i, m in enumerate(members):
        dst = out / Path(m.filename).name
        if dst.exists() and dst.stat().st_size == m.file_size:
            continue
        data = z.read(m)
        try:
            check(m.filename, data)
        except Exception as e:
            bad += 1
            print("BAD", m.filename, e, flush=True)
            continue
        dst.write_bytes(data)
        if i % 500 == 0:
            print(f"{i}/{len(members)} {time.time() - t0:.0f}s", flush=True)
    print(f"FETCH_DONE {len(members)} files, {bad} bad", flush=True)


if __name__ == "__main__":
    main()
