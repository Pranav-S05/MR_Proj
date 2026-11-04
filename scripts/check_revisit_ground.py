"""Is KITTI's ground truth consistent in height at loop revisits? (one-off check)

For each loop (later frame a, earlier frame b): fetch both LiDAR scans from the lab mirror, place them in the world
with KITTI's ground-truth poses, and compare the road-surface height of the two visits at the same spot. If the
ground truth is right, the two road surfaces coincide (difference about 0 m); a height error in the ground truth
shows up directly as a difference.
usage: python scripts/check_revisit_ground.py data/kitti results/mid/tables/loops_SS.csv SS <scan_dir>
(needs the IIIT network: the scans come from the RRC lab mirror of KITTI)
"""
import csv, io, struct, sys, time, urllib.request, zipfile, zlib
from pathlib import Path
import numpy as np

URL = "https://datasets.rrcx.tk/KITTI/odometry/data_odometry_velodyne.zip"
root, loops_csv, seq, scan_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])


def get(a, b):
    for t in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"}), timeout=120) as r:
                d = r.read()
            if len(d) == b - a + 1:
                return d
        except Exception:
            time.sleep(2 * (t + 1))
    raise RuntimeError("range failed")


class Remote(io.RawIOBase):
    def __init__(self):
        with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0: return b""
        d = get(self.pos, self.pos + n - 1); self.pos += n; return d


def scan(frame, infos):
    p = scan_dir / f"{seq}_{frame:06d}.bin"
    if not p.exists():
        i = infos[f"dataset/sequences/{seq}/velodyne/{frame:06d}.bin"]
        h = get(i.header_offset, i.header_offset + 29)
        n, m = struct.unpack("<HH", h[26:30])
        raw = get(i.header_offset + 30 + n + m, i.header_offset + 30 + n + m + i.compress_size - 1)
        data = raw if i.compress_type == 0 else zlib.decompress(raw, -15)
        assert zlib.crc32(data) == i.CRC
        scan_dir.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return np.fromfile(p, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)


def calib_tr(path):
    for line in open(path):
        if line.startswith("Tr:"):
            T = np.eye(4); T[:3] = np.array(line.split()[1:], float).reshape(3, 4); return T


poses = np.loadtxt(root / "poses" / f"{seq}.txt").reshape(-1, 3, 4)
P = np.tile(np.eye(4), (len(poses), 1, 1)); P[:, :3] = poses          # camera 0 -> world (world y points down)
Tr = calib_tr(root / "sequences" / seq / "calib.txt")                   # velodyne -> camera 0
infos = {i.filename: i for i in zipfile.ZipFile(Remote()).infolist()}


def ground_world(frame):
    v = scan(frame, infos)
    near = (np.hypot(v[:, 0], v[:, 1]) < 12) & (v[:, 2] < -1.3) & (v[:, 2] > -2.2)   # road just around the car
    w = (P[frame] @ Tr @ np.c_[v[near], np.ones(near.sum())].T).T[:, :3]
    return w


print(f"KITTI {seq}: road height (world, metres, y down) of both visits at the same spot")
print("  later/earlier | GT camera height gap | road height gap after GT placement")
for r in csv.DictReader(open(loops_csv)):
    a, b = int(r["frame_later"]), int(r["frame_earlier"])
    ga, gb = ground_world(a), ground_world(b)
    # compare the two road surfaces where they overlap: b's road points within 4 m (horizontally) of a's camera
    ca = P[a][:3, 3]
    sa = ga[np.hypot(ga[:, 0] - ca[0], ga[:, 2] - ca[2]) < 4]
    sb = gb[np.hypot(gb[:, 0] - ca[0], gb[:, 2] - ca[2]) < 4]
    cam_gap = P[b][1, 3] - P[a][1, 3]
    road_gap = np.median(sb[:, 1]) - np.median(sa[:, 1]) if len(sa) > 50 and len(sb) > 50 else float("nan")
    print(f"  {a:5d}/{b:5d}  | {cam_gap:+6.2f} m             | {road_gap:+6.2f} m   ({len(sa)}/{len(sb)} road points)")
