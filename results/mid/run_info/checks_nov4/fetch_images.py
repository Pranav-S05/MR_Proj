"""Fetch KITTI odometry colour images (camera 2) for selected sequences from the remote zip with HTTP range
requests (one-off helper). Writes <out>/sequences/<SS>/image_2/*.png and <out>/sequences/<SS>.done per sequence."""
import struct, sys, time, urllib.request, zipfile, zlib, io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://datasets.rrcx.tk/KITTI/odometry/data_odometry_color.zip"  # RRC lab mirror (internal network)
OUT = Path(sys.argv[1])
SEQS = sys.argv[2].split(",")
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 96
COUNTS = {"00": 4541, "02": 4661, "05": 2761, "07": 1101}


def get(a, b, tries=8):
    for t in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == b - a + 1:
                return data
        except Exception:
            pass
        time.sleep(2 * (t + 1))
    raise RuntimeError(f"range {a}-{b} failed")


class Remote(io.RawIOBase):
    def __init__(self):
        for _ in range(20):
            try:
                with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
                    self.size = int(r.headers["Content-Length"])
                break
            except Exception:
                time.sleep(10)
        self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos
    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        d = get(self.pos, self.pos + n - 1)
        self.pos += n
        return d


infos = {i.filename: i for i in zipfile.ZipFile(Remote()).infolist()}
print(len(infos), "entries in the zip", flush=True)


def fetch(name):
    i = infos[name]
    dst = OUT / name.replace("dataset/", "")
    if dst.exists() and dst.stat().st_size == i.file_size:
        return 0
    head = get(i.header_offset, i.header_offset + 29)
    n, m = struct.unpack("<HH", head[26:30])
    start = i.header_offset + 30 + n + m
    raw = get(start, start + i.compress_size - 1)
    data = raw if i.compress_type == 0 else zlib.decompress(raw, -15)
    assert len(data) == i.file_size and zlib.crc32(data) == i.CRC, name
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(dst)
    return len(raw)


for s in SEQS:
    want = [f"dataset/sequences/{s}/image_2/{f:06d}.png" for f in range(COUNTS[s])]
    missing = [w for w in want if w not in infos]
    if missing:
        sys.exit(f"{s}: {len(missing)} images not in the zip, e.g. {missing[:2]}")
    t0, nbytes = time.time(), 0
    with ThreadPoolExecutor(THREADS) as ex:
        for k, b in enumerate(ex.map(fetch, want), 1):
            nbytes += b
            if k % 500 == 0:
                print(f"{s}: {k}/{len(want)} {nbytes / 1e6 / (time.time() - t0):.2f} MB/s", flush=True)
    (OUT / "sequences" / f"{s}.done").write_text(str(len(want)))
    print(f"{s} DONE {len(want)} images in {time.time() - t0:.0f}s", flush=True)
print("IMAGES_DONE", flush=True)
