"""Fetch the image_2 files of one KITTI drive out of the remote colour zip with HTTP range requests.

usage: python fetch_seq_from_zip.py SEQ OUT_DIR
Only the zip's central directory and the wanted members are downloaded (about 1.5 GB for drive 09, instead of 64 GB).
Every written PNG is decoded with PIL as a check.
"""
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://datasets.rrcx.tk/KITTI/odometry/data_odometry_color.zip"
BLOCK = 8 << 20


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, k):
        if k not in self.cache:
            a, b = k * BLOCK, min(self.size, (k + 1) * BLOCK) - 1
            for attempt in range(8):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    if len(data) == b - a + 1:
                        break
                except Exception as e:  # network hiccup: retry
                    print("retry", k, e, flush=True)
                time.sleep(5 * (attempt + 1))
            else:
                raise IOError(f"block {k} failed")
            if len(self.cache) > 16:
                self.cache.pop(next(iter(self.cache)))
            self.cache[k] = data
        return self.cache[k]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            k, o = divmod(self.pos, BLOCK)
            chunk = self._block(k)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def main():
    seq, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    z = zipfile.ZipFile(HttpFile(URL))
    prefix = f"dataset/sequences/{seq}/image_2/"
    members = sorted((m for m in z.infolist() if m.filename.startswith(prefix) and m.filename.endswith(".png")), key=lambda m: m.header_offset)
    print(len(members), "members", flush=True)
    t0 = time.time()
    for i, m in enumerate(members):
        dst = out / Path(m.filename).name
        if dst.exists() and dst.stat().st_size == m.file_size:
            continue
        data = z.read(m)
        Image.open(io.BytesIO(data)).load()  # decode check
        dst.write_bytes(data)
        if i % 100 == 0:
            print(f"{i}/{len(members)} {time.time() - t0:.0f}s", flush=True)
    bad = 0
    for p in sorted(out.glob("*.png")):
        try:
            Image.open(p).load()
        except Exception:
            bad += 1
            print("BAD", p, flush=True)
    print(f"FETCH_DONE {len(members)} files, {bad} unreadable after writing", flush=True)


if __name__ == "__main__":
    main()
