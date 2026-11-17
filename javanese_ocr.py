#!/usr/bin/env python3
"""
Replikasi OCR aksara Jawa cetak:
  Mahastama & Krisnawati, "OCR for Printed Javanese Script using Projection
  Profile Segmentation and Nearest Centroid Classifier", ACCC 2020.

Pipeline:
  1. Binarisasi (threshold t=160)
  2. Segmentasi baris  : projection profile vertikal + filter + Q1 + merge/split
  3. Segmentasi karakter: projection profile horizontal per baris
  4. Fitur (10)        : 9 rasio piksel foreground per zona 3x3 + rasio luas
  5. Training          : centroid ("eigenface") = rata-rata fitur per kelas
  6. Klasifikasi       : Nearest Centroid (jarak Euclidean)
  7. Rekonstruksi      : label -> Unicode Jawa (+ aturan swap untuk taling dsb.)

Dependensi: pip install numpy pillow

Penggunaan:
  # 1) Potong halaman jadi segmen karakter untuk dilabeli manual
  python javanese_ocr.py segment halaman1.jpg halaman2.jpg --out work

  # 2) Susun hasil labeling: dataset/<nama_kelas>/*.png, lalu latih
  python javanese_ocr.py train dataset --model model.npz --max-per-class 10

  # 3) (opsional) evaluasi hold-out sederhana pada dataset berlabel
  python javanese_ocr.py evaluate dataset --max-per-class 10

  # 4) Prediksi halaman baru
  python javanese_ocr.py predict halaman_baru.jpg --model model.npz \
         --labels labels.csv --prefix-labels taling,dirgha_mure

labels.csv (tanpa header), contoh:
  na,ꦤ
  ka,ꦏ
  taling,ꦺ
  pdli,U+A9C8        <- boleh juga format U+XXXX (pisahkan beberapa dgn spasi)
"""
import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

THRESHOLD = 160          # paper: t = 160
FALSE_LINE_RATIO = 0.05  # paper: tinggi <= 5% rata-rata tinggi baris = noise


# ----------------------------------------------------------------------------
# 1. Pre-processing
# ----------------------------------------------------------------------------
def binarize(img: Image.Image, t: int = THRESHOLD) -> np.ndarray:
    """Foreground (teks gelap) = 1, background = 0."""
    gray = np.asarray(img.convert("L"))
    return (gray < t).astype(np.uint8)


def runs(profile: np.ndarray):
    """Pasangan (start, end_exclusive) untuk setiap deret nilai > 0."""
    nz = (profile > 0).astype(np.int8)
    d = np.diff(np.concatenate([[0], nz, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


# ----------------------------------------------------------------------------
# 2. Segmentasi baris
# ----------------------------------------------------------------------------
def segment_lines(bw: np.ndarray):
    proj = bw.sum(axis=1)
    cands = runs(proj)
    if not cands:
        return []

    # (a) buang kandidat palsu (noise): tinggi terlalu kecil
    avg_h = np.mean([e - s for s, e in cands])
    cands = [(s, e) for s, e in cands if (e - s) > FALSE_LINE_RATIO * avg_h]
    if not cands:
        return []

    # (b) "true line candidates": ordinat pertama & terakhir yang proyeksinya
    #     melebihi Q1 di dalam batas tiap kandidat
    true_lines = []
    for s, e in cands:
        seg = proj[s:e]
        q1 = np.percentile(seg, 25)
        idx = np.where(seg > q1)[0]
        if len(idx):
            true_lines.append((s + idx[0], s + idx[-1] + 1))
    h0 = np.mean([e - s for s, e in true_lines]) if true_lines else avg_h

    # (c) validasi: split / valid / merge
    valid, small = [], []
    for s, e in cands:
        h = e - s
        if h > 3 * h0:
            # baris menyatu -> bagi rata. Paper: int(tinggi / (3*h0)); minimal 2
            n = max(2, int(h // (3 * h0)))
            edges = np.linspace(s, e, n + 1).astype(int)
            valid += [(int(edges[i]), int(edges[i + 1])) for i in range(n)]
            continue
        has_true = any(ts < e and te > s for ts, te in true_lines)
        if has_true and h >= h0:
            valid.append((int(s), int(e)))
        elif h < h0:
            small.append((int(s), int(e)))
        else:
            valid.append((int(s), int(e)))
    # catatan: kondisi (a) di paper ambigu; silakan eksperimen dengan
    # variasi aturan ini (mis. cek `has_true` saja).

    if not valid:  # fallback: semuanya dianggap valid
        return sorted(small)

    # gabungkan segmen kecil (mis. diakritik terpisah) ke baris valid terdekat
    for s, e in small:
        c = (s + e) / 2
        j = min(range(len(valid)),
                key=lambda k: abs((valid[k][0] + valid[k][1]) / 2 - c))
        valid[j] = (min(valid[j][0], s), max(valid[j][1], e))
    return sorted(valid)


# ----------------------------------------------------------------------------
# 3. Segmentasi karakter
# ----------------------------------------------------------------------------
@dataclass
class Segment:
    img: np.ndarray   # biner, sudah di-trim
    line: int
    idx: int
    x: int
    y: int

    @property
    def h(self): return self.img.shape[0]
    @property
    def w(self): return self.img.shape[1]
    @property
    def area(self): return int(self.img.sum())


def segment_chars(bw: np.ndarray, lines):
    segs = []
    for ln, (y0, y1) in enumerate(lines):
        band = bw[y0:y1]
        for i, (x0, x1) in enumerate(runs(band.sum(axis=0))):
            box = band[:, x0:x1]
            rows = np.where(box.sum(axis=1) > 0)[0]  # trim whitespace vertikal
            if len(rows) == 0:
                continue
            r0, r1 = rows[0], rows[-1] + 1
            segs.append(Segment(box[r0:r1], ln, i, int(x0), int(y0 + r0)))
    return segs


# ----------------------------------------------------------------------------
# 4. Ekstraksi fitur (10 fitur)
# ----------------------------------------------------------------------------
def zone_features(seg_img: np.ndarray) -> np.ndarray:
    """f0..f8: rasio piksel foreground per zona pada grid 3x3."""
    h, w = seg_img.shape
    ys = np.linspace(0, h, 4).astype(int)
    xs = np.linspace(0, w, 4).astype(int)
    f = []
    for i in range(3):
        for j in range(3):
            z = seg_img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            f.append(z.sum() / z.size if z.size else 0.0)
    return np.array(f, dtype=np.float64)


def features(images, avg_w=None, avg_h=None) -> np.ndarray:
    """Matriks (N, 10). f9 = area / (avg_w * avg_h), Pers. (1) di paper."""
    if avg_w is None:
        avg_w = np.mean([im.shape[1] for im in images])
    if avg_h is None:
        avg_h = np.mean([im.shape[0] for im in images])
    rows = []
    for im in images:
        rows.append(np.append(zone_features(im), im.sum() / (avg_w * avg_h)))
    return np.array(rows)


# ----------------------------------------------------------------------------
# 5-6. Training (centroid) & Nearest Centroid Classifier
# ----------------------------------------------------------------------------
def load_dataset(root: Path, max_per_class=10, seed=0):
    rng = random.Random(seed)
    imgs, labels = [], []
    for cdir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = sorted(cdir.glob("*.png")) + sorted(cdir.glob("*.jpg"))
        if max_per_class and len(files) > max_per_class:
            files = rng.sample(files, max_per_class)  # paper: 10 acak/kelas
        for f in files:
            imgs.append(binarize(Image.open(f)))
            labels.append(cdir.name)
    return imgs, np.array(labels)


def fit_centroids(X, y):
    classes = np.array(sorted(set(y)))
    cents = np.stack([X[y == c].mean(axis=0) for c in classes])
    return classes, cents


def ncc_predict(X, classes, cents):
    d = np.sqrt(((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2))  # Pers. (2)
    return classes[d.argmin(axis=1)]


# ----------------------------------------------------------------------------
# 7. Rekonstruksi ke Unicode
# ----------------------------------------------------------------------------
def load_label_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            val = row[1].strip()
            if val.upper().startswith("U+"):
                val = "".join(chr(int(t[2:], 16)) for t in val.split())
            m[row[0].strip()] = val
    return m


def reconstruct(labels, label_map, prefix_labels=()):
    """Sandhangan seperti taling ditulis SEBELUM aksara dasar secara visual,
    tapi di Unicode urutannya SESUDAH -> tukar dengan label berikutnya."""
    labels = list(labels)
    i = 0
    while i < len(labels) - 1:
        if labels[i] in prefix_labels:
            labels[i], labels[i + 1] = labels[i + 1], labels[i]
            i += 2
        else:
            i += 1
    return "".join(label_map.get(l, f"[{l}]") for l in labels)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def cmd_segment(a):
    out = Path(a.out)
    (out / "unlabeled").mkdir(parents=True, exist_ok=True)
    for pi, p in enumerate(a.images):
        img = Image.open(p)
        bw = binarize(img)
        lines = segment_lines(bw)
        segs = segment_chars(bw, lines)
        stem = Path(p).stem
        dbg = img.convert("RGB")
        dr = ImageDraw.Draw(dbg)
        for (y0, y1) in lines:
            dr.rectangle([0, y0, bw.shape[1] - 1, y1], outline=(0, 0, 255))
        for s in segs:
            dr.rectangle([s.x, s.y, s.x + s.w, s.y + s.h], outline=(255, 0, 0))
            name = f"{stem}_l{s.line:02d}_c{s.idx:03d}.png"
            Image.fromarray(((1 - s.img) * 255).astype(np.uint8)).save(
                out / "unlabeled" / name)
        dbg.save(out / f"{stem}_debug.png")
        print(f"{p}: {len(lines)} baris, {len(segs)} segmen")
    print(f"Segmen di {out/'unlabeled'}. Pindahkan ke dataset/<nama_kelas>/ "
          "setelah dilabeli.")


def cmd_train(a):
    imgs, y = load_dataset(a.dataset, a.max_per_class)
    avg_w = np.mean([i.shape[1] for i in imgs])
    avg_h = np.mean([i.shape[0] for i in imgs])
    X = features(imgs, avg_w, avg_h)
    classes, cents = fit_centroids(X, y)
    np.savez(a.model, classes=classes, centroids=cents)
    print(f"{len(imgs)} contoh, {len(classes)} kelas -> {a.model}")


def cmd_evaluate(a):
    imgs, y = load_dataset(a.dataset, 0)  # semua contoh
    X = features(imgs)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    # batasi contoh training per kelas
    keep = []
    cnt = {}
    for i in tr:
        cnt[y[i]] = cnt.get(y[i], 0) + 1
        if not a.max_per_class or cnt[y[i]] <= a.max_per_class:
            keep.append(i)
    classes, cents = fit_centroids(X[keep], y[keep])
    pred = ncc_predict(X[te], classes, cents)
    print(f"Akurasi hold-out: {(pred == y[te]).mean()*100:.2f}% "
          f"({len(te)} uji, {len(classes)} kelas)")


def cmd_predict(a):
    m = np.load(a.model, allow_pickle=True)
    classes, cents = m["classes"], m["centroids"]
    lmap = load_label_map(a.labels) if a.labels else {}
    prefix = set(filter(None, a.prefix_labels.split(",")))
    bw = binarize(Image.open(a.image))
    lines = segment_lines(bw)
    segs = segment_chars(bw, lines)
    if not segs:
        print("Tidak ada segmen.")
        return
    avg_w = np.mean([s.w for s in segs])  # rata-rata per halaman (paper)
    avg_h = np.mean([s.h for s in segs])
    X = features([s.img for s in segs], avg_w, avg_h)
    pred = ncc_predict(X, classes, cents)
    for ln in range(len(lines)):
        lab = [p for p, s in zip(pred, segs) if s.line == ln]
        print(reconstruct(lab, lmap, prefix) if lmap else " ".join(lab))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)

    s = sub.add_parser("segment"); s.add_argument("images", nargs="+")
    s.add_argument("--out", default="work"); s.set_defaults(f=cmd_segment)

    t = sub.add_parser("train"); t.add_argument("dataset")
    t.add_argument("--model", default="model.npz")
    t.add_argument("--max-per-class", type=int, default=10)
    t.set_defaults(f=cmd_train)

    e = sub.add_parser("evaluate"); e.add_argument("dataset")
    e.add_argument("--max-per-class", type=int, default=10)
    e.set_defaults(f=cmd_evaluate)

    p = sub.add_parser("predict"); p.add_argument("image")
    p.add_argument("--model", default="model.npz")
    p.add_argument("--labels", help="CSV label,unicode")
    p.add_argument("--prefix-labels", default="",
                   help="label sandhangan yang ditukar urutannya, pisah koma")
    p.set_defaults(f=cmd_predict)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
