#!/usr/bin/env python3
"""
Membangun dataset GAMBAR berlabel dari dua CSV metadata + font Unicode Jawa.

  --chars     CSV level codepoint:  character,codepoint,unicode_name,
              unicode_general_category,combining_class,frequency,is_whitespace,
              functional_category
  --clusters  CSV level cluster:    cluster,codepoints,frequency,
              contains_only_javanese_block,contains_pangkon,contains_format_control

Peran kedua CSV:
  clusters -> DAFTAR KELAS (satu baris = satu kelas) + frekuensi korpus yang
              dipakai untuk menentukan jumlah sampel per kelas.
  chars    -> validasi/pembersihan (mis. buang whitespace), frekuensi untuk
              kelas tunggal hasil pemecahan segmen, dan nama Unicode.

Mode:
  cluster : seluruh cluster dirender dan disimpan sebagai SATU gambar/kelas.
            (cocok jika segmentasi Anda menghasilkan potongan level cluster)
  segment : hasil render disegmentasi dengan projection profile yang sama dgn
            paper; kelas mengikuti potongan yang benar-benar terbentuk
            (taling/tarung dsb. bisa menjadi kelas sendiri). Cluster yang
            hasil segmennya tidak cocok dilewati.

Jumlah sampel per kelas: n = clip(round(alpha * frekuensi**power), min, max)
  power=0 -> seragam; power=0.5 -> akar frekuensi (default); power=1 -> proporsional.

Contoh:
  python make_dataset.py --chars char.csv --clusters cluster.csv --font NotoSansJavanese-Regular.ttf --mode cluster --out dataset
Keluaran: dataset/<A9A4_A9B8>/*.png, dataset/labels.csv, dataset/class_freq.csv
"""
import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, features

from javanese_ocr import binarize, segment_chars
from make_synthetic_dataset import cname, label_segments, render

CP = re.compile(r"U\+([0-9A-Fa-f]{4,6})")


def parse_cps(s):
    return [int(x, 16) for x in CP.findall(s)]


def truthy(s):
    return str(s).strip().lower() == "true"


def trim(bw):
    rows = np.where(bw.sum(axis=1) > 0)[0]
    cols = np.where(bw.sum(axis=0) > 0)[0]
    if len(rows) == 0:
        return None
    return bw[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def load_chars(path):
    chars = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cps = parse_cps(r["codepoint"])
            if cps:
                chars[cps[0]] = r
    return chars


def load_clusters(path, chars, a):
    out, dropped = [], {"format": 0, "nonjv": 0, "space": 0, "freq": 0}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            codes = parse_cps(r["codepoints"])
            if not codes:
                continue
            if not a.keep_format_control and truthy(r["contains_format_control"]):
                dropped["format"] += 1
                continue
            if a.only_javanese and not truthy(r["contains_only_javanese_block"]):
                dropped["nonjv"] += 1
                continue
            if any(truthy(chars.get(c, {}).get("is_whitespace", "False")) for c in codes):
                dropped["space"] += 1
                continue
            freq = int(float(r["frequency"]))
            if freq < a.min_freq:
                dropped["freq"] += 1
                continue
            out.append((codes, freq))
    out.sort(key=lambda t: -t[1])
    print(f"Cluster dipakai: {len(out)} | dibuang: {dropped}")
    return out[:a.limit] if a.limit else out


def save(arr, path):
    Image.fromarray(((1 - arr) * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--font", action="append", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--mode", choices=["cluster", "segment"], default="cluster")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--min-per-class", type=int, default=10)
    ap.add_argument("--max-per-class", type=int, default=200)
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="hanya N cluster teratas")
    ap.add_argument("--only-javanese", action="store_true", default=True)
    ap.add_argument("--keep-format-control", action="store_true")
    ap.add_argument("--min-size", type=int, default=48)
    ap.add_argument("--max-size", type=int, default=110)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not features.check("raqm"):
        print("PERINGATAN: libraqm tidak tersedia; sandhangan/pasangan bisa salah render.")

    rng, np_rng = random.Random(a.seed), np.random.default_rng(a.seed)
    chars = load_chars(a.chars)
    clusters = load_clusters(a.clusters, chars, a)
    unknown = {c for codes, _ in clusters for c in codes if c not in chars}
    if unknown:
        print(f"Codepoint di cluster tapi tidak ada di chars.csv: "
              f"{sorted(f'U+{c:04X}' for c in unknown)[:10]}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target, freq_of, codes_of, counts = {}, {}, {}, {}
    for codes, freq in clusters:
        n = int(min(a.max_per_class, max(a.min_per_class,
                                          round(a.alpha * freq ** a.power))))
        name = cname(codes)
        target[name], freq_of[name], codes_of[name] = n, freq, codes
    skipped = []

    for codes, freq in clusters:
        n = target[cname(codes)]
        for k in range(n):
            font = a.font[k % len(a.font)]
            size = rng.randint(a.min_size, a.max_size)
            bw = binarize(render(codes, font, size, rng, np_rng, not a.no_aug))
            if a.mode == "cluster":
                items = []
                crop = trim(bw)
                if crop is not None:
                    items = [(cname(codes), crop)]
            else:
                segs = segment_chars(bw, [(0, bw.shape[0])])
                res = label_segments(codes, segs) if segs else None
                items = [(nm, s.img) for nm, s in res] if res else []
                if not items and k == 0:
                    skipped.append(cname(codes))
            for name, arr in items:
                if name not in codes_of:  # kelas hasil pemecahan (mark/basis)
                    codes_of[name] = [int(x, 16) for x in name.split("_")]
                    freq_of[name] = int(float(
                        chars.get(codes_of[name][0], {}).get("frequency", 0) or 0))
                if counts.get(name, 0) >= target.get(name, a.max_per_class):
                    continue
                d = out / name
                d.mkdir(exist_ok=True)
                save(arr, d / f"{name}_{counts.get(name, 0):04d}.png")
                counts[name] = counts.get(name, 0) + 1

    with open(out / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for name in sorted(counts):
            w.writerow([name, " ".join(f"U+{c:04X}" for c in codes_of[name]),
                        "".join(map(chr, codes_of[name]))])
    with open(out / "class_freq.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "frequency"])
        for name in sorted(counts):
            w.writerow([name, freq_of.get(name, 0)])

    print(f"{len(counts)} kelas, {sum(counts.values())} gambar -> {out}")
    if skipped:
        print(f"Cluster dilewati (segmen tidak cocok): {len(skipped)}, "
              f"contoh: {skipped[:8]}")


if __name__ == "__main__":
    main()