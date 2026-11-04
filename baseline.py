#!/usr/bin/env python3
"""
Training + testing + analisis performa Nearest Centroid Classifier (NCC)
pada dataset grapheme yang SUDAH berlabel (tanpa tahap segmentasi).

Format dataset (pilih salah satu):
  A) Folder per kelas:   dataset/<label>/*.png
        <label> boleh berupa codepoint, mis. A9A4, A9A4_A9B8, "U+A9A4 U+A9B8"
  B) Manifest CSV (header wajib: path,label):  --manifest data.csv
        path relatif terhadap folder dataset

Contoh:
  python train_eval.py dataset --test-ratio 0.2 --repeats 5 --out report
  python train_eval.py dataset --max-train-per-class 10 --features zone9
  python train_eval.py dataset --invert          # jika teks putih di atas hitam

Keluaran di folder --out:
  metrics.json       ringkasan metrik (seed pertama + rata-rata semua repeat)
  per_class.csv      recall / precision / F1 per kelas
  confusions.csv     pasangan kelas yang paling sering tertukar
dan model akhir (dilatih dari SELURUH data) di --model (default model.npz),
kompatibel dengan `javanese_ocr.py predict`.

Dependensi: numpy, pillow, dan javanese_ocr.py (satu folder).
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from javanese_ocr import zone_features  # 9 fitur zona 3x3 dari paper

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def pretty(label: str) -> str:
    """Ubah label codepoint menjadi karakter Jawa bila memungkinkan."""
    toks = label.upper().replace("U+", " ").replace("_", " ").replace("-", " ").split()
    try:
        cps = [int(t, 16) for t in toks]
        if cps and all(0xA980 <= c <= 0xA9DF for c in cps):
            return "".join(map(chr, cps))
    except ValueError:
        pass
    return label


def trim(bw: np.ndarray):
    rows = np.where(bw.sum(axis=1) > 0)[0]
    cols = np.where(bw.sum(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return bw[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def load_binary(path, threshold, invert):
    g = np.asarray(Image.open(path).convert("L"))
    bw = (g > threshold) if invert else (g < threshold)
    return trim(bw.astype(np.uint8))


def collect(root: Path, manifest):
    items = []
    if manifest:
        with open(manifest, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                items.append((root / r["path"], r["label"]))
    else:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in EXTS:
                    items.append((f, d.name))
    return items


def load_all(root, manifest, threshold, invert):
    Z, area, W, H, y, skipped = [], [], [], [], [], 0
    for path, label in collect(Path(root), manifest):
        bw = load_binary(path, threshold, invert)
        if bw is None:
            skipped += 1
            continue
        Z.append(zone_features(bw))
        area.append(int(bw.sum()))
        H.append(bw.shape[0])
        W.append(bw.shape[1])
        y.append(label)
    if skipped:
        print(f"Dilewati (gambar kosong): {skipped}")
    return (np.array(Z), np.array(area, float), np.array(W, float),
            np.array(H, float), np.array(y))


# ----------------------------------------------------------------------------
# Fitur, split, model
# ----------------------------------------------------------------------------
def make_X(Z, area, avg_w, avg_h, mode):
    if mode == "zone9":
        return Z
    return np.hstack([Z, (area / (avg_w * avg_h))[:, None]])  # f9, Pers. (1)


def stratified_split(y, test_ratio, rng):
    by_cls = defaultdict(list)
    for i, c in enumerate(y):
        by_cls[c].append(i)
    train, test = [], []
    for c, idx in by_cls.items():
        rng.shuffle(idx)
        if len(idx) < 2:              # kelas 1 sampel: hanya untuk training
            train += idx
            continue
        n_te = max(1, int(round(test_ratio * len(idx))))
        n_te = min(n_te, len(idx) - 1)
        test += idx[:n_te]
        train += idx[n_te:]
    return np.array(train), np.array(test)


def cap_per_class(idx, y, cap, rng):
    if not cap:
        return idx
    by_cls = defaultdict(list)
    for i in idx:
        by_cls[y[i]].append(i)
    out = []
    for c, lst in by_cls.items():
        out += rng.sample(lst, cap) if len(lst) > cap else lst
    return np.array(out)


def fit_centroids(X, y):
    classes = np.array(sorted(set(y)))
    cents = np.stack([X[y == c].mean(axis=0) for c in classes])
    return classes, cents


def distances(X, cents):
    """Jarak Euclidean (N, C) -- versi hemat memori."""
    d2 = (X ** 2).sum(1)[:, None] + (cents ** 2).sum(1)[None, :] - 2 * X @ cents.T
    return np.sqrt(np.maximum(d2, 0))


# ----------------------------------------------------------------------------
# Evaluasi
# ----------------------------------------------------------------------------
def evaluate(y_true, D, classes):
    order = np.argsort(D, axis=1)
    pred = classes[order[:, 0]]
    top3 = np.array([t in classes[order[i, :3]] for i, t in enumerate(y_true)])
    acc = float((pred == y_true).mean())

    stats = {}
    pred_cnt = Counter(pred)
    for c in sorted(set(y_true)):
        sup = int((y_true == c).sum())
        tp = int(((y_true == c) & (pred == c)).sum())
        rec = tp / sup if sup else 0.0
        prec = tp / pred_cnt[c] if pred_cnt[c] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        stats[c] = dict(support=sup, recall=rec, precision=prec, f1=f1)
    macro = lambda k: float(np.mean([s[k] for s in stats.values()]))
    conf = Counter((t, p) for t, p in zip(y_true, pred) if t != p)
    return dict(accuracy=acc, top3=float(top3.mean()),
                macro_precision=macro("precision"), macro_recall=macro("recall"),
                macro_f1=macro("f1"), n_test=len(y_true),
                n_classes_test=len(stats)), stats, conf


def run_once(Z, area, W, H, y, args, seed):
    rng = random.Random(seed)
    tr, te = stratified_split(y, args.test_ratio, rng)
    tr = cap_per_class(tr, y, args.max_train_per_class, rng)
    avg_w, avg_h = W[tr].mean(), H[tr].mean()      # dari data train saja
    Xtr = make_X(Z[tr], area[tr], avg_w, avg_h, args.features)
    Xte = make_X(Z[te], area[te], avg_w, avg_h, args.features)
    classes, cents = fit_centroids(Xtr, y[tr])
    D = distances(Xte, cents)
    m, stats, conf = evaluate(y[te], D, classes)
    m["n_train"] = int(len(tr))
    m["n_classes_train"] = int(len(classes))
    train_cnt = Counter(y[tr])
    return m, stats, conf, train_cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--manifest")
    ap.add_argument("--out", default="report")
    ap.add_argument("--model", default="model.npz")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=1,
                    help="jumlah split acak (seed 0..N-1) untuk mean +- std")
    ap.add_argument("--max-train-per-class", type=int, default=0,
                    help="0 = pakai semua; paper memakai 10")
    ap.add_argument("--features", choices=["full10", "zone9"], default="full10")
    ap.add_argument("--threshold", type=int, default=160)
    ap.add_argument("--invert", action="store_true",
                    help="teks terang di atas latar gelap")
    a = ap.parse_args()

    Z, area, W, H, y = load_all(a.dataset, a.manifest, a.threshold, a.invert)
    print(f"{len(y)} gambar, {len(set(y))} kelas")

    results = []
    for seed in range(a.repeats):
        m, stats, conf, train_cnt = run_once(Z, area, W, H, y, a, seed)
        results.append(m)
        print(f"[seed {seed}] acc={m['accuracy']*100:.2f}%  top3={m['top3']*100:.2f}%  "
              f"macroF1={m['macro_f1']*100:.2f}%  (train {m['n_train']}, "
              f"test {m['n_test']})")
        if seed == 0:
            first = (m, stats, conf, train_cnt)

    keys = ["accuracy", "top3", "macro_precision", "macro_recall", "macro_f1"]
    summary = {k: dict(mean=float(np.mean([r[k] for r in results])),
                       std=float(np.std([r[k] for r in results]))) for k in keys}
    print("\n== Ringkasan ({} repeat) ==".format(a.repeats))
    for k in keys:
        print(f"{k:16s} {summary[k]['mean']*100:6.2f}% +- {summary[k]['std']*100:.2f}")

    # ---- laporan detail (seed 0) ----
    m, stats, conf, train_cnt = first
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_class.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "char", "train_n", "test_n", "recall", "precision", "f1"])
        for c, s in sorted(stats.items(), key=lambda kv: kv[1]["f1"]):
            w.writerow([c, pretty(c), train_cnt.get(c, 0), s["support"],
                        f"{s['recall']:.3f}", f"{s['precision']:.3f}", f"{s['f1']:.3f}"])
    with open(out / "confusions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true", "true_char", "pred", "pred_char", "count"])
        for (t, p), n in conf.most_common(50):
            w.writerow([t, pretty(t), p, pretty(p), n])
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(a), first_split=m, summary=summary), f,
                  indent=2, ensure_ascii=False)

    print("\nKelas terlemah (F1):")
    for c, s in sorted(stats.items(), key=lambda kv: kv[1]["f1"])[:10]:
        print(f"  {pretty(c):>6s} {c:<18s} F1={s['f1']:.2f} "
              f"(test {s['support']}, train {train_cnt.get(c, 0)})")
    print("Pasangan paling sering tertukar:")
    for (t, p), n in conf.most_common(10):
        print(f"  {pretty(t)} ({t}) -> {pretty(p)} ({p}) : {n}x")

    # ---- model akhir: latih dari SELURUH data ----
    rng = random.Random(0)
    allidx = cap_per_class(np.arange(len(y)), y, a.max_train_per_class, rng)
    avg_w, avg_h = W[allidx].mean(), H[allidx].mean()
    X = make_X(Z[allidx], area[allidx], avg_w, avg_h, a.features)
    classes, cents = fit_centroids(X, y[allidx])
    np.savez(a.model, classes=classes, centroids=cents,
             avg_w=avg_w, avg_h=avg_h, features=a.features)
    print(f"\nModel akhir ({len(classes)} kelas) -> {a.model}; laporan -> {out}/")


if __name__ == "__main__":
    main()