"""
Pemeriksaan Tambahan Subbab 4.8: Dua Suku Skor Deteksi (tanpa pelatihan ulang)
==============================================================================
Persamaan 3.1 menyatakan skor deteksi = rata-rata (profil harapan - aktual) pada
kanal kWh dan arus ternormalisasi. Karena rata-rata itu dapat dipisah, skor sama
dengan deviasi dua kanal (pembanding tanpa pelatihan: -rata-rata aktual) ditambah
suku profil harapan (rata-rata keluaran model pada kedua kanal). Skrip ini menilai
kedua suku secara terpisah pada 4.959 window uji yang sama:
  - AUC tiap suku pada tingkat window (window pre = positif) dengan interval
    kepercayaan bootstrap klaster (case_id untuk positif, meter_id untuk negatif,
    protokol yang sama dengan analisis_v11.py) dan pada tingkat meter;
  - rata-rata suku profil harapan pada window pre, post, dan kontrol;
  - korelasi suku profil harapan dengan suku deviasi dan dengan rata-rata harian
    kWh aktual, tegangan, dan faktor daya ternormalisasi.
Masukan: output/detection_results.pkl (skor) dan data/processed_sequences.pkl
(window uji ternormalisasi). Keluaran: experiments/analysis_v11/suku_profil_harapan.json.
Jalankan dari code/:  ./venv/bin/python suku_profil_harapan.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from config import MODEL_FEATURES, OUTPUT_DIR, DATA_DIR

OUT = os.path.join(BASE, "experiments", "analysis_v11")
os.makedirs(OUT, exist_ok=True)
B_BOOT = int(os.environ.get("B_BOOT", 2000))
I_KWH, I_ARUS = MODEL_FEATURES.index("kWh"), MODEL_FEATURES.index("current")
I_TEG, I_PF = MODEL_FEATURES.index("voltage"), MODEL_FEATURES.index("power_factor")


def auc(pos, neg):
    """AUC-ROC via statistik Mann-Whitney (sama dengan analisis_v11.py)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_boot_auc(pos, pos_cl, neg, neg_cl, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pg = {k: pos[np.asarray(pos_cl) == k] for k in np.unique(pos_cl)}
    ng = {k: neg[np.asarray(neg_cl) == k] for k in np.unique(neg_cl)}
    pk, nk = list(pg), list(ng)
    v = np.empty(B)
    for b in range(B):
        ps = np.concatenate([pg[k] for k in rng.choice(pk, len(pk))])
        ns = np.concatenate([ng[k] for k in rng.choice(nk, len(nk))])
        v[b] = auc(ps, ns)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


det = pickle.load(open(os.path.join(OUTPUT_DIR, "detection_results.pkl"), "rb"))
seq = pickle.load(open(os.path.join(DATA_DIR, "processed_sequences.pkl"), "rb"))
X, y = seq["X_test"], np.asarray(seq["y_test"]).astype(int)
info = det["test_info"].reset_index(drop=True)
skor = np.asarray(det["re_test"], float)
assert len(skor) == len(X) == len(y) == len(info)

deviasi = -X[:, :, [I_KWH, I_ARUS]].mean(axis=(1, 2))   # pembanding deviasi dua kanal
profil = skor - deviasi                                  # suku profil harapan (sumbangan model)
pre = y == 1
neg_cl = info.loc[~pre, "meter_id"].to_numpy()
pos_cl = info.loc[pre, "case_id"].to_numpy()

hasil = {"n_window": int(len(y)), "n_pre": int(pre.sum()), "B_bootstrap": B_BOOT, "auc_window": {}}
for nama, v in (("skor_deteksi", skor), ("deviasi_dua_kanal", deviasi), ("suku_profil_harapan", profil)):
    lo, hi = cluster_boot_auc(v[pre], pos_cl, v[~pre], neg_cl)
    hasil["auc_window"][nama] = {"auc": auc(v[pre], v[~pre]), "ci95_klaster": [lo, hi]}
    print(f"AUC window {nama:22s} {hasil['auc_window'][nama]['auc']:.4f}  IK95 klaster [{lo:.3f}, {hi:.3f}]")

# tingkat meter: rata-rata per kasus (window pre) vs rata-rata per meter kontrol
df = info.assign(skor=skor, deviasi=deviasi, profil=profil)
kasus = df[df.periode == "pre_p2tl"].groupby("case_id")[["skor", "deviasi", "profil"]].mean()
kontrol = df[df.grp == "kontrol"].groupby("meter_id")[["skor", "deviasi", "profil"]].mean()
hasil["auc_meter"] = {c: auc(kasus[c], kontrol[c]) for c in ("skor", "deviasi", "profil")}
hasil["n_kasus"], hasil["n_kontrol_meter"] = int(len(kasus)), int(len(kontrol))
print("AUC tingkat meter:", {k: round(v, 3) for k, v in hasil["auc_meter"].items()})

kel = {"pre": pre, "post": (info.periode == "post_p2tl").to_numpy(), "kontrol": (info.grp == "kontrol").to_numpy()}
hasil["rata_rata_suku_profil_harapan"] = {k: float(profil[m].mean()) for k, m in kel.items()}
hasil["rata_rata_deviasi"] = {k: float(deviasi[m].mean()) for k, m in kel.items()}
hasil["simpangan_baku"] = {"deviasi": float(deviasi.std()), "suku_profil_harapan": float(profil.std())}
rerata = lambda i: X[:, :, i].mean(axis=1)
hasil["korelasi_pearson_suku_profil_harapan"] = {
    "dengan_deviasi": float(np.corrcoef(deviasi, profil)[0, 1]),
    "dengan_kwh_aktual": float(np.corrcoef(rerata(I_KWH), profil)[0, 1]),
    "dengan_tegangan": float(np.corrcoef(rerata(I_TEG), profil)[0, 1]),
    "dengan_faktor_daya": float(np.corrcoef(rerata(I_PF), profil)[0, 1]),
}
print("rata-rata suku profil harapan:", {k: round(v, 3) for k, v in hasil["rata_rata_suku_profil_harapan"].items()})
print("korelasi:", {k: round(v, 3) for k, v in hasil["korelasi_pearson_suku_profil_harapan"].items()})
with open(os.path.join(OUT, "suku_profil_harapan.json"), "w") as f:
    json.dump(hasil, f, indent=2)
print("tersimpan:", os.path.join(OUT, "suku_profil_harapan.json"))
