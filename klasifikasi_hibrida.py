"""
Klasifikasi Hibrida dengan Penyeimbangan Data
=============================================
Tahap klasifikasi di atas LSTM Autoencoder: skor window LSTM-AE (s_ae) dan
empat indikator per fasa (fitur_indikator.py) dipakai sebagai fitur regresi
logistik yang dilatih pada window berlabel dengan tiga perlakuan data latih:

  tanpa        : data latih apa adanya (668 pre vs 4.291 normal)
  random_over  : random oversampling kelas minoritas
  smote        : SMOTE (Chawla dkk. 2002), pustaka imbalanced-learn

Tiga himpunan fitur: AE (s_ae saja), Indikator (4 indikator), Hibrida (keduanya).

Protokol: validasi silang 5-fold dipisah per METER (StratifiedGroupKFold),
sehingga window dari meter yang sama tidak pernah berada di data latih dan
uji sekaligus; penskalaan dan resampling hanya dipasang pada bagian latih
tiap fold. Metrik tingkat window (AUC-ROC, PR-AUC, precision/recall/F1 pada
ambang 0,5) dan tingkat pelanggan (rata-rata probabilitas per unit: 32 kasus
vs 85 meter kontrol; AUC, kasus tertangkap pada k = 6, 12, 23). AUC window
dilaporkan dua cara: atas gabungan probabilitas out-of-fold kelima fold
(auc) dan rata-rata AUC yang dihitung di dalam tiap fold (auc_rata_fold);
keduanya berbeda karena skala probabilitas lima model fold tidak seragam.

Output: experiments/analysis_hibrida/
  tabel_hibrida.csv, tabel_cakupan_indikator.csv, ringkasan_hibrida.json,
  gambar_roc_hibrida.png, gambar_efek_penyeimbangan.png, oof_hibrida.csv

Jalankan dari folder code/:  ./venv/bin/python klasifikasi_hibrida.py

Penulis: Yudhi Armyndharis (220401010272)
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score, roc_curve)
from imblearn.over_sampling import RandomOverSampler, SMOTE

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
FITUR_FILE = os.path.join(BASE, "data", "fitur_harian_hibrida.csv")
OUT = os.path.join(BASE, "experiments", "analysis_hibrida")
os.makedirs(OUT, exist_ok=True)

SEED = 42
N_SPLITS = 5
INDIKATOR = ["fasa_mati", "teg_hilang", "rasio_register", "utilisasi"]
FITUR_SET = {"AE": ["s_ae"], "Indikator": INDIKATOR, "Hibrida": ["s_ae"] + INDIKATOR}
PENYEIMBANGAN = ["tanpa", "random_over", "smote"]
K_METER = [6, 12, 23]
WARNA = {"AE": "#2a78d6", "Indikator": "#eb6834", "Hibrida": "#1baf7a", "SMOTE": "#4a3aa7"}


def seimbangkan(X, y, cara):
    """Resampling hanya pada bagian latih; 'tanpa' mengembalikan data apa adanya."""
    if cara == "tanpa":
        return X, y
    s = RandomOverSampler(random_state=SEED) if cara == "random_over" \
        else SMOTE(random_state=SEED, k_neighbors=5)
    return s.fit_resample(X, y)


def metrik_window(p, y):
    pred = (p >= 0.5).astype(int)
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "n_flag": int(pred.sum())}


def skor_unit(p, F):
    """Rata-rata probabilitas per kasus (window pre) dan per meter kontrol."""
    d = F[["unit", "periode", "grp"]].copy()
    d["p"] = p
    pos = d[d.periode == "pre_p2tl"].groupby("unit")["p"].mean()
    neg = d[d.grp == "kontrol"].groupby("unit")["p"].mean()
    s = np.r_[pos.to_numpy(), neg.to_numpy()]
    lab = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    return s, lab


def auc_rata_fold(p, y, fold):
    """Rata-rata AUC-ROC yang dihitung terpisah di dalam tiap fold (estimand
    kedua di samping AUC atas gabungan probabilitas out-of-fold)."""
    return float(np.mean([roc_auc_score(y[fold == k], p[fold == k]) for k in np.unique(fold)]))


def metrik_meter(p, F):
    s, lab = skor_unit(p, F)
    urut = np.argsort(-s, kind="mergesort")
    m = {"auc_meter": roc_auc_score(lab, s), "n_kasus": int(lab.sum()), "n_kontrol": int((lab == 0).sum())}
    for k in K_METER:
        m[f"kasus_top{k}"] = int(lab[urut[:k]].sum())
    return m


def bagi_fold(F):
    """Nomor fold (0..N_SPLITS-1) tiap window: StratifiedGroupKFold per meter,
    seed tetap. Disimpan ke oof_hibrida.csv (kolom fold) agar pembagian yang
    dipakai naskah dapat diperiksa ulang; pembagian bergantung pada versi
    scikit-learn (lihat cek_variasi_fold.py untuk sebaran antar pembagian)."""
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold = np.full(len(F), -1, dtype=int)
    for k, (_, te) in enumerate(skf.split(F[["s_ae"]].to_numpy(float), F["y"].to_numpy(int),
                                          F["meter_id"].to_numpy())):
        fold[te] = k
    return fold


def jalankan_cv(F, kolom, cara, fold=None):
    """Validasi silang per meter; kembalikan probabilitas out-of-fold tiap window.
    Bila `fold` (dari bagi_fold) diberikan, pembagian itu yang dipakai; bila
    tidak, pembagian dibuat dengan seed yang sama (hasil identik)."""
    X = F[kolom].to_numpy(float)
    y = F["y"].to_numpy(int)
    if fold is None:
        fold = bagi_fold(F)
    oof = np.zeros(len(y))
    for k in range(N_SPLITS):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        sc = StandardScaler().fit(X[tr])
        Xtr, ytr = seimbangkan(sc.transform(X[tr]), y[tr], cara)
        model = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
        oof[te] = model.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def cakupan_indikator(F):
    """Lapisan aturan: ambang = persentil 95 unit kontrol; berapa kasus tertangkap."""
    arah = {"s_ae": 1, "fasa_mati": 1, "teg_hilang": 1, "rasio_register": -1, "utilisasi": -1}
    U = F.groupby(["unit", "periode", "grp"])[list(arah)].mean().reset_index()
    U = U[(U.periode == "pre_p2tl") | (U.grp == "kontrol")]
    pos, neg = U[U.periode == "pre_p2tl"], U[U.grp == "kontrol"]
    baris, tertangkap = [], {}
    for f, a in arah.items():
        thr = np.quantile(a * neg[f], 0.95)
        c = set(pos.loc[a * pos[f] > thr, "unit"].str.replace("_pre_p2tl", ""))
        tertangkap[f] = c
        baris.append({"indikator": f, "arah": "tinggi" if a > 0 else "rendah",
                      "ambang_p95_kontrol": float(a * thr),
                      "auc_unit": roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                                                np.r_[a * pos[f], a * neg[f]]),
                      "kasus_tertangkap": len(c), "kontrol_tertandai": int((a * neg[f] > thr).sum()),
                      "daftar_kasus": " ".join(sorted(c))})
    for nama, fs in {"skor + fasa_mati": ["s_ae", "fasa_mati"],
                     "semua": list(arah)}.items():
        u = set().union(*[tertangkap[f] for f in fs])
        baris.append({"indikator": "gabungan " + nama, "arah": "gabungan", "ambang_p95_kontrol": np.nan,
                      "auc_unit": np.nan, "kasus_tertangkap": len(u), "kontrol_tertandai": np.nan,
                      "daftar_kasus": " ".join(sorted(u))})
    # aturan keras: tepat satu fasa mati pada > 80% interval berbeban
    keras_pos = set(pos.loc[pos.fasa_mati > 0.8, "unit"].str.replace("_pre_p2tl", ""))
    baris.append({"indikator": "aturan fasa_mati > 0,8", "arah": "tinggi", "ambang_p95_kontrol": 0.8,
                  "auc_unit": np.nan, "kasus_tertangkap": len(keras_pos),
                  "kontrol_tertandai": int((neg.fasa_mati > 0.8).sum()), "daftar_kasus": " ".join(sorted(keras_pos))})
    return pd.DataFrame(baris)


def gambar_roc(F, oof):
    from visualize import sumbu_koma
    fig, ax = plt.subplots(figsize=(7, 6))
    y = F["y"].to_numpy(int)
    kurva = [("Skor LSTM-AE (tanpa klasifikasi)", F["s_ae"].to_numpy(), WARNA["AE"], "-"),
             ("Indikator, regresi logistik + SMOTE", oof[("Indikator", "smote")], WARNA["Indikator"], "--"),
             ("Hibrida, regresi logistik tanpa penyeimbangan", oof[("Hibrida", "tanpa")], WARNA["Hibrida"], "-"),
             ("Hibrida, regresi logistik + SMOTE", oof[("Hibrida", "smote")], WARNA["SMOTE"], "-.")]
    for label, p, warna, gaya in kurva:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=warna, linestyle=gaya, linewidth=2,
                label=f"{label} (AUC = {roc_auc_score(y, p):.3f})".replace(".", ","))
    ax.plot([0, 1], [0, 1], color="#9a9a9a", linestyle=":", linewidth=1, label="Acak (AUC = 0,5)")
    ax.set_xlabel("False positive rate (proporsi window normal yang ikut tertandai)", fontsize=10)
    ax.set_ylabel("True positive rate (proporsi window masa sebelum penertiban yang tertandai)", fontsize=10)
    ax.set_title("Kurva ROC Tingkat Window (validasi silang per meter)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    sumbu_koma(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "gambar_roc_hibrida.png"), dpi=150, bbox_inches="tight")
    plt.close()


def gambar_efek(tw):
    """Recall dan precision pada ambang 0,5 per perlakuan penyeimbangan (fitur Hibrida)."""
    d = tw[tw.fitur == "Hibrida"].set_index("penyeimbangan").loc[PENYEIMBANGAN]
    label = {"tanpa": "tanpa penyeimbangan", "random_over": "random oversampling", "smote": "SMOTE"}
    x = np.arange(len(PENYEIMBANGAN)); lebar = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (metrik, nama, warna) in enumerate([("recall", "Recall", WARNA["Hibrida"]), ("precision", "Precision", WARNA["SMOTE"])]):
        b = ax.bar(x + (i - 0.5) * lebar, d[metrik], lebar * 0.94, color=warna, label=nama)
        for rect, val in zip(b, d[metrik]):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01, f"{val:.2f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels([label[c] for c in PENYEIMBANGAN], fontsize=10)
    ax.set_ylabel("Nilai pada ambang 0,5", fontsize=10); ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=9)
    ax.set_title("Pengaruh Penyeimbangan Data Latih (regresi logistik, fitur hibrida)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "gambar_efek_penyeimbangan.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("KLASIFIKASI HIBRIDA DENGAN PENYEIMBANGAN DATA")
    print("=" * 70)
    F = pd.read_csv(FITUR_FILE, dtype={"meter_id": str, "case_id": str})
    y = F["y"].to_numpy(int)
    print(f"  Window: {len(F):,} | pre={int(y.sum())} | normal={int((y == 0).sum())} | meter={F.meter_id.nunique()}")

    # Acuan: skor LSTM-AE mentah (tanpa klasifikasi)
    acuan = {"auc": roc_auc_score(y, F.s_ae), "pr_auc": average_precision_score(y, F.s_ae)}
    acuan.update(metrik_meter(F.s_ae.to_numpy(), F))
    fold = bagi_fold(F)
    acuan["auc_rata_fold"] = auc_rata_fold(F.s_ae.to_numpy(), y, fold)
    print(f"\n  Acuan skor LSTM-AE: AUC window={acuan['auc']:.3f} | PR-AUC={acuan['pr_auc']:.3f} | "
          f"AUC meter={acuan['auc_meter']:.3f} | kasus top 6/12/23 = "
          f"{acuan['kasus_top6']}/{acuan['kasus_top12']}/{acuan['kasus_top23']}")

    print(f"  Fold (per meter, seed {SEED}): window per fold = {np.bincount(fold).tolist()}, "
          f"meter per fold = {[int(F.meter_id[fold == k].nunique()) for k in range(N_SPLITS)]}")
    baris, oof_semua = [], {}
    for fitur, kolom in FITUR_SET.items():
        for cara in PENYEIMBANGAN:
            oof = jalankan_cv(F, kolom, cara, fold)
            oof_semua[(fitur, cara)] = oof
            r = {"fitur": fitur, "penyeimbangan": cara}
            r.update(metrik_window(oof, y)); r.update(metrik_meter(oof, F))
            r["auc_rata_fold"] = auc_rata_fold(oof, y, fold)
            baris.append(r)
            print(f"  {fitur:9s} {cara:12s} AUC={r['auc']:.3f} (rata-rata per fold {r['auc_rata_fold']:.3f}) "
                  f"AP={r['pr_auc']:.3f} P={r['precision']:.3f} "
                  f"R={r['recall']:.3f} F1={r['f1']:.3f} | meter AUC={r['auc_meter']:.3f} "
                  f"top6/12/23={r['kasus_top6']}/{r['kasus_top12']}/{r['kasus_top23']}")

    tw = pd.DataFrame(baris)
    tw.to_csv(os.path.join(OUT, "tabel_hibrida.csv"), index=False)
    ck = cakupan_indikator(F)
    ck.to_csv(os.path.join(OUT, "tabel_cakupan_indikator.csv"), index=False)
    pd.DataFrame({f"{a}|{b}": v for (a, b), v in oof_semua.items()}) \
        .assign(unit=F.unit, meter_id=F.meter_id, date=F.date, y=F.y, fold=fold) \
        .to_csv(os.path.join(OUT, "oof_hibrida.csv"), index=False)
    gambar_roc(F, oof_semua)
    gambar_efek(tw)
    with open(os.path.join(OUT, "ringkasan_hibrida.json"), "w") as f:
        json.dump({"n_window": int(len(F)), "n_pre": int(y.sum()), "seed": SEED, "n_splits": N_SPLITS,
                   "acuan_lstm_ae": acuan, "tabel": tw.to_dict(orient="records"),
                   "cakupan_indikator": ck.drop(columns=["daftar_kasus"]).to_dict(orient="records")},
                  f, indent=2, default=float)

    print("\n  Cakupan indikator pada ambang P95 kontrol (tingkat pelanggan):")
    print(ck[["indikator", "arah", "auc_unit", "kasus_tertangkap", "kontrol_tertandai", "daftar_kasus"]].to_string(index=False))
    print(f"\n  Output: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
