"""
cek_variasi_fold.py
===================
Pemeriksaan tambahan tahap klasifikasi hibrida (Subbab 4.6): seberapa jauh
AUC bergeser bila pembagian meter ke dalam 5 fold diacak ulang.

Cara kerja: validasi silang 5-fold per meter (StratifiedGroupKFold) diulang
untuk 30 seed pengocokan (0 sampai 29). Pada tiap seed, regresi logistik
dilatih dengan SMOTE di dalam fold untuk tiga himpunan fitur: skor LSTM-AE
saja, indikator per fasa saja, dan hibrida (keduanya). Yang dilaporkan adalah
median dan rentang (minimum sampai maksimum) AUC gabungan out-of-fold, AUC
rata-rata per fold, dan AUC tingkat meter.

Masukan : data/fitur_harian_hibrida.csv (keluaran fitur_indikator.py)
Keluaran: experiments/analysis_hibrida/variasi_fold_30seed.csv
Jalankan: python cek_variasi_fold.py   (sekitar satu menit)
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

BASE = os.path.dirname(os.path.abspath(__file__))
FITUR_FILE = os.path.join(BASE, "data", "fitur_harian_hibrida.csv")
OUT_FILE = os.path.join(BASE, "experiments", "analysis_hibrida", "variasi_fold_30seed.csv")
N_SEED, N_SPLITS = 30, 5
FITUR_SET = {"AE": ["s_ae"],
             "Indikator": ["fasa_mati", "teg_hilang", "rasio_register", "utilisasi"],
             "Hibrida": ["s_ae", "fasa_mati", "teg_hilang", "rasio_register", "utilisasi"]}


def auc_meter(p, F):
    """AUC tingkat meter: rata-rata probabilitas per kasus (pre) vs per meter kontrol."""
    d = pd.DataFrame({"p": p, "unit": F["unit"], "meter": F["meter_id"], "periode": F["periode"]})
    kasus = d[d["periode"] == "pre_p2tl"].groupby("unit")["p"].mean()
    kontrol = d[d["periode"] == "normal"].groupby("meter")["p"].mean()
    y = np.r_[np.ones(len(kasus)), np.zeros(len(kontrol))]
    return roc_auc_score(y, np.r_[kasus.values, kontrol.values])


def main():
    F = pd.read_csv(FITUR_FILE, dtype={"meter_id": str, "case_id": str})
    y, g = F["y"].to_numpy(int), F["meter_id"].to_numpy()
    baris = []
    for seed in range(N_SEED):
        folds = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
                     .split(F[["s_ae"]], y, g))
        for nama, kolom in FITUR_SET.items():
            X, oof, auc_fold = F[kolom].to_numpy(float), np.zeros(len(y)), []
            for tr, te in folds:
                sc = StandardScaler().fit(X[tr])
                Xtr, ytr = SMOTE(random_state=42, k_neighbors=5).fit_resample(sc.transform(X[tr]), y[tr])
                model = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
                oof[te] = model.predict_proba(sc.transform(X[te]))[:, 1]
                auc_fold.append(roc_auc_score(y[te], oof[te]))
            baris.append({"seed": seed, "fitur": nama, "auc_gabungan": roc_auc_score(y, oof),
                          "auc_rata_fold": float(np.mean(auc_fold)), "auc_meter": auc_meter(oof, F)})
    hasil = pd.DataFrame(baris)
    hasil.to_csv(OUT_FILE, index=False)
    ringkas = hasil.groupby("fitur").agg(["median", "min", "max"]).drop(columns="seed")
    print(ringkas.round(3).to_string())


if __name__ == "__main__":
    main()
