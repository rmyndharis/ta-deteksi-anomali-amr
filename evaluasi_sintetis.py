"""
Evaluasi Anomali Sintetis Terkendali (pelengkap evaluasi lapangan)
===================================================================
Tujuan: mengukur sensitivitas model utama (masked LSTM-AE latih populasi)
terhadap penekanan energi yang JENIS dan BESARANNYA diketahui pasti,
terpisah dari kebisingan label lapangan. Dilaporkan terpisah dari metrik
lapangan dan tidak menggantikannya.

Basis: seluruh window meter kontrol dataset kasus-kontrol (tidak pernah
dilihat model karena model dilatih pada populasi 2025, dan berlabel
normal). Tiap window dibuat salinan termanipulasi pada
kanal energi (kWh dan arus, proporsional agar konsisten secara fisis);
tegangan, faktor daya, dan fitur temporal tidak diubah. Manipulasi
diterapkan pada NILAI MENTAH (skala balik memakai scaler per meter),
lalu diskalakan ulang dengan scaler yang sama.

Tipe manipulasi (mengadaptasi pola serangan sintetis pada literatur ETD,
rujukan [11] dan [20] pada naskah):
  T1  Skala tetap        : x' = alpha * x, alpha in {0,9; 0,7; 0,5; 0,3}
  T2  Skala acak/interval: x'_t = beta_t * x_t, beta_t ~ U(0,2, 0,8)
  T3  Padam sebagian     : x'_t = 0 pada 12 slot (6 jam) berurutan acak
  T4  Profil diratakan   : x'_t = rata-rata harian x (total harian TETAP)
  T5  Urutan dibalik     : x'_t = x_(T-1-t) (total harian TETAP)

T4 dan T5 sengaja disertakan sebagai kontrol negatif: skor residual
berarah rata-rata harian memang dirancang peka terhadap PENEKANAN energi,
bukan perubahan bentuk yang menjaga total harian.

Tabel 4.2 pada naskah melaporkan T1, T4, dan T5 sesuai rancangan pengujian
Bab III; T2 dan T3 dihitung sebagai pelengkap dan hasilnya hanya tersimpan
di tabel_sintetis.csv.

Artefak yang dipakai (aktif): models/ (model, ambang, scaler) dan
data/processed_sequences.pkl (set uji kasus-kontrol).
Keluaran: experiments/analysis_v11/sintetis/
  - tabel_sintetis.csv  (AUC, Recall pada ambang P95, median pergeseran skor)
  - auc_vs_alpha.png    (kurva AUC terhadap alpha untuk T1)

Jalankan: ./venv/bin/python evaluasi_sintetis.py
Env opsional (uji cepat): SINTETIS_MAX_WINDOWS=300, TA_SMOKE_OUT=<folder>
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import MODEL_FEATURES, RANDOM_SEED
from model import mask_energy_channels, MASKED_IDX
from artefak import load_processed, load_model_and_threshold, load_scalers

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("TA_SMOKE_OUT",
                     os.path.join(BASE, "experiments", "analysis_v11", "sintetis"))
os.makedirs(OUT, exist_ok=True)

IDX_KWH = MODEL_FEATURES.index("kWh")
IDX_CUR = MODEL_FEATURES.index("current")
ENERGY_IDX = [IDX_KWH, IDX_CUR]
CLIP = 10.0


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    _, _, X_test, _, ti = load_processed()
    kontrol = (ti["grp"] == "kontrol").to_numpy()
    X = X_test[kontrol].astype(np.float32)
    meters = ti.loc[kontrol, "meter_id"].astype(str).to_numpy()
    max_w = os.environ.get("SINTETIS_MAX_WINDOWS")
    if max_w:
        X, meters = X[:int(max_w)], meters[:int(max_w)]
    print(f"Window normal meter kontrol: {len(X)} dari {len(set(meters))} meter")

    scalers = load_scalers()
    med = np.zeros((len(X), 1, 2), np.float32)
    scl = np.ones((len(X), 1, 2), np.float32)
    for i, m in enumerate(meters):
        s = scalers[str(m)] if str(m) in scalers else scalers[m]
        med[i, 0, 0] = s["median"]["kWh"];     scl[i, 0, 0] = s["scale"]["kWh"]
        med[i, 0, 1] = s["median"]["current"]; scl[i, 0, 1] = s["scale"]["current"]

    def to_raw(Xs):
        return Xs[:, :, ENERGY_IDX] * scl + med

    def from_raw(raw, Xs):
        out = Xs.copy()
        out[:, :, ENERGY_IDX] = np.clip((raw - med) / scl, -CLIP, CLIP)
        return out

    mdl, thr = load_model_and_threshold()

    def score(Xs):
        rec = mdl.predict(mask_energy_channels(Xs), batch_size=512, verbose=0)
        return (rec[:, :, MASKED_IDX] - Xs[:, :, MASKED_IDX]).mean(axis=(1, 2))

    s0 = score(X)
    fpr0 = float((s0 > thr).mean())
    print(f"Ambang P95 = {thr:.4f} | flag rate window asli (FPR) = {fpr0:.3f}")

    raw0 = to_raw(X)
    T = X.shape[1]
    manip = []

    for a in [0.9, 0.7, 0.5, 0.3]:
        manip.append((f"T1 skala tetap {a:.1f}".replace(".", ","), from_raw(raw0 * a, X)))

    beta = rng.uniform(0.2, 0.8, size=(len(X), T, 1)).astype(np.float32)
    manip.append(("T2 skala acak U(0,2-0,8)", from_raw(raw0 * beta, X)))

    raw3 = raw0.copy()
    starts = rng.integers(0, T - 12, size=len(X))
    for i, st in enumerate(starts):
        raw3[i, st:st + 12, :] = 0.0
    manip.append(("T3 padam 6 jam berurutan", from_raw(raw3, X)))

    raw4 = np.repeat(raw0.mean(axis=1, keepdims=True), T, axis=1)
    manip.append(("T4 profil diratakan (total tetap)", from_raw(raw4, X)))

    manip.append(("T5 urutan dibalik (total tetap)", from_raw(raw0[:, ::-1, :], X)))

    rows = []
    for nama, Xa in manip:
        sa = score(np.ascontiguousarray(Xa))
        y = np.r_[np.zeros(len(s0)), np.ones(len(sa))]
        auc = roc_auc_score(y, np.r_[s0, sa])
        rec = float((sa > thr).mean())
        rows.append({"manipulasi": nama, "auc": round(auc, 3),
                     "recall_thr_p95": round(rec, 3),
                     "fpr_asli": round(fpr0, 3),
                     "median_pergeseran_skor": round(float(np.median(sa - s0)), 3)})
        print(f"  {nama:<34} AUC={auc:.3f}  Recall@P95={rec:.3f}")

    tab = pd.DataFrame(rows)
    tab["n_window"] = len(X)
    tab["n_meter"] = len(set(meters))
    tab.to_csv(os.path.join(OUT, "tabel_sintetis.csv"), index=False)

    alphas = [0.9, 0.7, 0.5, 0.3]
    aucs = [r["auc"] for r in rows[:4]]
    recs = [r["recall_thr_p95"] for r in rows[:4]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([1 - a for a in alphas], aucs, "o-", lw=2, label="AUC")
    ax.plot([1 - a for a in alphas], recs, "s--", lw=2, label="Recall pada ambang P95")
    ax.set_xlabel("Besaran penekanan energi (1 - alpha)")
    ax.set_ylabel("Nilai")
    ax.set_ylim(0, 1.02)
    ax.set_title("Sensitivitas Model terhadap Penekanan Energi Sintetis (T1)")
    ax.grid(alpha=.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "auc_vs_alpha.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSELESAI. Output di:", OUT)


if __name__ == "__main__":
    main()
