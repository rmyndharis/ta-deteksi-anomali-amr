"""
Indikator Fisik Per Fasa (Harian) + Skor LSTM Autoencoder
==========================================================
Menghitung indikator sederhana per hari dari dataset per fasa, lalu
menggabungkannya dengan skor window harian LSTM Autoencoder (artefak
output/detection_results.pkl) pada pasangan (meter_id, tanggal).

Empat indikator per interval (sisi sekunder meter):
  fasa_mati     : berbeban (Imax > 0,05 A) dan TEPAT satu fasa arusnya < 10% fasa
                  tertinggi (pola satu fasa tidak terukur)
  teg_hilang    : tegangan satu fasa < 50% fasa tertinggi
  rasio_register: kWh tercatat / (sum_i V_i I_i PF dt), konsistensi register energi
                  terhadap besaran ukur (nilai jauh < 1 = energi tidak tercatat penuh)
  utilisasi     : daya aktif rata-rata (kW, sisi primer = kWh x faktormeter / dt)
                  dibagi daya kontrak (kVA)
Agregasi per hari: fraksi (fasa_mati, teg_hilang), rasio jumlah harian
(rasio_register), rata-rata (utilisasi).

Output: data/fitur_harian_hibrida.csv (satu baris per window harian set uji)

Jalankan dari folder code/:  ./venv/bin/python fitur_indikator.py

Penulis: Yudhi Armyndharis (220401010272)
"""
import os
import numpy as np
import pandas as pd

from config import DATA_DIR
from artefak import load_detection

BASE = os.path.dirname(os.path.abspath(__file__))
FASA_FILE = os.path.join(os.path.dirname(BASE), "data", "AP2T", "dataset_lp_p2tl_fasa.csv")
OUT = os.path.join(DATA_DIR, "fitur_harian_hibrida.csv")

INDIKATOR = ["fasa_mati", "teg_hilang", "rasio_register", "utilisasi"]


def indikator_harian(df):
    V = df[["voltage_l1", "voltage_l2", "voltage_l3"]].to_numpy(float)
    I = df[["current_l1", "current_l2", "current_l3"]].to_numpy(float)
    PF = df["power_factor"].to_numpy(float)
    Vmax, Vmin = V.max(1), V.min(1)
    Isort = np.sort(I, axis=1)
    Imin, Imid, Imax = Isort[:, 0], Isort[:, 1], Isort[:, 2]
    berbeban = Imax > 0.05

    d = pd.DataFrame({"meter_id": df["meter_id"], "date": df["timestamp"].dt.normalize()})
    d["berbeban"] = berbeban.astype(float)
    d["fasa_mati"] = np.where(berbeban, ((Imin < 0.10 * Imax) & (Imid >= 0.10 * Imax)).astype(float), np.nan)
    d["teg_hilang"] = np.where(Vmax > 1.0, (Vmin < 0.5 * Vmax).astype(float), np.nan)
    d["e_calc"] = (V * I).sum(1) * PF * df["dt_h"].to_numpy() / 1000.0     # kWh sekunder
    d["kwh"] = df["kWh"].to_numpy(float)
    d["kw_prim"] = df["kWh"].to_numpy(float) * df["faktormeter"].to_numpy(float) / df["dt_h"].to_numpy()
    d["kva"] = df["daya"].to_numpy(float) / 1000.0

    g = d.groupby(["meter_id", "date"])
    H = pd.DataFrame({
        "n_interval": g.size(),
        "fasa_mati": g["fasa_mati"].mean(),
        "teg_hilang": g["teg_hilang"].mean(),
        "rasio_register": g["kwh"].sum() / g["e_calc"].sum().replace(0, np.nan),
        "utilisasi": g["kw_prim"].mean() / g["kva"].first(),
    }).reset_index()
    # hari tanpa beban / tanpa pembanding: indikator dianggap netral
    H["fasa_mati"] = H["fasa_mati"].fillna(0.0)
    H["teg_hilang"] = H["teg_hilang"].fillna(0.0)
    H["rasio_register"] = H["rasio_register"].clip(0, 5).fillna(1.0)
    H["utilisasi"] = H["utilisasi"].fillna(0.0)
    H["date"] = H["date"].dt.strftime("%Y-%m-%d")
    return H


def main():
    print("=" * 70)
    print("INDIKATOR FISIK PER FASA (HARIAN) + SKOR LSTM AUTOENCODER")
    print("=" * 70)
    df = pd.read_csv(FASA_FILE, low_memory=False, dtype={"meter_id": str, "case_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    diff = (df.sort_values(["meter_id", "timestamp"]).groupby("meter_id")["timestamp"]
              .diff().dt.total_seconds().div(60))
    itv = diff.groupby(df["meter_id"]).agg(lambda s: s[s > 0].mode().iloc[0] if (s > 0).any() else np.nan)
    df["dt_h"] = df["meter_id"].map(itv) / 60.0
    print(f"  Baris per fasa: {len(df):,} | meter: {df['meter_id'].nunique()}")

    H = indikator_harian(df)
    print(f"  Hari-meter dengan indikator: {len(H):,}")

    ti, thr = load_detection()
    ti = ti[["meter_id", "date", "grp", "periode", "case_id", "score"]].rename(columns={"score": "s_ae"})
    F = ti.merge(H, on=["meter_id", "date"], how="left")
    hilang = F["n_interval"].isna().sum()
    if hilang:
        print(f"  PERINGATAN: {hilang} window AE tanpa indikator (diisi netral)")
        F["n_interval"] = F["n_interval"].fillna(0)
        F[["fasa_mati", "teg_hilang"]] = F[["fasa_mati", "teg_hilang"]].fillna(0.0)
        F["rasio_register"] = F["rasio_register"].fillna(1.0)
        F["utilisasi"] = F["utilisasi"].fillna(F["utilisasi"].median())
    F["y"] = (F["periode"] == "pre_p2tl").astype(int)
    F["unit"] = np.where(F["grp"].eq("kontrol"), "K_" + F["meter_id"], F["case_id"] + "_" + F["periode"])
    F = F[["meter_id", "date", "case_id", "grp", "periode", "unit", "y", "s_ae", "n_interval"] + INDIKATOR]
    F.to_csv(OUT, index=False)

    print(f"\n  Window set uji: {len(F):,} | pre={int(F.y.sum())} | post="
          f"{int((F.periode == 'post_p2tl').sum())} | kontrol={int((F.grp == 'kontrol').sum())}")
    print(f"  Unit: kasus pre={F.loc[F.y == 1, 'unit'].nunique()} | meter kontrol="
          f"{F.loc[F.grp == 'kontrol', 'unit'].nunique()}")
    print("  Rata-rata indikator per kelompok:")
    print(F.groupby("periode")[["s_ae"] + INDIKATOR].mean().round(3).to_string())
    print(f"  Disimpan: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
