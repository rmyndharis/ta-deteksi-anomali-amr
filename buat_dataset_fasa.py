"""
Pembentukan Dataset Kasus-Kontrol PER FASA
==========================================
Sumber: data/AP2T/lp_anomali.csv dan lp_kontrol.csv (hasil tarikan load
profile AMR per meter-window, kolom per fasa dipertahankan), serta
dataset_lp_p2tl.csv untuk daftar kasus, tanggal laporan P2TL, tarif dan daya.

Perbedaan dengan dataset_lp_p2tl.csv: tegangan dan arus TIDAK dirata-rata
tiga fasa, melainkan disimpan per fasa (L1, L2, L3). Kolom lain disamakan:
kasus yang dipakai, aturan periode pre/post, label, dan dedup (meter_id, timestamp).

Output: data/AP2T/dataset_lp_p2tl_fasa.csv

Jalankan dari folder code/:  ./venv/bin/python buat_dataset_fasa.py

Penulis: Yudhi Armyndharis (220401010272)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
AP2T = os.path.join(os.path.dirname(BASE), "data", "AP2T")
OUT = os.path.join(AP2T, "dataset_lp_p2tl_fasa.csv")

NUM = ["voltage_l1", "voltage_l2", "voltage_l3", "current_l1", "current_l2",
       "current_l3", "kwh_export_total", "kwh_import_total",
       "kvarh_export_total", "pfaverage"]


def baca_lp(nama):
    df = pd.read_csv(os.path.join(AP2T, nama), low_memory=False,
                     dtype={"read_date": str, "meter_id": str, "case_id": str})
    df["timestamp"] = pd.to_datetime(df["read_date"].str[:12],
                                     format="%Y%m%d%H%M", errors="coerce")
    df["meter_id"] = df["meter_id"].str.strip()
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["timestamp"])


def baca_master():
    """faktormeter dan daya kontrak per meter dari tiga berkas master."""
    frames = []
    for fn in ["master_meter.csv", "master_meter_lp2025.csv",
               "master_meter_riwayat_pasang.csv"]:
        p = os.path.join(AP2T, fn)
        if not os.path.exists(p):
            continue
        m = pd.read_csv(p, low_memory=False, dtype=str)
        frames.append(pd.DataFrame({
            "meter_id": m["id_mtr"].astype(str).str.strip(),
            "faktormeter": pd.to_numeric(m["faktormeter"], errors="coerce"),
            "daya_master": pd.to_numeric(m["daya"], errors="coerce"),
            "tarif_master": m.get("golongan_tarif"),
        }))
    m = pd.concat(frames, ignore_index=True)
    m = m.dropna(subset=["faktormeter"]).drop_duplicates("meter_id")
    return m


def main():
    print("=" * 70)
    print("DATASET KASUS-KONTROL PER FASA")
    print("=" * 70)
    acuan = pd.read_csv(os.path.join(AP2T, "dataset_lp_p2tl.csv"), dtype=str,
                        usecols=["case_id", "grp", "tgl_lap_p2tl", "gol_p2tl",
                                 "tarip", "daya", "idpel"]).drop_duplicates("case_id")
    kasus = acuan[acuan["grp"] == "p2tl"].set_index("case_id")
    print(f"  Kasus acuan (dataset_lp_p2tl.csv): {len(kasus)}")

    A = baca_lp("lp_anomali.csv")
    K = baca_lp("lp_kontrol.csv")
    n_case_lp = A["case_id"].nunique()
    A = A[A["case_id"].isin(kasus.index)].copy()
    print(f"  lp_anomali: {n_case_lp} case_id, dipakai {A['case_id'].nunique()} "
          f"(sisanya duplikat kasus pada meter/tanggal yang sama)")
    print(f"  lp_kontrol: {K['meter_id'].nunique()} meter")

    df = pd.concat([A, K], ignore_index=True)
    df["tgl_lap_p2tl"] = pd.to_datetime(df["case_id"].map(kasus["tgl_lap_p2tl"]))
    df["periode"] = np.where(
        df["grp"].eq("kontrol"), "normal",
        np.where(df["timestamp"].dt.normalize() <= df["tgl_lap_p2tl"],
                 "pre_p2tl", "post_p2tl"))
    df["anomaly_label"] = (df["periode"] == "pre_p2tl").astype(int)

    m = baca_master()
    df = df.merge(m, on="meter_id", how="left")
    # daya: dari catatan P2TL untuk kasus (daya saat pelanggaran), master untuk kontrol
    daya_p2tl = pd.to_numeric(df["case_id"].map(kasus["daya"]), errors="coerce")
    df["daya"] = daya_p2tl.fillna(df["daya_master"])
    df["tarip"] = df["case_id"].map(kasus["tarip"]).fillna(df["tarif_master"])

    ds = pd.DataFrame({
        "timestamp": df["timestamp"],
        "meter_id": df["meter_id"],
        "case_id": df["case_id"],
        "grp": df["grp"],
        "periode": df["periode"],
        "anomaly_label": df["anomaly_label"],
        "voltage_l1": df["voltage_l1"] * 1000.0,   # kV -> Volt (sisi sekunder)
        "voltage_l2": df["voltage_l2"] * 1000.0,
        "voltage_l3": df["voltage_l3"] * 1000.0,
        "current_l1": df["current_l1"],            # Ampere (sisi sekunder)
        "current_l2": df["current_l2"],
        "current_l3": df["current_l3"],
        "kWh": df["kwh_export_total"],             # OBIS 1.0.1.29, kWh/interval sekunder
        "kwh_import": df["kwh_import_total"],
        "kvarh": df["kvarh_export_total"],
        "power_factor": df["pfaverage"].abs().clip(0, 1),   # tanda = arah daya reaktif; dipakai nilai mutlak
        "faktormeter": df["faktormeter"],
        "daya": df["daya"],
        "tarip": df["tarip"],
        "tgl_lap_p2tl": df["tgl_lap_p2tl"].dt.date,
    }).sort_values(["case_id", "meter_id", "timestamp"]).reset_index(drop=True)

    n0 = len(ds)
    ds = ds.drop_duplicates(subset=["meter_id", "timestamp"], keep="first")
    ds = ds.reset_index(drop=True)
    ds.to_csv(OUT, index=False)

    print(f"\n  Baris: {len(ds):,} (dibuang {n0 - len(ds):,} ganda) | meter: "
          f"{ds['meter_id'].nunique()} | kasus: {ds.loc[ds.grp == 'p2tl', 'case_id'].nunique()}")
    print(ds.groupby(["grp", "periode", "anomaly_label"])
            .agg(baris=("meter_id", "size"), meter=("meter_id", "nunique")).to_string())
    print(f"  Meter tanpa faktormeter: {ds.loc[ds.faktormeter.isna(), 'meter_id'].nunique()}"
          f" | tanpa daya: {ds.loc[ds.daya.isna(), 'meter_id'].nunique()}")
    print(f"  Disimpan: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
