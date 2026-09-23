"""Ranking operasional: skor model utama pada seluruh meter AMR 2025.

Semua meter master yang memiliki window harian lengkap diskor dengan model
di models/ (tanpa eksklusi; meter yang dikeluarkan dari data latih hanya
DITANDAI, bukan dibuang), lalu diagregasi per meter:
  flag_rate  = persentase hari dengan skor > ambang P95 (keterangan)
  mean_score = skor rata-rata
  rank       = urutan berdasarkan mean_score lalu flag_rate (menurun);
               sejak 9 September 2026 skor rata-rata menjadi kunci urutan
               utama karena ukuran inilah yang dievaluasi pada tingkat
               meter (AUC tingkat pelanggan) dan menemukan lebih banyak
               kasus pada K teratas set uji daripada persentase hari
               tertandai (analisis_v11.py, tabel_topk_peringkat.csv).
Dua penanda dibedakan (sejak 9 September 2026): p2tl_linked = meter yang
tercatat sebagai kasus P2TL pada sumber data (grp p2tl dataset kasus-kontrol,
rekap P2TL x AMR, pelanggan P2TL 2025), dan dikecualikan_latih = meter yang
dikeluarkan dari data latih oleh build_training_population (kasus P2TL DAN
meter kontrol set uji). Sebelumnya p2tl_linked memakai himpunan kedua,
sehingga meter kontrol ikut tertulis "riwayat P2TL".
Kasus P2TL yang tanggal laporannya berada pada periode data (2025/2026)
dilaporkan posisinya sebagai pemeriksaan wajar.
Meter yang register kWh-nya hampir selalu nol (aturan register nol di
build_training_population.zero_kwh_rule, fraksi slot nol > 0,5) TIDAK
diskor: energi nol yang terus-menerus adalah tanda register tidak
mencatat, bukan pola konsumsi. Meter-meter itu disimpan terpisah ke
output/meter_register_nol.csv sebagai daftar pemeriksaan register/pemasangan.

Output: output/population_ranking.csv dan output/population_ranking.pkl
        (dipakai halaman Daftar Prioritas pada dashboard),
        output/meter_register_nol.csv (meter yang dikecualikan aturan register nol).

Jalankan dari code/:  ./venv/bin/python rank_population.py
   --hanya-penanda : hitung ulang ketiga kolom penanda pada output yang sudah
                     ada (tanpa menskor ulang, tanpa TensorFlow).
Prasyarat: pipeline utama sudah dijalankan (models/ terisi) pada grid 30.
Env opsional (uji cepat): POP_MONTHS="lp_202506.csv", RANK_MAX_METERS=50,
                          TA_SMOKE_OUT=<folder output pengganti>.
Memori: seluruh slot bulanan (±21 juta baris) tetap di memori, tetapi
penyusunan kerangka data, normalisasi, window, dan penskoran dilakukan per
kelompok meter (RANK_CHUNK_METERS, bawaan 300), semua langkah itu bekerja
per meter sehingga hasilnya identik, sedangkan puncak pemakaian memori
turun jauh (aman untuk RAM ±12 GB seperti Google Colab).
"""
import gc
import os
import pickle
import sys

import numpy as np
import pandas as pd

from config import (INTERVAL_MINUTES, SEQUENCE_LENGTH, DATA_DIR, OUTPUT_DIR,
                    MODEL_FILE, THRESHOLD_FILE)
assert INTERVAL_MINUTES == 30 and SEQUENCE_LENGTH == 48, \
    "Jalankan `./venv/bin/python set_grid.py 30` dulu."

from preprocessing import scale_per_meter, create_sequences
import build_training_population as btp

OUT_DIR = os.environ.get("TA_SMOKE_OUT", OUTPUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
RANK_SCALER_FILE = os.path.join(OUT_DIR if "TA_SMOKE_OUT" in os.environ else DATA_DIR,
                                "ranking_scalers.pkl")


def detection_scores(mdl, X):
    from model import mask_energy_channels, MASKED_IDX   # TensorFlow hanya dimuat saat menskor
    Xr = mdl.predict(mask_energy_channels(X), batch_size=256, verbose=1)
    return (Xr[:, :, MASKED_IDX] - X[:, :, MASKED_IDX]).mean(axis=(1, 2))


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def baca_master():
    # id_pelanggan dibaca sebagai teks agar cocok dengan IDPEL laporan P2TL (koreksi 9 September 2026)
    master = pd.read_csv(btp.MASTER_FILE, dtype={"id_mtr": str, "id_pelanggan": str}, low_memory=False)
    master["id_mtr"] = master["id_mtr"].astype(str)
    master["id_pelanggan"] = master["id_pelanggan"].astype(str)
    return master


def penanda_p2tl(master):
    """Tiga himpunan meter: dikecualikan dari data latih (kasus P2TL dan meter
    kontrol set uji), beriwayat kasus P2TL pada sumber data, dan kasus 2025/2026."""
    ds = pd.read_csv(btp.RAW_DATA_FILE, usecols=["meter_id", "grp", "tgl_lap_p2tl"],
                     low_memory=False)
    ds["meter_id"] = ds["meter_id"].astype(str)
    rekap = pd.read_csv(btp.KASUS_FILE, usecols=["ID_MTR", "TGLLAPP2TL"])
    idpel = set(pd.read_csv(btp.P2TL_2025_FILE, usecols=["IDPEL"],
                            low_memory=False)["IDPEL"].astype(str))
    kasus_ds = ds[ds["grp"] == "p2tl"]
    riwayat = set(kasus_ds["meter_id"].unique())
    riwayat |= set(rekap["ID_MTR"].astype(str))
    riwayat |= set(master.loc[master["id_pelanggan"].isin(idpel), "id_mtr"])
    excl = set(ds["meter_id"].unique()) | riwayat      # kasus + kontrol + riwayat lain
    lap = pd.to_datetime(kasus_ds["tgl_lap_p2tl"], errors="coerce")
    kasus_periode = set(kasus_ds["meter_id"][lap >= "2025-01-01"].unique())
    lap_r = pd.to_datetime(rekap["TGLLAPP2TL"], errors="coerce")
    kasus_periode |= set(rekap.loc[lap_r >= "2025-01-01", "ID_MTR"].astype(str))
    print(f"Meter master: {len(master):,} | dikecualikan dari data latih: {len(excl):,} "
          f"| beriwayat kasus P2TL: {len(riwayat):,} | kasus 2025/2026: {len(kasus_periode)}")
    return excl, riwayat, kasus_periode


def tandai(df, excl, riwayat, kasus_periode):
    """Isi/perbarui kolom penanda; kolom dikecualikan_latih ditempatkan sesudah kasus_2025_26."""
    df["p2tl_linked"] = df["meter_id"].isin(riwayat)
    df["kasus_2025_26"] = df["meter_id"].isin(kasus_periode)
    df["dikecualikan_latih"] = df["meter_id"].isin(excl)
    kol = [c for c in df.columns if c != "dikecualikan_latih"]
    i = kol.index("kasus_2025_26") + 1
    return df[kol[:i] + ["dikecualikan_latih"] + kol[i:]]


def perbarui_penanda():
    """Hitung ulang kolom penanda pada population_ranking.* dan meter_register_nol.csv."""
    excl, riwayat, kasus_periode = penanda_p2tl(baca_master())
    pkl_out = os.path.join(OUT_DIR, "population_ranking.pkl")
    pk = load_pkl(pkl_out)
    pk["per_meter"] = tandai(pk["per_meter"], excl, riwayat, kasus_periode)
    pk["meter_register_nol"] = tandai(pk["meter_register_nol"], excl, riwayat, kasus_periode)
    pk["per_meter"].to_csv(os.path.join(OUT_DIR, "population_ranking.csv"), index=False)
    pk["meter_register_nol"].to_csv(os.path.join(OUT_DIR, "meter_register_nol.csv"), index=False)
    with open(pkl_out, "wb") as f:
        pickle.dump(pk, f, protocol=4)
    pm = pk["per_meter"]
    print(f"Penanda diperbarui: {int(pm['dikecualikan_latih'].sum())} meter dikecualikan dari data latih, "
          f"{int(pm['p2tl_linked'].sum())} di antaranya beriwayat kasus P2TL, "
          f"{int(pm['kasus_2025_26'].sum())} kasus 2025/2026 -> {OUT_DIR}")


def main():
    from model import load_model
    model = load_model(MODEL_FILE)
    thr = float(load_pkl(THRESHOLD_FILE))
    print(f"Ambang P95 model utama: {thr:.6f}")

    # ---------- meter master + penanda keterkaitan P2TL ----------
    master = baca_master()
    intervals = master.set_index("id_mtr")["lp2025_interval_min"]
    max_m = os.environ.get("RANK_MAX_METERS")
    if max_m:
        intervals = intervals.iloc[:int(max_m)]
    excl, riwayat, kasus_periode = penanda_p2tl(master)

    # ---------- window harian seluruh meter ----------
    parts = []
    for f in btp.list_month_files():
        part = btp.reduce_month(os.path.join(btp.EXPORT_DIR, f), intervals)
        if part is not None:
            # kolom hitungan (…_n, …_nz) cukup int32: hemat memori, nilai tak berubah
            for c in part.columns:
                if c.endswith(("_n", "_nz")):
                    part[c] = part[c].astype(np.int32)
            parts.append(part)
            print(f"  {f}: {len(part):,} slot")
    meters = sorted(set().union(*[set(p["meter_id"].unique()) for p in parts]))
    chunk_n = int(os.environ.get("RANK_CHUNK_METERS", 300))
    n_chunk = -(-len(meters) // chunk_n)
    print(f"Meter dengan slot: {len(meters):,} -> diproses per {chunk_n} meter ({n_chunk} kelompok)")

    # Per kelompok meter: pilih register -> kerangka data -> scaling -> window -> skor
    # (semua langkah bekerja per meter, jadi hasil identik dengan pemrosesan sekaligus)
    reg_l, info_l, score_l, scalers = [], [], [], {}
    for k in range(n_chunk):
        sel = set(meters[k * chunk_n:(k + 1) * chunk_n])
        print(f"\n--- Kelompok {k + 1}/{n_chunk} ({len(sel)} meter) ---")
        slots = pd.concat([p[p["meter_id"].isin(sel)] for p in parts], ignore_index=True)
        df, reg = btp.slots_to_frame(slots)
        del slots
        df, sc = scale_per_meter(df, scaler_file=RANK_SCALER_FILE)
        try:
            X, _, info = create_sequences(df)
        except ValueError as e:          # kelompok tanpa satu pun hari lengkap
            print(f"  dilewati: {e}")
            del df
            continue
        del df
        score_l.append(detection_scores(model, X))
        info_l.append(info)
        reg_l.append(reg)
        scalers.update(sc)
        del X
        gc.collect()
    del parts
    gc.collect()

    reg = pd.concat(reg_l, ignore_index=True)
    info = pd.concat(info_l, ignore_index=True)
    scores = np.concatenate(score_l)
    with open(RANK_SCALER_FILE, "wb") as f:
        pickle.dump(scalers, f)
    print(f"\nScaler per meter (gabungan) disimpan: {RANK_SCALER_FILE} ({len(scalers)} meter)")
    btp.print_register_summary(reg)
    info["meter_id"] = info["meter_id"].astype(str)
    print(f"Window ranking: {len(info):,} dari {info['meter_id'].nunique():,} meter")

    # ---------- meter yang dikecualikan aturan register nol ----------
    reg["meter_id"] = reg["meter_id"].astype(str)
    nol = tandai(reg[reg["status_populasi"] == btp.STATUS_REGISTER_NOL].copy(), excl, riwayat, kasus_periode)
    nol = nol.merge(master[["id_mtr", "golongan_tarif", "daya"]].drop_duplicates("id_mtr"),
                    left_on="meter_id", right_on="id_mtr", how="left").drop(columns=["id_mtr"])
    nol = nol.sort_values("fraksi_slot_kwh_nol", ascending=False).reset_index(drop=True)
    nol_out = os.path.join(OUT_DIR, "meter_register_nol.csv")
    nol.to_csv(nol_out, index=False)
    print(f"\nMeter dikecualikan aturan register nol (fraksi slot kWh nol > "
          f"{btp.ZERO_KWH_MAX_FRACTION}): {len(nol):,} meter, "
          f"{int(nol['dikecualikan_latih'].sum())} di antaranya dikecualikan dari data latih "
          f"({int(nol['p2tl_linked'].sum())} beriwayat kasus P2TL), "
          f"{int(nol['kasus_2025_26'].sum())} kasus 2025/2026 -> {nol_out}")

    # ---------- agregasi per meter ----------
    flags = (scores > thr).astype(int)
    agg = pd.DataFrame({"meter_id": info["meter_id"], "score": scores, "flag": flags})
    per_meter = agg.groupby("meter_id").agg(
        n_windows=("score", "size"),
        mean_score=("score", "mean"),
        flag_rate=("flag", "mean"),
    ).reset_index()
    order = per_meter.sort_values(["mean_score", "flag_rate"], ascending=False).reset_index()
    rk = pd.Series(np.arange(1, len(order) + 1), index=order["index"])
    per_meter["rank"] = per_meter.index.map(rk)
    n_m = len(per_meter)
    per_meter["pctl"] = 100 * (1 - (per_meter["rank"] - 1) / n_m)
    per_meter = tandai(per_meter, excl, riwayat, kasus_periode)
    per_meter = per_meter.merge(
        master[["id_mtr", "golongan_tarif", "daya"]].drop_duplicates("id_mtr"),
        left_on="meter_id", right_on="id_mtr", how="left").drop(columns=["id_mtr"])
    per_meter = per_meter.merge(reg[["meter_id", "register_energi", "fraksi_slot_kwh_nol"]],
                                on="meter_id", how="left")
    per_meter = per_meter.sort_values("rank").reset_index(drop=True)

    print(f"\nMeter ter-ranking: {n_m:,} | meter dengan >=1 hari tertandai: "
          f"{int((per_meter['flag_rate'] > 0).sum()):,} | dikecualikan dari data latih: "
          f"{int(per_meter['dikecualikan_latih'].sum())} ({int(per_meter['p2tl_linked'].sum())} beriwayat kasus P2TL)")
    print("\n=== POSISI KASUS P2TL 2025/2026 DALAM PERINGKAT ===")
    sub = per_meter[per_meter["kasus_2025_26"]]
    for _, r in sub.iterrows():
        print(f"  {r['meter_id']:>18} | n={int(r['n_windows']):4d} | rank {int(r['rank']):4d}/{n_m} "
              f"(pctl {r['pctl']:5.1f}, flag {r['flag_rate']:.2f}, skor {r['mean_score']:.3f})")
    missing = sorted(kasus_periode - set(per_meter["meter_id"]))
    if missing:
        nol_ids = set(nol["meter_id"])
        print("  Dikecualikan aturan register nol:", [m for m in missing if m in nol_ids])
        print("  Tanpa window lengkap (tidak ter-ranking):", [m for m in missing if m not in nol_ids])

    print("\n=== 15 TERATAS ===")
    for _, r in per_meter.head(15).iterrows():
        tanda = "P2TL" if r["p2tl_linked"] else ("uji" if r["dikecualikan_latih"] else "-")
        print(f"  #{int(r['rank']):3d} {r['meter_id']:>18} {str(r['golongan_tarif']):>5} "
              f"daya={r['daya']} n={int(r['n_windows']):4d} flag={r['flag_rate']:.2f} "
              f"mean={r['mean_score']:.3f} reg={r['register_energi']} "
              f"nol={r['fraksi_slot_kwh_nol']:.2f} [{tanda}]")
    top20 = per_meter.head(20)
    print(f"\n20 teratas: register 1.0.2.29 = {int((top20['register_energi'] == '1.0.2.29').sum())} meter; "
          f"fraksi slot kWh nol median {top20['fraksi_slot_kwh_nol'].median():.3f}, "
          f"maksimum {top20['fraksi_slot_kwh_nol'].max():.3f}")

    csv_out = os.path.join(OUT_DIR, "population_ranking.csv")
    per_meter.to_csv(csv_out, index=False)
    with open(os.path.join(OUT_DIR, "population_ranking.pkl"), "wb") as f:
        pickle.dump({"per_meter": per_meter, "info_windows": info,
                     "scores": scores, "threshold": thr, "register_map": reg,
                     "meter_register_nol": nol}, f, protocol=4)
    print(f"\nDisimpan: {csv_out}")
    print(f"Disimpan: {os.path.join(OUT_DIR, 'population_ranking.pkl')}")
    print("SELESAI.")


if __name__ == "__main__":
    if "--hanya-penanda" in sys.argv[1:]:
        perbarui_penanda()
    else:
        main()
