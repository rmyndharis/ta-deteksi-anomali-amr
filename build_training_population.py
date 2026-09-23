"""
Pembangunan Data Latih Populasi dari Ekspor AMR 2025
=====================================================
Membangun window harian NORMAL dari load profile AMR 2025 (2.781 meter)
untuk memperbesar data latih model, mengikuti aturan kurasi yang sama
dengan dataset kasus-kontrol:

- Register energi dan keluarga kolom konteks dipetakan OTOMATIS per meter.
  Kandidat energi: 1.0.1.29 (kwh_export_total, kWh per interval, +A),
  1.0.1.25 (active_power_import_25, daya rata-rata per interval, kW),
  serta register 1.0.2.29 dan 1.0.2.25. Pada seri meter tertentu
  (EM1I, H310, HXF3) register 1.0.1.x selalu nol sedangkan energi
  tercatat pada register 1.0.2.x yang berkorelasi kuat dengan arus
  terukur (diagnostik Juni 2025: r = 0,83 untuk 1.0.2.29 pada seri
  EM1I, dibanding r = 0,99 untuk 1.0.1.29 pada seri lain). Register
  1.0.1.x diprioritaskan; register 1.0.2.x hanya dipakai bila register
  1.0.1.x hampir selalu nol. Keluarga konteks (tegangan, arus, faktor
  daya) dipilih dari keluarga kolom yang terisi. PENTING: energi hanya
  diambil dari register kWh (x.29). Register daya (x.25, kW) TIDAK
  dipakai sebagai pengganti bila register kWh kosong, karena besaran
  yang diukur berbeda; meter yang register kWh-nya tidak pernah lengkap
  satu hari gugur pada aturan kelengkapan hari (lihat
  ENERGY_REGISTER_KWH_ONLY). Peta register per meter disimpan
  ke data/population_register_map.csv.
- Agregasi energi ke slot grid memakai min_count=1 sehingga slot tanpa
  pembacaan energi valid menjadi NaN (bukan nol), dan k_n adalah jumlah
  pembacaan energi VALID pada register terpilih (bukan jumlah timestamp).
- Meter yang register kWh-nya HAMPIR SELALU NOL dikeluarkan (aturan
  register nol, ZERO_KWH_MAX_FRACTION): bila lebih dari separuh slot kWh
  valid sebuah meter bernilai nol, meter itu tidak dipakai sebagai contoh
  normal dan tidak diskor pada peringkat populasi, melainkan dilaporkan
  terpisah untuk pemeriksaan register/pemasangan (temuan jalan resmi
  September 2026: meter seperti ini mendominasi puncak daftar prioritas
  karena "energi nol padahal arus mengalir" bukan pola konsumsi, melainkan
  register yang tidak mencatat). Status tiap meter dicatat pada kolom
  status_populasi dan fraksi_slot_kwh_nol di peta register.
- Tegangan = rata-rata fasa terisi; arus = rata-rata fasa terisi
  (selaras kurasi dataset kasus-kontrol).
- Nilai dibiarkan pada skala aslinya (sekunder/primer campuran);
  normalisasi per meter menyerap perbedaan skala.
- Meter yang PERNAH terkait kasus P2TL (dataset kasus-kontrol, rekap
  115 kasus, semua tahun) DIKECUALIKAN, sehingga data latih berisi
  meter yang diasumsikan normal (asumsi mayoritas-normal, dicatat
  sebagai keterbatasan).
- Grid 60/30 menit (diatur set_grid.py; kadensi per meter berubah
  antar era), window harian penuh dengan syarat kadensi seragam per
  hari untuk energi, interpolasi terbatas hanya untuk fitur konteks,
  maksimum WINDOWS_PER_METER hari per meter (sampel acak ber-seed)
  agar meter berdata-lengkap tidak mendominasi.

Output: data/population_windows.pkl {"X": float32, "info": DataFrame}
        data/population_register_map.csv (register energi per meter)

Jalankan: python build_training_population.py
Variabel lingkungan opsional (untuk uji cepat): POP_MONTHS="lp_202506.csv"

Penulis: Yudhi Armyndharis (220401010272)
"""

import os
import pickle
import numpy as np
import pandas as pd

from config import (
    RAW_DATA_FILE, DATA_DIR, MODEL_FEATURES, ELECTRICAL_FEATURES,
    RANDOM_SEED,
)
from preprocessing import scale_per_meter, create_sequences

# Grid data populasi; ubah BERSAMA config.INTERVAL_MINUTES lewat
# `python set_grid.py 60|30`.
POPULATION_INTERVAL_MINUTES = 30
POPULATION_SEQ_LEN = (24 * 60) // POPULATION_INTERVAL_MINUTES

AP2T_DIR = os.path.dirname(RAW_DATA_FILE)
EXPORT_DIR = os.path.join(AP2T_DIR, "export_amr")
MASTER_FILE = os.path.join(AP2T_DIR, "master_meter_lp2025.csv")
KASUS_FILE = os.path.join(AP2T_DIR, "p2tl_x_amr_meter_115kasus.csv")
P2TL_2025_FILE = os.path.join(AP2T_DIR, "p2tl_2025_alll.csv")

POPULATION_FILE = os.path.join(DATA_DIR, "population_windows.pkl")
POPULATION_SCALER_FILE = os.path.join(os.path.dirname(POPULATION_FILE),
                                      "population_scalers.pkl")
REGISTER_MAP_FILE = os.path.join(DATA_DIR, "population_register_map.csv")
WINDOWS_PER_METER = 60

# Register 1.0.2.x hanya dipakai bila jumlah nilai non-nol register
# 1.0.1.x kurang dari MIN_RATIO_PLUS_A kali jumlah non-nol register 1.0.2.x
MIN_RATIO_PLUS_A = 0.05
# --- Seleksi register energi (CRISP-DM Data Preparation / Select Data) ---
# Energi per interval HANYA diambil dari register energi kWh (x.29).
# Register daya (x.25, kW) TIDAK dipakai sebagai pengganti bila register
# kWh kosong: besaran yang diukur berbeda (daya rata-rata, bukan energi)
# dan kesetaraannya tidak dapat diverifikasi dari isi ekspor. Meter yang
# register kWh-nya tidak pernah lengkap 48 pembacaan dalam satu hari
# gugur sendiri pada aturan kelengkapan hari, bukan ditambal.
ENERGY_REGISTER_KWH_ONLY = True
# Register kWh (x.29) dipilih selama jumlah non-nolnya >= PREFER_KWH_RATIO
# kali jumlah non-nol register daya (x.25) pada pasangan yang sama.
PREFER_KWH_RATIO = 0.8
# --- Aturan register nol (CRISP-DM Data Preparation / Clean Data) ---
# Meter yang fraksi slot kWh valid bernilai nol-nya LEBIH BESAR dari
# ZERO_KWH_MAX_FRACTION dikeluarkan dari populasi latih dan dari peringkat
# populasi. Energi nol yang hampir terus-menerus (sementara arus terukur)
# adalah tanda register tidak mencatat, bukan pola konsumsi; meter seperti
# ini dilaporkan terpisah (output/meter_register_nol.csv) untuk pemeriksaan
# register atau pemasangan. Nilai 0,5 = "lebih dari separuh pembacaan
# energinya nol". Set None untuk menonaktifkan aturan.
ZERO_KWH_MAX_FRACTION = 0.5
STATUS_DIPAKAI = "dipakai"
STATUS_REGISTER_NOL = "dikecualikan_register_nol"

# Kandidat register energi (kueri OBIS): kolom "kwh_export_total" =
# 1.0.1.29 = +A per interval; "kwh_import_total" = 1.0.2.29; keluarga
# _25 adalah daya rata-rata per interval (1.0.1.25 / 1.0.2.25).
# Penamaan CSV berasal dari sisi sistem - jangan tertukar.
ENERGY_CANDIDATES = {
    "ea": ("kwh_export_total", "1.0.1.29", "A"),
    "eb": ("active_power_import_25", "1.0.1.25", "B"),
    "ea2": ("kwh_import_total", "1.0.2.29", "A"),
    "eb2": ("active_power_export_25", "1.0.2.25", "B"),
}
FAM = {
    "A": {"voltage": ["voltage_l1", "voltage_l2", "voltage_l3"],
          "current": ["current_l1", "current_l2", "current_l3"],
          "pf": "pfaverage"},
    "B": {"voltage": ["voltage_l1_25", "voltage_l2_25", "voltage_l3_25"],
          "current": ["current_l1_25", "current_l2_25", "current_l3_25"],
          "pf": "pfaverage_25"},
}
USECOLS = (["meter_id", "read_date"]
           + [c[0] for c in ENERGY_CANDIDATES.values()]
           + FAM["A"]["voltage"] + FAM["A"]["current"] + [FAM["A"]["pf"]]
           + FAM["B"]["voltage"] + FAM["B"]["current"] + [FAM["B"]["pf"]])


def select_meters():
    """Meter populasi: interval <= grid, tanpa keterkaitan P2TL apa pun."""
    # id_pelanggan dibaca sebagai teks agar cocok dengan IDPEL laporan P2TL (koreksi 9 September 2026)
    m = pd.read_csv(MASTER_FILE, dtype={"id_mtr": str, "id_pelanggan": str}, low_memory=False)
    m["id_mtr"] = m["id_mtr"].astype(str)
    m["id_pelanggan"] = m["id_pelanggan"].astype(str)

    excl = set()
    ds = pd.read_csv(RAW_DATA_FILE, usecols=["meter_id"], low_memory=False)
    excl |= set(ds["meter_id"].astype(str).unique())
    kasus = pd.read_csv(KASUS_FILE)
    excl |= set(kasus["ID_MTR"].astype(str))
    p2tl = pd.read_csv(P2TL_2025_FILE, usecols=["IDPEL"], low_memory=False)
    idpel_p2tl = set(p2tl["IDPEL"].astype(str))
    excl |= set(m.loc[m["id_pelanggan"].isin(idpel_p2tl), "id_mtr"])

    ok = m[(m["lp2025_interval_min"] <= POPULATION_INTERVAL_MINUTES)
           & ~m["id_mtr"].isin(excl)]
    intervals = ok.set_index("id_mtr")["lp2025_interval_min"].astype(int)

    print(f"Master           : {len(m):,} meter")
    print(f"Dikecualikan P2TL: {len(excl):,} meter")
    print(f"Meter populasi   : {len(intervals):,} "
          f"(interval <= {POPULATION_INTERVAL_MINUTES} menit)")
    return intervals


def list_month_files():
    """Berkas ekspor bulanan; POP_MONTHS membatasi daftar (uji cepat)."""
    months = sorted(f for f in os.listdir(EXPORT_DIR)
                    if f.startswith("lp_") and f.endswith(".csv"))
    sel = os.environ.get("POP_MONTHS")
    if sel:
        months = [f for f in months if f in sel.split(",")]
    return months


def reduce_month(filepath, intervals):
    """
    Baca satu file bulanan, reduksi ke slot grid populasi per meter dengan
    nilai keempat kandidat register energi dan kedua keluarga konteks
    (keputusan register/keluarga diambil belakangan per meter).
    Energi per slot = sum dengan min_count=1 (slot tanpa nilai valid = NaN);
    *_n = jumlah pembacaan valid; *_nz = jumlah pembacaan non-nol.
    """
    frames = []
    for chunk in pd.read_csv(filepath, usecols=USECOLS, chunksize=1_000_000,
                             low_memory=False):
        chunk["meter_id"] = chunk["meter_id"].astype(str)
        chunk = chunk[chunk["meter_id"].isin(intervals.index)]
        if not len(chunk):
            continue
        # read_date: string 12 digit YYYYMMDDHHMM; sebagian meter menulis
        # 14 digit (dengan detik) -> dipotong ke 12 digit dulu.
        ts = pd.to_datetime(chunk["read_date"].astype(str).str.slice(0, 12),
                            format="%Y%m%d%H%M", errors="coerce")
        chunk = chunk.assign(ts=ts).dropna(subset=["ts"])

        out = pd.DataFrame({"meter_id": chunk["meter_id"], "ts": chunk["ts"]})
        for key, (col, _, _) in ENERGY_CANDIDATES.items():
            out[key] = pd.to_numeric(chunk[col], errors="coerce")
        for fam, cols in FAM.items():
            out[f"v{fam}"] = chunk[cols["voltage"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            out[f"i{fam}"] = chunk[cols["current"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            out[f"p{fam}"] = pd.to_numeric(chunk[cols["pf"]], errors="coerce")
        frames.append(out)

    if not frames:
        return None
    month = pd.concat(frames, ignore_index=True)

    # Selaraskan ke grid 15 menit (grid terhalus yang ada) untuk
    # menyatukan jitter waktu baca, lalu dedup pembacaan ganda.
    # Kadensi aktual per meter BERUBAH antar era sehingga interval di
    # master (agregat setahun) tidak dipakai sebagai acuan per baris.
    month["ts"] = month["ts"].dt.floor("15min")
    month = month.drop_duplicates(subset=["meter_id", "ts"], keep="last")

    # Agregasi ke slot grid populasi
    month["slot"] = month["ts"].dt.floor(f"{POPULATION_INTERVAL_MINUTES}min")
    # Semua agregasi memakai fungsi bawaan (sum/count/mean) agar cepat;
    # sum dengan min_count=1 ditiru lewat count (count 0 -> NaN).
    agg = {}
    for key in ENERGY_CANDIDATES:
        month[f"{key}_nz"] = (month[key].fillna(0) != 0).astype(np.int32)
        agg[key] = (key, "sum")
        agg[f"{key}_n"] = (key, "count")
        agg[f"{key}_nz"] = (f"{key}_nz", "sum")
    for fam in FAM:
        agg[f"v{fam}"] = (f"v{fam}", "mean")
        agg[f"i{fam}"] = (f"i{fam}", "mean")
        agg[f"p{fam}"] = (f"p{fam}", "mean")
        agg[f"i{fam}_n"] = (f"i{fam}", "count")
    g = month.groupby(["meter_id", "slot"], sort=False).agg(**agg).reset_index()
    for key in ENERGY_CANDIDATES:
        g.loc[g[f"{key}_n"] == 0, key] = np.nan      # min_count=1
    return g


def choose_registers(slots):
    """
    Keputusan per meter: register energi dan keluarga konteks.
    Bila ENERGY_REGISTER_KWH_ONLY, kandidat dibatasi pada register kWh
    (1.0.1.29 dan 1.0.2.29); 1.0.1.29 diprioritaskan, 1.0.2.29 dipakai
    hanya bila 1.0.1.29 hampir selalu nol (< MIN_RATIO_PLUS_A).
    Register daya (x.25) tidak pernah menjadi pengganti register kWh.
    Keluarga konteks = keluarga dengan pembacaan arus valid terbanyak.
    """
    nz = slots.groupby("meter_id")[[f"{k}_nz" for k in ENERGY_CANDIDATES]].sum()
    cn = slots.groupby("meter_id")[["iA_n", "iB_n"]].sum()
    # Ringkasan keterisian per keluarga register (kolom informasi peta register)
    plus_a = nz[["ea_nz", "eb_nz"]].max(axis=1)
    minus_a = nz[["ea2_nz", "eb2_nz"]].max(axis=1)

    if ENERGY_REGISTER_KWH_ONLY:
        # Kandidat dibatasi pada register kWh saja: 1.0.1.29 dan 1.0.2.29.
        use_2x = (nz["ea_nz"] < MIN_RATIO_PLUS_A * nz["ea2_nz"]) & (nz["ea2_nz"] > 0)
        energy = pd.Series(np.where(use_2x, "ea2", "ea"), index=nz.index)
        kosong = (nz["ea_nz"] + nz["ea2_nz"]) == 0
        if int(kosong.sum()):
            print(f"Meter tanpa register kWh terisi sama sekali: "
                  f"{int(kosong.sum()):,} (gugur pada aturan kelengkapan hari)")
    else:
        use_2x = (plus_a < MIN_RATIO_PLUS_A * minus_a) & (minus_a > 0)

        # Dalam satu pasangan register, register energi (x.29, kWh) lebih
        # disukai daripada register daya (x.25) kecuali jauh lebih jarang terisi.
        energy = pd.Series(np.where(
            use_2x,
            np.where(nz["ea2_nz"] >= PREFER_KWH_RATIO * nz["eb2_nz"], "ea2", "eb2"),
            np.where(nz["ea_nz"] >= PREFER_KWH_RATIO * nz["eb_nz"], "ea", "eb")), index=nz.index)
    context = pd.Series(np.where(cn["iA_n"] >= cn["iB_n"], "A", "B"), index=nz.index)

    reg = pd.DataFrame({
        "meter_id": nz.index,
        "register_energi": energy.map({k: v[1] for k, v in ENERGY_CANDIDATES.items()}).to_numpy(),
        "kolom_energi": energy.map({k: v[0] for k, v in ENERGY_CANDIDATES.items()}).to_numpy(),
        "keluarga_konteks": context.to_numpy(),
        "n_nonnol_1.0.1.x": plus_a.to_numpy(),
        "n_nonnol_1.0.2.x": minus_a.to_numpy(),
    })
    return energy, context, reg


def zero_kwh_rule(df, reg):
    """
    Aturan register nol: hitung fraksi slot kWh valid yang bernilai nol
    per meter, catat pada peta register (kolom n_slot_kwh_valid,
    fraksi_slot_kwh_nol, status_populasi), lalu buang baris meter yang
    fraksinya melebihi ZERO_KWH_MAX_FRACTION. Mengembalikan (df, reg).
    """
    valid = df["kWh"].notna()
    n_valid = valid.groupby(df["meter_id"]).sum()
    n_zero = (valid & (df["kWh"] == 0)).groupby(df["meter_id"]).sum()
    frac = n_zero / n_valid.replace(0, np.nan)
    reg = reg.copy()
    reg["n_slot_kwh_valid"] = reg["meter_id"].map(n_valid).fillna(0).astype(int)
    reg["fraksi_slot_kwh_nol"] = reg["meter_id"].map(frac).round(4)
    if ZERO_KWH_MAX_FRACTION is None:
        reg["status_populasi"] = STATUS_DIPAKAI
        return df, reg
    buang = reg["fraksi_slot_kwh_nol"] > ZERO_KWH_MAX_FRACTION
    reg["status_populasi"] = np.where(buang, STATUS_REGISTER_NOL, STATUS_DIPAKAI)
    keluar = set(reg.loc[buang, "meter_id"])
    n_before = len(df)
    df = df[~df["meter_id"].isin(keluar)]
    print(f"Aturan register nol (fraksi slot kWh nol > {ZERO_KWH_MAX_FRACTION:g}): "
          f"{len(keluar):,} meter dikeluarkan, {n_before - len(df):,} slot dibuang; "
          f"per register: "
          f"{reg.loc[buang, 'register_energi'].value_counts().to_dict()}")
    return df, reg


def slots_to_frame(slots):
    """
    Slot grid (semua kandidat) -> DataFrame fitur per meter dan slot:
    kWh (register terpilih), voltage, current, power_factor, plus
    penjagaan kadensi seragam per hari untuk energi dan aturan register
    nol (meter yang kWh-nya hampir selalu nol dikeluarkan, lihat
    zero_kwh_rule). Mengembalikan (df, register_map).
    """
    energy, context, reg = choose_registers(slots)
    e = slots["meter_id"].map(energy)
    c = slots["meter_id"].map(context)

    kwh = pd.Series(np.nan, index=slots.index)
    k_n = pd.Series(0, index=slots.index)
    for key in ENERGY_CANDIDATES:
        sel = (e == key).to_numpy()
        kwh[sel] = slots.loc[sel, key]
        k_n[sel] = slots.loc[sel, f"{key}_n"]
    isB = (c == "B").to_numpy()
    df = pd.DataFrame({
        "meter_id": slots["meter_id"],
        "timestamp": slots["slot"],
        "kWh": kwh,
        "voltage": np.where(isB, slots["vB"], slots["vA"]),
        "current": np.where(isB, slots["iB"], slots["iA"]),
        "power_factor": np.where(isB, slots["pB"], slots["pA"]),
        "k_n": k_n,
    })

    # Guard keseragaman kadensi PER HARI: energi antar slot dalam satu
    # hari hanya sebanding bila jumlah pembacaan energi valid per slot
    # seragam hari itu (kadensi berubah antar era; hari campuran dan
    # slot tanpa pembacaan valid dibuang lewat NaN).
    day = df["timestamp"].dt.normalize()
    grpday = df.groupby(["meter_id", day])["k_n"]
    mixed = (grpday.transform("min") != grpday.transform("max")) | (df["k_n"] == 0)
    df.loc[mixed, "kWh"] = np.nan
    print(f"Slot pada hari berkadensi campuran / tanpa energi valid: "
          f"{int(mixed.sum()):,} -> NaN")
    df = df.drop(columns=["k_n"])

    df.loc[df["kWh"] < 0, "kWh"] = np.nan
    df.loc[df["current"] < 0, "current"] = np.nan
    # Faktor daya bertanda negatif (konvensi arah daya reaktif, hampir
    # seluruhnya seri EMK1) dipakai nilai mutlaknya, sama seperti
    # preprocessing.load_dataset (perubahan 9 September 2026).
    df["power_factor"] = df["power_factor"].abs().clip(0, 1)

    # Aturan register nol: meter yang kWh validnya hampir selalu nol
    # bukan contoh normal dan tidak diskor (dilaporkan terpisah).
    df, reg = zero_kwh_rule(df, reg)

    # Fitur konteks (V/I/PF) hanya terekam di sebagian pembacaan;
    # interpolasi terbatas (maks 3 slot) per meter. Energi TIDAK
    # diinterpolasi: kWh adalah target deteksi, harus data nyata.
    df = df.sort_values(["meter_id", "timestamp"])
    for col in ["voltage", "current", "power_factor"]:
        df[col] = df.groupby("meter_id")[col].transform(
            lambda s: s.interpolate(limit=3, limit_area="inside"))

    df["anomaly_label"] = 0
    df["grp"] = "populasi"
    df["periode"] = "normal"
    df["case_id"] = ""
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df, reg


def print_register_summary(reg):
    seri = reg["meter_id"].str[:4]
    print("\nPeta register energi per meter:")
    print(reg["register_energi"].value_counts().to_string())
    print("Keluarga konteks:", reg["keluarga_konteks"].value_counts().to_dict())
    tab = pd.crosstab(seri, reg["register_energi"])
    print("Per seri meter (4 karakter awal):")
    print(tab.to_string())
    if "status_populasi" in reg.columns:
        print("Status aturan register nol:", reg["status_populasi"].value_counts().to_dict())
        dip = reg[reg["status_populasi"] == STATUS_DIPAKAI]
        print("Register energi pada meter yang dipakai:",
              dip["register_energi"].value_counts().to_dict())
        f = reg["fraksi_slot_kwh_nol"].dropna()
        print("Fraksi slot kWh nol per meter: median "
              f"{f.median():.3f}; jumlah meter dengan fraksi > 0,1 / 0,3 / 0,5 / 0,9: "
              f"{int((f > 0.1).sum())} / {int((f > 0.3).sum())} / "
              f"{int((f > 0.5).sum())} / {int((f > 0.9).sum())}")


def build():
    intervals = select_meters()

    parts = []
    for f in list_month_files():
        part = reduce_month(os.path.join(EXPORT_DIR, f), intervals)
        if part is not None:
            parts.append(part)
            print(f"  {f}: {len(part):,} slot")
    slots = pd.concat(parts, ignore_index=True)
    del parts

    df, reg = slots_to_frame(slots)
    del slots
    print_register_summary(reg)
    reg.to_csv(REGISTER_MAP_FILE, index=False)
    print(f"Peta register disimpan: {REGISTER_MAP_FILE}")

    df, _ = scale_per_meter(df, scaler_file=POPULATION_SCALER_FILE)
    X, y, info = create_sequences(df, sequence_length=POPULATION_SEQ_LEN)

    # Sampel maksimum WINDOWS_PER_METER hari per meter
    rng = np.random.default_rng(RANDOM_SEED)
    keep = np.zeros(len(info), dtype=bool)
    for _, idx in info.groupby("meter_id").indices.items():
        if len(idx) <= WINDOWS_PER_METER:
            keep[idx] = True
        else:
            keep[rng.choice(idx, size=WINDOWS_PER_METER, replace=False)] = True
    X, info = X[keep], info[keep].reset_index(drop=True)

    print(f"\nWindow populasi final: {len(X):,} dari "
          f"{info['meter_id'].nunique():,} meter")
    seri = info["meter_id"].astype(str).str[:4]
    print("Meter per seri dalam data latih:",
          info.groupby(seri)["meter_id"].nunique().to_dict())

    with open(POPULATION_FILE, "wb") as f:
        pickle.dump({"X": X, "info": info}, f, protocol=4)
    print(f"Disimpan: {POPULATION_FILE} "
          f"({os.path.getsize(POPULATION_FILE) / 1024**2:.1f} MB)")


if __name__ == "__main__":
    build()
