"""
Preprocessing Dataset Kasus-Kontrol P2TL (AP2T/AMR)
==============================================================
Input: dataset kurasi load profile AMR dengan desain kasus-kontrol:
- grp 'p2tl'   : meter dengan kasus P2TL terkonfirmasi;
                 periode 'pre_p2tl' (masa pelanggaran, label 1) dan
                 'post_p2tl' (setelah penertiban, label 0)
- grp 'kontrol': meter normal pembanding (label 0)

Tahapan:
1. Load & cleaning (tipe data, duplikat, validasi rentang fisik;
   faktor daya bertanda negatif dipakai nilai mutlaknya)
2. Penyeragaman grid interval (meter 15-menit di-downsample ke 30-menit;
   meter berinterval lebih besar dibuang)
3. Robust scaling per meter (bebas label) + fitur temporal deterministik
4. Pembentukan window harian penuh per meter
5. Split berbasis kelompok meter:
   train = mayoritas meter kontrol (window normal)
   test  = seluruh meter p2tl + sebagian meter kontrol yang ditahan
           (untuk mengukur FPR pada meter normal tak terlihat)

Mode pemeriksaan data:
    python preprocessing.py --check [path_csv]

Penulis: Yudhi Armyndharis (220401010272)
"""

import numpy as np
import pandas as pd
import pickle
import os

from config import (
    RAW_DATA_FILE, DATA_DIR, SCALER_FILE,
    MODEL_FEATURES, ELECTRICAL_FEATURES,
    TARGET_COLUMN, SEQUENCE_LENGTH, RECORDS_PER_DAY,
    INTERVAL_MINUTES, KONTROL_TEST_FRACTION, RANDOM_SEED, SCALED_CLIP,
)

META_COLUMNS = ["grp", "periode", "case_id"]


def load_dataset(filepath=None, nrows=None):
    """Load dataset kasus-kontrol dan validasi kolom."""
    filepath = filepath or RAW_DATA_FILE
    print(f"\n[1/5] Loading dataset dari {filepath}...")

    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)

    required = ["timestamp", "meter_id", TARGET_COLUMN, "grp", "periode"] + ELECTRICAL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}. "
                         f"Kolom tersedia: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["meter_id"] = df["meter_id"].astype(str)
    df = df.dropna(subset=["timestamp"])

    for col in ELECTRICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COLUMN] = (
        pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int).clip(0, 1)
    )
    if "case_id" not in df.columns:
        df["case_id"] = ""

    # Validasi rentang fisik. Faktor daya bertanda negatif (konvensi arah
    # daya reaktif pada seri EMK1; besaran mutlaknya bermedian sekitar 0,96,
    # sama dengan pembacaan bertanda positif) dipakai NILAI MUTLAKNYA, bukan
    # disetel nol (perubahan 9 September 2026; sebelumnya clip(0, 1) membuat
    # faktor daya 21% record menjadi nol).
    for col in ["kWh", "current"]:
        df.loc[df[col] < 0, col] = np.nan
    df["power_factor"] = df["power_factor"].abs().clip(0, 1)

    n0 = len(df)
    df = df.drop_duplicates(subset=["meter_id", "timestamp"], keep="last")
    if n0 - len(df):
        print(f"  Duplikat dihapus: {n0 - len(df):,}")

    print(f"  Record: {len(df):,} | meter: {df['meter_id'].nunique()} "
          f"| kasus: {df.loc[df['grp'] == 'p2tl', 'case_id'].nunique()}")
    return df


def meter_intervals(df):
    """Interval pencatatan dominan (menit) per meter."""
    diffs = (df.sort_values(["meter_id", "timestamp"])
               .groupby("meter_id")["timestamp"].diff()
               .dt.total_seconds().div(60))
    return diffs.groupby(df["meter_id"]).agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)


def resample_to_grid(df):
    """
    Seragamkan semua meter ke grid INTERVAL_MINUTES:
    - meter dengan interval lebih rapat di-downsample
      (kWh dijumlah, V/I/PF dirata-rata, label = max)
    - slot kWh yang readingnya tidak lengkap di-NaN-kan
      (agar tidak tercipta 'drop' palsu), harinya nanti dibuang
    - meter dengan interval lebih renggang dari grid dibuang
    """
    print(f"\n[2/5] Penyeragaman grid {INTERVAL_MINUTES} menit...")

    itv = meter_intervals(df)
    ok = itv[itv <= INTERVAL_MINUTES].index
    dropped = sorted(set(df["meter_id"]) - set(ok))
    if dropped:
        print(f"  Meter dibuang (interval > {INTERVAL_MINUTES} menit): {len(dropped)}")
    df = df[df["meter_id"].isin(ok)].copy()

    df["slot"] = df["timestamp"].dt.floor(f"{INTERVAL_MINUTES}min")
    g = df.groupby(["meter_id", "slot"], sort=False).agg(
        kWh=("kWh", "sum"),
        k_n=("kWh", "count"),
        voltage=("voltage", "mean"),
        current=("current", "mean"),
        power_factor=("power_factor", "mean"),
        anomaly_label=(TARGET_COLUMN, "max"),
        grp=("grp", "first"),
        periode=("periode", "first"),
        case_id=("case_id", "first"),
    ).reset_index().rename(columns={"slot": "timestamp"})

    # Slot energi tidak lengkap -> NaN (jumlah reading < yang diharapkan)
    expected = (INTERVAL_MINUTES / itv).round().astype(int)
    g["expected"] = g["meter_id"].map(expected).fillna(1).astype(int)
    incomplete = g["k_n"] < g["expected"]
    g.loc[incomplete, "kWh"] = np.nan
    print(f"  Slot energi tidak lengkap: {int(incomplete.sum()):,} -> NaN")

    g = g.drop(columns=["k_n", "expected"])
    g["hour"] = g["timestamp"].dt.hour
    g["day_of_week"] = g["timestamp"].dt.dayofweek

    print(f"  Record setelah grid: {len(g):,} | meter: {g['meter_id'].nunique()}")
    return g


def scale_per_meter(df, scaler_file=SCALER_FILE):
    """
    Robust scaling fitur kelistrikan PER METER: (x - median) / IQR,
    dihitung dari seluruh riwayat meter itu sendiri (bebas label,
    dapat diterapkan operasional). Fitur temporal diskalakan
    deterministik: hour/23, day_of_week/6.
    """
    print(f"\n[3/5] Robust scaling per meter...")

    scalers = {}
    for meter_id, g in df.groupby("meter_id", sort=False):
        med = g[ELECTRICAL_FEATURES].median()
        iqr = g[ELECTRICAL_FEATURES].quantile(0.75) - g[ELECTRICAL_FEATURES].quantile(0.25)
        std = g[ELECTRICAL_FEATURES].std()
        scale = iqr.where(iqr > 1e-9, std).where(lambda s: s > 1e-9, 1.0)
        idx = g.index
        df.loc[idx, ELECTRICAL_FEATURES] = (
            ((g[ELECTRICAL_FEATURES] - med) / scale).clip(-SCALED_CLIP, SCALED_CLIP)
        )
        scalers[meter_id] = {"median": med.to_dict(), "scale": scale.to_dict()}

    df["hour"] = df["hour"] / 23.0
    df["day_of_week"] = df["day_of_week"] / 6.0

    with open(scaler_file, "wb") as f:
        pickle.dump(scalers, f)
    print(f"  Scaler per meter disimpan: {scaler_file} ({len(scalers)} meter)")
    return df, scalers


def create_sequences(df, sequence_length=SEQUENCE_LENGTH):
    """
    Window harian penuh per meter: RECORDS_PER_DAY timestep x fitur.
    Hari tidak lengkap atau memuat NaN dibuang.

    Returns:
        X (n, seq, fitur) float32; y (n,) 1 bila ada record anomali;
        info_df per window (meter, tanggal, grp, periode, case, titik anomali)
    """
    print(f"\n[4/5] Pembentukan window harian ({sequence_length} timestep)...")

    dates = df["timestamp"].dt.normalize()
    X_list, y_list, info_list = [], [], []
    n_dropped = 0

    for (meter_id, date), g in df.groupby(["meter_id", dates], sort=False):
        if len(g) != sequence_length or g[MODEL_FEATURES].isna().values.any():
            n_dropped += 1
            continue
        g = g.sort_values("timestamp")
        X_list.append(g[MODEL_FEATURES].to_numpy(dtype=np.float32))
        n_anom = int(g[TARGET_COLUMN].sum())
        y_list.append(1 if n_anom > 0 else 0)
        info_list.append({
            "meter_id": meter_id,
            "date": date,
            "start_time": g["timestamp"].iloc[0],
            "end_time": g["timestamp"].iloc[-1],
            "grp": g["grp"].iloc[0],
            "periode": g["periode"].iloc[0],
            "case_id": g["case_id"].iloc[0],
            "n_anomaly_points": n_anom,
        })

    if not X_list:
        raise ValueError("Tidak ada hari lengkap yang terbentuk; periksa "
                         "INTERVAL_MINUTES dan kelengkapan data.")

    X = np.stack(X_list)
    y = np.array(y_list)
    info_df = pd.DataFrame(info_list)

    print(f"  Window terbentuk : {len(X):,} (shape {X.shape})")
    print(f"  Hari dibuang     : {n_dropped:,} (tidak lengkap)")
    print(f"  Window anomali   : {int(y.sum()):,} ({y.mean() * 100:.2f}%)")
    return X, y, info_df


def split_by_group(X, y, info):
    """
    Split berbasis kelompok meter:
    - train: (1 - KONTROL_TEST_FRACTION) meter kontrol
    - test : seluruh meter p2tl + meter kontrol yang ditahan
    """
    print(f"\n[5/5] Split berbasis kelompok meter...")

    rng = np.random.default_rng(RANDOM_SEED)
    kontrol_meters = np.array(sorted(info.loc[info["grp"] == "kontrol", "meter_id"].unique()))
    rng.shuffle(kontrol_meters)
    n_test = int(round(len(kontrol_meters) * KONTROL_TEST_FRACTION))
    kontrol_test = set(kontrol_meters[:n_test])
    kontrol_train = set(kontrol_meters[n_test:])

    train_mask = info["meter_id"].isin(kontrol_train).to_numpy()
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    train_info = info[train_mask].reset_index(drop=True)
    test_info = info[test_mask].reset_index(drop=True)

    print(f"  Meter kontrol train : {len(kontrol_train)}")
    print(f"  Meter kontrol test  : {len(kontrol_test)} (ukur FPR meter tak terlihat)")
    print(f"  Meter p2tl (test)   : {info.loc[info['grp'] == 'p2tl', 'meter_id'].nunique()}")
    print(f"  Window train        : {len(X_train):,} (anomali: {int(y_train.sum()):,})")
    print(f"  Window test         : {len(X_test):,} (anomali: {int(y_test.sum()):,}, "
          f"{y_test.mean() * 100:.1f}%)")
    print(f"  Test per periode    : "
          f"{test_info.groupby(['grp', 'periode']).size().to_dict()}")

    return X_train, X_test, y_train, y_test, train_info, test_info


def run_preprocessing(data_filepath=None):
    """
    Jalankan seluruh pipeline preprocessing.

    Returns:
        X_train, X_test, y_train, y_test, scalers, train_info, test_info
    """
    print("=" * 70)
    print("PREPROCESSING PIPELINE (dataset kasus-kontrol P2TL)")
    print("=" * 70)

    df = load_dataset(data_filepath)
    df = resample_to_grid(df)
    df, scalers = scale_per_meter(df)
    X, y, info = create_sequences(df)
    X_train, X_test, y_train, y_test, train_info, test_info = split_by_group(X, y, info)

    processed = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_info": train_info,
        "test_info": test_info,
    }
    processed_file = os.path.join(DATA_DIR, "processed_sequences.pkl")
    with open(processed_file, "wb") as f:
        pickle.dump(processed, f, protocol=4)
    print(f"\n  Data terproses disimpan: {processed_file} "
          f"({os.path.getsize(processed_file) / 1024**2:.1f} MB)")
    print("=" * 70)

    return X_train, X_test, y_train, y_test, scalers, train_info, test_info


def run_preprocessing_population(data_filepath=None):
    """
    Varian eksperimen populasi:
    - Train : window populasi 2025 (meter non-P2TL, diasumsikan normal)
      dari data/population_windows.pkl (bangun dulu dengan
      `python build_training_population.py`).
    - Test  : SELURUH window dataset kasus-kontrol (semua meter kontrol
      kini tak pernah dilihat model -> FPR bersih; pre_p2tl = 1).

    Returns: X_train, X_test, y_train, y_test, scalers, train_info, test_info
    """
    print("=" * 70)
    print("PREPROCESSING PIPELINE (train populasi 2025, test kasus-kontrol)")
    print("=" * 70)

    population_file = os.path.join(DATA_DIR, "population_windows.pkl")
    if not os.path.exists(population_file):
        raise FileNotFoundError(
            f"{population_file} belum ada. Jalankan dulu: "
            f"python build_training_population.py")
    with open(population_file, "rb") as f:
        pop = pickle.load(f)
    X_train, train_info = pop["X"], pop["info"]
    if X_train.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Window populasi {X_train.shape[1]} timestep, config "
            f"SEQUENCE_LENGTH={SEQUENCE_LENGTH}. Eksperimen populasi "
            f"berjalan pada grid 60 menit: set INTERVAL_MINUTES = 60 "
            f"di config.py.")
    y_train = np.zeros(len(X_train), dtype=int)
    print(f"\n  Window latih populasi: {len(X_train):,} "
          f"({train_info['meter_id'].nunique():,} meter)")

    df = load_dataset(data_filepath)
    df = resample_to_grid(df)
    df, scalers = scale_per_meter(df)
    X_test, y_test, test_info = create_sequences(df)
    print(f"  Window uji kasus-kontrol: {len(X_test):,} "
          f"(anomali {int(y_test.sum()):,} = {y_test.mean() * 100:.1f}%)")
    print(f"  Uji per subgrup: "
          f"{test_info.groupby(['grp', 'periode']).size().to_dict()}")

    processed = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_info": train_info,
        "test_info": test_info,
    }
    processed_file = os.path.join(DATA_DIR, "processed_sequences.pkl")
    with open(processed_file, "wb") as f:
        pickle.dump(processed, f, protocol=4)
    print(f"\n  Data terproses disimpan: {processed_file}")
    print("=" * 70)

    return X_train, X_test, y_train, y_test, scalers, train_info, test_info


def check_data(filepath=None):
    """Pemeriksaan kesiapan dataset tanpa menjalankan pipeline penuh."""
    print("=" * 70)
    print("PEMERIKSAAN DATASET")
    print("=" * 70)

    df = load_dataset(filepath)

    print(f"\n  Periode data : {df['timestamp'].min()} s/d {df['timestamp'].max()}")
    print(f"  Grup         : {df['grp'].value_counts().to_dict()}")
    print(f"  Periode      : {df['periode'].value_counts().to_dict()}")
    n_anom = df[TARGET_COLUMN].sum()
    print(f"  Label anomali: {n_anom:,} record ({n_anom / len(df) * 100:.2f}%)")

    itv = meter_intervals(df)
    print(f"  Interval per meter: {itv.value_counts().to_dict()}")
    n_out = (itv > INTERVAL_MINUTES).sum()
    if n_out:
        print(f"  PERINGATAN: {n_out} meter berinterval > {INTERVAL_MINUTES} menit "
              f"akan dibuang.")

    missing = df[ELECTRICAL_FEATURES].isna().mean() * 100
    print(f"  Sel kosong per fitur (%): {missing.round(3).to_dict()}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--check"]
    filepath = args[0] if args else None
    if "--check" in sys.argv:
        check_data(filepath)
    else:
        run_preprocessing(filepath)
