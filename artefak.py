"""Pemuat artefak pipeline (model, ambang, set uji) untuk skrip analisis.

Skrip analisis memakai artefak AKTIF di models/, output/, dan data/ sehingga
selalu selaras dengan pipeline yang terakhir dijalankan. Pemuatan pickle
mempunyai cadangan: bila DataFrame di dalam pickle tidak dapat dibaca oleh
versi pandas yang berbeda, hanya array NumPy yang dimuat dan info window
diambil dari output/detection_results.pkl (urutannya identik).

Penulis: Yudhi Armyndharis (220401010272)
"""
import os
import pickle

import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, MODEL_FILE, THRESHOLD_FILE, SCALER_FILE

PROCESSED_FILE = os.path.join(DATA_DIR, "processed_sequences.pkl")
DETECTION_FILE = os.path.join(OUTPUT_DIR, "detection_results.pkl")


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        pass


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Ganti objek pandas dengan stub agar array NumPy tetap terbaca."""

    def find_class(self, mod, name):
        if mod.startswith("pandas"):
            return _Stub
        return super().find_class(mod, name)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_processed():
    """X_train, y_train, X_test, y_test, test_info (DataFrame)."""
    try:
        p = load_pkl(PROCESSED_FILE)
        ti = p.get("test_info")
    except Exception as e:                     # pickle pandas beda versi
        print(f"  processed_sequences: DataFrame tidak terbaca ({type(e).__name__}); "
              f"memuat array saja, info window dari detection_results.pkl")
        with open(PROCESSED_FILE, "rb") as f:
            p = _NumpyOnlyUnpickler(f).load()
        ti = None
    if ti is None or not isinstance(ti, pd.DataFrame):
        ti = load_pkl(DETECTION_FILE)["test_info"]
    ti = ti.reset_index(drop=True).copy()
    for c in ["meter_id", "grp", "periode", "case_id"]:
        if c in ti:
            ti[c] = ti[c].astype(str)
    assert len(ti) == len(p["X_test"]), "info window tidak selaras dengan X_test"
    return (np.asarray(p["X_train"]), np.asarray(p["y_train"]),
            np.asarray(p["X_test"]), np.asarray(p["y_test"]), ti)


def load_detection():
    """test_info dengan kolom score dan flag, plus ambang."""
    d = load_pkl(DETECTION_FILE)
    ti = d["test_info"].reset_index(drop=True).copy()
    for c in ["meter_id", "grp", "periode", "case_id"]:
        ti[c] = ti[c].astype(str)
    ti["date"] = pd.to_datetime(ti["date"]).dt.strftime("%Y-%m-%d")
    ti["score"] = np.asarray(d["re_test"], float)
    thr = float(d["threshold"])
    ti["flag"] = (ti["score"] > thr).astype(int)
    return ti, thr


def load_model_and_threshold():
    from model import load_model
    return load_model(MODEL_FILE), float(load_pkl(THRESHOLD_FILE))


def load_scalers():
    return load_pkl(SCALER_FILE)
