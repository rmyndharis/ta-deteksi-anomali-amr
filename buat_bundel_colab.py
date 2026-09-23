"""Membangun bundel ringan untuk Google Colab dari folder TA.

Bundel = salinan terpangkas folder TA (struktur sama) yang cukup untuk
menjalankan Penjelasan_Sistem_TA.ipynb tanpa pkl besar (~50 MB):

    TA_colab/
      code/            modul pipeline, notebook, models/, output/ (aktif = arsip)
      code/experiments/<arsip>/   artefak jalan resmi (+ population_info.csv)
      code/experiments/analysis_hibrida/, code/data/fitur_harian_hibrida.csv
      data/AP2T/dataset_lp_p2tl.csv
      manifest_bundel.json        daftar berkas + md5 untuk pemeriksaan di notebook

Jalankan dari folder code/ (venv sudah ada):
    ./venv/bin/python buat_bundel_colab.py                 # arsip terbaru
    ./venv/bin/python buat_bundel_colab.py --arsip experiments/20260906_..._v11_groupsplit_register
    ./venv/bin/python buat_bundel_colab.py --lengkap        # ikutkan pkl besar (+320 MB)
Hasil: ../TA_colab/  -> unggah folder ini ke Google Drive (MyDrive/TA_colab).

Penulis: Yudhi Armyndharis (220401010272)
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import shutil

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))          # .../TA/code
TA = os.path.dirname(BASE)                                  # .../TA

MODUL = ["config.py", "preprocessing.py", "model.py", "detect.py", "evaluate.py",
         "artefak.py", "klasifikasi_hibrida.py", "fitur_indikator.py", "train.py",
         "Penjelasan_Sistem_TA.ipynb"]
ARSIP_KECIL = ["models/lstm_autoencoder.keras", "models/threshold.pkl", "models/training_history.pkl",
               "models/scaler.pkl", "output/detection_results.pkl", "output/evaluation_metrics.pkl",
               "output/per_channel_loss.json", "output/population_ranking.csv",
               "population_register_map.csv", "analysis_v11/ringkasan_v11.json",
               "analysis_v11/tabel_detektabilitas_kasus.csv", "analysis_v11/sintetis/tabel_sintetis.csv",
               "v11_02_build.log", "v11_03_main.log"]
ARSIP_BESAR = ["population_windows.pkl", "processed_sequences.pkl"]
AKTIF = ["models/lstm_autoencoder.keras", "models/threshold.pkl", "output/detection_results.pkl",
         "output/evaluation_metrics.pkl", "output/population_ranking.csv"]
HIBRIDA = ["data/fitur_harian_hibrida.csv", "experiments/analysis_hibrida/tabel_hibrida.csv",
           "experiments/analysis_hibrida/tabel_cakupan_indikator.csv",
           "experiments/analysis_hibrida/ringkasan_hibrida.json",
           "experiments/analysis_hibrida/gambar_kasus_fasa.png",
           "experiments/analysis_hibrida/gambar_roc_hibrida.png",
           "experiments/analysis_hibrida/gambar_efek_penyeimbangan.png"]
DATASET = "data/AP2T/dataset_lp_p2tl.csv"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def salin(src, dst, manifest, rel):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    manifest[rel] = {"bytes": os.path.getsize(dst), "md5": md5(dst)}
    print(f"  {rel} ({os.path.getsize(dst) / 1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arsip", default=None, help="folder arsip (default: experiments/*_v11_groupsplit_register terbaru)")
    ap.add_argument("--lengkap", action="store_true", help="ikutkan population_windows.pkl & processed_sequences.pkl")
    ap.add_argument("--tujuan", default=os.path.join(TA, "TA_colab"))
    a = ap.parse_args()

    arsip = a.arsip or sorted(glob.glob(os.path.join(BASE, "experiments", "*_v11_groupsplit_register")))[-1]
    arsip = os.path.relpath(os.path.abspath(arsip), BASE)          # "experiments/<stamp>_v11_groupsplit_register"
    tujuan = os.path.abspath(a.tujuan)
    if os.path.exists(tujuan):
        shutil.rmtree(tujuan)
    print(f"Arsip : {arsip}\nTujuan: {tujuan}\n")

    manifest = {}
    for rel in MODUL:
        salin(os.path.join(BASE, rel), os.path.join(tujuan, "code", rel), manifest, f"code/{rel}")
    for rel in ARSIP_KECIL + (ARSIP_BESAR if a.lengkap else []):
        salin(os.path.join(BASE, arsip, rel), os.path.join(tujuan, "code", arsip, rel), manifest, f"code/{arsip}/{rel}")
    for rel in AKTIF:                                              # artefak aktif = salinan arsip (seperti cp -a run_v11.sh)
        salin(os.path.join(BASE, arsip, rel), os.path.join(tujuan, "code", rel), manifest, f"code/{rel}")
    for rel in HIBRIDA:
        salin(os.path.join(BASE, rel), os.path.join(tujuan, "code", rel), manifest, f"code/{rel}")
    salin(os.path.join(TA, DATASET), os.path.join(tujuan, DATASET), manifest, DATASET)

    # ringkasan window populasi (pengganti population_windows.pkl 150 MB): hanya info per window
    with open(os.path.join(BASE, arsip, "population_windows.pkl"), "rb") as f:
        pop = pickle.load(f)
    info = pop["info"].copy()
    info["meter_id"] = info["meter_id"].astype(str)
    assert len(info) == len(pop["X"]) and pop["X"].shape[1:] == (48, 6), pop["X"].shape
    rel = f"code/{arsip}/population_info.csv"
    dst = os.path.join(tujuan, rel)
    info[["meter_id", "date", "grp", "periode", "case_id"]].to_csv(dst, index=False)
    manifest[rel] = {"bytes": os.path.getsize(dst), "md5": md5(dst),
                     "catatan": f"info {len(info):,} window x {pop['X'].shape[1]} langkah x {pop['X'].shape[2]} fitur dari population_windows.pkl"}
    print(f"  {rel} ({os.path.getsize(dst) / 1e6:.2f} MB) <- ringkasan population_windows.pkl")

    with open(os.path.join(tujuan, "manifest_bundel.json"), "w") as f:
        json.dump({"arsip": arsip, "lengkap": a.lengkap,
                   "dibuat": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "berkas": manifest}, f, indent=1)
    total = sum(v["bytes"] for v in manifest.values()) / 1e6
    print(f"\nSelesai: {len(manifest)} berkas, {total:.1f} MB -> {tujuan}")
    print("Unggah folder TA_colab ke Google Drive (MyDrive/TA_colab), lalu buka code/Penjelasan_Sistem_TA.ipynb di Colab.")


if __name__ == "__main__":
    main()
