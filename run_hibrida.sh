#!/bin/bash
# Pipeline tahap klasifikasi hibrida (29 Agustus 2026): dataset per fasa,
# indikator harian + skor LSTM-AE, klasifikasi dengan penyeimbangan data
# (regresi logistik, validasi silang per meter), dan gambar studi kasus per fasa.
# Prasyarat: lp_anomali.csv dan lp_kontrol.csv di ../data/AP2T/, artefak
# pipeline v11 di models/ dan output/, serta pustaka imbalanced-learn:
#     ./venv/bin/pip install imbalanced-learn
# Jalankan dari folder code/:   bash run_hibrida.sh
# Langkah tambahan (7 September 2026): cek_variasi_fold.py mengulang validasi
# silang untuk 30 pembagian fold (Subbab 4.6, rentang AUC antar pembagian).
# Perkiraan waktu di MacBook Pro M-series: 4 sampai 6 menit.
set -euo pipefail
cd "$(dirname "$0")"
PY=./venv/bin/python
mkdir -p logs
echo "=== [1/5] dataset kasus-kontrol per fasa ==="
$PY buat_dataset_fasa.py 2>&1 | tee logs/hibrida_01_dataset.log
echo "=== [2/5] indikator fisik harian + skor LSTM-AE ==="
$PY fitur_indikator.py 2>&1 | tee logs/hibrida_02_fitur.log
echo "=== [3/5] klasifikasi hibrida dengan penyeimbangan data ==="
$PY klasifikasi_hibrida.py 2>&1 | tee logs/hibrida_03_klasifikasi.log
echo "=== [4/5] gambar studi kasus per fasa ==="
$PY gambar_kasus_fasa.py 2>&1 | tee logs/hibrida_04_gambar.log
echo "=== [5/5] variasi AUC antar 30 pembagian fold (Subbab 4.6) ==="
$PY cek_variasi_fold.py 2>&1 | tee logs/hibrida_05_variasi_fold.log
echo "=== SELESAI. Hasil: experiments/analysis_hibrida/ ==="
