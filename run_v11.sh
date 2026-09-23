#!/bin/bash
# Pipeline utama (29 Agustus 2026, dirapikan 30 Agustus 2026): populasi latih
# dengan pemetaan register energi, pelatihan LSTM Autoencoder dengan validasi
# per meter, deteksi dan evaluasi, uji anomali sintetis, analisis kasus, dan
# peringkat populasi. Jalankan dari folder code/:   bash run_v11.sh
# Perkiraan waktu di MacBook Pro M-series: 25 sampai 35 menit.
# PERHATIAN: pelatihan memakai bobot awal acak, sehingga angka evaluasi dapat
# sedikit berbeda dari naskah. Hasil yang dipakai naskah diarsipkan di
# experiments/20260910_1709_v11_colab/ (jalan resmi Google Colab T4, 10 September 2026).
set -euo pipefail
cd "$(dirname "$0")"
PY=./venv/bin/python
mkdir -p logs experiments/analysis_v11
STAMP=$(date +%Y%m%d_%H%M)

echo "=== [1/6] set grid 30 menit ==="
$PY set_grid.py 30

echo "=== [2/6] build window populasi (register, min_count, k_n) ==="
$PY build_training_population.py 2>&1 | tee logs/v11_02_build.log

echo "=== [3/6] pipeline utama: praproses, latih (validasi per meter), deteksi, evaluasi ==="
$PY main.py --population 2>&1 | tee logs/v11_03_main.log

echo "=== [4/6] uji anomali sintetis ==="
$PY evaluasi_sintetis.py 2>&1 | tee logs/v11_05_sintetis.log

echo "=== [5/6] analisis kasus P2TL (Tabel 4.3, Lampiran C) ==="
$PY -u analisis_v11.py 2>&1 | tee logs/v11_06_analisis.log

echo "=== [6/6] peringkat populasi untuk dashboard ==="
$PY rank_population.py 2>&1 | tee logs/v11_08_ranking.log

DEST="experiments/${STAMP}_v11_groupsplit_register"
mkdir -p "$DEST"
cp -a models output "$DEST"/
cp -a data/processed_sequences.pkl data/population_windows.pkl \
      data/population_scalers.pkl data/population_register_map.csv \
      data/ranking_scalers.pkl "$DEST"/
cp -a experiments/analysis_v11 "$DEST"/analysis_v11
cp -a logs/v11_*.log "$DEST"/
echo "=== SELESAI. Arsip: $DEST ==="
echo "Dashboard: ./venv/bin/streamlit run dashboard.py"
