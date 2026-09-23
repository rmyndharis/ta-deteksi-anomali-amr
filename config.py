"""
Konfigurasi Global untuk Pipeline Deteksi Anomali Konsumsi Energi Listrik
Berbasis LSTM Autoencoder pada Data Smart Meter AMR

Penulis: Yudhi Armyndharis (220401010272)
Universitas Siber Asia
"""

import os

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Buat direktori jika belum ada
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# DATASET CONFIGURATION
# ============================================================
# Dataset kasus-kontrol hasil kurasi ekspor AP2T/AMR:
# - grp 'p2tl'   : meter dengan kasus P2TL terkonfirmasi;
#                  periode 'pre_p2tl' (masa pelanggaran, label 1)
#                  dan 'post_p2tl' (setelah penertiban, label 0)
# - grp 'kontrol': meter normal pembanding (label 0)
RAW_DATA_FILE = os.path.join(
    os.path.dirname(BASE_DIR), "data", "AP2T", "dataset_lp_p2tl.csv"
)

# Grid interval umum. Grid 30 menit dipakai karena data 15 menit dapat
# digabung tanpa membuat nilai baru sedangkan data 60 menit tidak dapat
# dipecah; hari yang kadensinya lebih kasar dari grid otomatis gugur.
# Meter 15-menit di-downsample (kWh dijumlah, V/I/PF dirata-rata).
INTERVAL_MINUTES = 30
RECORDS_PER_DAY = (24 * 60) // INTERVAL_MINUTES

# Porsi meter kontrol yang ditahan sebagai bagian test set,
# untuk mengukur false positive rate pada meter normal
# yang tidak pernah dilihat model.
KONTROL_TEST_FRACTION = 0.2

# ============================================================
# FEATURE CONFIGURATION
# ============================================================
# Fitur utama kelistrikan
ELECTRICAL_FEATURES = ["kWh", "voltage", "current", "power_factor"]

# Fitur temporal (di-derive dari timestamp).
# 'month' tidak dipakai: split train/test berbasis waktu membuat nilai
# bulan pada data uji selalu di luar rentang data latih.
TEMPORAL_FEATURES = ["hour", "day_of_week"]

# Semua fitur yang digunakan model
MODEL_FEATURES = ELECTRICAL_FEATURES + TEMPORAL_FEATURES

# Masked autoencoder: kanal energi DISEMBUNYIKAN dari input (di-nol-kan,
# yaitu nilai median meter pada skala robust) dan model merekonstruksinya
# dari konteks (voltage, power_factor, jam, hari). Autoencoder identitas
# dapat menyalin input sehingga pola pencurian (konsumsi tertekan)
# ikut tersalin dan tidak terdeteksi; dengan masking, rekonstruksi
# menjadi profil normal yang diharapkan dan residualnya bermakna.
MASKED_FEATURES = ["kWh", "current"]

# Kolom target (label anomali P2TL - hanya untuk kurasi data latih
# dan evaluasi, bukan untuk loss training)
TARGET_COLUMN = "anomaly_label"

# ============================================================
# PREPROCESSING CONFIGURATION
# ============================================================
# Normalisasi fitur kelistrikan: robust scaling PER METER
# ((x - median) / IQR pada riwayat meter itu sendiri, bebas label).
# Skala absolut antar meter sangat heterogen (satuan sisi metering,
# pelanggan TM vs TR, CT/PT berbeda) sehingga skala global akan
# didominasi meter besar; model dimaksudkan belajar BENTUK pola.
# Fitur temporal diskalakan deterministik (hour/23, day_of_week/6).
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")

# Winsorisasi nilai ter-skala. Meter berkonsumsi nyaris nol punya IQR
# sangat kecil sehingga nilai ter-skala bisa ratusan IQR dan mendominasi
# loss MSE. Nilai ter-skala dipotong ke +-SCALED_CLIP (dipilih a-priori
# dari statistik data latih: 99% |kWh ter-skala| < 10).
SCALED_CLIP = 10.0

# Sequence window: satu jendela = satu hari penuh.
# Hari dengan record tidak lengkap dibuang, tidak dikarang datanya.
SEQUENCE_LENGTH = RECORDS_PER_DAY

# ============================================================
# MODEL CONFIGURATION (LSTM Autoencoder)
# ============================================================
# Arsitektur encoder
ENCODER_LSTM_1_UNITS = 64
ENCODER_LSTM_2_UNITS = 32
LATENT_DIM = 16

# Arsitektur decoder
DECODER_LSTM_1_UNITS = 32
DECODER_LSTM_2_UNITS = 64

# Training hyperparameters
LOSS_FUNCTION = "mse"              # Mean Squared Error
OPTIMIZER = "adam"
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50
VALIDATION_SPLIT = 0.1             # 10% METER latih ditahan sebagai validasi (group split per meter)

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-5

# File model
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_autoencoder.keras")
TRAINING_HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.pkl")

# ============================================================
# ANOMALY DETECTION CONFIGURATION
# ============================================================
# Threshold berdasarkan persentil reconstruction error pada data training (normal).
# P95 dipilih sebagai kompromi: persentil lebih tinggi membuat penurunan sedang
# mudah terlewat, lebih rendah menambah hari normal yang harus ditinjau petugas
# (dipilih a-priori, BUKAN disetel terhadap data uji).
THRESHOLD_PERCENTILE = 95
THRESHOLD_FILE = os.path.join(MODEL_DIR, "threshold.pkl")

# ============================================================
# EVALUATION CONFIGURATION
# ============================================================
CONFUSION_MATRIX_FILE = os.path.join(FIGURE_DIR, "confusion_matrix.png")
ROC_CURVE_FILE = os.path.join(FIGURE_DIR, "roc_curve.png")
TRAINING_LOSS_FILE = os.path.join(FIGURE_DIR, "training_loss.png")
DAILY_PATTERN_FILE = os.path.join(FIGURE_DIR, "daily_pattern.png")
DISTRIBUTION_FILE = os.path.join(FIGURE_DIR, "distribution.png")
ANOMALY_TIMESERIES_FILE = os.path.join(FIGURE_DIR, "anomaly_timeseries.png")
RE_DISTRIBUTION_FILE = os.path.join(FIGURE_DIR, "re_distribution.png")

# ============================================================
# DASHBOARD CONFIGURATION
# ============================================================
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501

# ============================================================
# RANDOM SEED (untuk reprodusibilitas)
# ============================================================
RANDOM_SEED = 42
