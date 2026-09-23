"""
Deteksi Anomali Berdasarkan Reconstruction Error (Masked Autoencoder)
======================================================================
Model menerima window dengan kanal energi (kWh, current) di-mask dan
merekonstruksi profil konsumsi normal yang diharapkan dari konteks
(voltage, power_factor, jam, hari). Skor deteksi adalah residual
BERARAH energi: ekspektasi dikurangi aktual,

    skor = mean_t mean_e (x_hat_{t,e} - x_{t,e}) ,  e in MASKED_FEATURES

Positif besar = konsumsi tercatat konsisten DI BAWAH profil normal
yang diharapkan, pola khas Non-Technical Loss (pencurian/tampering
menekan energi yang tercatat meter). Residual simetris (kuadrat)
sengaja tidak dipakai: ia buta arah, sedangkan deviasi ke atas bukan
indikasi pencurian. Mean antar-timestep dipakai karena pelanggaran
P2TL berlangsung berhari-hari.
Threshold = persentil P95 skor pada window latih normal (a-priori,
tidak disetel terhadap data uji).

Penulis: Yudhi Armyndharis (220401010272)
"""

import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    THRESHOLD_PERCENTILE, THRESHOLD_FILE,
    MODEL_FILE, RE_DISTRIBUTION_FILE, ANOMALY_TIMESERIES_FILE,
    FIGURE_DIR, ELECTRICAL_FEATURES
)


def calculate_reconstruction_error(model, X):
    """
    Hitung reconstruction error per window (masked autoencoder).

    Input model = window dengan kanal energi di-mask; rekonstruksi =
    profil normal yang diharapkan. Skor = residual BERARAH energi
    (ekspektasi - aktual), dirata-rata antar timestep dan kanal:

        skor = mean_t mean_e (x_hat_{t,e} - x_{t,e}) ,  e in MASKED_FEATURES

    Positif besar = konsumsi tercatat di bawah profil normal (pola NTL).
    Mean antar-timestep dipakai (bukan max) karena pelanggaran P2TL
    berlangsung berhari-hari.

    Args:
        model: Trained masked LSTM Autoencoder
        X: Window AKTUAL (n_samples, seq_length, n_features), belum di-mask

    Returns:
        re_scores: Array skor residual per window
    """
    from model import mask_energy_channels, MASKED_IDX

    print(f"\n  Menghitung skor residual untuk {len(X):,} windows...")

    X_reconstructed = model.predict(mask_energy_channels(X), batch_size=64, verbose=1)

    re_scores = (X_reconstructed[:, :, MASKED_IDX] - X[:, :, MASKED_IDX]).mean(axis=(1, 2))

    print(f"  RE min  : {re_scores.min():.6f}")
    print(f"  RE max  : {re_scores.max():.6f}")
    print(f"  RE mean : {re_scores.mean():.6f}")
    print(f"  RE std  : {re_scores.std():.6f}")

    return re_scores


def determine_threshold(re_train, percentile=THRESHOLD_PERCENTILE):
    """
    Tentukan threshold deteksi anomali berdasarkan persentil
    reconstruction error pada data training.

    Threshold = persentil ke-95 dari distribusi RE pada data training.
    Data dengan RE > threshold dikategorikan sebagai anomali.

    Args:
        re_train: Reconstruction error pada data training
        percentile: Persentil untuk threshold (default: 95)

    Returns:
        threshold: Nilai threshold
    """
    threshold = np.percentile(re_train, percentile)

    print(f"\n  Threshold Detection:")
    print(f"  Persentil          : {percentile}")
    print(f"  Threshold          : {threshold:.6f}")
    print(f"  Data > threshold   : {(re_train > threshold).sum():,} / {len(re_train):,} "
          f"({(re_train > threshold).sum() / len(re_train) * 100:.2f}%)")

    # Simpan threshold
    with open(THRESHOLD_FILE, "wb") as f:
        pickle.dump(threshold, f)
    print(f"  Threshold disimpan : {THRESHOLD_FILE}")

    return threshold


def detect_anomalies(re_scores, threshold):
    """
    Klasifikasi sequence sebagai normal atau anomali
    berdasarkan threshold reconstruction error.

    Args:
        re_scores: Array reconstruction error per sequence
        threshold: Nilai threshold

    Returns:
        predictions: Array binary (0=normal, 1=anomali)
    """
    predictions = (re_scores > threshold).astype(int)

    n_anomaly = predictions.sum()
    n_normal = len(predictions) - n_anomaly

    print(f"\n  Hasil Deteksi:")
    print(f"  Total sequences : {len(predictions):,}")
    print(f"  Normal          : {n_normal:,} ({n_normal/len(predictions)*100:.2f}%)")
    print(f"  Anomali         : {n_anomaly:,} ({n_anomaly/len(predictions)*100:.2f}%)")

    return predictions


def run_detection(model, X_train, X_test, y_train):
    """
    Jalankan pipeline deteksi anomali lengkap.

    Returns:
        re_test, threshold, predictions
    """
    print("=" * 70)
    print("DETEKSI ANOMALI BERDASARKAN RECONSTRUCTION ERROR")
    print("=" * 70)

    # 1. Hitung RE pada data training (hanya normal untuk threshold)
    print("\n--- Reconstruction Error (Training, window normal) ---")
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    re_train_normal = calculate_reconstruction_error(model, X_train_normal)

    # 2. Tentukan threshold
    threshold = determine_threshold(re_train_normal)

    # 3. Hitung RE pada data test
    print("\n--- Reconstruction Error (Testing) ---")
    re_test = calculate_reconstruction_error(model, X_test)

    # 4. Deteksi anomali pada data testing
    print("\n--- Deteksi pada Testing Set ---")
    predictions = detect_anomalies(re_test, threshold)

    # 5. Plot distribusi RE
    plot_re_distribution(re_train_normal, re_test, threshold)
    plot_anomaly_timeseries(re_test, threshold, predictions)

    print("=" * 70)
    return re_test, threshold, predictions


def plot_re_distribution(re_train, re_test, threshold):
    """Plot distribusi skor deteksi (residual berarah) dengan garis ambang."""
    from visualize import sumbu_koma, koma
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    label_ambang = f"Ambang (P{THRESHOLD_PERCENTILE:g}) = {koma(threshold, 4)}"

    # Plot 1: Distribusi skor pada window latih normal
    axes[0].hist(re_train, bins=100, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=label_ambang)
    axes[0].set_xlabel("Skor deteksi (tebakan model dikurangi kenyataan)", fontsize=11)
    axes[0].set_ylabel("Frekuensi", fontsize=11)
    axes[0].set_title("Distribusi Skor Deteksi pada Data Latih (Normal)", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Distribusi skor pada data uji
    axes[1].hist(re_test, bins=100, alpha=0.7, color="darkorange", edgecolor="black", linewidth=0.5)
    axes[1].axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=label_ambang)
    axes[1].set_xlabel("Skor deteksi (tebakan model dikurangi kenyataan)", fontsize=11)
    axes[1].set_ylabel("Frekuensi", fontsize=11)
    axes[1].set_title("Distribusi Skor Deteksi pada Data Uji", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    for a in axes:
        sumbu_koma(a)

    plt.tight_layout()
    plt.savefig(RE_DISTRIBUTION_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot distribusi RE disimpan: {RE_DISTRIBUTION_FILE}")


def plot_anomaly_timeseries(re_test, threshold, predictions):
    """Plot skor deteksi per window harian data uji dengan titik anomali."""
    from visualize import sumbu_koma, koma
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(re_test))
    normal_mask = predictions == 0
    anomaly_mask = predictions == 1

    ax.scatter(x[normal_mask], re_test[normal_mask], c="steelblue",
               s=5, alpha=0.5, label="Normal")
    ax.scatter(x[anomaly_mask], re_test[anomaly_mask], c="red",
               s=20, alpha=0.8, label="Anomali", zorder=5)
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Ambang = {koma(threshold, 4)}")

    ax.set_xlabel("Indeks window harian (data uji)", fontsize=11)
    ax.set_ylabel("Skor deteksi", fontsize=11)
    ax.set_title("Hasil Deteksi Anomali pada Window Harian Data Uji\nKonsumsi Energi Listrik", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    sumbu_koma(ax)

    plt.tight_layout()
    plt.savefig(ANOMALY_TIMESERIES_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot anomali time-series disimpan: {ANOMALY_TIMESERIES_FILE}")
