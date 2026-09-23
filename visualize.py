"""
Visualisasi Eksplorasi Data
============================
Menghasilkan visualisasi untuk BAB IV subbab 4.1:
- Pola konsumsi harian rata-rata
- Distribusi konsumsi energi
- Pola mingguan
- Korelasi antar fitur

Penulis: Yudhi Armyndharis (220401010272)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import (
    DATA_DIR, FIGURE_DIR, DAILY_PATTERN_FILE, DISTRIBUTION_FILE
)


def koma(v, n=None):
    """Format angka dengan koma desimal (gaya Indonesia): koma(0.585, 3) -> '0,585'."""
    s = f"{v:.{n}f}" if n is not None else f"{v:g}"
    return s.replace(".", ",")


def sumbu_koma(ax, sumbu="xy"):
    """Label tick sumbu memakai koma desimal dan titik ribuan (gaya Indonesia)."""
    from matplotlib.ticker import FuncFormatter

    def _f(v, _pos):
        if float(v).is_integer():
            return f"{int(v):,}".replace(",", ".")
        return f"{v:g}".replace(".", ",")
    if "x" in sumbu:
        ax.xaxis.set_major_formatter(FuncFormatter(_f))
    if "y" in sumbu:
        ax.yaxis.set_major_formatter(FuncFormatter(_f))


def plot_daily_pattern(df, save_path=None):
    """
    Visualisasi pola konsumsi harian rata-rata.
    Gambar IV.1 dalam BAB IV.
    """
    save_path = save_path or DAILY_PATTERN_FILE

    hourly_avg = df.groupby("hour")["kWh"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(hourly_avg.index, hourly_avg.values, "o-",
            color="steelblue", linewidth=2, markersize=6)

    # Highlight area puncak
    ax.axvspan(6, 9, alpha=0.1, color="orange", label="Puncak Pagi")
    ax.axvspan(18, 21, alpha=0.1, color="red", label="Puncak Malam")
    ax.axvspan(0, 5, alpha=0.1, color="blue", label="Periode Rendah")

    ax.set_xlabel("Jam", fontsize=12)
    ax.set_ylabel("Rata-rata konsumsi energi (kWh)", fontsize=12)
    ax.set_title("Pola Konsumsi Energi Listrik Harian Rata-rata", fontsize=14)
    ax.set_xticks(range(0, 24))
    sumbu_koma(ax, "y")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot pola harian disimpan: {save_path}")


def plot_distribution(df, save_path=None):
    """
    Visualisasi distribusi konsumsi energi (kWh).
    Gambar IV.2 dalam BAB IV.
    """
    save_path = save_path or DISTRIBUTION_FILE

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram kWh
    axes[0, 0].hist(df["kWh"].dropna(), bins=100, color="steelblue",
                    edgecolor="black", linewidth=0.3, alpha=0.7)
    axes[0, 0].set_xlabel("Energi per interval (kWh)")
    axes[0, 0].set_ylabel("Frekuensi")
    axes[0, 0].set_title("Distribusi Konsumsi Energi (kWh)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Histogram voltage
    axes[0, 1].hist(df["voltage"].dropna(), bins=100, color="darkorange",
                    edgecolor="black", linewidth=0.3, alpha=0.7)
    axes[0, 1].set_xlabel("Tegangan (V)")
    axes[0, 1].set_ylabel("Frekuensi")
    axes[0, 1].set_title("Distribusi Tegangan Listrik")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Histogram current
    axes[1, 0].hist(df["current"].dropna(), bins=100, color="forestgreen",
                    edgecolor="black", linewidth=0.3, alpha=0.7)
    axes[1, 0].set_xlabel("Arus (A)")
    axes[1, 0].set_ylabel("Frekuensi")
    axes[1, 0].set_title("Distribusi Arus Listrik")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Histogram power_factor
    axes[1, 1].hist(df["power_factor"].dropna(), bins=100, color="purple",
                    edgecolor="black", linewidth=0.3, alpha=0.7)
    axes[1, 1].set_xlabel("Faktor daya")
    axes[1, 1].set_ylabel("Frekuensi")
    axes[1, 1].set_title("Distribusi Faktor Daya")
    axes[1, 1].grid(True, alpha=0.3)
    for a in axes.ravel():
        sumbu_koma(a)

    plt.suptitle("Distribusi Nilai Konsumsi Energi Listrik", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot distribusi disimpan: {save_path}")


def plot_weekly_pattern(df, save_path=None):
    """Visualisasi pola konsumsi per hari dalam seminggu."""
    save_path = save_path or os.path.join(FIGURE_DIR, "weekly_pattern.png")

    day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    daily_avg = df.groupby("day_of_week")["kWh"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["steelblue"] * 5 + ["darkorange"] * 2
    ax.bar(range(7), daily_avg.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_xlabel("Hari", fontsize=12)
    ax.set_ylabel("Rata-rata konsumsi (kWh)", fontsize=12)
    ax.set_title("Pola Konsumsi Energi Listrik per Hari dalam Seminggu", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    sumbu_koma(ax, "y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot pola mingguan disimpan: {save_path}")


def plot_correlation_matrix(df, save_path=None):
    """Matriks korelasi antar fitur kelistrikan."""
    save_path = save_path or os.path.join(FIGURE_DIR, "correlation_matrix.png")

    features = ["kWh", "voltage", "current", "power_factor"]
    nama = ["Energi (kWh)", "Tegangan", "Arus", "Faktor daya"]
    corr = df[features].corr()
    corr.index = nama
    corr.columns = nama

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, square=True, ax=ax, linewidths=0.5)
    for t in ax.texts:                      # koma desimal pada angka korelasi
        t.set_text(t.get_text().replace(".", ","))
    cb = ax.collections[0].colorbar
    if cb is not None:
        sumbu_koma(cb.ax, "y")
    ax.set_title("Matriks Korelasi Fitur Kelistrikan", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot korelasi disimpan: {save_path}")


def run_visualization(data_file=None):
    """Jalankan semua visualisasi eksplorasi."""
    from config import RAW_DATA_FILE
    from preprocessing import load_dataset

    data_file = data_file or RAW_DATA_FILE

    print("=" * 70)
    print("VISUALISASI EKSPLORASI DATA")
    print("=" * 70)

    # Load sample data (EDA tidak butuh seluruh dataset)
    df = load_dataset(data_file, nrows=2_000_000)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    print(f"  Sample size: {len(df):,} record")

    plot_daily_pattern(df)
    plot_distribution(df)
    plot_weekly_pattern(df)
    plot_correlation_matrix(df)

    print(f"\n  Semua visualisasi disimpan di: {FIGURE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_visualization()
