"""
Evaluasi Performa Model
========================
Menghitung metrik evaluasi berdasarkan confusion matrix:
- Precision: ketepatan prediksi anomali
- Recall: kemampuan mendeteksi seluruh anomali
- F1-Score: harmonic mean precision dan recall
- AUC-ROC: kemampuan diskriminasi model

AUC-ROC dihitung menggunakan reconstruction error sebagai
skor probabilistik (continuous score), BUKAN dari confusion matrix tunggal.

Penulis: Yudhi Armyndharis (220401010272)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)

from config import (
    CONFUSION_MATRIX_FILE, ROC_CURVE_FILE, FIGURE_DIR
)


def evaluate_model(y_true, y_pred, re_scores=None):
    """
    Evaluasi lengkap performa model deteksi anomali.

    Args:
        y_true: Ground truth labels (0=normal, 1=anomali)
        y_pred: Predicted labels dari threshold
        re_scores: Reconstruction error scores (untuk AUC-ROC)

    Returns:
        metrics: Dictionary berisi semua metrik evaluasi
    """
    print("=" * 70)
    print("EVALUASI PERFORMA MODEL")
    print("=" * 70)

    # 1. Confusion Matrix (labels eksplisit agar tetap 2x2
    # meski salah satu kelas kosong)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix:")
    print(f"  {'':>20} {'Predicted Normal':>18} {'Predicted Anomali':>18}")
    print(f"  {'Actual Normal':>20} {tn:>18,} {fp:>18,}")
    print(f"  {'Actual Anomali':>20} {fn:>18,} {tp:>18,}")

    # 2. Metrik Evaluasi
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC menggunakan reconstruction error sebagai continuous score
    # (BUKAN dari binary prediction)
    auc_roc = None
    if re_scores is not None and len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, re_scores)

    print(f"\n  Metrik Evaluasi:")
    print(f"  {'Precision':>15} : {precision:.4f}")
    print(f"  {'Recall':>15} : {recall:.4f}")
    print(f"  {'F1-Score':>15} : {f1:.4f}")
    if auc_roc is not None:
        print(f"  {'AUC-ROC':>15} : {auc_roc:.4f}")

    print(f"\n  Interpretasi:")
    print(f"  - Precision {precision:.2f}: {precision*100:.0f}% data yang diprediksi anomali benar-benar anomali")
    print(f"  - Recall {recall:.2f}: {recall*100:.0f}% anomali dalam dataset berhasil terdeteksi")
    print(f"  - F1-Score {f1:.2f}: keseimbangan antara precision dan recall")
    if auc_roc is not None:
        print(f"  - AUC-ROC {auc_roc:.2f}: kemampuan diskriminasi model antara normal dan anomali")

    # 3. Classification Report
    print(f"\n  Classification Report:")
    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["Normal", "Anomali"], zero_division=0
    )
    print(report)

    # 4. Plot
    plot_confusion_matrix(cm)
    if re_scores is not None and len(np.unique(y_true)) > 1:
        plot_roc_curve(y_true, re_scores, auc_roc)

    # 5. Compile metrics
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }

    print("=" * 70)
    return metrics


def plot_confusion_matrix(cm):
    """Plot confusion matrix sebagai heatmap (angka dengan titik ribuan)."""
    from visualize import sumbu_koma
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Normal", "Anomali"],
        yticklabels=["Normal", "Anomali"],
        annot_kws={"size": 14},
        ax=ax
    )
    for t in ax.texts:                      # 3,633 -> 3.633 (titik ribuan)
        t.set_text(t.get_text().replace(",", "."))
    cb = ax.collections[0].colorbar
    if cb is not None:
        sumbu_koma(cb.ax, "y")

    ax.set_xlabel("Label prediksi", fontsize=12)
    ax.set_ylabel("Label sebenarnya", fontsize=12)
    ax.set_title("Confusion Matrix\nHasil Deteksi Anomali Konsumsi Energi Listrik", fontsize=14)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot confusion matrix disimpan: {CONFUSION_MATRIX_FILE}")


def plot_roc_curve(y_true, re_scores, auc_roc):
    """
    Plot kurva ROC.
    ROC dihitung dari berbagai threshold pada reconstruction error,
    bukan dari confusion matrix tunggal.
    """
    from visualize import sumbu_koma, koma
    fpr, tpr, thresholds = roc_curve(y_true, re_scores)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="darkorange", linewidth=2,
            label=f"Kurva ROC (AUC = {koma(auc_roc, 3)})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
            label="Tebakan acak (AUC = 0,5)")

    ax.fill_between(fpr, tpr, alpha=0.15, color="darkorange")

    ax.set_xlabel("False positive rate (proporsi hari normal yang ikut tertandai)", fontsize=10)
    ax.set_ylabel("True positive rate (proporsi hari masa sebelum penertiban yang tertandai)", fontsize=10)
    ax.set_title("Kurva ROC\nModel Deteksi Anomali Konsumsi Energi Listrik", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    sumbu_koma(ax)

    plt.tight_layout()
    plt.savefig(ROC_CURVE_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot ROC curve disimpan: {ROC_CURVE_FILE}")
