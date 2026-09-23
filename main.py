"""
Pipeline Utama - End-to-End
============================
Menjalankan seluruh pipeline deteksi anomali konsumsi energi listrik
pada data smart meter AMR:

1. Pemeriksaan data   -> Validasi file ekspor AMR
2. Visualisasi EDA    -> Plot eksplorasi data
3. Preprocessing      -> Cleaning, grid 30 menit, normalisasi, window harian
4. Training Model     -> LSTM Autoencoder (dilatih pada window normal)
5. Deteksi Anomali    -> Reconstruction error + threshold P95
6. Evaluasi Model     -> Precision, Recall, F1, AUC-ROC (label P2TL)
7. Dashboard          -> (Opsional) Jalankan dengan streamlit

Usage:
    python main.py                          # Pakai data/AP2T/dataset_lp_p2tl.csv
    python main.py --data path/ke/file.csv  # Pakai file lain
    python main.py --dashboard              # Jalankan dashboard setelah pipeline

Sebelum run penuh, periksa kesiapan data:
    python preprocessing.py --check data/AP2T/dataset_lp_p2tl.csv

Penulis: Yudhi Armyndharis (220401010272)
Universitas Siber Asia
"""

import os
import sys
import time
import pickle
import argparse

# Pastikan working directory benar
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR, FIGURE_DIR,
    MODEL_FILE, RAW_DATA_FILE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline Deteksi Anomali Konsumsi Energi Listrik"
    )
    parser.add_argument("--data", default=RAW_DATA_FILE,
                        help=f"Path file CSV data smart meter AMR (default: {RAW_DATA_FILE})")
    parser.add_argument("--population", action="store_true",
                        help="Latih pada window populasi 2025 "
                             "(build_training_population.py), uji pada "
                             "seluruh dataset kasus-kontrol")
    parser.add_argument("--dashboard", action="store_true",
                        help="Jalankan dashboard setelah pipeline")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  PIPELINE DETEKSI ANOMALI KONSUMSI ENERGI LISTRIK")
    print("  LSTM Autoencoder")
    print("  Tugas Akhir - Yudhi Armyndharis (220401010272)")
    print("=" * 70)

    data_file = args.data
    if not os.path.exists(data_file):
        print(f"\nERROR: File data tidak ditemukan: {data_file}")
        print("Letakkan file ekspor AMR di path tersebut, atau jalankan dengan")
        print("  python main.py --data <path_file_csv>")
        sys.exit(1)

    total_start = time.time()

    # ============================================================
    # STEP 1: VISUALISASI EDA
    # ============================================================
    from visualize import run_visualization
    run_visualization(data_file)

    # ============================================================
    # STEP 2: PREPROCESSING
    # ============================================================
    if args.population:
        from preprocessing import run_preprocessing_population as run_prep
    else:
        from preprocessing import run_preprocessing as run_prep
    X_train, X_test, y_train, y_test, scaler, train_info, test_info = \
        run_prep(data_file)

    # ============================================================
    # STEP 3: TRAINING MODEL
    # ============================================================
    from train import train_model
    model, history = train_model(X_train, y_train, train_info)

    # ============================================================
    # STEP 4: DETEKSI ANOMALI
    # ============================================================
    from detect import run_detection
    re_test, threshold, predictions = run_detection(model, X_train, X_test, y_train)

    # Simpan hasil deteksi + mapping window -> pelanggan/waktu untuk dashboard
    test_info = test_info.copy()
    test_info["re"] = re_test
    test_info["prediction"] = predictions
    detection_results = {
        "re_test": re_test,
        "threshold": threshold,
        "predictions": predictions,
        "test_info": test_info,
    }
    detection_file = os.path.join(OUTPUT_DIR, "detection_results.pkl")
    with open(detection_file, "wb") as f:
        pickle.dump(detection_results, f)

    # ============================================================
    # STEP 5: EVALUASI MODEL
    # ============================================================
    from evaluate import evaluate_model
    metrics = evaluate_model(y_test, predictions, re_scores=re_test)

    metrics_file = os.path.join(OUTPUT_DIR, "evaluation_metrics.pkl")
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)

    # ============================================================
    # RINGKASAN
    # ============================================================
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("  PIPELINE SELESAI!")
    print("=" * 70)
    print(f"\n  Waktu total  : {total_time:.1f} detik ({total_time/60:.1f} menit)")
    print(f"\n  Hasil Evaluasi:")
    print(f"  - Precision  : {metrics['precision']:.4f}")
    print(f"  - Recall     : {metrics['recall']:.4f}")
    print(f"  - F1-Score   : {metrics['f1_score']:.4f}")
    if metrics.get('auc_roc') is not None:
        print(f"  - AUC-ROC    : {metrics['auc_roc']:.4f}")
    print(f"\n  Output files:")
    print(f"  - Model      : {MODEL_FILE}")
    print(f"  - Figures    : {FIGURE_DIR}/")
    print(f"  - Data       : {DATA_DIR}/")
    print(f"\n  Untuk dashboard:")
    print(f"  streamlit run dashboard.py")
    print("=" * 70)

    # ============================================================
    # STEP 6: DASHBOARD (opsional)
    # ============================================================
    if args.dashboard:
        print("\n  Menjalankan dashboard Streamlit...")
        os.system("streamlit run dashboard.py")


if __name__ == "__main__":
    main()
