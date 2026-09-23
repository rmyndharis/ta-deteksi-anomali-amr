"""
Training Model LSTM Autoencoder
================================
Melatih model LSTM Autoencoder pada data konsumsi energi listrik normal.
Model belajar merekonstruksi pola normal sehingga data anomali
akan menghasilkan reconstruction error yang tinggi.

Catatan: pendekatan ini semi-supervised. Label P2TL dipakai untuk
KURASI data latih (hanya window berlabel normal yang dilatihkan),
tetapi tidak pernah masuk ke loss function; model murni belajar
merekonstruksi pola normal.

Validasi (untuk early stopping) dipisah BERDASARKAN METER (group split):
VALIDATION_SPLIT dari seluruh meter latih ditahan sebagai meter validasi,
sehingga window dari meter yang sama tidak pernah muncul di data latih
dan validasi sekaligus. Loss per kanal fitur dihitung setelah pelatihan
(bobot terbaik) dan disimpan bersama riwayat pelatihan.

Penulis: Yudhi Armyndharis (220401010272)
"""

import json
import numpy as np
import pickle
import time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import (
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    MODEL_FILE, TRAINING_HISTORY_FILE, TRAINING_LOSS_FILE,
    MODEL_DIR, OUTPUT_DIR, RANDOM_SEED, MODEL_FEATURES
)
from model import build_lstm_autoencoder, mask_energy_channels

PER_CHANNEL_LOSS_FILE = os.path.join(OUTPUT_DIR, "per_channel_loss.json")


def prepare_training_data(X_train, y_train):
    """
    Persiapkan data training.
    PENTING: Hanya window berlabel NORMAL (P2TL = 0) yang dilatihkan.
    Model belajar pola normal -> anomali = reconstruction error tinggi.
    """
    # Filter hanya sequence normal (y=0) untuk training
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    print(f"  Total training sequences  : {len(X_train):,}")
    print(f"  Normal sequences (dipakai): {len(X_train_normal):,}")
    print(f"  Anomali sequences (skip)  : {(~normal_mask).sum():,}")
    print(f"  Input shape               : {X_train_normal.shape}")

    return X_train_normal


def split_validation_by_meter(X, meter_ids, fraction=VALIDATION_SPLIT,
                              seed=RANDOM_SEED):
    """
    Group split: pilih `fraction` dari METER (bukan window) sebagai meter
    validasi, sehingga tidak ada meter yang muncul di kedua himpunan.
    """
    meter_ids = np.asarray(meter_ids).astype(str)
    meters = np.unique(meter_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(meters)
    n_val = max(1, int(round(len(meters) * fraction)))
    val_meters = set(meters[:n_val])
    val_mask = np.isin(meter_ids, list(val_meters))
    return X[~val_mask], X[val_mask], sorted(val_meters), val_mask


def per_channel_loss(model, X, batch_size=512):
    """MSE rekonstruksi per kanal fitur (input di-mask, target window utuh)."""
    if len(X) == 0:
        return {}
    rec = model.predict(mask_energy_channels(X), batch_size=batch_size, verbose=0)
    mse = ((rec - X) ** 2).mean(axis=(0, 1))
    return {f: float(v) for f, v in zip(MODEL_FEATURES, mse)}


def train_model(X_train, y_train, train_info=None):
    """
    Melatih model LSTM Autoencoder.

    Args:
        X_train: Training sequences (n_samples, seq_length, n_features)
        y_train: Labels (hanya untuk filter, TIDAK untuk training)
        train_info: DataFrame info per window (kolom meter_id) untuk
                    group split validasi per meter; bila None, dipakai
                    validation_split biasa (acak per window)

    Returns:
        model: Trained model
        history: Training history (dict), termasuk meter validasi dan
                 loss per kanal pada bobot terbaik
    """
    print("=" * 70)
    print("TRAINING MODEL LSTM AUTOENCODER")
    print("=" * 70)

    # 0. Seed numpy + tensorflow agar training reproducible
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # 1. Persiapkan data (hanya normal)
    X_train_normal = prepare_training_data(X_train, y_train)
    meters_normal = None
    if train_info is not None and "meter_id" in train_info:
        meters_normal = train_info["meter_id"].to_numpy()[np.asarray(y_train) == 0]

    # 1b. Group split validasi per meter
    if meters_normal is not None:
        X_fit, X_val, val_meters, _ = split_validation_by_meter(
            X_train_normal, meters_normal)
        print(f"  Validasi per meter        : {len(val_meters):,} meter "
              f"({VALIDATION_SPLIT:.0%}), {len(X_val):,} window; "
              f"latih {len(X_fit):,} window dari "
              f"{len(np.unique(meters_normal)) - len(val_meters):,} meter")
    else:
        X_fit, X_val, val_meters = X_train_normal, None, []
        print(f"  Validasi                  : validation_split={VALIDATION_SPLIT} "
              f"(acak per window; tanpa info meter)")

    # 2. Build model
    n_features = X_train_normal.shape[2]
    sequence_length = X_train_normal.shape[1]
    model = build_lstm_autoencoder(sequence_length, n_features)

    # 3. Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]

    # 4. Training
    # Autoencoder: input = output (model belajar merekonstruksi input)
    print(f"\nTraining dimulai...")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Epochs (max)     : {EPOCHS}")
    print(f"  Validation       : {VALIDATION_SPLIT:.0%} "
          f"{'meter (group split)' if X_val is not None else 'window (acak)'}")
    print(f"  Early stopping   : patience={EARLY_STOPPING_PATIENCE}")

    start_time = time.time()

    # Masked autoencoder: kanal energi disembunyikan dari input;
    # target = window utuh, sehingga model belajar merekonstruksi
    # profil konsumsi normal dari konteks (bukan menyalin input).
    fit_kwargs = dict(epochs=EPOCHS, batch_size=BATCH_SIZE,
                      callbacks=callbacks, shuffle=True, verbose=1)
    if X_val is not None:
        fit_kwargs["validation_data"] = (mask_energy_channels(X_val), X_val)
    else:
        fit_kwargs["validation_split"] = VALIDATION_SPLIT
    history = model.fit(
        mask_energy_channels(X_fit),   # Input (energi di-mask)
        X_fit,                         # Target = window utuh
        **fit_kwargs
    )

    training_time = time.time() - start_time

    # 5. Simpan model dan history (bobot terbaik sudah dipulihkan oleh
    #    EarlyStopping restore_best_weights=True)
    model.save(MODEL_FILE)
    print(f"\nModel disimpan: {MODEL_FILE}")

    hist = dict(history.history)
    hist["val_meters"] = list(val_meters)
    hist["n_train_windows"] = int(len(X_fit))
    hist["n_val_windows"] = int(len(X_val)) if X_val is not None else None

    # 5b. Loss per kanal fitur pada bobot terbaik (latih dan validasi)
    pcl = {"train": per_channel_loss(model, X_fit),
           "val": per_channel_loss(model, X_val) if X_val is not None else {}}
    for split, d in pcl.items():
        tot = sum(d.values())
        if tot > 0:
            d_share = {k: v / tot for k, v in d.items()}
            print(f"  Loss per kanal ({split}): " + ", ".join(
                f"{k}={v:.4f} ({d_share[k]:.0%})" for k, v in d.items()))
    hist["per_channel_loss"] = pcl
    os.makedirs(os.path.dirname(PER_CHANNEL_LOSS_FILE), exist_ok=True)
    with open(PER_CHANNEL_LOSS_FILE, "w") as f:
        json.dump(pcl, f, indent=2)
    print(f"  Loss per kanal disimpan: {PER_CHANNEL_LOSS_FILE}")

    with open(TRAINING_HISTORY_FILE, "wb") as f:
        pickle.dump(hist, f)
    print(f"Training history disimpan: {TRAINING_HISTORY_FILE}")

    # 6. Print ringkasan training
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    best_epoch = np.argmin(history.history["val_loss"]) + 1

    print(f"\n{'='*70}")
    print(f"TRAINING SELESAI")
    print(f"{'='*70}")
    print(f"  Waktu training    : {training_time:.1f} detik ({training_time/60:.1f} menit)")
    print(f"  Epochs dijalankan : {len(history.history['loss'])}")
    print(f"  Best epoch        : {best_epoch}")
    print(f"  Final train loss  : {final_train_loss:.6f}")
    print(f"  Final val loss    : {final_val_loss:.6f}")
    print(f"  Overfitting check : {'OK (val ≈ train)' if abs(final_val_loss - final_train_loss) < final_train_loss * 0.5 else 'WARNING: possible overfitting'}")
    print(f"{'='*70}")

    # 7. Plot training loss
    plot_training_history(history.history)

    return model, hist


def plot_training_history(history):
    """Plot kurva training loss dan validation loss."""
    plt.figure(figsize=(10, 6))

    from visualize import sumbu_koma, koma

    epochs = np.arange(1, len(history["loss"]) + 1)
    plt.plot(epochs, history["loss"], label="Loss data latih", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Loss data validasi", linewidth=2)

    best_idx = int(np.argmin(history["val_loss"]))
    best_epoch = best_idx + 1
    best_val_loss = history["val_loss"][best_idx]
    plt.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5,
                label=f"Epoch terbaik ({best_epoch}), loss validasi {koma(best_val_loss, 4)}")
    plt.scatter([best_epoch], [best_val_loss], color="red", zorder=5, s=100)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Kurva Loss Data Latih dan Data Validasi\nModel LSTM Autoencoder", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    sumbu_koma(plt.gca())
    plt.tight_layout()

    plt.savefig(TRAINING_LOSS_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot training loss disimpan: {TRAINING_LOSS_FILE}")


if __name__ == "__main__":
    # Load processed data
    import pickle as pkl
    from config import DATA_DIR

    processed_file = os.path.join(DATA_DIR, "processed_sequences.pkl")
    with open(processed_file, "rb") as f:
        data = pkl.load(f)

    model, history = train_model(data["X_train"], data["y_train"],
                                 data.get("train_info"))
