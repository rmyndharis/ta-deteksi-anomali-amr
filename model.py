"""
LSTM Autoencoder Model
=======================
Arsitektur model LSTM Autoencoder untuk deteksi anomali
konsumsi energi listrik.

Arsitektur:
- Encoder: LSTM(64) → LSTM(32) → Latent(16)
- Decoder: LSTM(32) → LSTM(64) → TimeDistributed(Dense)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

Penulis: Yudhi Armyndharis (220401010272)
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense
)
from tensorflow.keras.optimizers import Adam

from config import (
    ENCODER_LSTM_1_UNITS, ENCODER_LSTM_2_UNITS, LATENT_DIM,
    DECODER_LSTM_1_UNITS, DECODER_LSTM_2_UNITS,
    SEQUENCE_LENGTH, LEARNING_RATE, LOSS_FUNCTION,
    MODEL_FEATURES, MASKED_FEATURES
)

MASKED_IDX = [MODEL_FEATURES.index(f) for f in MASKED_FEATURES]


def mask_energy_channels(X):
    """
    Kembalikan salinan X dengan kanal energi (MASKED_FEATURES) di-nol-kan.
    Pada skala robust per meter, nol = nilai median meter tersebut, sehingga
    model tidak melihat level konsumsi aktual hari itu dan dipaksa
    merekonstruksi profil normal dari konteks.
    """
    import numpy as np
    X_masked = np.array(X, copy=True)
    X_masked[:, :, MASKED_IDX] = 0.0
    return X_masked


def build_lstm_autoencoder(sequence_length=SEQUENCE_LENGTH, n_features=None):
    """
    Membangun model LSTM Autoencoder.

    Arsitektur:
    Input (48, 6) → Encoder LSTM(64) → Encoder LSTM(32) → Latent(16)
    → RepeatVector(48) → Decoder LSTM(32) → Decoder LSTM(64)
    → TimeDistributed Dense(6)

    Args:
        sequence_length: Panjang sequence input (default: 48)
        n_features: Jumlah fitur (default: len(MODEL_FEATURES) = 6)

    Returns:
        model: Compiled Keras Model
    """
    if n_features is None:
        n_features = len(MODEL_FEATURES)

    print(f"\n{'='*70}")
    print("LSTM AUTOENCODER ARCHITECTURE")
    print(f"{'='*70}")
    print(f"Input shape: ({sequence_length}, {n_features})")

    # --- ENCODER ---
    inputs = Input(shape=(sequence_length, n_features), name="encoder_input")

    # Encoder LSTM Layer 1 (64 units)
    encoded = LSTM(
        ENCODER_LSTM_1_UNITS,
        activation="tanh",
        return_sequences=True,
        name="encoder_lstm_1"
    )(inputs)

    # Encoder LSTM Layer 2 (32 units)
    encoded = LSTM(
        ENCODER_LSTM_2_UNITS,
        activation="tanh",
        return_sequences=False,
        name="encoder_lstm_2"
    )(encoded)

    # Latent Representation (16 units)
    latent = Dense(
        LATENT_DIM,
        activation="relu",
        name="latent_representation"
    )(encoded)

    # --- DECODER ---
    # Repeat latent vector untuk setiap timestep
    decoded = RepeatVector(sequence_length, name="repeat_vector")(latent)

    # Decoder LSTM Layer 1 (32 units)
    decoded = LSTM(
        DECODER_LSTM_1_UNITS,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm_1"
    )(decoded)

    # Decoder LSTM Layer 2 (64 units)
    decoded = LSTM(
        DECODER_LSTM_2_UNITS,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm_2"
    )(decoded)

    # Output Layer: TimeDistributed Dense → rekonstruksi input
    outputs = TimeDistributed(
        Dense(n_features),
        name="output_reconstruction"
    )(decoded)

    # Build model
    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Autoencoder")

    # Compile
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)

    # Print summary
    model.summary()

    print(f"\nEncoder: {ENCODER_LSTM_1_UNITS} → {ENCODER_LSTM_2_UNITS} → {LATENT_DIM}")
    print(f"Decoder: {DECODER_LSTM_1_UNITS} → {DECODER_LSTM_2_UNITS} → {n_features}")
    print(f"Loss: {LOSS_FUNCTION} | Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"{'='*70}")

    return model


def load_model(model_path):
    """Load model yang sudah di-train.

    Bila arsip .keras tidak dapat dideserialisasi (perbedaan versi Keras),
    arsitektur dibangun ulang dari build_lstm_autoencoder() dan hanya
    bobotnya yang dimuat.
    """
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError) as e:
        print(f"  load_model gagal ({type(e).__name__}); membangun ulang "
              f"arsitektur dan memuat bobot saja.")
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_lstm_autoencoder(SEQUENCE_LENGTH, len(MODEL_FEATURES))
        model.load_weights(model_path)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)
    print(f"Model loaded from: {model_path}")
    return model


if __name__ == "__main__":
    model = build_lstm_autoencoder()
