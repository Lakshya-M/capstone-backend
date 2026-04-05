"""
Definition of the LSTM autoencoder model used for anomaly detection.
"""

from __future__ import annotations

from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed


def build_lstm_autoencoder(
    input_shape: Tuple[int, int],
    latent_dim: int = 32,
) -> Model:
    """
    Build a simple LSTM autoencoder for time-series reconstruction.

    Args:
        input_shape (Tuple[int, int]): Tuple of (timesteps, num_features).
        latent_dim (int): Dimension of the latent space.

    Returns:
        tensorflow.keras.Model: Compiled LSTM autoencoder model.
    """
    timesteps, num_features = input_shape

    inputs = Input(shape=(timesteps, num_features))

    # Encoder
    encoded = LSTM(latent_dim, activation="tanh", return_sequences=False)(inputs)

    # Repeat the latent vector across timesteps
    repeated = RepeatVector(timesteps)(encoded)

    # Decoder
    decoded = LSTM(latent_dim, activation="tanh", return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(num_features))(decoded)

    model = Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model

