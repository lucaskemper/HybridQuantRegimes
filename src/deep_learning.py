from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DeepLearningConfig:
    """Configuration for deep learning models"""

    sequence_length: int = 21  # Lookback window
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    n_regimes: int = 3
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    validation_split: float = 0.2
    early_stopping_patience: int = 10


class LSTMRegimeDetector:
    """LSTM-based market regime detection"""

    def __init__(self, config: DeepLearningConfig):
        self.config = config
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.n_features = 7  # Number of features we'll create
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                self.config.hidden_dims[0],
                input_shape=(self.config.sequence_length, self.n_features),
                return_sequences=True
            ),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.LSTM(self.config.hidden_dims[1]),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(self.config.n_regimes, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare features for LSTM"""
        # Validate input
        if len(returns) < self.config.sequence_length:
            raise ValueError(f"Input length must be >= {self.config.sequence_length}")

        features = pd.DataFrame(index=returns.index)
        
        # Basic features
        features["returns"] = returns
        features["volatility"] = returns.rolling(window=21).std()
        features["ewm_volatility"] = returns.ewm(span=21).std()
        features["momentum"] = returns.rolling(window=21).mean()
        features["rsi"] = self._calculate_rsi(returns)
        features["skewness"] = returns.rolling(window=21).skew()
        features["kurtosis"] = returns.rolling(window=21).kurt()

        # Forward fill then backward fill any remaining NaN values
        features = features.fillna(method="ffill").fillna(method="bfill")

        # Scale features
        scaled_features = self.scaler.fit_transform(features) if not self._is_fitted else self.scaler.transform(features)
        
        return scaled_features

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM"""
        X = []
        for i in range(len(features) - self.config.sequence_length):
            sequence = features[i:(i + self.config.sequence_length)]
            X.append(sequence)
        return np.array(X)

    @staticmethod
    def _calculate_rsi(returns: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = returns.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI

    def fit(self, returns: pd.Series, initial_regimes: np.ndarray) -> None:
        """
        Fit the LSTM model using initial regime classifications
        
        Args:
            returns: Time series of returns
            initial_regimes: Initial regime classifications (e.g., from HMM)
        """
        try:
            # Prepare features
            features = self._prepare_features(returns)
            
            # Create sequences
            X = self._create_sequences(features)
            
            # Prepare labels (shift by sequence_length to align with sequences)
            y = initial_regimes[self.config.sequence_length:]
            y = tf.keras.utils.to_categorical(y, self.config.n_regimes)
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.validation_split,
                shuffle=False
            )

            # Create early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            )

            # Train model
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=[early_stopping],
                verbose=1
            )

            self._is_fitted = True

        except Exception as e:
            self._is_fitted = False
            raise RuntimeError(f"Training failed: {str(e)}") from e

    def predict(self, returns: pd.Series) -> pd.Series:
        """Predict market regimes"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Prepare features
            features = self._prepare_features(returns)
            
            # Create sequences
            X = self._create_sequences(features)
            
            # Make predictions
            predictions = self.model.predict(X)
            regime_predictions = np.argmax(predictions, axis=1)
            
            # Create Series with proper index
            prediction_index = returns.index[self.config.sequence_length:]
            regime_series = pd.Series(
                regime_predictions,
                index=prediction_index,
                name="regime"
            )
            
            return regime_series

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

    def predict_proba(self, returns: pd.Series) -> pd.DataFrame:
        """Predict regime probabilities"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Prepare features
            features = self._prepare_features(returns)
            
            # Create sequences
            X = self._create_sequences(features)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Create DataFrame with proper index
            prediction_index = returns.index[self.config.sequence_length:]
            regime_probs = pd.DataFrame(
                predictions,
                index=prediction_index,
                columns=[f"regime_{i}" for i in range(self.config.n_regimes)]
            )
            
            return regime_probs

        except Exception as e:
            raise RuntimeError(f"Probability prediction failed: {str(e)}") from e
