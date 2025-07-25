from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

import logging

from src.features import calculate_enhanced_features

import tensorflow_probability as tfp

tfd = tfp.distributions


@dataclass
class DeepLearningConfig:
    """Configuration for deep learning models"""

    sequence_length: int = 21  # Lookback window
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32, 16])  # Enhanced architecture
    n_regimes: int = 4  # Updated to 4 regimes
    batch_size: int = 64  # Increased batch size
    epochs: int = 250  # More epochs for better convergence
    learning_rate: float = 0.0005  # Reduced learning rate
    dropout_rate: float = 0.3  # Increased dropout
    validation_split: float = 0.2
    early_stopping_patience: int = 30  # Increased patience
    
    # Enhanced features
    use_attention: bool = True
    bidirectional: bool = True
    residual_connections: bool = True
    batch_normalization: bool = True
    l2_regularization: float = 0.001
    gradient_clipping: float = 1.0
    
    # Learning rate schedule
    learning_rate_schedule: Dict[str, Any] = field(default_factory=lambda: {
        "type": "cosine_annealing",
        "T_max": 50,
        "eta_min": 0.00001
    })


class LSTMRegimeDetector:
    """Enhanced LSTM-based market regime detection"""

    def __init__(self, config: DeepLearningConfig):
        self.config = config
        self.sequence_length = config.sequence_length  # Ensure this attribute is always set
        self.scaler = RobustScaler()  # Changed to RobustScaler
        self._is_fitted = False
        self.n_features = 15  # Fixed to exactly 15 features
        self.model = self._build_model()
        self.logger = logging.getLogger(__name__)

    def _build_model(self) -> tf.keras.Model:
        """Build enhanced LSTM model architecture"""
        inputs = tf.keras.layers.Input(shape=(self.config.sequence_length, self.n_features))
        
        # First LSTM layer with attention
        if self.config.bidirectional:
            lstm1 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.config.hidden_dims[0],
                    return_sequences=True,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                )
            )(inputs)
        else:
            lstm1 = tf.keras.layers.LSTM(
                self.config.hidden_dims[0],
                return_sequences=True,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            )(inputs)
        
        if self.config.batch_normalization:
            lstm1 = tf.keras.layers.BatchNormalization()(lstm1)
        
        # Attention mechanism
        if self.config.use_attention:
            attention = tf.keras.layers.Attention()([lstm1, lstm1])
            lstm1 = tf.keras.layers.Add()([lstm1, attention])  # Residual connection
        
        # Second LSTM layer
        if self.config.bidirectional:
            lstm2 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.config.hidden_dims[1],
                    return_sequences=True,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                )
            )(lstm1)
        else:
            lstm2 = tf.keras.layers.LSTM(
                self.config.hidden_dims[1],
                return_sequences=True,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            )(lstm1)
        
        if self.config.batch_normalization:
            lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
        
        # Third LSTM layer
        if self.config.bidirectional:
            lstm3 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.config.hidden_dims[2],
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                )
            )(lstm2)
        else:
            lstm3 = tf.keras.layers.LSTM(
                self.config.hidden_dims[2],
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            )(lstm2)
        
        if self.config.batch_normalization:
            lstm3 = tf.keras.layers.BatchNormalization()(lstm3)
        
        # Dense layers
        # Use the last value in hidden_dims for the dense layer for flexibility
        dense1 = tf.keras.layers.Dense(
            self.config.hidden_dims[-1],
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)
        )(lstm3)
        dense1 = tf.keras.layers.Dropout(self.config.dropout_rate)(dense1)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.config.n_regimes, 
            activation="softmax"
        )(dense1)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with AdamW optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            clipnorm=self.config.gradient_clipping
        )
        try:
            from tensorflow_addons.metrics import F1Score
            metrics = [
                "accuracy",
                tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
                F1Score(num_classes=self.config.n_regimes, average="macro", name="f1_macro")
            ]
        except ImportError:
            metrics = ["accuracy", tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=metrics
        )
        
        return model

    def _prepare_enhanced_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare enhanced features for LSTM with better preprocessing (centralized)"""
        print("[LSTMRegimeDetector._prepare_enhanced_features] returns type:", type(returns))
        print("[LSTMRegimeDetector._prepare_enhanced_features] returns shape:", getattr(returns, 'shape', None))
        if isinstance(returns, pd.DataFrame):
            print("[LSTMRegimeDetector._prepare_enhanced_features] WARNING: DataFrame received, converting to first column.")
            returns = returns.iloc[:, 0]
        features = calculate_enhanced_features(returns)
        print("[LSTMRegimeDetector._prepare_enhanced_features] features type:", type(features))
        print("[LSTMRegimeDetector._prepare_enhanced_features] features shape:", getattr(features, 'shape', None))
        print("[LSTMRegimeDetector._prepare_enhanced_features] features columns:", getattr(features, 'columns', None))
        # Ensure exactly 15 features BEFORE scaling
        if len(features.columns) > 15:
            features = features.iloc[:, :15]
        elif len(features.columns) < 15:
            for _ in range(15 - len(features.columns)):
                features[f"pad_{len(features.columns)}"] = 0
        # Scale features with robust scaling
        scaled_features = self.scaler.fit_transform(features) if not self._is_fitted else self.scaler.transform(features)
        scaled_features = np.clip(scaled_features, -10, 10)
        return scaled_features

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM"""
        X = []
        for i in range(len(features) - self.config.sequence_length):
            sequence = features[i:(i + self.config.sequence_length)]
            X.append(sequence)
        return np.array(X)

    def fit(self, returns: pd.Series, initial_regimes: np.ndarray) -> None:
        """
        Fit the LSTM model using initial regime classifications
        
        Args:
            returns: Time series of returns
            initial_regimes: Initial regime classifications (e.g., from HMM)
        """
        try:
            # Prepare features
            features = self._prepare_enhanced_features(returns)
            
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

            # Logging training start
            self.logger.info(f"Starting LSTM training: X shape={X.shape}, y shape={y.shape}, epochs={self.config.epochs}")

            # Create callbacks
            callbacks = []
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
            # Learning rate schedule
            if self.config.learning_rate_schedule["type"] == "cosine_annealing":
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=self.config.learning_rate_schedule["eta_min"]
                )
                callbacks.append(lr_scheduler)
            
            # Train model
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )

            self._is_fitted = True

        except Exception as e:
            self._is_fitted = False
            self.logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}") from e

    def predict(self, returns: pd.Series) -> pd.Series:
        """Predict market regimes"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Prepare features
            features = self._prepare_enhanced_features(returns)
            
            # Create sequences
            X = self._create_sequences(features)
            
            # Make predictions
            predictions = self.model.predict(X)
            print('[LSTMRegimeDetector.predict] predictions shape:', getattr(predictions, 'shape', None), 'type:', type(predictions))
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
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
            features = self._prepare_enhanced_features(returns)
            # Create sequences
            X = self._create_sequences(features)
            # Make predictions
            predictions = self.model.predict(X)
            print('[LSTMRegimeDetector.predict_proba] predictions shape:', getattr(predictions, 'shape', None), 'type:', type(predictions))
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            # Use the index after the lookback window to match the number of predictions
            prediction_index = returns.index[self.config.sequence_length:]
            print('[LSTMRegimeDetector.predict_proba] prediction_index shape:', getattr(prediction_index, 'shape', None))
            regime_probs = pd.DataFrame(
                predictions,
                index=prediction_index,
                columns=[f"regime_{i}" for i in range(self.config.n_regimes)]
            )
            print('[LSTMRegimeDetector.predict_proba] regime_probs shape:', getattr(regime_probs, 'shape', None), 'columns:', regime_probs.columns)
            return regime_probs
        except Exception as e:
            raise RuntimeError(f"Probability prediction failed: {str(e)}") from e

    def predict_latest(self, returns: pd.Series) -> np.ndarray:
        """
        Predict regime probabilities for the latest point using the last sequence_length returns.
        Returns the regime probabilities for the latest point.
        """
        if len(returns) < self.config.sequence_length:
            raise ValueError(f"Need at least {self.config.sequence_length} points for real-time LSTM prediction.")
        features = self._prepare_enhanced_features(returns)
        last_window = features[-self.config.sequence_length:]
        X = np.expand_dims(last_window, axis=0)  # shape (1, sequence_length, n_features)
        predictions = self.model.predict(X)
        return predictions[0]  # regime probabilities for the latest point

    def save(self, path: str):
        """Save the LSTM model and scaler to the specified path"""
        import os
        import joblib
        os.makedirs(path, exist_ok=True)
        # Save Keras model
        self.model.save(os.path.join(path, "lstm_model.h5"))
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        # Save config
        joblib.dump(self.config, os.path.join(path, "config.pkl"))

    def load(self, path: str):
        """Load the LSTM model and scaler from the specified path"""
        import os
        import joblib
        from tensorflow import keras
        try:
            self.model = keras.models.load_model(os.path.join(path, "lstm_model.h5"))
            self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
            self.config = joblib.load(os.path.join(path, "config.pkl"))
            self._is_fitted = True
        except Exception as e:
            self._is_fitted = False
            self.logger.error(f"Failed to load model or dependencies: {e}")
            raise RuntimeError(f"Failed to load model or dependencies: {e}")


class TransformerRegimeDetector:
    """Transformer-based market regime detection"""

    def __init__(self, config: DeepLearningConfig):
        self.config = config
        self.scaler = RobustScaler()
        self._is_fitted = False
        self.n_features = 15  # Fixed to exactly 15 features
        self.model = self._build_model()
        self.logger = logging.getLogger(__name__)

    def _build_model(self) -> tf.keras.Model:
        """Build Transformer model architecture"""
        inputs = tf.keras.layers.Input(shape=(self.config.sequence_length, self.n_features))
        
        # Positional encoding
        pos_encoding = self._positional_encoding(self.config.sequence_length, self.n_features)
        inputs_with_pos = inputs + pos_encoding
        
        # Transformer encoder layers
        x = inputs_with_pos
        for i in range(4):  # 4 transformer layers
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=8,
                key_dim=16,
                dropout=self.config.dropout_rate
            )(x, x)
            
            # Add & Norm
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed forward
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation="gelu"),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(self.n_features)
            ])
            
            # Add & Norm
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(128, activation="gelu")(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        x = tf.keras.layers.Dense(64, activation="gelu")(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.config.n_regimes, 
            activation="softmax"
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.gradient_clipping
        )
        try:
            from tensorflow_addons.metrics import F1Score
            metrics = [
                "accuracy",
                tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
                F1Score(num_classes=self.config.n_regimes, average="macro", name="f1_macro")
            ]
        except ImportError:
            metrics = ["accuracy", tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=metrics
        )
        
        return model

    def _positional_encoding(self, position: int, d_model: int) -> np.ndarray:
        """Generate positional encoding for transformer"""
        pos_encoding = np.zeros((position, d_model))
        for pos in range(position):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        return pos_encoding

    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare features for Transformer with better preprocessing (centralized)"""
        features = calculate_enhanced_features(returns)
        # Ensure exactly 15 features BEFORE scaling
        if len(features.columns) > 15:
            features = features.iloc[:, :15]
        elif len(features.columns) < 15:
            for _ in range(15 - len(features.columns)):
                features[f"pad_{len(features.columns)}"] = 0
        scaled_features = self.scaler.fit_transform(features) if not self._is_fitted else self.scaler.transform(features)
        scaled_features = np.clip(scaled_features, -10, 10)
        return scaled_features

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """Create sequences for Transformer"""
        X = []
        for i in range(len(features) - self.config.sequence_length):
            sequence = features[i:(i + self.config.sequence_length)]
            X.append(sequence)
        return np.array(X)

    def fit(self, returns: pd.Series, initial_regimes: np.ndarray) -> None:
        """Fit the Transformer model"""
        try:
            # Prepare features
            features = self._prepare_features(returns)
            
            # Create sequences
            X = self._create_sequences(features)
            
            # Prepare labels
            y = initial_regimes[self.config.sequence_length:]
            y = tf.keras.utils.to_categorical(y, self.config.n_regimes)
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.validation_split,
                shuffle=False
            )

            # Logging training start
            self.logger.info(f"Starting Transformer training: X shape={X.shape}, y shape={y.shape}, epochs={self.config.epochs}")

            # Create callbacks
            callbacks = []
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

            # Train model
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )

            self._is_fitted = True

        except Exception as e:
            self._is_fitted = False
            self.logger.error(f"Training failed: {str(e)}")
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

    def predict_latest(self, returns: pd.Series) -> np.ndarray:
        """Predict regime probabilities for the latest point"""
        if len(returns) < self.config.sequence_length:
            raise ValueError(f"Need at least {self.config.sequence_length} points for real-time Transformer prediction.")
        features = self._prepare_features(returns)
        last_window = features[-self.config.sequence_length:]
        X = np.expand_dims(last_window, axis=0)
        predictions = self.model.predict(X)
        return predictions[0]

    def save(self, path: str):
        """Save the Transformer model and scaler"""
        import os
        import joblib
        os.makedirs(path, exist_ok=True)
        # Save Keras model
        self.model.save(os.path.join(path, "transformer_model.h5"))
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        # Save config
        joblib.dump(self.config, os.path.join(path, "config.pkl"))

    def load(self, path: str):
        """Load the Transformer model and scaler"""
        import os
        import joblib
        from tensorflow import keras
        try:
            self.model = keras.models.load_model(os.path.join(path, "transformer_model.h5"))
            self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
            self.config = joblib.load(os.path.join(path, "config.pkl"))
            self._is_fitted = True
        except Exception as e:
            self._is_fitted = False
            self.logger.error(f"Failed to load model or dependencies: {e}")
            raise RuntimeError(f"Failed to load model or dependencies: {e}")


class BayesianLSTMRegimeForecaster:
    """Bayesian LSTM for risk forecasting with predictive uncertainty."""
    def __init__(self, config: DeepLearningConfig):
        self.config = config
        self.n_features = 15
        self.model = self._build_bayesian_lstm_model()
        self.scaler = RobustScaler()
        self._is_fitted = False

    def _build_bayesian_lstm_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=(self.config.sequence_length, self.n_features))
        # Bayesian DenseVariational layer as input
        x = tfp.layers.DenseVariational(
            units=self.config.hidden_dims[0],
            make_prior_fn=tfp.layers.default_mean_field_normal_fn(),
            make_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
            kl_weight=1/self.config.sequence_length,
            activation='tanh'
        )(inputs)
        # Standard LSTM layer
        x = tf.keras.layers.LSTM(self.config.hidden_dims[1], return_sequences=False)(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        # Output: Predict mean and std for risk metric (e.g., volatility)
        mean = tf.keras.layers.Dense(1)(x)
        std = tf.keras.layers.Dense(1, activation='softplus')(x)
        outputs = tf.keras.layers.Concatenate()([mean, std])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=self._nll_loss)
        return model

    def _nll_loss(self, y_true, y_pred):
        mean = y_pred[:, 0]
        std = y_pred[:, 1] + 1e-6
        dist = tfd.Normal(loc=mean, scale=std)
        return -dist.log_prob(y_true)

    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        features = calculate_enhanced_features(returns)
        # Ensure exactly 15 features BEFORE scaling
        if len(features.columns) > 15:
            features = features.iloc[:, :15]
        elif len(features.columns) < 15:
            for _ in range(15 - len(features.columns)):
                features[f"pad_{len(features.columns)}"] = 0
        scaled_features = self.scaler.fit_transform(features) if not self._is_fitted else self.scaler.transform(features)
        scaled_features = np.clip(scaled_features, -10, 10)
        return scaled_features

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        X = []
        for i in range(len(features) - self.config.sequence_length):
            sequence = features[i:(i + self.config.sequence_length)]
            X.append(sequence)
        return np.array(X)

    def fit(self, returns: pd.Series, y: np.ndarray) -> None:
        """
        Fit the Bayesian LSTM model to predict a risk metric (e.g., next-period volatility or VaR).
        Args:
            returns: pd.Series of returns
            y: np.ndarray of target risk metric (aligned with sequences)
        """
        features = self._prepare_features(returns)
        X = self._create_sequences(features)
        # Align y to match X
        y = y[self.config.sequence_length:]
        self.model.fit(X, y, epochs=self.config.epochs, batch_size=self.config.batch_size, verbose=1)
        self._is_fitted = True

    def predict(self, returns: pd.Series, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Predict risk metric with uncertainty. Returns predictive mean and std for each sequence.
        Args:
            returns: pd.Series of returns
            n_samples: Number of MC samples for uncertainty estimation
        Returns:
            Dict with 'mean', 'std', and 'all_samples' arrays
        """
        features = self._prepare_features(returns)
        X = self._create_sequences(features)
        preds = []
        for _ in range(n_samples):
            pred = self.model(X, training=True).numpy()
            preds.append(pred)
        preds = np.stack(preds, axis=0)  # shape (n_samples, n_obs, 2)
        mean_pred = preds[..., 0].mean(axis=0)
        std_pred = preds[..., 0].std(axis=0)
        return {'mean': mean_pred, 'std': std_pred, 'all_samples': preds}

    def predict_latest(self, returns: pd.Series, n_samples: int = 100) -> Dict[str, float]:
        """
        Predict risk metric for the latest window with uncertainty.
        Returns dict with mean and std.
        """
        features = self._prepare_features(returns)
        last_window = features[-self.config.sequence_length:]
        X = np.expand_dims(last_window, axis=0)
        preds = []
        for _ in range(n_samples):
            pred = self.model(X, training=True).numpy()[0]
            preds.append(pred)
        preds = np.stack(preds, axis=0)  # shape (n_samples, 2)
        mean_pred = preds[:, 0].mean()
        std_pred = preds[:, 0].std()
        return {'mean': float(mean_pred), 'std': float(std_pred), 'all_samples': preds}
