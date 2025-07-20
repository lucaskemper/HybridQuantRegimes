from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler


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
        self.scaler = RobustScaler()  # Changed to RobustScaler
        self._is_fitted = False
        self.n_features = 15  # Fixed to exactly 15 features
        self.model = self._build_model()

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
        dense1 = tf.keras.layers.Dense(
            self.config.hidden_dims[3], 
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
            weight_decay=0.01
        )
        
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model

    def _prepare_enhanced_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare enhanced features for LSTM with better preprocessing"""
        # Validate input
        if len(returns) < self.config.sequence_length:
            raise ValueError(f"Input length must be >= {self.config.sequence_length}")

        features = pd.DataFrame(index=returns.index)
        
        # Basic features with better handling of extreme values
        features["returns"] = returns
        features["log_returns"] = np.log(np.abs(returns) + 1e-8)  # Prevent log(0)
        
        # Volatility features with clipping
        vol_21 = returns.rolling(window=21).std()
        vol_21 = vol_21.clip(lower=1e-8, upper=vol_21.quantile(0.99))  # Clip extreme values
        features["volatility"] = vol_21
        features["ewm_volatility"] = returns.ewm(span=21).std().clip(lower=1e-8)
        
        # Momentum features with clipping (reduced to fit 15 features)
        for period in [5, 20]:  # Reduced from 4 periods to 2
            momentum = returns.rolling(window=period).mean()
            momentum = momentum.clip(lower=momentum.quantile(0.01), upper=momentum.quantile(0.99))
            features[f"momentum_{period}d"] = momentum
            
            roc = (returns / returns.shift(period) - 1)
            roc = roc.clip(lower=-1, upper=10)  # Clip extreme ROC values
            features[f"roc_{period}d"] = roc
        
        # Technical indicators with better handling
        features["rsi_14"] = self._calculate_rsi(returns, 14)
        features["rsi_30"] = self._calculate_rsi(returns, 30)
        features["macd_signal"] = self._calculate_macd(returns)
        features["bollinger_position"] = self._calculate_bollinger_position(returns)
        features["williams_r"] = self._calculate_williams_r(returns)
        
        # Skewness and kurtosis with clipping
        skew = returns.rolling(window=21).skew()
        skew = skew.clip(lower=-5, upper=5)  # Clip extreme skewness
        features["skewness"] = skew
        
        kurt = returns.rolling(window=21).kurt()
        kurt = kurt.clip(lower=-10, upper=10)  # Clip extreme kurtosis
        features["kurtosis"] = kurt

        # Forward fill then backward fill any remaining NaN values
        features = features.ffill().bfill()
        
        # Ensure exactly 15 features
        if len(features.columns) > 15:
            # Keep only the first 15 features
            features = features.iloc[:, :15]
        elif len(features.columns) < 15:
            # Pad with zeros if needed
            while len(features.columns) < 15:
                features[f"pad_{len(features.columns)}"] = 0
        
        # Additional clipping to prevent infinity values
        for col in features.columns:
            features[col] = features[col].clip(
                lower=features[col].quantile(0.001),
                upper=features[col].quantile(0.999)
            )

        # Scale features with robust scaling
        scaled_features = self.scaler.fit_transform(features) if not self._is_fitted else self.scaler.transform(features)
        
        # Final clipping to ensure no infinity values
        scaled_features = np.clip(scaled_features, -10, 10)
        
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
        
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI

    @staticmethod
    def _calculate_macd(returns: pd.Series) -> pd.Series:
        """Calculate MACD signal line"""
        ema12 = returns.ewm(span=12).mean()
        ema26 = returns.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return signal

    @staticmethod
    def _calculate_bollinger_position(returns: pd.Series) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = returns.rolling(20).mean()
        std = returns.rolling(20).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        position = (returns - lower) / (upper - lower)
        return position

    @staticmethod
    def _calculate_williams_r(returns: pd.Series) -> pd.Series:
        """Calculate Williams %R"""
        high = returns.rolling(14).max()
        low = returns.rolling(14).min()
        williams_r = ((high - returns) / (high - low)) * -100
        return williams_r

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
            
            # Gradient clipping
            if self.config.gradient_clipping > 0:
                optimizer = self.model.optimizer
                optimizer.clipnorm = self.config.gradient_clipping

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
        # Load Keras model
        self.model = keras.models.load_model(os.path.join(path, "lstm_model.h5"))
        # Load scaler
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        # Load config
        self.config = joblib.load(os.path.join(path, "config.pkl"))
        self._is_fitted = True


class TransformerRegimeDetector:
    """Transformer-based market regime detection"""

    def __init__(self, config: DeepLearningConfig):
        self.config = config
        self.scaler = RobustScaler()
        self._is_fitted = False
        self.n_features = 15  # Fixed to exactly 15 features
        self.model = self._build_model()

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
            learning_rate=self.config.learning_rate
        )
        
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
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
        """Prepare features for Transformer with better preprocessing"""
        if len(returns) < self.config.sequence_length:
            raise ValueError(f"Input length must be >= {self.config.sequence_length}")

        features = pd.DataFrame(index=returns.index)
        
        # Basic features with better handling of extreme values
        features["returns"] = returns
        features["log_returns"] = np.log(np.abs(returns) + 1e-8)  # Prevent log(0)
        
        # Volatility features with clipping
        vol_21 = returns.rolling(window=21).std()
        vol_21 = vol_21.clip(lower=1e-8, upper=vol_21.quantile(0.99))  # Clip extreme values
        features["volatility"] = vol_21
        features["ewm_volatility"] = returns.ewm(span=21).std().clip(lower=1e-8)
        
        # Momentum features with clipping (reduced to fit 15 features)
        for period in [5, 20]:  # Reduced from 4 periods to 2
            momentum = returns.rolling(window=period).mean()
            momentum = momentum.clip(lower=momentum.quantile(0.01), upper=momentum.quantile(0.99))
            features[f"momentum_{period}d"] = momentum
            
            roc = (returns / returns.shift(period) - 1)
            roc = roc.clip(lower=-1, upper=10)  # Clip extreme ROC values
            features[f"roc_{period}d"] = roc
        
        # Technical indicators with better handling
        features["rsi_14"] = self._calculate_rsi(returns, 14)
        features["rsi_30"] = self._calculate_rsi(returns, 30)
        features["macd_signal"] = self._calculate_macd(returns)
        features["bollinger_position"] = self._calculate_bollinger_position(returns)
        features["williams_r"] = self._calculate_williams_r(returns)
        
        # Skewness and kurtosis with clipping
        skew = returns.rolling(window=21).skew()
        skew = skew.clip(lower=-5, upper=5)  # Clip extreme skewness
        features["skewness"] = skew
        
        kurt = returns.rolling(window=21).kurt()
        kurt = kurt.clip(lower=-10, upper=10)  # Clip extreme kurtosis
        features["kurtosis"] = kurt

        # Forward fill then backward fill any remaining NaN values
        features = features.ffill().bfill()
        
        # Ensure exactly 15 features
        if len(features.columns) > 15:
            # Keep only the first 15 features
            features = features.iloc[:, :15]
        elif len(features.columns) < 15:
            # Pad with zeros if needed
            while len(features.columns) < 15:
                features[f"pad_{len(features.columns)}"] = 0

        # Additional clipping to prevent infinity values
        for col in features.columns:
            features[col] = features[col].clip(
                lower=features[col].quantile(0.001),
                upper=features[col].quantile(0.999)
            )

        # Scale features with robust scaling
        scaled_features = self.scaler.fit_transform(features) if not self._is_fitted else self.scaler.transform(features)
        
        # Final clipping to ensure no infinity values
        scaled_features = np.clip(scaled_features, -10, 10)
        
        return scaled_features

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """Create sequences for Transformer"""
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
        
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def _calculate_macd(returns: pd.Series) -> pd.Series:
        """Calculate MACD signal line"""
        ema12 = returns.ewm(span=12).mean()
        ema26 = returns.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return signal

    @staticmethod
    def _calculate_bollinger_position(returns: pd.Series) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = returns.rolling(20).mean()
        std = returns.rolling(20).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        position = (returns - lower) / (upper - lower)
        return position

    @staticmethod
    def _calculate_williams_r(returns: pd.Series) -> pd.Series:
        """Calculate Williams %R"""
        high = returns.rolling(14).max()
        low = returns.rolling(14).min()
        williams_r = ((high - returns) / (high - low)) * -100
        return williams_r

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
        # Load Keras model
        self.model = keras.models.load_model(os.path.join(path, "transformer_model.h5"))
        # Load scaler
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        # Load config
        self.config = joblib.load(os.path.join(path, "config.pkl"))
        self._is_fitted = True
