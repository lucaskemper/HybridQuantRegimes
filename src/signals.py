# src/signals.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.regime import MarketRegimeDetector
from src.features import calculate_enhanced_features

class SignalGenerator:
    def __init__(self, lookback_fast: int = 10, lookback_slow: int = 21, normalize: bool = False, 
                 use_regime: bool = True, regime_detector: Optional[MarketRegimeDetector] = None,
                 regime_sharpe_multipliers: Optional[dict] = None,
                 scaler_k: float = 2.5,
                 scaling_method: str = 'tanh',
                 vol_normalize: bool = True):
        self.lookback_fast = lookback_fast
        self.lookback_slow = lookback_slow
        self.normalize = normalize
        self.use_regime = use_regime
        self.regime_detector = regime_detector
        self.regime_sharpe_multipliers = regime_sharpe_multipliers or {}
        self.scaler_k = scaler_k  # For smooth scaling
        self.scaling_method = scaling_method  # 'tanh' or 'clip'
        self.vol_normalize = vol_normalize

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        print("[DIAGNOSTIC] Entering generate_signals")
        """Generate regime-aware, feature-rich composite signals"""
        returns = market_data['returns']
        prices = market_data['prices']
        features_dict = market_data.get('features', None)
        # Diagnostics for returns
        print("[DIAGNOSTIC] Returns head:\n", returns.head(10))
        print("[DIAGNOSTIC] Returns describe:\n", returns.describe())
        # Diagnostics for features
        if features_dict is not None and len(features_dict) > 0:
            first_ticker = list(features_dict.keys())[0]
            print(f"[DIAGNOSTIC] Features head for {first_ticker}:\n", features_dict[first_ticker].head(10))
            print(f"[DIAGNOSTIC] Features describe for {first_ticker}:\n", features_dict[first_ticker].describe())
        signals = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

        required_features = [
            "momentum_20d", "rsi_14", "macd_signal", "semiconductor_pmi", "term_structure_slope",
            "realized_volatility", "williams_r"
        ]

        def safe_feature(features, key, default=0):
            if key in features:
                return features[key]
            else:
                raise KeyError(f"Feature '{key}' missing for ticker. Available: {list(features.columns)}")

        def z(x):
            return (x - x.mean()) / (x.std() + 1e-4)

        # Learn regime-specific feature weights using Ridge regression
        # Use the required features for learning
        regime_series_dict = {}
        if self.use_regime and self.regime_detector:
            for ticker in returns.columns:
                ticker_returns = returns[ticker]
                try:
                    regime_series_dict[ticker] = self.regime_detector.predict(ticker_returns, output_labels=True)
                except Exception:
                    regime_series_dict[ticker] = pd.Series([self.regime_detector.regime_labels[1]] * len(ticker_returns), index=ticker_returns.index)
        else:
            for ticker in returns.columns:
                regime_series_dict[ticker] = pd.Series(['Default'] * len(returns), index=returns.index)
        # Fit weights per regime
        feature_weights_per_regime = fit_regime_feature_weights_ridge(
            features_dict,
            returns,
            regime_series_dict,
            alpha=1.0,
            forward_lag=1
        )
        # Main signal generation loop
        for i, ticker in enumerate(returns.columns):
            ticker_price = prices[ticker]
            ticker_returns = returns[ticker]
            # Get features for this ticker
            if features_dict is not None and ticker in features_dict:
                features = features_dict[ticker]
            else:
                features = calculate_enhanced_features(ticker_returns)
            # Check all required features are present
            missing = [f for f in required_features if f not in features.columns]
            if missing:
                raise KeyError(f"Missing features for {ticker}: {missing}. Available: {list(features.columns)}")
            # Get regime series for this ticker
            regime_series = regime_series_dict[ticker]
            # For each time, get regime and use corresponding weights
            composite_signal = pd.Series(0.0, index=features.index)
            for idx in features.index:
                regime = regime_series.loc[idx] if idx in regime_series.index else list(feature_weights_per_regime.keys())[0]
                weights = feature_weights_per_regime.get(regime)
                if weights is None:
                    # Fallback: equal weights for required features
                    weights = {f: 1.0 / len(required_features) for f in required_features}
                # Weighted sum of features (z-scored)
                val = 0.0
                for feat, w in weights.items():
                    if feat in features.columns:
                        x = features.loc[idx, feat]
                        # Z-score normalization per feature
                        x = (x - features[feat].mean()) / (features[feat].std() + 1e-4)
                        val += w * x
                composite_signal.loc[idx] = val
            # DEBUG PATCH: Print diagnostics for the first ticker only
            if i == 0:
                print(f"[DEBUG] Features head for {ticker}:")
                print(features.head())
                print(f"[DEBUG] Features describe for {ticker}:")
                print(features.describe())
                print(f"[DEBUG] {ticker} composite_signal head:")
                print(composite_signal.head(10))
                print(f"[DEBUG] {ticker} composite_signal min/max:", composite_signal.min(), composite_signal.max())
            # Regime-specific formulas (for regime adjustment below)
            momentum_formula = np.tanh(z(safe_feature(features, "momentum_20d")))
            meanrev_formula = 0.5 * np.tanh(z(safe_feature(features, "rsi_14") / 100 - 0.5)) + \
                             0.5 * np.tanh(z(safe_feature(features, "williams_r") / 100))
            vol = safe_feature(features, "realized_volatility")
            # Volatility normalization (optional)
            if self.vol_normalize:
                signal = composite_signal / (vol + 1e-4)
            else:
                signal = composite_signal.copy()
            # Dynamic threshold for signal strength (regime- and asset-specific)
            rolling_vol = vol.rolling(window=20, min_periods=5).mean().fillna(method='bfill')
            threshold = 0.2 * rolling_vol
            # Soft thresholding: retain some edge for weak signals
            signal = signal * (np.abs(signal) > threshold) + 0.1 * signal * (np.abs(signal) <= threshold)
            # Scaling: tanh or clip
            if self.scaling_method == 'tanh':
                signal = np.tanh(signal * self.scaler_k)
            elif self.scaling_method == 'clip':
                signal = np.clip(signal, -1, 1)
            else:
                raise ValueError(f"Unknown scaling_method: {self.scaling_method}")
            # DEBUG PATCH: Print diagnostics for the first ticker only (after scaling)
            if i == 0:
                print(f"[DEBUG] {ticker} signal after scaling head:")
                print(signal.head(10))
                print(f"[DEBUG] {ticker} signal after scaling min/max:", signal.min(), signal.max())
            # Diagnostics before regime adjustment
            if ticker == "SMH":
                print(f"[DIAGNOSTIC] {ticker} composite_signal head:\n", composite_signal.head(10))
                print(f"[DIAGNOSTIC] {ticker} signal before regime adjustment head:\n", signal.head(10))

            if self.use_regime and self.regime_detector:
                try:
                    regime_series = self.regime_detector.predict(ticker_returns, output_labels=True)
                    try:
                        regime_probs = self.regime_detector.predict_proba(ticker_returns)
                    except Exception:
                        regime_probs = None
                except Exception:
                    regime_series = pd.Series([self.regime_detector.regime_labels[1]] * len(signal), index=signal.index)
                    regime_probs = None
                for regime in self.regime_detector.regime_labels:
                    mask = regime_series == regime
                    # Regime-specific signal logic
                    if regime == "High Vol":
                        # Use mean-reversion only
                        signal[mask] = meanrev_formula[mask]
                    elif regime == "Low Vol":
                        # Use momentum only
                        signal[mask] = momentum_formula[mask]
                    elif regime == "Medium Vol":
                        # Use a blend
                        signal[mask] = 0.5 * momentum_formula[mask] + 0.5 * meanrev_formula[mask]
                    # Optionally scale by regime Sharpe multiplier
                    sharpe_mult = self.regime_sharpe_multipliers.get(regime, 1.0)
                    signal[mask] *= sharpe_mult
                    # Regime confidence weighting (if available)
                    if regime_probs is not None and regime in regime_probs.columns:
                        regime_prob_values = regime_probs[regime][mask].fillna(1.0)
                        signal[mask] *= regime_prob_values
                # Diagnostics after regime adjustment
                if ticker == "SMH":
                    print(f"[DIAGNOSTIC] {ticker} regime_series value counts:\n", regime_series.value_counts())
                    if regime_probs is not None:
                        print(f"[DIAGNOSTIC] {ticker} regime_probs head:\n", regime_probs.head(10))
                    print(f"[DIAGNOSTIC] {ticker} signal after regime adjustment head:\n", signal.head(10))

            signals[ticker] = signal.fillna(0)

        # Diagnostic prints (add before return if you want to see diagnostics)
        print("\n[DIAGNOSTIC] Signals head:\n", signals.head(20))
        print("\n[DIAGNOSTIC] Signals describe:\n", signals.describe())
        print("\n[DIAGNOSTIC] Nonzero signals count:", (signals != 0).sum().sum())

        return signals

    def diagnose_signals(self, signals: pd.DataFrame, returns: pd.DataFrame, regime_series: Optional[pd.Series] = None) -> Dict:
        """Diagnostic function to check signal quality, including regime-split stats"""
        diagnostics = {}
        diagnostics['signal_stats'] = {
            'mean': signals.mean().mean(),
            'std': signals.std().mean(),
            'min': signals.min().min(),
            'max': signals.max().max(),
            'non_zero_count': (signals != 0).sum().sum(),
            'total_count': signals.size
        }
        # Global correlations
        correlations = []
        for col in signals.columns:
            if col in returns.columns:
                forward_return = returns[col].shift(-1)
                corr = signals[col].corr(forward_return)
                if not np.isnan(corr):
                    correlations.append(corr)
        diagnostics['correlations'] = {
            'individual': correlations,
            'mean': np.mean(correlations) if correlations else 0,
            'std': np.std(correlations) if correlations else 0
        }
        # Regime-split diagnostics
        if regime_series is not None:
            regime_stats = {}
            for regime in regime_series.unique():
                mask = regime_series == regime
                regime_signals = signals[mask]
                regime_returns = returns[mask].shift(-1)
                regime_corr = regime_signals.corrwith(regime_returns).mean()
                regime_stats[regime] = {
                    'signal_mean': regime_signals.mean().mean(),
                    'signal_std': regime_signals.std().mean(),
                    'forward_corr': regime_corr
                }
            diagnostics['regime_stats'] = regime_stats
        return diagnostics

    def generate_signal_components(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generate different signal types from features for a single asset.
        Returns a dict: {signal_type: pd.Series}
        """
        # Defensive: fill missing features with 0
        f = features.fillna(0)
        # Momentum: e.g., momentum_20d, macd_signal
        momentum = 0.6 * np.tanh((f.get('momentum_20d', 0) - f.get('momentum_20d', 0).mean()) / (f.get('momentum_20d', 0).std() + 1e-4)) \
                 + 0.4 * np.tanh((f.get('macd_signal', 0) - f.get('macd_signal', 0).mean()) / (f.get('macd_signal', 0).std() + 1e-4))
        # Mean-reversion: e.g., rsi_14, williams_r
        meanrev = 0.5 * np.tanh(f.get('rsi_14', 0) / 100 - 0.5) + 0.5 * np.tanh(f.get('williams_r', 0) / 100)
        # Macro: e.g., semiconductor_pmi, term_structure_slope
        macro = 0.7 * np.tanh(f.get('semiconductor_pmi', 0) / 100 - 0.5) + 0.3 * np.tanh(f.get('term_structure_slope', 0))
        # Volatility breakout: realized_volatility, e.g., high vol = breakout
        vol_breakout = np.tanh((f.get('realized_volatility', 0) - f.get('realized_volatility', 0).rolling(20, min_periods=5).mean()) / (f.get('realized_volatility', 0).rolling(20, min_periods=5).std() + 1e-4))
        # Regime jump: placeholder (could use regime probability changes, etc.)
        regime_jump = f.get('regime_jump', pd.Series(0, index=f.index))
        return {
            'momentum': momentum,
            'meanrev': meanrev,
            'macro': macro,
            'vol_breakout': vol_breakout,
            'regime_jump': regime_jump
        }

    def blend_signals_with_meta_model(self, signal_components: Dict[str, pd.Series],
                                      forward_returns: pd.Series,
                                      model_type: str = 'ridge', alpha: float = 1.0) -> pd.Series:
        """
        Blend multiple signal types using a meta-model (Ridge, Lasso, GradientBoosting, XGBoost, or MLP).
        signal_components: dict of {signal_type: pd.Series}
        forward_returns: pd.Series (aligned)
        model_type: 'ridge', 'lasso', 'gbr', 'xgb', 'mlp'
        Returns: pd.Series of blended signal
        """
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        import warnings
        # Stack signals into DataFrame
        X = pd.DataFrame(signal_components)
        y = forward_returns
        valid = X.notna().all(axis=1) & y.notna()
        if valid.sum() < 10:
            return pd.Series(0, index=X.index)
        X_valid = X[valid]
        y_valid = y[valid]
        model = None
        if model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha)
        elif model_type == 'gbr':
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        elif model_type == 'mlp':
            model = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', max_iter=500, random_state=42)
        elif model_type == 'xgb':
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
            except ImportError:
                raise ImportError("xgboost is not installed. Please install it to use model_type='xgb'.")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_valid, y_valid)
        # Predict blended signal
        blended = pd.Series(model.predict(X), index=X.index)
        # Smooth scaling
        blended = np.tanh(blended * self.scaler_k)
        return blended

def expand_with_lagged_and_cross_features(features: pd.DataFrame, base_features: list, lags: list = [1,2,3,4,5], cross_pairs: list = None) -> pd.DataFrame:
    """
    Expand a feature DataFrame with lagged features and cross-lag interactions.
    Args:
        features: pd.DataFrame of base features
        base_features: list of feature names to lag
        lags: list of lags to use
        cross_pairs: list of (feature1, feature2) tuples for cross-lag interactions
    Returns:
        Expanded pd.DataFrame
    """
    df = features.copy()
    # Add lagged features
    for feat in base_features:
        if feat in df.columns:
            for lag in lags:
                df[f"{feat}_lag{lag}"] = df[feat].shift(lag)
    # Add cross-lag interactions
    if cross_pairs is not None:
        for (f1, f2) in cross_pairs:
            for lag in lags:
                if f1 in df.columns and f2 in df.columns:
                    df[f"{f1}_lag{lag}_x_{f2}"] = df[f1].shift(lag) * df[f2]
    return df

def select_top_features_per_regime(features_dict: Dict[str, pd.DataFrame],
                                   returns: pd.DataFrame,
                                   regime_series_dict: Dict[str, pd.Series],
                                   N: int = 5,
                                   forward_lag: int = 1) -> Dict[str, list]:
    """
    For each regime, rank features by mutual information with forward returns and select top N.
    Args:
        features_dict: Dict[ticker, pd.DataFrame] of features
        returns: pd.DataFrame of returns (index aligned)
        regime_series_dict: Dict[ticker, pd.Series] of regime labels (index aligned)
        N: number of top features to select
        forward_lag: int, number of periods to look ahead for forward returns
    Returns:
        Dict[regime, list of top N features]
    """
    from collections import defaultdict
    from sklearn.feature_selection import mutual_info_regression
    regime_feature_scores = defaultdict(lambda: defaultdict(list))
    all_features = set()
    # For each ticker
    for ticker, features in features_dict.items():
        if ticker not in returns.columns or ticker not in regime_series_dict:
            continue
        ticker_returns = returns[ticker]
        regime_series = regime_series_dict[ticker]
        # Align indices
        features, ticker_returns, regime_series = features.align(ticker_returns, join='inner', axis=0)
        features, regime_series = features.align(regime_series, join='inner', axis=0)
        forward_return = ticker_returns.shift(-forward_lag)
        for regime in regime_series.unique():
            mask = regime_series == regime
            if mask.sum() < 10:
                continue  # skip if too few samples
            X = features[mask]
            y = forward_return[mask]
            for feature in X.columns:
                all_features.add(feature)
                x = X[feature]
                valid = x.notna() & y.notna()
                if valid.sum() < 10:
                    continue
                score = mutual_info_regression(x[valid].values.reshape(-1, 1), y[valid].values, discrete_features=False)[0]
                regime_feature_scores[regime][feature].append(score)
    # Aggregate (mean) across tickers and select top N
    result = {}
    for regime, feature_scores in regime_feature_scores.items():
        avg_scores = {feature: np.nanmean(scores) for feature, scores in feature_scores.items() if scores}
        top_features = sorted(avg_scores, key=avg_scores.get, reverse=True)[:N]
        result[regime] = top_features
    return result

def generate_multi_horizon_signals(signal_components: Dict[str, pd.Series],
                                   forward_returns: pd.Series,
                                   regime_series: pd.Series,
                                   horizons: list = [1, 3, 5],
                                   regime_horizon_weights: dict = None,
                                   model_type: str = 'ridge', alpha: float = 1.0,
                                   scaler_k: float = 1.5) -> pd.Series:
    """
    Generate and blend multi-horizon signals, with regime-dependent horizon weights.
    Args:
        signal_components: dict of {signal_type: pd.Series}
        forward_returns: pd.Series of returns (aligned)
        regime_series: pd.Series of regime labels (aligned)
        horizons: list of forward return horizons
        regime_horizon_weights: dict {regime: {horizon: weight}}
        model_type: meta-model type for blending
        alpha: regularization for meta-model
        scaler_k: scaling for final signal
    Returns:
        pd.Series of blended multi-horizon signal
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    import warnings
    # For each horizon, fit a meta-model and get predicted signal
    horizon_signals = {}
    for h in horizons:
        y = forward_returns.shift(-h)
        X = pd.DataFrame(signal_components)
        valid = X.notna().all(axis=1) & y.notna()
        if valid.sum() < 10:
            horizon_signals[h] = pd.Series(0, index=X.index)
            continue
        X_valid = X[valid]
        y_valid = y[valid]
        if model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'gbr':
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        elif model_type == 'mlp':
            model = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_valid, y_valid)
        horizon_signals[h] = pd.Series(model.predict(X), index=X.index)
    # Blend horizons using regime-dependent weights
    blended = pd.Series(0, index=forward_returns.index, dtype=float)
    for idx in blended.index:
        regime = regime_series.loc[idx] if idx in regime_series.index else None
        if regime_horizon_weights and regime in regime_horizon_weights:
            weights = regime_horizon_weights[regime]
        else:
            weights = {h: 1.0/len(horizons) for h in horizons}
        total_weight = sum(weights.get(h, 0) for h in horizons)
        if total_weight == 0:
            continue
        val = sum(horizon_signals[h].get(idx, 0) * weights.get(h, 0) for h in horizons) / total_weight
        blended.loc[idx] = val
    # Smooth scaling
    blended = np.tanh(blended * scaler_k)
    return blended

def update_weights_from_rolling_performance(signal_components_dict: Dict[str, pd.DataFrame],
                                            returns: pd.DataFrame,
                                            regime_series_dict: Dict[str, pd.Series],
                                            window: int = 60,
                                            min_periods: int = 20,
                                            base_weight: float = 1.0,
                                            upweight_factor: float = 1.2) -> Dict[str, Dict[str, float]]:
    """
    Track rolling performance of each signal component per regime and suggest updated weights for blending.
    Args:
        signal_components_dict: Dict[ticker, pd.DataFrame of signal components]
        returns: pd.DataFrame of returns (tickers as columns)
        regime_series_dict: Dict[ticker, pd.Series] of regime labels
        window: rolling window size
        min_periods: minimum periods for valid performance
        base_weight: default weight for each component
        upweight_factor: multiplier for best-performing component
    Returns:
        Dict[regime, Dict[component, new_weight]]
    """
    from collections import defaultdict
    regime_component_perf = defaultdict(lambda: defaultdict(list))
    # For each ticker
    for ticker, sig_df in signal_components_dict.items():
        if ticker not in returns.columns or ticker not in regime_series_dict:
            continue
        ret = returns[ticker]
        regime_series = regime_series_dict[ticker]
        # Align indices
        sig_df, ret, regime_series = sig_df.align(ret, join='inner', axis=0)
        sig_df, regime_series = sig_df.align(regime_series, join='inner', axis=0)
        for component in sig_df.columns:
            for regime in regime_series.unique():
                mask = regime_series == regime
                # Rolling window performance (mean signal * forward return)
                perf = (sig_df[component][mask] * ret.shift(-1)[mask]).rolling(window, min_periods=min_periods).mean()
                # Use last value as recent performance
                if not perf.empty and not np.isnan(perf.iloc[-1]):
                    regime_component_perf[regime][component].append(perf.iloc[-1])
    # Aggregate and suggest weights
    result = {}
    for regime, comp_perf in regime_component_perf.items():
        avg_perf = {comp: np.nanmean(perfs) for comp, perfs in comp_perf.items() if perfs}
        if not avg_perf:
            continue
        # Upweight best-performing component
        best = max(avg_perf, key=avg_perf.get)
        weights = {comp: base_weight for comp in avg_perf}
        weights[best] = base_weight * upweight_factor
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            for comp in weights:
                weights[comp] /= total
        result[regime] = weights
    return result

def compute_drawdown(equity_curve: pd.Series, window: int = 10) -> pd.Series:
    """
    Compute rolling drawdown over a window.
    Args:
        equity_curve: pd.Series of cumulative returns or equity
        window: rolling window size
    Returns:
        pd.Series of drawdown (as positive values)
    """
    roll_max = equity_curve.rolling(window, min_periods=1).max()
    drawdown = (roll_max - equity_curve).clip(lower=0)
    return drawdown

def compute_rolling_beta(asset_returns: pd.Series, index_returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling beta of asset to index.
    Args:
        asset_returns: pd.Series
        index_returns: pd.Series (aligned)
        window: rolling window size
    Returns:
        pd.Series of rolling beta
    """
    cov = asset_returns.rolling(window).cov(index_returns)
    var = index_returns.rolling(window).var()
    beta = cov / (var + 1e-8)
    return beta

def apply_risk_feedback_scaling(signal: pd.Series,
                                drawdown: pd.Series = None,
                                beta: pd.Series = None,
                                drawdown_thresh: float = 0.05,
                                beta_thresh: float = 1.5,
                                min_scale: float = 0.2,
                                fast_drawdown: pd.Series = None,
                                fast_drawdown_thresh: float = 0.03,
                                recovery_speed: float = 0.05) -> pd.Series:
    """
    Reduce signal exposure based on drawdown and beta thresholds, with asymmetric scaling and fast drawdown feedback.
    Args:
        signal: pd.Series of signals
        drawdown: pd.Series of rolling drawdown (same index)
        beta: pd.Series of rolling beta (same index)
        drawdown_thresh: threshold for drawdown (e.g., 5%)
        beta_thresh: threshold for beta
        min_scale: minimum scaling factor
        fast_drawdown: pd.Series of fast (e.g., 5-day) drawdown
        fast_drawdown_thresh: threshold for fast drawdown
        recovery_speed: how quickly to increase scale after drawdown (0-1)
    Returns:
        pd.Series of scaled signals
    """
    scale = pd.Series(1.0, index=signal.index)
    # Fast drawdown feedback (cut quickly)
    if fast_drawdown is not None:
        scale[fast_drawdown > fast_drawdown_thresh] *= 0.2
    # Standard drawdown feedback (cut quickly, recover slowly)
    if drawdown is not None:
        in_drawdown = drawdown > drawdown_thresh
        scale[in_drawdown] *= 0.5
        # Asymmetric: slow recovery
        recovering = (~in_drawdown) & (drawdown.shift(1) > drawdown_thresh)
        scale[recovering] = scale[recovering] * (1 - recovery_speed) + recovery_speed
    # Beta feedback
    if beta is not None:
        scale[beta > beta_thresh] *= 0.5
    scale = scale.clip(lower=min_scale, upper=1.0)
    return signal * scale

def compute_regime_feature_relevance(features_dict: Dict[str, pd.DataFrame],
                                    returns: pd.DataFrame,
                                    regime_series_dict: Dict[str, pd.Series],
                                    method: str = 'correlation',
                                    forward_lag: int = 1) -> Dict[str, Dict[str, float]]:
    """
    For each regime, compute the predictive power (correlation or mutual information)
    between each feature and forward returns, across all tickers.
    Args:
        features_dict: Dict[ticker, pd.DataFrame] of features
        returns: pd.DataFrame of returns (index aligned with features)
        regime_series_dict: Dict[ticker, pd.Series] of regime labels (index aligned)
        method: 'correlation' or 'mutual_info'
        forward_lag: int, number of periods to look ahead for forward returns
    Returns:
        Dict[regime, Dict[feature, score]]
    """
    from collections import defaultdict
    from sklearn.feature_selection import mutual_info_regression
    regime_feature_scores = defaultdict(lambda: defaultdict(list))
    all_features = set()
    # For each ticker
    for ticker, features in features_dict.items():
        if ticker not in returns.columns or ticker not in regime_series_dict:
            continue
        ticker_returns = returns[ticker]
        regime_series = regime_series_dict[ticker]
        # Align indices
        features, ticker_returns, regime_series = features.align(ticker_returns, join='inner', axis=0)
        features, regime_series = features.align(regime_series, join='inner', axis=0)
        forward_return = ticker_returns.shift(-forward_lag)
        for regime in regime_series.unique():
            mask = regime_series == regime
            if mask.sum() < 10:
                continue  # skip if too few samples
            X = features[mask]
            y = forward_return[mask]
            for feature in X.columns:
                all_features.add(feature)
                x = X[feature]
                valid = x.notna() & y.notna()
                if valid.sum() < 10:
                    continue
                if method == 'correlation':
                    score = np.corrcoef(x[valid], y[valid])[0, 1]
                elif method == 'mutual_info':
                    score = mutual_info_regression(x[valid].values.reshape(-1, 1), y[valid].values, discrete_features=False)[0]
                else:
                    raise ValueError(f"Unknown method: {method}")
                regime_feature_scores[regime][feature].append(score)
    # Aggregate (mean) across tickers
    result = {}
    for regime, feature_scores in regime_feature_scores.items():
        result[regime] = {}
        for feature in all_features:
            scores = feature_scores.get(feature, [])
            if scores:
                result[regime][feature] = float(np.nanmean(scores))
            else:
                result[regime][feature] = np.nan
    return result

def fit_regime_feature_weights_ridge(features_dict: Dict[str, pd.DataFrame],
                                     returns: pd.DataFrame,
                                     regime_series_dict: Dict[str, pd.Series],
                                     alpha: float = 1.0,
                                     forward_lag: int = 1) -> Dict[str, Dict[str, float]]:
    """
    For each regime, fit a Ridge regression to learn feature weights for predicting forward returns.
    Args:
        features_dict: Dict[ticker, pd.DataFrame] of features
        returns: pd.DataFrame of returns (index aligned with features)
        regime_series_dict: Dict[ticker, pd.Series] of regime labels (index aligned)
        alpha: Ridge regularization parameter
        forward_lag: int, number of periods to look ahead for forward returns
    Returns:
        Dict[regime, Dict[feature, weight]]
    """
    from collections import defaultdict
    from sklearn.linear_model import Ridge
    import warnings
    regime_X = defaultdict(list)
    regime_y = defaultdict(list)
    all_features = set()
    # Gather data per regime across all tickers
    for ticker, features in features_dict.items():
        if ticker not in returns.columns or ticker not in regime_series_dict:
            continue
        ticker_returns = returns[ticker]
        regime_series = regime_series_dict[ticker]
        # Align indices
        features, ticker_returns = features.align(ticker_returns, join='inner', axis=0)
        features, regime_series = features.align(regime_series, join='inner', axis=0)
        ticker_returns, regime_series = ticker_returns.align(regime_series, join='inner', axis=0)
        forward_return = ticker_returns.shift(-forward_lag)
        for regime in regime_series.unique():
            mask = regime_series == regime
            if mask.sum() < 10:
                continue  # skip if too few samples
            X = features[mask]
            y = forward_return[mask]
            valid = X.notna().all(axis=1) & y.notna()
            if valid.sum() < 10:
                continue
            regime_X[regime].append(X[valid])
            regime_y[regime].append(y[valid])
            all_features.update(X.columns)
    # Fit Ridge per regime
    result = {}
    for regime in regime_X:
        X_regime = pd.concat(regime_X[regime], axis=0)
        y_regime = pd.concat(regime_y[regime], axis=0)
        # Ensure all features present (fill missing with 0)
        X_regime = X_regime.reindex(columns=sorted(all_features), fill_value=0)
        if len(X_regime) < 10:
            continue
        model = Ridge(alpha=alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_regime, y_regime)
        weights = dict(zip(X_regime.columns, model.coef_))
        result[regime] = weights
    return result

def compute_signal_forward_return_alignment(signals: pd.DataFrame,
                                            returns: pd.DataFrame,
                                            regime_series_dict: Dict[str, pd.Series],
                                            lags: list = [1, 3, 5]) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    For each ticker and regime, compute the mean product of signal[t] * return[t+lag] for given lags.
    Args:
        signals: pd.DataFrame of signals (tickers as columns)
        returns: pd.DataFrame of returns (tickers as columns)
        regime_series_dict: Dict[ticker, pd.Series] of regime labels (index aligned)
        lags: list of lags to test
    Returns:
        Dict[ticker, Dict[regime, Dict[lag, mean_product]]]
    """
    result = {}
    for ticker in signals.columns:
        if ticker not in returns.columns or ticker not in regime_series_dict:
            continue
        sig = signals[ticker]
        ret = returns[ticker]
        regime_series = regime_series_dict[ticker]
        # Align indices
        sig, ret, regime_series = sig.align(ret, join='inner', axis=0)
        sig, regime_series = sig.align(regime_series, join='inner', axis=0)
        ticker_result = {}
        for regime in regime_series.unique():
            mask = regime_series == regime
            regime_result = {}
            for lag in lags:
                forward_ret = ret.shift(-lag)
                valid = mask & sig.notna() & forward_ret.notna()
                if valid.sum() < 10:
                    regime_result[lag] = float('nan')
                else:
                    regime_result[lag] = float(np.nanmean(sig[valid] * forward_ret[valid]))
            ticker_result[regime] = regime_result
        result[ticker] = ticker_result
    return result
