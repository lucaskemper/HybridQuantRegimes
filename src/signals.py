import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from typing import Dict, Optional

# --- Signal component functions ---
def momentum_signal(f):
    # Z-score momentum_20d and MACD
    m = f.get('momentum_20d', pd.Series(0, index=f.index))
    macd = f.get('macd_signal', pd.Series(0, index=f.index))
    m_z = (m - m.mean()) / (m.std() + 1e-6)
    macd_z = (macd - macd.mean()) / (macd.std() + 1e-6)
    return 0.6 * np.tanh(m_z) + 0.4 * np.tanh(macd_z)

def meanrev_signal(f):
    # RSI and Williams %R
    rsi = f.get('rsi_14', pd.Series(50, index=f.index))
    willr = f.get('williams_r', pd.Series(0, index=f.index))
    rsi_score = -np.tanh((rsi / 100 - 0.5) * 2)
    willr_score = np.tanh(willr / 100)
    return 0.5 * rsi_score + 0.5 * willr_score

def macro_signal(f):
    # Term structure slope, VIX percentile, semiconductor PMI
    slope = f.get('term_structure_slope', pd.Series(0, index=f.index))
    vix = f.get('vix_percentile', pd.Series(0, index=f.index))
    pmi = f.get('semiconductor_pmi', pd.Series(50, index=f.index))
    return 0.4 * np.tanh(slope) + 0.3 * (vix - 0.5) + 0.3 * (pmi / 100 - 0.5)

def vol_breakout_signal(f):
    # Realized volatility vs rolling mean
    rv = f.get('realized_volatility', pd.Series(0, index=f.index))
    rv_mean = rv.rolling(20, min_periods=5).mean()
    rv_std = rv.rolling(20, min_periods=5).std() + 1e-6
    breakout = (rv - rv_mean) / rv_std
    return np.tanh(breakout)

def cross_sectional_signal(f):
    # Momentum rank, sector spreads
    rank = f.get('momentum_rank', pd.Series(0, index=f.index))
    spread = f.get('nvda_amd_spread', pd.Series(0, index=f.index))
    return 0.7 * (rank - 0.5) + 0.3 * np.tanh(spread)

# --- Regime-specific blending weights ---
REGIME_WEIGHTS = {
    'High Vol': {'momentum': 0.35, 'meanrev': 0.1, 'macro': 0.1, 'vol_breakout': 0.35, 'cross': 0.1},
    'Medium Vol': {'momentum': 0.45, 'meanrev': 0.15, 'macro': 0.2, 'vol_breakout': 0.1, 'cross': 0.1},
    'Low Vol': {'momentum': 0.35, 'meanrev': 0.15, 'macro': 0.3, 'vol_breakout': 0.1, 'cross': 0.1},
    'Default': {'momentum': 0.3, 'meanrev': 0.2, 'macro': 0.2, 'vol_breakout': 0.2, 'cross': 0.1},
}

# --- Meta-model blender ---
def meta_blend(X: pd.DataFrame, y: pd.Series):
    # Drop all rows with any NaNs in X or y
    mask = X.notna().all(axis=1) & y.notna()
    X_valid = X[mask]
    y_valid = y[mask]
    if len(X_valid) < 20:
        return pd.Series(0.0, index=X.index), None
    model = RidgeCV(alphas=np.logspace(-4, 1, 20)).fit(X_valid, y_valid)
    # Predict only on rows without NaNs in X
    pred = pd.Series(0.0, index=X.index)
    pred[mask] = model.predict(X_valid)
    return pred.fillna(0), model

# --- Risk overlay (optional) ---
def risk_overlay(signal, f, drawdown_window=20, min_scale=0.3):
    eq = (signal * f.get('returns', 0)).cumsum()
    dd = eq - eq.cummax()
    dd_pct = dd / (eq.cummax().replace(0, np.nan) + 1e-6)
    scale = np.clip(1 - dd_pct, min_scale, 1)
    return signal * scale

# --- Main improved signal generator ---
def generate_signals_core(returns, features_dict, regime_series_dict):
    required = ["momentum_20d", "realized_volatility", "bollinger_position"]
    for ticker, df in features_dict.items():
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"[Feature Check] {ticker} missing: {missing}")
        if "momentum_20d" not in df:
            if "returns" in df:
                df["momentum_20d"] = df["returns"].rolling(20).mean()
        if "realized_volatility" not in df:
            if "returns" in df:
                df["realized_volatility"] = df["returns"].rolling(20).std()
        if "bollinger_position" not in df:
            if "price" in df:
                df["bollinger_position"] = (df["price"] - df["price"].rolling(20).mean()) / (df["price"].rolling(20).std() + 1e-6)
        still_missing = [col for col in required if col not in df.columns]
        assert not still_missing, f"{ticker} missing features: {still_missing}"
        non_numeric = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]
        if non_numeric:
            print(f"[Feature Check] {ticker} has non-numeric features: {non_numeric}")
    for ticker in returns.columns:
        r_idx = returns[ticker].index
        f_idx = features_dict[ticker].index
        reg_idx = regime_series_dict[ticker].index if ticker in regime_series_dict else r_idx
        if not (r_idx.equals(f_idx) and r_idx.equals(reg_idx)):
            print(f"[Feature Check] {ticker} has misaligned indices")
    output = pd.DataFrame(index=returns.index, columns=returns.columns)
    feature_importances = {}
    for ticker in returns.columns:
        r, f = returns[ticker], features_dict[ticker].copy()
        regimes = regime_series_dict[ticker] if ticker in regime_series_dict else pd.Series("Default", index=r.index)
        # Compute signal components
        components = {
            'momentum': momentum_signal(f),
            'meanrev': meanrev_signal(f),
            'macro': macro_signal(f),
            'vol_breakout': vol_breakout_signal(f),
            'cross': cross_sectional_signal(f),
        }
        X = pd.DataFrame(components)
        y = r.shift(-1)
        # Regime-specific blending
        blended = pd.Series(0.0, index=f.index)
        for regime in regimes.unique():
            mask = regimes == regime
            weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['Default'])
            X_reg = X.loc[mask]
            y_reg = y.loc[mask]
            # Meta-model blend for this regime
            meta_pred, model = meta_blend(X_reg, y_reg)
            # Save feature importances
            feature_importances[(ticker, regime)] = dict(zip(X.columns, getattr(model, 'coef_', [0]*len(X.columns))))
            # Weighted blend
            blend = sum(weights[k] * X[k] for k in X.columns)
            # Combine meta-model and weighted blend
            final = 0.5 * blend + 0.5 * meta_pred
            # Optional: risk overlay
            final = risk_overlay(final, f)
            blended[mask] = final[mask]
        output[ticker] = blended.fillna(0)
    # Diagnostics: print feature importances summary
    print("\n[Signal Diagnostics] Feature importances (mean abs by regime):")
    for regime in REGIME_WEIGHTS:
        vals = [abs(feature_importances[(t, regime)][k]) for t in returns.columns if (t, regime) in feature_importances for k in X.columns]
        if vals:
            print(f"  {regime}: mean abs coef = {np.mean(vals):.4f}")
    return output.fillna(0)

# --- Pipeline-compatible SignalGenerator class remains unchanged ---
class SignalGenerator:
    def __init__(self, **kwargs):
        pass
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
        returns = market_data['returns']
        features_dict = market_data.get('features', {})
        regime_series_dict = market_data.get('regime_series_dict', {})
        if not regime_series_dict:
            regime_series_dict = {t: pd.Series("Default", index=returns.index) for t in returns.columns}
        return generate_signals_core(returns, features_dict, regime_series_dict)
    def diagnose_signals(self, signals: pd.DataFrame, returns: pd.DataFrame, regime_series: Optional[pd.Series] = None) -> Dict:
        diagnostics = {}
        diagnostics['signal_stats'] = {
            'mean': signals.mean().mean(),
            'std': signals.std().mean(),
            'min': signals.min().min(),
            'max': signals.max().max(),
            'non_zero_count': (signals != 0).sum().sum(),
            'total_count': signals.size
        }
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
