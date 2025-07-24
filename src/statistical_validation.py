"""
Statistical Validation Framework
===============================

This module provides rigorous statistical significance testing for regime detection and model performance.

Mathematical Formulations:
--------------------------

1. **Binomial Test for Regime Detection Accuracy**
   - Null hypothesis: $H_0$: accuracy = random chance $= 1/K$ (K = number of regimes)
   - Test statistic: $X \sim \text{Binomial}(n, p_0)$
   - $p$-value: $P(X \geq x_{obs} | H_0)$

2. **Paired t-test for Model Outperformance**
   - Let $d_i = r^{\text{model}}_i - r^{\text{baseline}}_i$
   - $H_0: \mathbb{E}[d] \leq 0$
   - $H_1: \mathbb{E}[d] > 0$
   - $t = \frac{\overline{d}}{s_d / \sqrt{n}}$
   - $p$-value from $t$-distribution

3. **Bootstrap Confidence Intervals**
   - Resample $d_i$ with replacement, compute mean for each sample
   - CI: 2.5th and 97.5th percentiles of bootstrap means

4. **Bayesian Model Averaging (used in regime.py)**
   - Given model probabilities $P_1(y|x), P_2(y|x)$ and weights $w_1, w_2$:
   - $P_{\text{BMA}}(y|x) = w_1 P_1(y|x) + w_2 P_2(y|x)$
   - Weights can be based on log-evidence or entropy (see regime.py)

"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple

class StatisticalValidator:
    """Statistical significance testing for regime detection and performance"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def test_regime_detection_accuracy(self, 
                                     true_regimes: pd.Series, 
                                     predicted_regimes: pd.Series) -> Dict:
        """Test statistical significance of regime detection accuracy"""
        
        # Align series
        common_idx = true_regimes.index.intersection(predicted_regimes.index)
        true_aligned = true_regimes.loc[common_idx]
        pred_aligned = predicted_regimes.loc[common_idx]
        
        # Calculate accuracy
        accuracy = (true_aligned == pred_aligned).mean()
        
        # Null hypothesis: random classification
        n_regimes = len(true_aligned.unique())
        null_accuracy = 1 / n_regimes
        
        # Binomial test
        n_correct = (true_aligned == pred_aligned).sum()
        n_total = len(true_aligned)
        
        p_value = stats.binom_test(n_correct, n_total, null_accuracy, alternative='greater')
        
        # Confidence interval for accuracy
        ci_lower, ci_upper = stats.binom.interval(
            1 - self.significance_level, n_total, accuracy
        )
        ci_lower /= n_total
        ci_upper /= n_total
        
        return {
            'accuracy': accuracy,
            'null_accuracy': null_accuracy,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'confidence_interval': (ci_lower, ci_upper),
            'n_samples': n_total
        }
    
    def test_performance_significance(self, 
                                    strategy_returns: pd.Series,
                                    benchmark_returns: pd.Series) -> Dict:
        """Test statistical significance of strategy outperformance"""
        
        # Align returns
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strat_returns = strategy_returns.loc[common_idx]
        bench_returns = benchmark_returns.loc[common_idx]
        
        # Calculate excess returns
        excess_returns = strat_returns - bench_returns
        
        # T-test for mean excess return
        t_stat, p_value = stats.ttest_1samp(excess_returns.dropna(), 0)
        
        # Information ratio
        info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Bootstrap confidence interval for information ratio
        n_bootstrap = 1000
        bootstrap_info_ratios = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(excess_returns.dropna(), 
                                              size=len(excess_returns.dropna()), 
                                              replace=True)
            bootstrap_ir = (np.mean(bootstrap_sample) / np.std(bootstrap_sample) * 
                           np.sqrt(252))
            bootstrap_info_ratios.append(bootstrap_ir)
        
        ci_lower = np.percentile(bootstrap_info_ratios, 2.5)
        ci_upper = np.percentile(bootstrap_info_ratios, 97.5)
        
        return {
            'excess_return_mean': excess_returns.mean() * 252,
            'information_ratio': info_ratio,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'ir_confidence_interval': (ci_lower, ci_upper),
            'n_observations': len(excess_returns.dropna())
        }
    
    def cross_validate_regime_detection(self, 
                                       features: pd.DataFrame, 
                                       regime_detector,
                                       macro_data=None,
                                       n_splits: int = 5) -> Dict:
        """Time series cross-validation for regime detection (now uses features DataFrame directly)"""
        from src.regime import MarketRegimeDetector
        tscv = TimeSeriesSplit(n_splits=n_splits)
        stabilities = []
        features = features.dropna()
        for train_idx, test_idx in tscv.split(features):
            train_features = features.iloc[train_idx]
            test_features = features.iloc[test_idx]
            # Create a fresh detector for each fold
            fold_detector = MarketRegimeDetector(regime_detector.config)
            fold_detector.fit(train_features)
            test_regimes = fold_detector.predict(test_features)
            regime_changes = (test_regimes != test_regimes.shift()).sum()
            stability = 1 - (regime_changes / len(test_regimes))
            stabilities.append(stability)
        return {
            'mean_stability': np.mean(stabilities),
            'std_stability': np.std(stabilities),
            'cv_scores': stabilities,
            'stability_significant': (np.mean(stabilities) > 0.7)  # Arbitrary threshold
        }

    def test_model_performance_hypothesis(self, model_returns: pd.Series, baseline_returns: pd.Series, alternative: str = 'greater') -> Dict:
        """
        Hypothesis test for model performance vs. baseline.
        H0: E[model_returns - baseline_returns] <= 0
        H1: E[model_returns - baseline_returns] > 0 (default, one-sided)
        Uses paired t-test and bootstrap for robustness.
        Args:
            model_returns: pd.Series of returns from the Bayesian ensemble model
            baseline_returns: pd.Series of returns from the baseline (e.g., HMM only or random)
            alternative: 'greater', 'less', or 'two-sided'
        Returns:
            Dict with t-statistic, p-value, bootstrap CI, and significance
        Mathematical Formulation:
            Let d_i = model_returns_i - baseline_returns_i
            H0: E[d] <= 0
            H1: E[d] > 0
            t = mean(d) / (std(d) / sqrt(n))
        """
        # Align
        common_idx = model_returns.index.intersection(baseline_returns.index)
        model_aligned = model_returns.loc[common_idx]
        base_aligned = baseline_returns.loc[common_idx]
        diff = model_aligned - base_aligned
        n = len(diff.dropna())
        # Paired t-test
        t_stat, p_value = stats.ttest_1samp(diff.dropna(), 0, alternative=alternative)
        # Bootstrap CI
        n_bootstrap = 1000
        boot_means = [np.mean(np.random.choice(diff.dropna(), size=n, replace=True)) for _ in range(n_bootstrap)]
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        return {
            'mean_difference': diff.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'bootstrap_confidence_interval': (ci_lower, ci_upper),
            'n_samples': n
        } 