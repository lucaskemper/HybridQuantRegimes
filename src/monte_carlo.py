# src/monte_carlo.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy import stats as scipy_stats
from scipy.stats import t as students_t
from arch import arch_model  # Add this import for GARCH modeling

@dataclass
class SimConfig:
    def __init__(self, n_sims: int = 10000, n_days: int = 252, 
                 risk_free_rate: float = 0.05,
                 confidence_levels: tuple = (0.05, 0.25, 0.5, 0.75, 0.95)):
        """
        Initialize simulation configuration.
        
        Args:
            n_sims: Number of Monte Carlo simulations
            n_days: Number of trading days to simulate
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            confidence_levels: Tuple of confidence levels for intervals
        """
        # Validate inputs
        if n_sims <= 0:
            raise ValueError("Number of simulations must be positive")
        if n_days <= 0:
            raise ValueError("Number of days must be positive")
        if not all(0 < level < 1 for level in confidence_levels):
            raise ValueError("Confidence levels must be between 0 and 1")
        
        self.n_sims = n_sims
        self.n_days = n_days
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels

class MonteCarlo:
    def __init__(self, config: SimConfig):
        self.config = config
        self.distribution = 'normal'  # Add default distribution type
    
    def simulate(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run Monte Carlo simulation"""
        print("\nRunning Monte Carlo simulation...")
        
        # Add input validation
        if not isinstance(market_data, dict) or 'returns' not in market_data:
            raise ValueError("market_data must be a dictionary containing 'returns' DataFrame")
        
        returns = market_data['returns']
        if returns.empty:
            raise ValueError("Returns data is empty")
        
        # Validate distribution type first
        if self.distribution not in ['normal', 't']:
            raise ValueError("Unsupported distribution type")
        
        # Initialize arrays
        n_assets = len(returns.columns)
        paths = np.zeros((self.config.n_sims, n_assets, self.config.n_days))
        
        # Get parameters
        mu = returns.mean().values
        sigma = returns.std().values
        corr = returns.corr().values
        
        # Generate correlated returns
        for i in range(self.config.n_days):
            if self.distribution == 'normal':
                z = np.random.standard_normal((self.config.n_sims, n_assets))
            elif self.distribution == 't':
                # Reduce degrees of freedom to create heavier tails
                degrees_of_freedom = 3  # Lower value = heavier tails
                z = scipy_stats.t.rvs(df=degrees_of_freedom, size=(self.config.n_sims, n_assets))
            
            # Apply GARCH volatility forecasting
            sigma_t = self._forecast_volatility(returns)
            L = np.linalg.cholesky(corr)
            paths[:, :, i] = np.dot(z, L.T) * sigma_t
        
        # Apply drift and volatility
        for i in range(n_assets):
            paths[:, i, :] = (mu[i] - 0.5 * sigma[i]**2) + sigma[i] * paths[:, i, :]
        
        # Convert to cumulative returns
        paths = np.exp(np.cumsum(paths, axis=2))
        
        # Calculate results
        final_values = paths[:, :, -1]
        portfolio_values = np.sum(final_values, axis=1)
        
        # Calculate confidence intervals
        confidence_intervals = np.percentile(portfolio_values, 
                                          [level * 100 for level in self.config.confidence_levels])
        
        results = {
            'paths': paths,
            'final_values': portfolio_values,
            'confidence_intervals': dict(zip(self.config.confidence_levels, confidence_intervals)),
            'expected_return': np.mean(portfolio_values),
            'simulation_volatility': np.std(portfolio_values),
            'var_95': np.percentile(portfolio_values, 5),
            'var_99': np.percentile(portfolio_values, 1),
            'statistics': self._calculate_statistics(portfolio_values),
            'validation': self.validate_simulation(paths, market_data)
        }
        
        return results
    
    def _forecast_volatility(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Forecast volatility using GARCH(1,1) model with proper scaling.
        
        Args:
            returns (pd.DataFrame): Historical returns data
        
        Returns:
            np.ndarray: Array of volatility forecasts for each asset
        
        Notes:
            - Uses dynamic scaling to ensure numerical stability
            - Falls back to historical volatility if GARCH fitting fails
            - Automatically handles proper rescaling of results
        """
        forecasted_vol = np.zeros(len(returns.columns))
        
        for i, col in enumerate(returns.columns):
            try:
                # Scale returns to be between 1 and 1000 as recommended by arch
                scale_factor = 1000 / returns[col].std()  # This will make std around 1000
                scaled_returns = returns[col] * scale_factor
                
                # Fit GARCH model with rescaling disabled to avoid warnings
                model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
                results = model.fit(disp='off')
                
                # Extract variance forecast and properly handle array conversion
                variance = results.forecast().variance.values[-1]
                if isinstance(variance, np.ndarray):
                    variance = variance.item()  # Safely convert to scalar
                    
                # Rescale back to original scale
                forecasted_vol[i] = np.sqrt(variance) / scale_factor
                
            except Exception as e:
                print(f"GARCH fitting failed for {col}: {str(e)}")
                # Fallback to historical volatility if GARCH fitting fails
                forecasted_vol[i] = returns[col].std()
        
        return forecasted_vol
    
    def _calculate_statistics(self, portfolio_values: np.ndarray) -> Dict:
        """Calculate detailed statistics of simulation results"""
        statistics = {
            'mean': np.mean(portfolio_values),
            'median': np.median(portfolio_values),
            'std': np.std(portfolio_values),
            'skew': scipy_stats.skew(portfolio_values),
            'kurtosis': scipy_stats.kurtosis(portfolio_values),
            'sharpe_ratio': (np.mean(portfolio_values) - self.config.risk_free_rate) / np.std(portfolio_values)
        }
        
        # Add bias-variance analysis
        bias_variance = self._calculate_bias_variance(portfolio_values)
        
        return {**statistics, **bias_variance}
    
    def _calculate_bias_variance(self, portfolio_values: np.ndarray) -> Dict:
        """Calculate bias-variance decomposition"""
        # Using k-fold cross-validation approach
        k_folds = 5
        fold_size = len(portfolio_values) // k_folds
        
        predictions = []
        for i in range(k_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            fold_pred = np.mean(portfolio_values[start_idx:end_idx])
            predictions.append(fold_pred)
        
        predictions = np.array(predictions)
        expected_pred = np.mean(predictions)
        
        return {
            'bias': np.mean(predictions - np.mean(portfolio_values)),
            'variance': np.var(predictions),
            'cross_val_score': np.std(predictions) / np.mean(predictions)  # CV score
        }
    
    def validate_simulation(self, paths: np.ndarray, market_data: Dict) -> Dict:
        """Validate simulation results"""
        try:
            historical_corr = market_data['returns'].corr()
            simulated_returns = np.diff(np.log(paths), axis=2)
            simulated_corr = np.corrcoef(simulated_returns.mean(axis=0))
            
            validation = {
                'positive_values': np.all(paths > 0),
                'correlation_preservation': np.allclose(historical_corr, simulated_corr, atol=0.1),
                'reasonable_returns': self._check_return_reasonability(paths),
                'volatility_alignment': self._check_volatility_alignment(paths, market_data)
            }
            
            return validation
        
        except Exception as e:
            return {
                'positive_values': False,
                'correlation_preservation': False,
                'reasonable_returns': False,
                'volatility_alignment': False,
                'error': str(e)
            }
    
    def _check_return_reasonability(self, paths: np.ndarray) -> bool:
        """Check if returns are within reasonable bounds"""
        returns = np.diff(np.log(paths), axis=2)
        return np.all((-0.5 <= returns) & (returns <= 0.5))
    
    def _check_volatility_alignment(self, paths: np.ndarray, market_data: Dict) -> bool:
        """Check if simulation volatility aligns with historical volatility"""
        historical_vol = market_data['returns'].std().values
        simulated_vol = np.std(np.diff(np.log(paths), axis=2), axis=(0, 2))
        return np.allclose(historical_vol, simulated_vol, rtol=0.2)