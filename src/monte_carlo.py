# src/monte_carlo.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats as scipy_stats
from scipy.stats import t as students_t
import os
from datetime import datetime


@dataclass
class SimConfig:
    def __init__(
        self,
        n_sims=10000,
        n_days=1260,
        risk_free_rate=0.05,
        confidence_levels=None,
        distribution="t",
        simulation_mode="block_bootstrap",  # Add simulation_mode
        **kwargs,                # Ignore extra keys
    ):
        self.n_sims = n_sims
        self.n_days = n_days
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels or [0.05, 0.25, 0.5, 0.75, 0.95]
        self.distribution = distribution
        self.simulation_mode = simulation_mode


class MonteCarlo:
    def __init__(self, config: SimConfig):
        self.config = config
        self.distribution = config.distribution

    def simulate(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run Monte Carlo simulation with realistic parameters and multiple modes."""
        print("\nRunning Monte Carlo simulation...")

        # Check for None first
        if market_data is None:
            raise TypeError("market_data cannot be None")

        if not isinstance(market_data, dict) or "returns" not in market_data:
            raise ValueError(
                "market_data must be a dictionary containing 'returns' DataFrame"
            )

        returns = market_data["returns"]
        if returns.empty:
            raise ValueError("Returns data is empty")

        # --- Simulation config ---
        simulation_mode = getattr(self.config, 'simulation_mode', 'block_bootstrap')  # Default to block_bootstrap for realism
        n_days = max(self.config.n_days, 1260)  # At least 5 years
        n_sims = self.config.n_sims
        n_assets = len(returns.columns)
        block_size = 21  # 1 month
        risk_premium = 0.02
        vol_scale = 0.5  # Lower vol_scale for more realistic volatility

        # --- Lower mean return and volatility ---
        mu = (np.array(returns.mean())) - (risk_premium / 252)  # dailyized
        sigma = np.array(returns.std()) * vol_scale  # dailyized std
        corr = returns.corr().values

        # --- Block Bootstrapping ---
        if simulation_mode == "block_bootstrap":
            print("Using block bootstrapping for simulation.")
            # Use log returns for compounding
            log_returns = np.log(1 + returns)
            log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
            # Cap extreme returns
            lower, upper = np.nanpercentile(log_returns.values.flatten(), [1, 99])
            log_returns = log_returns.clip(lower, upper)
            n_days = self.config.n_days
            n_sims = self.config.n_sims
            n_assets = log_returns.shape[1] if len(log_returns.shape) > 1 else 1
            block_size = getattr(self.config, 'block_size', 21)  # 1 month default
            # Precompute blocks
            n_blocks = len(log_returns) - block_size + 1
            blocks = [log_returns.iloc[i:i+block_size].values for i in range(n_blocks)]
            # Simulate
            sim_paths = np.zeros((n_sims, n_days, n_assets))
            for sim in range(n_sims):
                idx = 0
                while idx < n_days:
                    block = blocks[np.random.randint(0, n_blocks)]
                    take = min(block_size, n_days - idx)
                    sim_paths[sim, idx:idx+take, :] = block[:take]
                    idx += take
            # Portfolio: equally weighted
            port_log_returns = sim_paths.mean(axis=2) if n_assets > 1 else sim_paths[:, :, 0]
            # Cumulative log return
            port_cum = np.cumsum(port_log_returns, axis=1)
            port_value = np.exp(port_cum)
            # Compute metrics
            final_value = port_value[:, -1]
            total_return = final_value - 1
            n_years = n_days / 252
            # Annualized return/vol from daily log returns
            ann_return = np.exp(port_log_returns.mean(axis=1) * 252) - 1
            ann_vol = port_log_returns.std(axis=1) * np.sqrt(252)
            sharpe = np.where(ann_vol > 0, ann_return / ann_vol, 0)
            # Success rate: percent of paths with positive return
            success_rate = (total_return > 0).mean() * 100
            # Max drawdown
            roll_max = np.maximum.accumulate(port_value, axis=1)
            drawdown = (port_value - roll_max) / roll_max
            max_drawdown = drawdown.min(axis=1)
            # VaR/CVaR and confidence intervals on total_return (not final_value)
            var_95 = np.percentile(total_return, 5)
            cvar_95 = total_return[total_return <= var_95].mean()
            var_99 = np.percentile(total_return, 1)
            cvar_99 = total_return[total_return <= var_99].mean()
            confidence_intervals = np.percentile(total_return, [level * 100 for level in self.config.confidence_levels])
            # Convert log returns to price paths for each asset for validation
            sim_price_paths = np.exp(np.cumsum(sim_paths, axis=1))  # (n_sims, n_days, n_assets)
            sim_price_paths = np.transpose(sim_price_paths, (0, 2, 1))  # (n_sims, n_assets, n_days)
            return {
                "final_values": final_value,
                "port_value": port_value,
                "total_return": total_return,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe_ratio": np.mean(sharpe),
                "max_drawdown": max_drawdown,
                "success_rate": success_rate,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "var_99": var_99,
                "cvar_99": cvar_99,
                "confidence_intervals": dict(zip(self.config.confidence_levels, confidence_intervals)),
                "paths": port_value,  # Add this line for downstream analysis
                "expected_return": np.mean(total_return),
                "simulation_volatility": np.mean(ann_vol),
                "annualized_return": np.mean(ann_return),
                "annualized_vol": np.mean(ann_vol),
                "validation": self.validate_simulation(sim_price_paths, market_data),
            }

        # --- Regime Switching ---
        elif simulation_mode == 'regime_switching':
            print("Using regime switching for simulation.")
            # Get regimes from market_data if available
            regimes = market_data.get("regimes")
            if regimes is None:
                raise ValueError("Regime labels must be provided in market_data['regimes'] for regime switching simulation.")
            # Align regimes index to returns index
            regimes = pd.Series(regimes, index=regimes.index) if not isinstance(regimes, pd.Series) else regimes
            regimes = regimes.reindex(returns.index)
            # Drop any NaNs (dates where regime is not defined)
            valid_mask = regimes.notna()
            regimes = regimes[valid_mask]
            returns_aligned = returns.loc[regimes.index]
            # Now use returns_aligned and regimes for regime-specific stats
            unique_regimes = pd.unique(regimes)
            n_regimes = len(unique_regimes)
            regime_means = []
            regime_stds = []
            for reg in unique_regimes:
                mask = regimes == reg
                # Use DAILY mean and std for simulation (not annualized)
                regime_means.append(returns_aligned[mask].mean().values - risk_premium / 252)
                regime_stds.append(returns_aligned[mask].std().values * vol_scale)
            # Estimate regime transition matrix
            regime_labels = regimes.values  # Use the aligned regimes as labels
            trans_mat = np.zeros((n_regimes, n_regimes))
            for i in range(1, len(regime_labels)):
                prev_reg = np.where(unique_regimes == regime_labels[i-1])[0][0]
                curr_reg = np.where(unique_regimes == regime_labels[i])[0][0]
                trans_mat[prev_reg, curr_reg] += 1
            # Normalize and enforce minimum self-transition probability for realism
            for i in range(n_regimes):
                row_sum = trans_mat[i].sum()
                if row_sum > 0:
                    trans_mat[i] /= row_sum
                # Enforce minimum self-transition probability (e.g., 0.7)
                if trans_mat[i, i] < 0.7:
                    diff = 0.7 - trans_mat[i, i]
                    trans_mat[i, i] += diff
                    # Reduce other probabilities proportionally
                    if n_regimes > 1:
                        for j in range(n_regimes):
                            if j != i:
                                trans_mat[i, j] *= (1 - 0.7) / (1 - trans_mat[i, i] + diff)
            # Simulate regime paths
            paths = np.zeros((n_sims, n_assets, n_days))
            for sim in range(n_sims):
                regime = np.random.choice(n_regimes)
                log_returns = np.zeros((n_assets, n_days))
                for t in range(n_days):
                    # Draw regime
                    if t > 0:
                        regime = np.random.choice(n_regimes, p=trans_mat[regime])
                    # Simulate correlated daily log returns for this regime
                    z = np.random.standard_normal(n_assets)
                    L = np.linalg.cholesky(corr)
                    asset_ret = regime_means[regime] + regime_stds[regime] * np.dot(L, z)
                    log_returns[:, t] = asset_ret
                # Compound log returns
                paths[sim, :, :] = np.exp(np.cumsum(log_returns, axis=1))
            # Block bootstrapping: for even more realism, set simulation_mode: block_bootstrap in config

        # --- Standard GBM (default) ---
        else:
            print("Using standard GBM for simulation.")
            paths = np.zeros((n_sims, n_assets, n_days))
            for i in range(n_days):
                z = np.random.standard_normal((n_sims, n_assets))
                sigma_t = sigma
                L = np.linalg.cholesky(corr)
                paths[:, :, i] = np.dot(z, L.T) * sigma_t
            for i in range(n_assets):
                daily_mu = mu[i]
                daily_sigma = sigma[i]
                paths[:, i, :] = (daily_mu - 0.5 * daily_sigma ** 2) + daily_sigma * paths[:, i, :]
            paths = np.exp(np.cumsum(paths, axis=2))

        # --- Portfolio aggregation and stats (same as before) ---
        portfolio_paths = np.sum(paths, axis=1)
        initial_value = portfolio_paths[:, 0]
        final_value = portfolio_paths[:, -1]
        portfolio_return = (final_value / initial_value) - 1
        n_years = n_days / 252
        annualized_return = (final_value / initial_value) ** (1 / n_years) - 1
        daily_log_returns = np.diff(np.log(portfolio_paths), axis=1)
        annualized_vol = np.std(daily_log_returns) * np.sqrt(252)
        mean_annualized_return = np.mean(annualized_return)
        mean_annualized_vol = np.mean(annualized_vol)
        mean_portfolio_return = np.mean(portfolio_return)
        sharpe_ratio = (mean_annualized_return - self.config.risk_free_rate) / mean_annualized_vol if mean_annualized_vol > 0 else 0
        sharpe_ratio = min(sharpe_ratio, 3.0)
        # VaR, CVaR, confidence intervals on returns
        var_95 = np.percentile(portfolio_return, 5)
        cvar_95 = portfolio_return[portfolio_return <= var_95].mean()
        var_99 = np.percentile(portfolio_return, 1)
        cvar_99 = portfolio_return[portfolio_return <= var_99].mean()
        confidence_intervals = np.percentile(portfolio_return, [level * 100 for level in self.config.confidence_levels])
        results = {
            "paths": paths,
            "final_values": final_value,
            "confidence_intervals": dict(zip(self.config.confidence_levels, confidence_intervals)),
            "expected_return": mean_portfolio_return,
            "simulation_volatility": mean_annualized_vol,
            "annualized_return": mean_annualized_return,
            "annualized_vol": mean_annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "var_95": var_95,
            "var_99": var_99,
            "statistics": self._calculate_statistics(final_value, sharpe_ratio),
            "validation": self.validate_simulation(paths, market_data),
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
                scale_factor = (
                    1000 / returns[col].std()
                )  # This will make std around 1000
                scaled_returns = returns[col] * scale_factor

                # Fit GARCH model with rescaling disabled to avoid warnings
                model = arch_model(scaled_returns, vol="GARCH", p=1, q=1, rescale=False)
                results = model.fit(disp="off")

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

    def _calculate_statistics(self, portfolio_values: np.ndarray, sharpe_ratio: float) -> Dict:
        """Calculate detailed statistics of simulation results"""
        statistics = {
            "mean": np.mean(portfolio_values),
            "median": np.median(portfolio_values),
            "std": np.std(portfolio_values),
            "skew": scipy_stats.skew(portfolio_values),
            "kurtosis": scipy_stats.kurtosis(portfolio_values),
            "sharpe_ratio": sharpe_ratio,
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
            "bias": np.mean(predictions - np.mean(portfolio_values)),
            "variance": np.var(predictions),
            "cross_val_score": np.std(predictions) / np.mean(predictions),  # CV score
        }

    def validate_simulation(self, paths: np.ndarray, market_data: Dict) -> Dict:
        """Validate simulation results"""
        try:
            historical_corr = market_data["returns"].corr()
            simulated_returns = np.diff(np.log(paths), axis=2)
            simulated_corr = np.corrcoef(simulated_returns.mean(axis=0))

            validation = {
                "positive_values": np.all(paths > 0),
                "correlation_preservation": np.allclose(
                    historical_corr, simulated_corr, atol=0.1
                ),
                "reasonable_returns": self._check_return_reasonability(paths),
                "volatility_alignment": self._check_volatility_alignment(
                    paths, market_data
                ),
            }
            # Add overall_valid key
            validation["overall_valid"] = all(validation.values())
            return validation

        except Exception as e:
            return {
                "positive_values": False,
                "correlation_preservation": False,
                "reasonable_returns": False,
                "volatility_alignment": False,
                "overall_valid": False,
                "error": str(e),
            }

    def _check_return_reasonability(self, paths: np.ndarray) -> bool:
        """Check if returns are within reasonable bounds"""
        returns = np.diff(np.log(paths), axis=2)
        return np.all((-0.5 <= returns) & (returns <= 0.5))

    def _check_volatility_alignment(self, paths: np.ndarray, market_data: Dict) -> bool:
        """Check if simulation volatility aligns with historical volatility"""
        historical_vol = market_data["returns"].std().values
        simulated_vol = np.std(np.diff(np.log(paths), axis=2), axis=(0, 2))
        return np.allclose(historical_vol, simulated_vol, rtol=0.2)

    def _validate_results(self, results: Dict) -> bool:
        """
        Validate simulation results.

        Args:
            results: Dictionary containing simulation results

        Returns:
            bool: True if results are valid, False otherwise
        """
        try:
            # Check if all required keys are present
            required_keys = [
                "paths",
                "final_values",
                "confidence_intervals",
                "expected_return",
                "simulation_volatility",
                "var_95",
                "var_99",
                "statistics",
                "validation",
            ]
            if not all(key in results for key in required_keys):
                return False

            # Check if values are reasonable
            if not (
                isinstance(results["paths"], np.ndarray)
                and isinstance(results["final_values"], np.ndarray)
                and isinstance(results["confidence_intervals"], dict)
                and isinstance(results["statistics"], dict)
                and isinstance(results["validation"], dict)
            ):
                return False

            # Check if numerical values are finite
            if not (
                np.isfinite(results["expected_return"])
                and np.isfinite(results["simulation_volatility"])
                and np.isfinite(results["var_95"])
                and np.isfinite(results["var_99"])
            ):
                return False

            # Check if paths have correct shape
            if len(results["paths"].shape) != 3:  # (n_sims, n_assets, n_days)
                return False

            return True

        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def analyze_results(self, results: Dict) -> Dict[str, float]:
        """
        Perform detailed analysis of simulation results.

        Args:
            results: Dictionary containing simulation results

        Returns:
            Dict containing additional analysis metrics
        """
        portfolio_values = results["final_values"]
        # Calculate additional risk metrics
        cvar_95 = np.mean(portfolio_values[portfolio_values <= results["var_95"]])
        cvar_99 = np.mean(portfolio_values[portfolio_values <= results["var_99"]])
        # Robust axis handling for all path-based calculations
        paths = results["paths"]
        if paths.ndim == 3:
            # (n_sims, n_assets, n_days)
            daily_returns = np.diff(np.log(paths), axis=2)
            cumulative_returns = np.cumprod(1 + daily_returns, axis=2)
            rolling_max = np.maximum.accumulate(cumulative_returns, axis=2)
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            # For portfolio-level metrics, aggregate over assets
            # Use mean across assets for each sim and day
            portfolio_daily_returns = daily_returns.mean(axis=1)
        elif paths.ndim == 2:
            # (n_sims, n_days)
            daily_returns = np.diff(np.log(paths), axis=1)
            cumulative_returns = np.cumprod(1 + daily_returns, axis=1)
            rolling_max = np.maximum.accumulate(cumulative_returns, axis=1)
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            portfolio_daily_returns = daily_returns
        else:
            raise ValueError(f"Unexpected shape for paths: {paths.shape}")
        max_drawdown = np.min(drawdowns)
        # Annualized return/volatility from daily returns
        mean_daily_return = np.mean(portfolio_daily_returns)
        std_daily_return = np.std(portfolio_daily_returns)
        annualized_return = (1 + mean_daily_return) ** 252 - 1
        annualized_vol = std_daily_return * np.sqrt(252)
        # Information ratio (assuming risk-free rate as benchmark)
        excess_returns = portfolio_daily_returns - self.config.risk_free_rate / 252
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else np.nan
        # Sortino ratio
        downside_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else np.nan
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_vol if downside_vol and downside_vol > 0 else np.nan
        analysis = {
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "max_drawdown": max_drawdown,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "information_ratio": information_ratio,
            "sortino_ratio": sortino_ratio,
            "success_rate": np.mean(portfolio_values > 1.0),  # Probability of positive return
        }
        return analysis

    def save_results(self, results: Dict, filepath: str) -> None:
        """
        Save simulation results to file.

        Args:
            results: Dictionary containing simulation results
            filepath: Path to save results
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {
                "confidence_intervals": results["confidence_intervals"],
                "expected_return": float(results["expected_return"]),
                "simulation_volatility": float(results["simulation_volatility"]),
                "var_95": float(results["var_95"]),
                "var_99": float(results["var_99"]),
                "statistics": {
                    k: float(v) if isinstance(v, np.number) else v
                    for k, v in results["statistics"].items()
                },
                "validation": results["validation"],
                "analysis": self.analyze_results(results)
            }
            
            # Save to file
            import json
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            
            print(f"Results saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            raise

    def get_summary_statistics(self, results: Dict) -> pd.DataFrame:
        """
        Generate summary statistics of simulation results.

        Args:
            results: Dictionary containing simulation results

        Returns:
            DataFrame containing summary statistics
        """
        analysis = self.analyze_results(results)
        
        summary = pd.DataFrame({
            "Metric": [
                "Expected Return",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Information Ratio",
                "VaR (95%)",
                "CVaR (95%)",
                "Maximum Drawdown",
                "Success Rate"
            ],
            "Value": [
                f"{results['expected_return']:.2%}",
                f"{analysis['annualized_return']:.2%}",
                f"{analysis['annualized_volatility']:.2%}",
                f"{results['statistics']['sharpe_ratio']:.2f}",
                f"{analysis['sortino_ratio']:.2f}",
                f"{analysis['information_ratio']:.2f}",
                f"{results['var_95']:.2%}",
                f"{analysis['cvar_95']:.2%}",
                f"{analysis['max_drawdown']:.2%}",
                f"{analysis['success_rate']:.2%}"
            ]
        })
        
        return summary

    def __str__(self) -> str:
        """String representation of the Monte Carlo simulation configuration"""
        return (
            f"Monte Carlo Simulation:\n"
            f"  Number of simulations: {self.config.n_sims}\n"
            f"  Number of days: {self.config.n_days}\n"
            f"  Risk-free rate: {self.config.risk_free_rate:.2%}\n"
            f"  Distribution: {self.config.distribution}\n"
            f"  Confidence levels: {self.config.confidence_levels}"
        )

    def plot_simulation_paths(self, results: Dict, title: str = "Monte Carlo Simulation Paths") -> None:
        """
        Plot simulation paths.

        Args:
            results: Dictionary containing simulation results
            title: Title for the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")
            
            # Plot a subset of paths for better visualization
            n_paths_to_plot = min(100, self.config.n_sims)
            paths = results["paths"]
            
            # Calculate portfolio values over time
            portfolio_paths = np.sum(paths, axis=1)
            time_points = np.arange(self.config.n_days)
            
            # Plot paths
            for i in range(n_paths_to_plot):
                plt.plot(time_points, portfolio_paths[i], alpha=0.1, color='blue')
                
            # Plot mean path
            mean_path = np.mean(portfolio_paths, axis=0)
            plt.plot(time_points, mean_path, color='red', linewidth=2, label='Mean Path')
            
            # Plot confidence intervals
            percentiles = np.percentile(portfolio_paths, [5, 95], axis=0)
            plt.fill_between(time_points, percentiles[0], percentiles[1], 
                           color='gray', alpha=0.2, label='90% Confidence Interval')
            
            plt.title(title)
            plt.xlabel('Trading Days')
            plt.ylabel('Portfolio Value')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Please install matplotlib and seaborn to use plotting functionality")
        except Exception as e:
            print(f"Error plotting simulation paths: {str(e)}")

    def plot_distribution(self, results: Dict, title: str = "Final Portfolio Value Distribution") -> None:
        """
        Plot the distribution of final portfolio values.

        Args:
            results: Dictionary containing simulation results
            title: Title for the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid")
            
            # Plot distribution
            sns.histplot(results["final_values"], kde=True)
            
            # Add vertical lines for key statistics
            plt.axvline(results["expected_return"], color='red', linestyle='--', 
                       label=f'Expected Return: {results["expected_return"]:.2%}')
            plt.axvline(results["var_95"], color='orange', linestyle='--',
                       label=f'95% VaR: {results["var_95"]:.2%}')
            
            # Add labels and title
            plt.title(title)
            plt.xlabel('Portfolio Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Please install matplotlib and seaborn to use plotting functionality")
        except Exception as e:
            print(f"Error plotting distribution: {str(e)}")

    def plot_risk_metrics(self, results: Dict) -> None:
        """
        Plot key risk metrics.

        Args:
            results: Dictionary containing simulation results
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            analysis = self.analyze_results(results)
            
            # Prepare data
            metrics = {
                'VaR (95%)': results['var_95'],
                'CVaR (95%)': analysis['cvar_95'],
                'Max Drawdown': analysis['max_drawdown'],
                'Annualized Vol': analysis['annualized_volatility']
            }
            
            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid")
            
            # Create bar plot
            plt.bar(metrics.keys(), [abs(v) for v in metrics.values()])
            
            # Customize plot
            plt.title('Risk Metrics')
            plt.xticks(rotation=45)
            plt.ylabel('Absolute Value')
            
            # Add value labels on top of bars
            for i, (metric, value) in enumerate(metrics.items()):
                plt.text(i, abs(value), f'{value:.2%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Please install matplotlib and seaborn to use plotting functionality")
        except Exception as e:
            print(f"Error plotting risk metrics: {str(e)}")

    def generate_report(self, results: Dict, output_dir: str = "reports") -> None:
        """
        Generate a comprehensive report of simulation results.

        Args:
            results: Dictionary containing simulation results
            output_dir: Directory to save the report
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f"monte_carlo_report_{timestamp}")
            
            # Save numerical results
            self.save_results(results, f"{report_path}.json")
            
            # Save summary statistics to CSV
            summary_stats = self.get_summary_statistics(results)
            summary_stats.to_csv(f"{report_path}_summary.csv", index=False)
            
            # Generate plots
            import matplotlib.pyplot as plt
            
            # Save simulation paths plot
            self.plot_simulation_paths(results)
            plt.savefig(f"{report_path}_paths.png")
            plt.close()
            
            # Save distribution plot
            self.plot_distribution(results)
            plt.savefig(f"{report_path}_distribution.png")
            plt.close()
            
            # Save risk metrics plot
            self.plot_risk_metrics(results)
            plt.savefig(f"{report_path}_risk_metrics.png")
            plt.close()
            
            print(f"Report generated successfully in {output_dir}")
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise
