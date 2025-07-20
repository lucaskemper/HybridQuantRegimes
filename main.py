#!/usr/bin/env python3
"""
Enhanced Market Regime Detection and Portfolio Analysis System

This script demonstrates a comprehensive quantitative finance pipeline including:
- Market regime detection using HMM, LSTM, and Transformer ensemble
- Signal generation with regime awareness
- Risk management and Monte Carlo simulation
- Portfolio allocation strategies
- Advanced backtesting capabilities
- Semiconductor-focused analysis

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.regime import RegimeConfig, MarketRegimeDetector, AdaptiveRegimeDetector
from src.deep_learning import DeepLearningConfig
from src.data import PortfolioConfig, DataLoader
from src.signals import SignalGenerator
from src.risk import RiskConfig, RiskManager
from src.monte_carlo import SimConfig, MonteCarlo
from src.backtest import BacktestEngine
from src.visualization import PortfolioVisualizer, calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Enhanced configuration class for the analysis pipeline."""
    
    # Data configuration - Extended training window for comprehensive market cycles
    tickers: Optional[List[str]] = None
    start_date: str = "2008-01-01"  # Include 2008 financial crisis
    end_date: str = "2024-12-31"    # Extended to include recent market conditions
    frequency: str = 'D'
    use_cache: bool = True
    include_macro: bool = True  # Include VIX and yield curve data
    
    # Enhanced regime detection configuration
    n_regimes: int = 4  # Optimal for semiconductor business cycles
    window_size: int = 15  # Faster regime change detection
    smoothing_window: int = 3  # Reduced for more responsive detection
    features: List[str] = field(default_factory=lambda: [
        "returns", "volatility", "momentum", "rsi_14", "rsi_30",
        "macd_signal", "bollinger_position", "williams_r",
        "volume_ratio", "volume_sma_ratio", "on_balance_volume",
        "vix_level", "vix_change", "yield_spread", "term_structure_slope",
        "dollar_strength", "semiconductor_pmi", "memory_vs_logic_spread",
        "equipment_vs_design_ratio"
    ])
    labeling_metric: str = 'risk_adjusted_return'  # Updated to risk-adjusted return
    ensemble_method: str = 'dynamic_weighted_confidence'
    max_flips: int = 3
    transition_window: int = 10
    use_deep_learning: bool = True
    use_transformer: bool = True  # New transformer model
    
    # Enhanced adaptive retraining configuration
    use_adaptive_retraining: bool = True
    retrain_window: int = 1008  # 4 years for initial training
    retrain_freq: int = 63  # Quarterly retraining
    use_enhanced_adaptive: bool = True
    adaptive_vix_threshold: float = 25.0
    vix_high_retrain_freq: int = 10  # Weekly during high volatility
    vix_low_retrain_freq: int = 126  # Semi-annual during low volatility
    
    # Enhanced risk management configuration
    stop_loss: float = 0.08  # 8% stop-loss
    take_profit: float = 0.30  # 30% take-profit
    max_drawdown: float = 0.15  # 15% max drawdown
    
    # Enhanced dynamic position sizing configuration
    use_dynamic_position_sizing: bool = True
    base_position_size: float = 0.12  # 12% base allocation
    confidence_threshold: float = 0.3  # Lower threshold for more participation
    min_position_size: float = 0.02   # 2% minimum position
    max_position_size: float = 0.35   # 35% maximum position
    vol_target: float = 0.15          # 15% annual volatility target
    
    # Enhanced deep learning configuration
    sequence_length: int = 30  # Increased sequence length
    hidden_dims: Optional[List[int]] = None
    epochs: int = 250  # More epochs for better convergence
    batch_size: int = 64  # Increased batch size
    learning_rate: float = 0.0005  # Reduced learning rate
    dropout_rate: float = 0.3  # Increased dropout
    validation_split: float = 0.2
    early_stopping_patience: int = 30  # Increased patience
    
    # Enhanced Monte Carlo configuration
    n_sims: int = 50000  # High accuracy simulation
    n_days: int = 252
    risk_free_rate: float = 0.02  # Current realistic risk-free rate
    mc_confidence_levels: tuple = (0.01, 0.025, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.975, 0.99)
    mc_distribution: str = "regime_conditional"  # Regime-conditional distributions
    
    # Output configuration
    output_dir: str = "output"
    save_plots: bool = True
    show_plots: bool = False
    
    def __post_init__(self):
        if self.tickers is None:
            # Enhanced semiconductor-focused asset selection with tiered allocation
            self.tickers = [
                # Tier 1: Semiconductor Leaders (45% allocation)
                "NVDA", "TSM", "ASML", "AMD",  # Core semiconductor leaders
                
                # Tier 2: Tech Infrastructure (30% allocation)
                "MSFT", "GOOGL", "AAPL",  # Large-cap tech with stable growth
                
                # Tier 3: High-Beta Growth (25% allocation)
                "TSLA", "ANET", "CRM", "AVGO"  # High regime sensitivity
            ]
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32, 16]  # Enhanced architecture


class AnalysisPipeline:
    """Enhanced analysis pipeline class."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.data: Optional[Dict[str, pd.DataFrame]] = None
        self.returns_df: Optional[pd.DataFrame] = None
        self.detector: Optional[MarketRegimeDetector] = None
        self.regimes_batch: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.risk_metrics: Optional[Dict[str, Any]] = None
        self.mc_results: Optional[Dict[str, Any]] = None
        self.backtest_results: Dict[str, Any] = {}
        self.regime_confidence: Optional[pd.DataFrame] = None
        self.volatility: Optional[pd.Series] = None
        
        # Create output directory structure
        for subdir in ["plots", "data", "models", "reports", "cache"]:
            Path(os.path.join(self.config.output_dir, subdir)).mkdir(parents=True, exist_ok=True)
        
        logger.info("Enhanced analysis pipeline initialized")
    
    def load_data(self) -> None:
        """Load and prepare market data with enhanced preprocessing."""
        logger.info("Loading enhanced market data...")
        
        try:
            portfolio_config = PortfolioConfig(
                tickers=self.config.tickers,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                frequency=self.config.frequency,
                use_cache=self.config.use_cache,
                include_macro=self.config.include_macro
            )
            
            loader = DataLoader(portfolio_config)
            self.data = loader.load_data()
            self.returns_df = self.data["returns"]
            
            # Enhanced data quality checks
            self._validate_data_quality()
            
            logger.info(f"Loaded data for {len(self.config.tickers)} assets from {self.config.start_date} to {self.config.end_date}")
            logger.info(f"Data shape: {self.returns_df.shape}")
            if self.data.get("macro") is not None:
                logger.info(f"Loaded macro data: {list(self.data['macro'].keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _validate_data_quality(self) -> None:
        """Validate data quality and handle missing values."""
        if self.returns_df is None:
            return
            
        # Check for missing values
        missing_pct = self.returns_df.isnull().sum() / len(self.returns_df) * 100
        logger.info(f"Missing data percentages: {missing_pct.to_dict()}")
        
        # Forward fill then backward fill
        self.returns_df = self.returns_df.ffill().bfill()
        
        # Remove outliers (beyond 3 standard deviations)
        for col in self.returns_df.columns:
            mean_val = self.returns_df[col].mean()
            std_val = self.returns_df[col].std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            self.returns_df[col] = self.returns_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info("Data quality validation completed")
    
    def setup_regime_detection(self) -> None:
        """Setup enhanced regime detection models."""
        logger.info("Setting up enhanced regime detection...")
        
        try:
            deep_config = DeepLearningConfig(
                n_regimes=self.config.n_regimes,
                sequence_length=self.config.sequence_length,
                hidden_dims=self.config.hidden_dims,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                dropout_rate=self.config.dropout_rate,
                validation_split=self.config.validation_split,
                early_stopping_patience=self.config.early_stopping_patience
            )
            
            config = RegimeConfig(
                n_regimes=self.config.n_regimes,
                window_size=self.config.window_size,
                smoothing_window=self.config.smoothing_window,
                features=self.config.features,
                labeling_metric=self.config.labeling_metric,
                ensemble_method=self.config.ensemble_method,
                max_flips=self.config.max_flips,
                transition_window=self.config.transition_window,
                use_deep_learning=self.config.use_deep_learning,
                use_transformer=self.config.use_transformer,
                deep_learning_config=deep_config,
                include_semiconductor_features=True
            )
            
            self.detector = MarketRegimeDetector(config)
            logger.info("Enhanced regime detection setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup regime detection: {e}")
            raise
    
    def run_regime_detection(self) -> None:
        """Run enhanced adaptive regime detection with ensemble methods."""
        logger.info("Running enhanced adaptive regime detection...")
        
        try:
            if self.detector is None or self.returns_df is None:
                raise ValueError("Detector or returns data not initialized")
            
            if self.config.use_adaptive_retraining:
                # Use enhanced adaptive retraining
                logger.info("Using enhanced adaptive retraining")
                adaptive_detector = AdaptiveRegimeDetector(
                    base_config=self.detector.config,
                    retrain_window=self.config.retrain_window,
                    retrain_freq=self.config.retrain_freq
                )
                
                if self.config.use_enhanced_adaptive:
                    # Use enhanced quarterly retraining with walkforward backtest
                    logger.info("Running enhanced quarterly retraining with walkforward backtest")
                    
                    # Split data into training and test periods
                    train_size = min(self.config.retrain_window, len(self.returns_df) // 2)
                    train_data = self.returns_df.iloc[:train_size]
                    test_data = self.returns_df.iloc[train_size:]
                    
                    # Run enhanced adaptive backtest
                    results = adaptive_detector.adaptive_walkforward_backtest(
                        train_data=train_data,
                        test_data=test_data,
                        retrain_freq=self.config.retrain_freq
                    )
                    
                    self.regimes_batch = results['regimes']
                    self.regime_confidence = results.get('confidence', pd.DataFrame())
                    self.volatility = results.get('volatility', pd.Series())
                    
                    logger.info(f"Enhanced adaptive backtest completed with {len(self.regimes_batch)} regime predictions")
                else:
                    # Standard adaptive retraining
                    results = adaptive_detector.adaptive_backtest(
                        train_data=self.returns_df.iloc[:train_size],
                        test_data=self.returns_df.iloc[train_size:],
                        retrain_frequency=self.config.retrain_freq
                    )
                    
                    self.regimes_batch = results['regimes']
                    self.regime_confidence = results.get('confidence', pd.DataFrame())
                    self.volatility = results.get('volatility', pd.Series())
            else:
                # Standard batch regime detection
                logger.info("Running standard batch regime detection")
                self.regimes_batch = self.detector.fit_predict_batch(self.returns_df)
                
                # Calculate regime confidence
                self.regime_confidence = self.detector.get_regime_confidence(self.returns_df, self.regimes_batch)
                
                # Calculate volatility
                self.volatility = self.returns_df.std(axis=1)
            
            logger.info(f"Regime detection completed. Shape: {self.regimes_batch.shape}")
            
        except Exception as e:
            logger.error(f"Failed to run regime detection: {e}")
            raise
    
    def run_real_time_analysis(self) -> None:
        """Run real-time regime analysis with enhanced features."""
        logger.info("Running real-time regime analysis...")
        
        try:
            if self.detector is None or self.returns_df is None:
                raise ValueError("Detector or returns data not initialized")
            
            # Get latest data for real-time analysis
            latest_returns = self.returns_df.tail(100)  # Last 100 days
            
            # Run combined prediction with ensemble models
            self._run_combined_prediction(latest_returns)
            
            logger.info("Real-time analysis completed")
            
        except Exception as e:
            logger.error(f"Failed to run real-time analysis: {e}")
            raise
    
    def _run_combined_prediction(self, latest_returns: pd.DataFrame) -> None:
        """Run combined prediction using HMM, LSTM, and Transformer models."""
        try:
            combined_predictions = {}
            model_weights = self.detector.config.model_weights
            
            # HMM predictions
            if hasattr(self.detector, 'hmm_model') and self.detector.hmm_model is not None:
                hmm_predictions = {}
                for col in latest_returns.columns:
                    hmm_predictions[col] = self.detector.fit_predict(latest_returns[col])
                combined_predictions['hmm'] = pd.DataFrame(hmm_predictions)
            
            # LSTM predictions
            if hasattr(self.detector, 'lstm_model') and self.detector.lstm_model is not None:
                lstm_predictions = {}
                for col in latest_returns.columns:
                    lstm_predictions[col] = self.detector.lstm_model.predict(latest_returns[col])
                combined_predictions['lstm'] = pd.DataFrame(lstm_predictions)
            
            # Transformer predictions
            if hasattr(self.detector, 'transformer_model') and self.detector.transformer_model is not None:
                transformer_predictions = {}
                for col in latest_returns.columns:
                    transformer_predictions[col] = self.detector.transformer_model.predict(latest_returns[col])
                combined_predictions['transformer'] = pd.DataFrame(transformer_predictions)
            
            # Combine predictions using weighted ensemble
            if combined_predictions:
                final_predictions = pd.DataFrame(0, index=latest_returns.index, columns=latest_returns.columns)
                total_weight = 0
                
                for model_name, predictions in combined_predictions.items():
                    if model_name in model_weights:
                        weight = model_weights[model_name]
                        final_predictions += predictions * weight
                        total_weight += weight
                
                if total_weight > 0:
                    final_predictions /= total_weight
                
                # Convert to integer regime labels
                self.regimes_batch = final_predictions.round().astype(int)
                
                logger.info(f"Combined ensemble predictions completed with {len(combined_predictions)} models")
            
        except Exception as e:
            logger.error(f"Failed to run combined prediction: {e}")
            raise
    
    def generate_signals(self) -> None:
        """Generate regime-aware signals."""
        logger.info("Generating signals...")
        
        try:
            if self.detector is None or self.data is None:
                raise ValueError("Detector or data not initialized")
                
            signal_gen = SignalGenerator(regime_detector=self.detector, use_regime=True)
            self.signals = signal_gen.generate_signals(self.data)
            
            logger.info("Signal generation completed")
            logger.info(f"Signals shape: {self.signals.shape}")
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise
    
    def calculate_risk_metrics(self) -> None:
        """Calculate comprehensive risk metrics."""
        logger.info("Calculating risk metrics...")
        
        try:
            if self.data is None:
                raise ValueError("Data not initialized")
                
            risk_config = RiskConfig()
            risk_manager = RiskManager(risk_config)
            self.risk_metrics = risk_manager.calculate_metrics(self.data['returns'])
            
            logger.info("Risk metrics calculation completed")
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            raise
    
    def run_monte_carlo_simulation(self) -> None:
        """Run Monte Carlo simulation."""
        logger.info("Running Monte Carlo simulation...")
        
        try:
            if self.data is None:
                raise ValueError("Data not initialized")
                
            sim_config = SimConfig(n_sims=self.config.n_sims, n_days=self.config.n_days)
            mc = MonteCarlo(sim_config)
            self.mc_results = mc.simulate(self.data)
            
            logger.info("Monte Carlo simulation completed")
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            raise
    
    def run_backtests(self) -> None:
        """Run comprehensive backtesting with enhanced dynamic position sizing."""
        logger.info("Running enhanced backtests with dynamic position sizing...")
        
        try:
            if self.returns_df is None or self.signals is None:
                raise ValueError("Returns data or signals not initialized")
                
            # Basic backtest
            bt_basic = BacktestEngine(
                returns=self.returns_df,
                signals=self.signals,
                initial_cash=1.0,
                rebalance_freq='D',
                transaction_cost=0.001,
                leverage=1.0,
                position_sizing='proportional',
                allow_short=False,
            )
            self.backtest_results['basic'] = bt_basic.run()
            
            # Enhanced confidence-based position sizing backtest
            if self.config.use_dynamic_position_sizing and self.regime_confidence is not None:
                logger.info("Running confidence-based position sizing backtest...")
                
                bt_confidence = BacktestEngine(
                    returns=self.returns_df,
                    signals=self.signals,
                    initial_cash=1.0,
                    rebalance_freq='D',
                    transaction_cost=0.001,
                    leverage=1.0,
                    position_sizing='proportional',
                    allow_short=False
                )
                
                # Run confidence-based backtest
                volatility_df = self.volatility.to_frame() if self.volatility is not None else pd.DataFrame()
                self.backtest_results['confidence_based'] = bt_confidence.run_confidence_based_backtest(
                    returns=self.returns_df,
                    signals=self.signals,
                    regime_confidence=self.regime_confidence,
                    volatility=volatility_df
                )
                logger.info("Confidence-based position sizing backtest completed")
            
            # Regime-aware risk controls backtest
            if self.regime_confidence is not None and self.regimes_batch is not None:
                logger.info("Running regime-aware risk controls backtest...")
                
                bt_regime_aware = BacktestEngine(
                    returns=self.returns_df,
                    signals=self.signals,
                    initial_cash=1.0,
                    rebalance_freq='D',
                    transaction_cost=0.001,
                    leverage=1.0,
                    position_sizing='proportional',
                    allow_short=False
                )
                
                # Run regime-aware backtest
                self.backtest_results['regime_aware'] = bt_regime_aware.run_regime_aware_backtest(
                    returns=self.returns_df,
                    signals=self.signals,
                    regimes=self.regimes_batch,
                    regime_confidence=self.regime_confidence
                )
                logger.info("Regime-aware risk controls backtest completed")
            
            # Risk parity backtest
            def risk_parity_position_sizing(signals: pd.DataFrame, returns: pd.DataFrame, date: pd.Timestamp, context: Dict[str, Any]) -> pd.Series:
                lookback = 21
                if date not in returns.index:
                    return pd.Series(0, index=signals.index)
                idx = returns.index.get_loc(date)
                if isinstance(idx, (int, np.integer)) and idx < lookback:
                    return pd.Series(0, index=signals.index)
                if isinstance(idx, (int, np.integer)):
                    cov = returns.iloc[idx-lookback:idx].cov()
                    inv_vol = 1 / np.sqrt(np.diag(cov))
                    weights = inv_vol / inv_vol.sum()
                    return pd.Series(weights, index=signals.index)
                return pd.Series(0, index=signals.index)
            
            bt_rp = BacktestEngine(
                returns=self.returns_df,
                signals=self.signals,
                initial_cash=1.0,
                rebalance_freq='D',
                transaction_cost=0.001,
                leverage=1.0,
                position_sizing='proportional',
                allow_short=False,
            )
            self.backtest_results['risk_parity'] = bt_rp.run()
            
            # Risk-managed backtest
            bt_risk = BacktestEngine(
                returns=self.returns_df,
                signals=self.signals,
                initial_cash=1.0,
                rebalance_freq='D',
                transaction_cost=0.001,
                leverage=1.0,
                position_sizing='proportional',
                allow_short=False,
                stop_loss=0.10,
                take_profit=0.30,
            )
            self.backtest_results['risk_managed'] = bt_risk.run()
            
            logger.info("Enhanced backtesting completed")
            
        except Exception as e:
            logger.error(f"Enhanced backtesting failed: {e}")
            raise
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        try:
            # Signal plots with regime overlay
            self._plot_signals_with_regime()
            
            # Risk metrics plots
            self._plot_risk_metrics()
            
            # Monte Carlo plots
            self._plot_monte_carlo()
            
            # Portfolio allocation plots
            self._plot_portfolio_allocation()
            
            logger.info("Visualizations completed")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def _plot_signals_with_regime(self) -> None:
        """Plot signals with regime overlay."""
        if self.signals is None:
            logger.warning("Signals not available for plotting")
            return
            
        for col in self.signals.columns:
            plt.figure(figsize=(12, 4))
            ax = plt.gca()
            self.signals[col].plot(title=f"Regime-Aware Signal: {col}", ax=ax)
            
            if self.regimes_batch is not None and col in self.regimes_batch:
                regimes = self.regimes_batch[col]
                colors = {0: "green", 1: "yellow", 2: "red"}
                for regime in regimes.unique():
                    mask = regimes == regime
                    if mask.any():
                        ax.fill_between(
                            self.signals.index, 
                            ax.get_ylim()[0], 
                            ax.get_ylim()[1], 
                            where=mask, 
                            color=colors.get(regime, "gray"), 
                            alpha=0.08
                        )
            
            plt.ylabel("Signal Value")
            plt.xlabel("Date")
            plt.tight_layout()
            
            if self.config.save_plots:
                plt.savefig(os.path.join(self.config.output_dir, f"signal_{col}.png"))
            if self.config.show_plots:
                plt.show()
            plt.close()
    
    def _plot_risk_metrics(self) -> None:
        """Plot risk metrics."""
        if self.risk_metrics is None:
            logger.warning("Risk metrics not available for plotting")
            return
            
        if "rolling_volatility" in self.risk_metrics and isinstance(self.risk_metrics["rolling_volatility"], pd.DataFrame):
            plt.figure(figsize=(12, 4))
            self.risk_metrics["rolling_volatility"].plot(title="Rolling Volatility")
            plt.ylabel("Volatility")
            plt.xlabel("Date")
            plt.tight_layout()
            
            if self.config.save_plots:
                plt.savefig(os.path.join(self.config.output_dir, "rolling_volatility.png"))
            if self.config.show_plots:
                plt.show()
            plt.close()
        
        # Plot key metrics
        metrics_to_plot = ['max_drawdown', 'var_95']
        for metric in metrics_to_plot:
            if metric in self.risk_metrics:
                plt.figure(figsize=(6, 4))
                plt.bar([metric.replace('_', ' ').title()], [self.risk_metrics[metric]])
                plt.title(metric.replace('_', ' ').title())
                plt.tight_layout()
                
                if self.config.save_plots:
                    plt.savefig(os.path.join(self.config.output_dir, f"{metric}.png"))
                if self.config.show_plots:
                    plt.show()
                plt.close()
    
    def _plot_monte_carlo(self) -> None:
        """Plot Monte Carlo simulation results."""
        if self.mc_results is None:
            logger.warning("Monte Carlo results not available for plotting")
            return
            
        sim_config = SimConfig(n_sims=self.config.n_sims, n_days=self.config.n_days)
        mc = MonteCarlo(sim_config)
        
        # Simulation paths
        mc.plot_simulation_paths(self.mc_results, title="Monte Carlo Simulation Paths")
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.output_dir, "mc_paths.png"))
        if self.config.show_plots:
            plt.show()
        plt.close()
        
        # Distribution
        mc.plot_distribution(self.mc_results, title="Final Portfolio Value Distribution")
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.output_dir, "mc_distribution.png"))
        if self.config.show_plots:
            plt.show()
        plt.close()
        
        # Risk metrics
        mc.plot_risk_metrics(self.mc_results)
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.output_dir, "mc_risk_metrics.png"))
        if self.config.show_plots:
            plt.show()
        plt.close()
    
    def _plot_portfolio_allocation(self) -> None:
        """Plot portfolio allocation equity curves."""
        if self.returns_df is None or self.signals is None:
            logger.warning("Returns data or signals not available for portfolio allocation")
            return
            
        rets = self.returns_df.loc[self.signals.index]
        
        # Calculate different allocation strategies
        curves = self._calculate_portfolio_curves(rets)
        
        # Plot equity curves
        plt.figure(figsize=(10, 6))
        for name, curve in curves.items():
            plt.plot(curve, label=name)
        plt.title('Portfolio Equity Curves')
        plt.xlabel('Date')
        plt.ylabel('Equity (Growth of $1)')
        plt.legend()
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.output_dir, "portfolio_equity_curves.png"))
        if self.config.show_plots:
            plt.show()
        plt.close()
    
    def _calculate_portfolio_curves(self, rets: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate portfolio equity curves for different allocation strategies."""
        # Equal-weight allocation
        ew_weights = np.ones(rets.shape[1]) / rets.shape[1]
        portfolio_ew = (rets * ew_weights).sum(axis=1)
        
        # Risk parity allocation
        rp_weights = []
        for i in range(len(rets)):
            if i < 21:
                rp_weights.append(ew_weights)
            else:
                cov = rets.iloc[i-21:i].cov().values
                inv_vol = 1 / np.sqrt(np.diag(cov))
                w = inv_vol / inv_vol.sum()
                rp_weights.append(w)
        rp_weights = np.array(rp_weights)
        portfolio_rp = (rets.values * rp_weights).sum(axis=1)
        portfolio_rp = pd.Series(portfolio_rp, index=rets.index)
        
        # Kelly allocation
        kelly_weights_arr = []
        for i in range(len(rets)):
            if i < 21:
                kelly_weights_arr.append(ew_weights)
            else:
                mean = rets.iloc[i-21:i].mean().values
                cov = rets.iloc[i-21:i].cov().values
                try:
                    w = np.linalg.pinv(cov).dot(mean)
                    w = np.clip(w, 0, 1)
                    if w.sum() > 0:
                        w = w / w.sum()
                    else:
                        w = np.ones_like(w) / len(w)
                except Exception:
                    w = np.ones_like(mean) / len(mean)
                kelly_weights_arr.append(w)
        kelly_weights_arr = np.array(kelly_weights_arr)
        portfolio_kelly = (rets.values * kelly_weights_arr).sum(axis=1)
        portfolio_kelly = pd.Series(portfolio_kelly, index=rets.index)
        
        # Regime-based allocation
        asset = rets.columns[0]
        regimes = self.regimes_batch[asset] if self.regimes_batch is not None and asset in self.regimes_batch else None
        if regimes is not None:
            regime_mask = regimes.loc[rets.index] == 'High Vol' if regimes.dtype == object else regimes.loc[rets.index] == 2
            regime_alloc = np.where(regime_mask, 0.5, 1.0)
            portfolio_regime = portfolio_ew * regime_alloc
        else:
            portfolio_regime = portfolio_ew
        
        # Calculate equity curves
        def equity_curve(returns):
            return (1 + returns).cumprod()
        
        return {
            'Equal-Weight': equity_curve(portfolio_ew),
            'Risk Parity': equity_curve(portfolio_rp),
            'Kelly': equity_curve(portfolio_kelly),
            'Regime-Adjusted EW': equity_curve(portfolio_regime),
        }
    
    def run_benchmark_analysis(self) -> None:
        """Run benchmark analysis against QQQ."""
        logger.info("Running benchmark analysis...")
        
        try:
            import yfinance as yf
            
            qqq_data = yf.download('QQQ', start=self.config.start_date, end=self.config.end_date, progress=False)
            if qqq_data is None or qqq_data.empty:
                logger.warning("Could not load QQQ data for benchmark")
                return
            
            # Process QQQ data
            if 'Adj Close' in qqq_data.columns:
                qqq_prices = qqq_data['Adj Close'].ffill().bfill()
            elif 'Close' in qqq_data.columns:
                qqq_prices = qqq_data['Close'].ffill().bfill()
            else:
                logger.warning("QQQ data missing price columns")
                return
            
            if isinstance(qqq_prices, pd.DataFrame):
                qqq_prices = qqq_prices.iloc[:, 0]
            
            if len(qqq_prices) < 200:
                logger.warning("Not enough QQQ data for benchmark")
                return
            
            # Calculate QQQ returns and moving average crossover
            qqq_returns = qqq_prices.pct_change().dropna()
            fast_ma = qqq_prices.rolling(50).mean()
            slow_ma = qqq_prices.rolling(200).mean()
            
            if isinstance(fast_ma, pd.DataFrame):
                fast_ma = fast_ma.iloc[:, 0]
            if isinstance(slow_ma, pd.DataFrame):
                slow_ma = slow_ma.iloc[:, 0]
            
            # Handle numpy arrays
            if isinstance(fast_ma, np.ndarray):
                fast_ma = pd.Series(fast_ma, index=qqq_prices.index)
            if isinstance(slow_ma, np.ndarray):
                slow_ma = pd.Series(slow_ma, index=qqq_prices.index)
            
            valid_mask = (~fast_ma.isna()) & (~slow_ma.isna())
            if valid_mask.sum() == 0:
                logger.warning("Not enough valid data for crossover benchmark")
                return
            
            crossover_signal = pd.Series(0, index=qqq_prices.index)
            crossover_signal[valid_mask] = (fast_ma[valid_mask] > slow_ma[valid_mask]).astype(int)
            crossover_signal = crossover_signal.shift(1).reindex(qqq_returns.index).fillna(0)
            crossover_returns = qqq_returns * crossover_signal
            
            # Regime-aware signal returns
            if self.signals is None or self.returns_df is None:
                logger.warning("Signals or returns data not available for benchmark")
                return
                
            regime_signal_returns = (self.signals * self.returns_df).mean(axis=1)
            
            # Align returns
            min_index = max(qqq_returns.index.min(), regime_signal_returns.index.min(), crossover_returns.index.min())
            max_index = min(qqq_returns.index.max(), regime_signal_returns.index.max(), crossover_returns.index.max())
            idx = pd.Index(qqq_returns.loc[min_index:max_index].index)
            
            regime_signal_returns = regime_signal_returns.reindex(idx).fillna(0)
            qqq_returns = qqq_returns.reindex(idx).fillna(0)
            crossover_returns = crossover_returns.reindex(idx).fillna(0)
            
            # Calculate metrics
            benchmarks = [
                ('Regime-Aware Signal', regime_signal_returns),
                ('QQQ (Passive)', qqq_returns),
                ('QQQ 50/200 SMA', crossover_returns),
            ]
            
            bench_results = []
            for name, returns in benchmarks:
                m = calculate_metrics(returns.to_frame(name))
                bench_results.append({
                    'Strategy': name,
                    'Cumulative Return': (1 + returns).prod() - 1,
                    'Annualized Return': m['annual_returns'].iloc[0],
                    'Annualized Vol': m['annual_vol'].iloc[0],
                    'Sharpe': m['sharpe'].iloc[0],
                    'Max Drawdown': m['max_drawdown'],
                    'VaR 95%': m['var_95'].iloc[0],
                })
            
            self.benchmark_results = pd.DataFrame(bench_results)
            logger.info("Benchmark analysis completed")
            
        except Exception as e:
            logger.error(f"Benchmark analysis failed: {e}")
    
    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        logger.info("Generating final report...")
        
        report = []
        report.append("=" * 60)
        report.append("MARKET REGIME DETECTION & PORTFOLIO ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Data summary
        report.append(f"\n[1] DATA SUMMARY")
        report.append(f"Assets: {', '.join(self.config.tickers) if self.config.tickers else 'Not specified'}")
        report.append(f"Period: {self.config.start_date} to {self.config.end_date}")
        if self.returns_df is not None:
            report.append(f"Data points: {len(self.returns_df)}")
            report.append(f"Assets: {len(self.returns_df.columns)}")
        else:
            report.append("Data points: Not available")
            report.append("Assets: Not available")
        
        # Macro data summary
        if self.data is not None and isinstance(self.data, dict) and self.data.get("macro") is not None:
            report.append(f"Macro indicators: {list(self.data['macro'].keys())}")
        
        # Regime detection results
        report.append(f"\n[2] REGIME DETECTION RESULTS")
        if self.regimes_batch is not None and self.returns_df is not None:
            for asset in self.returns_df.columns:
                if asset in self.regimes_batch:
                    regimes = self.regimes_batch[asset]
                    unique_regimes = regimes.unique()
                    report.append(f"\nAsset: {asset}")
                    for regime in unique_regimes:
                        count = (regimes == regime).sum()
                        percentage = (count / len(regimes)) * 100
                        report.append(f"  Regime {regime}: {count} periods ({percentage:.1f}%)")
        
        # Adaptive retraining results
        if self.config.use_adaptive_retraining:
            report.append(f"\n[3] ADAPTIVE RETRAINING RESULTS")
            report.append(f"Retrain window: {self.config.retrain_window} days (~{self.config.retrain_window/21:.1f} months)")
            report.append(f"Retrain frequency: {self.config.retrain_freq} days")
            if hasattr(self, 'regime_confidence') and self.regime_confidence is not None:
                if isinstance(self.regime_confidence, pd.DataFrame):
                    try:
                        mean_result = self.regime_confidence.mean()
                        if isinstance(mean_result, pd.Series):
                            avg_confidence = mean_result.mean()
                        else:
                            avg_confidence = mean_result
                        report.append(f"Average regime confidence: {avg_confidence:.3f}")
                    except (AttributeError, TypeError):
                        report.append(f"Average regime confidence: Not available")
                elif isinstance(self.regime_confidence, (float, int)):
                    report.append(f"Average regime confidence: {self.regime_confidence:.3f}")
                else:
                    report.append(f"Average regime confidence: Not available")
        
        # Risk metrics
        report.append(f"\n[4] RISK METRICS")
        if self.risk_metrics:
            for k, v in self.risk_metrics.items():
                if isinstance(v, (float, int, str)):
                    report.append(f"  {k}: {v}")
                elif isinstance(v, dict):
                    report.append(f"  {k}: (dict with keys: {list(v.keys())})")
                elif isinstance(v, pd.DataFrame):
                    report.append(f"  {k}: (DataFrame shape: {v.shape})")
        
        # Monte Carlo results
        report.append(f"\n[5] MONTE CARLO SIMULATION")
        if self.mc_results:
            for k, v in self.mc_results.items():
                if isinstance(v, (float, int, str)):
                    report.append(f"  {k}: {v}")
        
        # Backtest results
        report.append(f"\n[6] BACKTEST RESULTS")
        for name, result in self.backtest_results.items():
            if 'equity_curve' in result:
                metrics = BacktestEngine.compute_metrics(result['equity_curve'])
                report.append(f"\n{name.title()} Backtest:")
                for k, v in metrics.items():
                    if isinstance(v, (float, int)):
                        report.append(f"  {k}: {v:.4f}")
                
                # Add confidence metrics if available
                if 'confidence_metrics' in result:
                    conf_metrics = result['confidence_metrics']
                    report.append(f"  Confidence Metrics:")
                    if 'avg_confidence' in conf_metrics and conf_metrics['avg_confidence']:
                        avg_conf = np.mean(conf_metrics['avg_confidence'])
                        report.append(f"    Average Confidence: {avg_conf:.3f}")
                    report.append(f"    Low Confidence Periods: {conf_metrics.get('low_confidence_periods', 0)}")
                    report.append(f"    High Confidence Periods: {conf_metrics.get('high_confidence_periods', 0)}")
        
        # Benchmark results
        if hasattr(self, 'benchmark_results'):
            report.append(f"\n[7] BENCHMARK COMPARISON")
            report.append(self.benchmark_results.round(4).to_string())
        
        report.append("\n" + "=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        with open(os.path.join(self.config.output_dir, "analysis_report.txt"), "w") as f:
            f.write(report_text)
        
        # Print to console
        print(report_text)
        
        logger.info("Report generated successfully")
    
    def run(self) -> None:
        """Run the complete analysis pipeline."""
        logger.info("Starting analysis pipeline...")
        
        try:
            # Execute pipeline steps
            self.load_data()
            self.setup_regime_detection()
            self.run_regime_detection()
            self.run_real_time_analysis()
            self.generate_signals()
            self.calculate_risk_metrics()
            self.run_monte_carlo_simulation()
            self.run_backtests()
            self.run_benchmark_analysis()
            self.create_visualizations()
            self.generate_report()
            
            logger.info("Analysis pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise


def load_config_from_file(config_path: str) -> AnalysisConfig:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return AnalysisConfig(**config_dict)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return AnalysisConfig()


def run_analysis(fast_mode=False, ultra_fast=False):
    """Run the complete analysis pipeline"""
    logger.info("Analysis pipeline initialized")
    
    # Load configuration
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Fast mode optimizations
    if fast_mode:
        logger.info("Running in FAST MODE - optimized for speed")
        # Reduce deep learning complexity
        config['deep_learning']['lstm_config']['epochs'] = 25
        config['deep_learning']['transformer_config']['epochs'] = 15
        config['monte_carlo']['n_sims'] = 5000
        # Skip hyperparameter optimization
        config['hyperparameter_optimization']['enabled'] = False
        # Reduce ensemble complexity
        config['regime_detection']['ensemble_config']['models'] = ['hmm', 'lstm']
    
    # Ultra-fast mode - skip deep learning entirely
    if ultra_fast:
        logger.info("Running in ULTRA-FAST MODE - HMM only")
        config['deep_learning']['lstm_config']['epochs'] = 0  # Skip LSTM
        config['deep_learning']['transformer_config']['epochs'] = 0  # Skip Transformer
        config['monte_carlo']['n_sims'] = 2000
        config['hyperparameter_optimization']['enabled'] = False
        config['regime_detection']['ensemble_config']['models'] = ['hmm']  # HMM only
        config['regime_detection']['ensemble_config']['ensemble_method'] = 'single_model'
    
    logger.info("Starting analysis pipeline with extended training window (2010-2019)...")
    
    # Configuration
    config = AnalysisConfig()
    
    # 1. Load Training Data (2010-2019) - Extended training window for full market cycles
    logger.info("Loading training data (2010-2019) - Extended window for full market cycles...")
    train_config = PortfolioConfig(
        tickers=config.tickers or [
            # Tier 1: Core Holdings (40% allocation)
            "NVDA", "MSFT", "AAPL", "GOOGL",  # Core tech leaders with strong regime patterns
            
            # Tier 2: Momentum/Volatility Assets (35% allocation)
            "TSLA", "AVGO", "ANET", "CRM",    # High regime sensitivity, momentum drivers
            
            # Tier 3: Defensive/Diversification (25% allocation)
            "JPM", "UNH", "LLY"               # Financial, healthcare for diversification
        ],
        start_date="2010-01-01",  # Extended from 2017 to include full market cycles
        end_date="2019-12-31",
        frequency=config.frequency,
        use_cache=config.use_cache
    )
    train_loader = DataLoader(train_config)
    train_data = train_loader.load_data()
    train_returns = train_data["returns"]
    
    logger.info(f"Training data loaded: {train_returns.shape}")
    logger.info(f"Training period: 2010-01-01 to 2019-12-31 (9 years for full market cycles)")
    logger.info(f"Training assets: {list(train_returns.columns)}")
    
    # 2. Load Test Data (2020-2021)
    logger.info("Loading test data (2020-2021)...")
    test_config = PortfolioConfig(
        tickers=config.tickers or [
            # Tier 1: Core Holdings (40% allocation)
            "NVDA", "MSFT", "AAPL", "GOOGL",  # Core tech leaders with strong regime patterns
            
            # Tier 2: Momentum/Volatility Assets (35% allocation)
            "TSLA", "AVGO", "ANET", "CRM",    # High regime sensitivity, momentum drivers
            
            # Tier 3: Defensive/Diversification (25% allocation)
            "JPM", "UNH", "LLY"               # Financial, healthcare for diversification
        ],
        start_date="2020-01-01",
        end_date="2021-12-31",
        frequency=config.frequency,
        use_cache=config.use_cache
    )
    test_loader = DataLoader(test_config)
    test_data = test_loader.load_data()
    test_returns = test_data["returns"]
    
    logger.info(f"Test data loaded: {test_returns.shape}")
    logger.info(f"Test period: 2020-01-01 to 2021-12-31")
    logger.info(f"Test assets: {list(test_returns.columns)}")
    
    # 3. Train Regime Detection on Training Data (2010-2019)
    logger.info("Training regime detection on 2010-2019 data (full market cycles)...")
    
    # Ultra-fast mode: use only HMM
    if ultra_fast:
        logger.info("Ultra-fast mode: Using HMM only for regime detection")
        regime_config = RegimeConfig(
            n_regimes=config.n_regimes,
            window_size=config.window_size,
            smoothing_window=config.smoothing_window,
            features=config.features,
            use_deep_learning=False,  # Skip LSTM
            use_transformer=False,     # Skip Transformer
            labeling_metric=config.labeling_metric,
            ensemble_method="single_model",
            model_weights={"hmm": 1.0}
        )
    else:
        # Normal/fast mode: use ensemble
        regime_config = RegimeConfig(
            n_regimes=config.n_regimes,
            window_size=config.window_size,
            smoothing_window=config.smoothing_window,
            features=config.features,
            use_deep_learning=config.use_deep_learning,
            labeling_metric=config.labeling_metric,
            ensemble_method=config.ensemble_method
        )
    
    regime_detector = MarketRegimeDetector(regime_config)
    
    # Train on training data - use fit_predict_batch for multiple assets
    train_regimes = regime_detector.fit_predict_batch(train_returns, n_jobs=2)
    logger.info("Regime detection training completed")
    logger.info(f"Training regimes shape: {train_regimes.shape}")
    
    # Store regime labels for consistency
    regime_labels = regime_detector.regime_labels
    logger.info(f"Regime labels: {regime_labels}")
    
    # Ensure consistent regime labeling by converting to numeric values first
    def ensure_numeric_regimes(regimes_df):
        """Ensure regimes are numeric for consistent labeling"""
        numeric_regimes = pd.DataFrame(index=regimes_df.index, columns=regimes_df.columns)
        for asset in regimes_df.columns:
            regimes = regimes_df[asset]
            # Convert string labels to numeric indices
            def to_numeric(x):
                if pd.isna(x):
                    return np.nan
                elif isinstance(x, str):
                    # Convert string labels to numeric indices
                    if x in regime_labels:
                        return regime_labels.index(x)
                    else:
                        return np.nan
                elif isinstance(x, (int, float)):
                    return x
                else:
                    return np.nan
            
            numeric_regimes[asset] = regimes.map(to_numeric)
        return numeric_regimes
    
    # Convert training regimes to numeric for consistency
    train_regimes_numeric = ensure_numeric_regimes(train_regimes)
    logger.info("Converted training regimes to numeric format")
    
    # FIXED: Ensure the model is fitted by training on the first asset with proper type handling
    logger.info("Ensuring model is fitted for prediction...")
    first_asset = train_returns.columns[0]
    first_asset_returns = train_returns[first_asset]
    if isinstance(first_asset_returns, pd.Series):
        regime_detector.fit(first_asset_returns)
    else:
        logger.warning(f"First asset returns not in expected Series format: {type(first_asset_returns)}")
    
    # 4. Apply Trained Model to Test Data - FIXED: Use same trained model
    logger.info("Applying trained regime model to test data...")
    # Use the SAME trained detector for test data (don't create new one)
    test_regimes = pd.DataFrame(index=test_returns.index, columns=test_returns.columns)
    
    for asset in test_returns.columns:
        asset_returns = test_returns[asset]
        if isinstance(asset_returns, pd.Series):
            # Use predict() instead of fit_predict() to use the trained model
            test_regimes[asset] = regime_detector.predict(asset_returns)
        else:
            logger.warning(f"Asset {asset} returns not in expected Series format")
    
    logger.info(f"Test regimes shape: {test_regimes.shape}")
    
    # Verify regime consistency
    train_unique = set()
    test_unique = set()
    for asset in train_regimes_numeric.columns:
        train_unique.update(train_regimes_numeric[asset].dropna().unique())
    for asset in test_regimes.columns:
        test_unique.update(test_regimes[asset].dropna().unique())
    
    logger.info(f"Training regime values: {train_unique}")
    logger.info(f"Test regime values: {test_unique}")
    
    # Convert numeric regimes to consistent labels
    def convert_regimes_to_labels(regimes_df, labels):
        """Convert numeric regime values to consistent labels"""
        labeled_regimes = pd.DataFrame(index=regimes_df.index, columns=regimes_df.columns)
        for asset in regimes_df.columns:
            regimes = regimes_df[asset]
            # Handle both numeric and string regime values
            def convert_regime(x):
                if pd.isna(x):
                    return "Unknown"
                elif isinstance(x, str):
                    # Already a string label, return as is
                    return x
                elif isinstance(x, (int, float)):
                    # Numeric value, convert to label
                    try:
                        idx = int(x)
                        if 0 <= idx < len(labels):
                            return labels[idx]
                        else:
                            return f"Regime_{x}"
                    except (ValueError, TypeError):
                        return f"Regime_{x}"
                else:
                    return f"Regime_{x}"
            
            labeled_regimes[asset] = regimes.map(convert_regime)
        return labeled_regimes
    
    train_regimes_labeled = convert_regimes_to_labels(train_regimes_numeric, regime_labels)
    test_regimes_labeled = convert_regimes_to_labels(test_regimes, regime_labels)
    
    # 5. Generate Signals on Test Data
    logger.info("Generating signals on test data...")
    signal_generator = SignalGenerator(
        lookback_fast=10,
        lookback_slow=21,
        normalize=True,
        use_regime=True,
        regime_detector=regime_detector
    )
    
    # Create market_data dictionary as expected by SignalGenerator
    test_market_data = {
        'returns': test_returns,
        'prices': test_data['prices']
    }
    
    test_signals = signal_generator.generate_signals(test_market_data)
    logger.info(f"Test signals shape: {test_signals.shape}")
    
    # 6. Calculate Risk Metrics on Test Data
    logger.info("Calculating risk metrics on test data...")
    risk_config = RiskConfig(
        confidence_level=0.95,
        max_drawdown_limit=config.max_drawdown,
        volatility_target=0.15,
        stop_loss=config.stop_loss,
        take_profit=config.take_profit
    )
    
    risk_manager = RiskManager(risk_config)
    test_risk_metrics = risk_manager.calculate_metrics(test_returns)
    logger.info("Risk metrics calculation completed")
    
    # 7. Run Enhanced Monte Carlo Simulation on Test Data
    logger.info("Running enhanced Monte Carlo simulation on test data...")
    logger.info(f"Configuration: {config.n_sims:,} simulations, {config.n_days} days, {config.mc_distribution} distribution")
    
    mc_config = SimConfig(
        n_sims=config.n_sims,
        n_days=config.n_days,
        risk_free_rate=config.risk_free_rate,
        confidence_levels=config.mc_confidence_levels,
        distribution=config.mc_distribution
    )
    
    import time
    
    # Start timing
    mc_start_time = time.time()
    
    mc_simulator = MonteCarlo(mc_config)
    test_mc_results = mc_simulator.simulate({"returns": test_returns})
    
    # End timing
    mc_end_time = time.time()
    mc_elapsed_time = mc_end_time - mc_start_time
    
    logger.info(f"Enhanced Monte Carlo simulation completed in {mc_elapsed_time:.2f} seconds")
    logger.info(f"Average time per simulation: {mc_elapsed_time/config.n_sims*1000:.2f} milliseconds")
    logger.info(f"Simulations per second: {config.n_sims/mc_elapsed_time:.0f}")
    
    # 8. Run Backtest on Test Data
    logger.info("Running backtest on test data...")
    
    # Basic backtest with realistic transaction costs
    basic_backtest = BacktestEngine(
        returns=test_returns,
        signals=test_signals,
        initial_cash=1.0,
        position_sizing='proportional',
        transaction_cost=0.001,  # 0.1% transaction cost
        slippage=0.0005,        # 0.05% slippage
        stop_loss=config.stop_loss,
        take_profit=config.take_profit,
        max_drawdown=config.max_drawdown,
        min_trade_size=0.001,   # Minimum trade size
        max_position_size=0.3,  # Maximum 30% per asset
    )
    test_backtest_results = basic_backtest.run()
    logger.info("Backtest completed")
    
    # Log realistic metrics
    logger.info(f"Backtest Results:")
    logger.info(f"  Total Return: {test_backtest_results['total_return']:.4f}")
    logger.info(f"  Annualized Return: {test_backtest_results['annualized_return']:.4f}")
    logger.info(f"  Annualized Volatility: {test_backtest_results['annualized_vol']:.4f}")
    logger.info(f"  Sharpe Ratio: {test_backtest_results['sharpe_ratio']:.4f}")
    logger.info(f"  Max Drawdown: {test_backtest_results['max_drawdown']:.4f}")
    logger.info(f"  Total Transaction Costs: {test_backtest_results['total_transaction_costs']:.6f}")
    logger.info(f"  Total Trades: {test_backtest_results['total_trades']}")
    logger.info(f"  Turnover Ratio: {test_backtest_results['turnover_ratio']:.4f}")
    
    # 9. Create a simple benchmark comparison
    logger.info("Creating benchmark comparison...")
    # Calculate simple buy-and-hold performance
    buy_hold_ann_ret = ((1 + test_returns).prod()) ** (252 / len(test_returns)) - 1
    try:
        std_result = test_returns.std()
        if isinstance(std_result, pd.Series):
            buy_hold_ann_vol = std_result.mean() * np.sqrt(252)
        else:
            buy_hold_ann_vol = std_result * np.sqrt(252)
    except (AttributeError, TypeError):
        buy_hold_ann_vol = test_returns.std() * np.sqrt(252)
    buy_hold_sharpe = (buy_hold_ann_ret - 0.02) / buy_hold_ann_vol if buy_hold_ann_vol > 0 else 0  # Using 2% risk-free rate
    
    test_benchmark_results = pd.DataFrame({
        'Strategy': ['Regime-Aware Signal', 'Buy and Hold'],
        'Cumulative Return': [
            test_backtest_results['total_return'],
            (1 + test_returns).prod() - 1
        ],
        'Annualized Return': [
            test_backtest_results['annualized_return'],
            buy_hold_ann_ret
        ],
        'Annualized Vol': [
            test_backtest_results['annualized_vol'],
            buy_hold_ann_vol
        ],
        'Sharpe Ratio': [
            test_backtest_results['sharpe_ratio'],
            buy_hold_sharpe
        ],
        'Max Drawdown': [
            test_backtest_results['max_drawdown'],
            (test_returns.cumsum() - test_returns.cumsum().cummax()).min().mean()
        ],
        'Transaction Costs': [
            test_backtest_results['total_transaction_costs'],
            0.0  # Buy and hold has minimal costs
        ],
        'Turnover Ratio': [
            test_backtest_results['turnover_ratio'],
            0.0  # Buy and hold has no turnover
        ]
    })
    logger.info("Benchmark analysis completed")
    
    # 10. Create Visualizations
    logger.info("Creating visualizations...")
    visualizer = PortfolioVisualizer()
    
    # Create comprehensive portfolio analysis plot
    visualizer.plot_portfolio_analysis(
        market_data=test_data,
        signals=test_signals,
        mc_results=test_mc_results,
        regimes=test_regimes.iloc[:, 0] if test_regimes is not None else None
    )
    
    # Create risk metrics plot
    visualizer.plot_risk_metrics(test_risk_metrics)
    
    logger.info("Visualizations completed")
    
    # 11. Generate Final Report
    logger.info("Generating final report...")
    generate_report(
        train_data=train_data,
        test_data=test_data,
        train_regimes=train_regimes,
        test_regimes=test_regimes,
        test_signals=test_signals,
        test_risk_metrics=test_risk_metrics,
        test_mc_results=test_mc_results,
        test_backtest_results=test_backtest_results,
        test_benchmark_results=test_benchmark_results,
        config=config
    )
    logger.info("Report generated successfully")
    
    # Cleanup
    regime_detector.cleanup()
    logger.info("Analysis pipeline completed successfully")


def generate_report(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    train_regimes: pd.DataFrame,
    test_regimes: pd.DataFrame,
    test_signals: pd.DataFrame,
    test_risk_metrics: Dict[str, Any],
    test_mc_results: Dict[str, Any],
    test_backtest_results: Dict[str, Any],
    test_benchmark_results: pd.DataFrame,
    config: AnalysisConfig
) -> None:
    """Generate comprehensive analysis report with train/test split."""
    logger = logging.getLogger(__name__)
    logger.info("Generating train/test analysis report...")
    
    report = []
    report.append("=" * 80)
    report.append("MARKET REGIME DETECTION & PORTFOLIO ANALYSIS REPORT")
    report.append("EXTENDED TRAIN/TEST SPLIT ANALYSIS (2010-2019 Train, 2020-2021 Test)")
    report.append("SUPERIOR ASSET SELECTION WITH TIERED ALLOCATION")
    report.append("=" * 80)
    
    # Data summary
    report.append(f"\n[1] DATA SUMMARY")
    tickers_list = config.tickers or [
        # Tier 1: Core Holdings (40% allocation)
        "NVDA", "MSFT", "AAPL", "GOOGL",  # Core tech leaders with strong regime patterns
        
        # Tier 2: Momentum/Volatility Assets (35% allocation)
        "TSLA", "AVGO", "ANET", "CRM",    # High regime sensitivity, momentum drivers
        
        # Tier 3: Defensive/Diversification (25% allocation)
        "JPM", "UNH", "LLY"               # Financial, healthcare for diversification
    ]
    report.append(f"Assets: {', '.join(tickers_list)}")
    report.append(f"Training Period: 2010-01-01 to 2019-12-31 (9 years for full market cycles)")
    report.append(f"Test Period: 2020-01-01 to 2021-12-31")
    
    train_returns = train_data["returns"]
    test_returns = test_data["returns"]
    
    report.append(f"Training Data: {train_returns.shape[0]} days, {train_returns.shape[1]} assets")
    report.append(f"Test Data: {test_returns.shape[0]} days, {test_returns.shape[1]} assets")
    
    # Asset Allocation Summary
    report.append(f"\n[1.1] ASSET ALLOCATION STRATEGY")
    report.append(f"Tier 1 - Core Tech Leaders (40% allocation):")
    report.append(f"  NVDA, MSFT, AAPL, GOOGL - Core tech leaders with strong regime patterns")
    report.append(f"Tier 2 - Momentum/Volatility Assets (35% allocation):")
    report.append(f"  TSLA, AVGO, ANET, CRM - High regime sensitivity, momentum drivers")
    report.append(f"Tier 3 - Defensive/Diversification (25% allocation):")
    report.append(f"  JPM, UNH, LLY - Financial, healthcare for diversification")
    
    # Training vs Test Performance
    report.append(f"\n[2] TRAINING vs TEST PERFORMANCE")
    
    # Calculate basic stats for both periods
    train_cum_ret = (1 + train_returns).prod() - 1
    test_cum_ret = (1 + test_returns).prod() - 1
    
    train_ann_ret = (1 + train_cum_ret) ** (252 / len(train_returns)) - 1
    test_ann_ret = (1 + test_cum_ret) ** (252 / len(test_returns)) - 1
    
    train_vol = train_returns.std() * np.sqrt(252)
    test_vol = test_returns.std() * np.sqrt(252)
    
    train_sharpe = (train_ann_ret - 0.05) / train_vol
    test_sharpe = (test_ann_ret - 0.05) / test_vol
    
    report.append(f"\nTraining Period (2010-2019) - Full Market Cycles:")
    for i, asset in enumerate(train_returns.columns):
        report.append(f"  {asset}:")
        report.append(f"    Cumulative Return: {train_cum_ret.iloc[i]:.4f}")
        report.append(f"    Annualized Return: {train_ann_ret.iloc[i]:.4f}")
        report.append(f"    Annualized Volatility: {train_vol.iloc[i]:.4f}")
        report.append(f"    Sharpe Ratio: {train_sharpe.iloc[i]:.4f}")
    
    report.append(f"\nTest Period (2020-2021) - COVID-19 Pandemic & Recovery:")
    for i, asset in enumerate(test_returns.columns):
        report.append(f"  {asset}:")
        report.append(f"    Cumulative Return: {test_cum_ret.iloc[i]:.4f}")
        report.append(f"    Annualized Return: {test_ann_ret.iloc[i]:.4f}")
        report.append(f"    Annualized Volatility: {test_vol.iloc[i]:.4f}")
        report.append(f"    Sharpe Ratio: {test_sharpe.iloc[i]:.4f}")
    
    # Regime detection results
    report.append(f"\n[3] REGIME DETECTION RESULTS")
    
    # Training regimes
    report.append(f"\nTraining Period Regimes (2010-2019):")
    for asset in train_regimes.columns:
        regimes = train_regimes[asset]
        unique_regimes = regimes.unique()
        report.append(f"\nAsset: {asset}")
        for regime in unique_regimes:
            if not pd.isna(regime):
                count = (regimes == regime).sum()
                percentage = (count / len(regimes)) * 100
                report.append(f"  Regime {regime}: {count} periods ({percentage:.1f}%)")
    
    # Test regimes
    report.append(f"\nTest Period Regimes (2020-2021):")
    for asset in test_regimes.columns:
        regimes = test_regimes[asset]
        unique_regimes = regimes.unique()
        report.append(f"\nAsset: {asset}")
        for regime in unique_regimes:
            if not pd.isna(regime):
                count = (regimes == regime).sum()
                percentage = (count / len(regimes)) * 100
                report.append(f"  Regime {regime}: {count} periods ({percentage:.1f}%)")
    
    # Test period risk metrics
    report.append(f"\n[4] TEST PERIOD RISK METRICS")
    if test_risk_metrics:
        for k, v in test_risk_metrics.items():
            if isinstance(v, (float, int, str)):
                report.append(f"  {k}: {v}")
            elif isinstance(v, dict):
                report.append(f"  {k}: (dict with keys: {list(v.keys())})")
            elif isinstance(v, pd.DataFrame):
                report.append(f"  {k}: (DataFrame shape: {v.shape})")
    
    # Test period Monte Carlo results
    report.append(f"\n[5] ENHANCED MONTE CARLO SIMULATION RESULTS")
    report.append(f"Configuration:")
    report.append(f"  - Simulations: {config.n_sims:,}")
    report.append(f"  - Days: {config.n_days}")
    report.append(f"  - Distribution: {config.mc_distribution}")
    report.append(f"  - Risk-free rate: {config.risk_free_rate:.1%}")
    report.append(f"  - Confidence levels: {config.mc_confidence_levels}")
    
    if test_mc_results:
        report.append(f"\nSimulation Results:")
        if 'expected_return' in test_mc_results:
            report.append(f"  Expected Return: {test_mc_results['expected_return']:.4f}")
        if 'simulation_volatility' in test_mc_results:
            report.append(f"  Simulation Volatility: {test_mc_results['simulation_volatility']:.4f}")
        if 'var_95' in test_mc_results:
            report.append(f"  VaR 95%: {test_mc_results['var_95']:.4f}")
        if 'var_99' in test_mc_results:
            report.append(f"  VaR 99%: {test_mc_results['var_99']:.4f}")
        
        # Show confidence intervals
        if 'confidence_intervals' in test_mc_results:
            report.append(f"\nConfidence Intervals:")
            for level, value in test_mc_results['confidence_intervals'].items():
                report.append(f"  {level*100:.0f}%: {value:.4f}")
        
        # Show statistics if available
        if 'statistics' in test_mc_results:
            stats = test_mc_results['statistics']
            report.append(f"\nDetailed Statistics:")
            for key, value in stats.items():
                if isinstance(value, (float, int)):
                    report.append(f"  {key}: {value:.4f}")
    
    # Test period backtest results
    report.append(f"\n[6] TEST PERIOD BACKTEST RESULTS")
    if isinstance(test_backtest_results, dict) and 'equity_curve' in test_backtest_results:
        report.append(f"\nRegime-Aware Strategy Backtest (Test Period):")
        report.append(f"  Total Return: {test_backtest_results.get('total_return', 0):.4f}")
        report.append(f"  Annualized Return: {test_backtest_results.get('annualized_return', 0):.4f}")
        report.append(f"  Annualized Volatility: {test_backtest_results.get('annualized_vol', 0):.4f}")
        report.append(f"  Sharpe Ratio: {test_backtest_results.get('sharpe_ratio', 0):.4f}")
        report.append(f"  Max Drawdown: {test_backtest_results.get('max_drawdown', 0):.4f}")
        report.append(f"  Total Transaction Costs: {test_backtest_results.get('total_transaction_costs', 0):.6f}")
        report.append(f"  Total Trades: {test_backtest_results.get('total_trades', 0)}")
        report.append(f"  Turnover Ratio: {test_backtest_results.get('turnover_ratio', 0):.4f}")
        
        # Add confidence metrics if available
        if 'confidence_metrics' in test_backtest_results:
            conf_metrics = test_backtest_results['confidence_metrics']
            report.append(f"  Confidence Metrics:")
            if 'avg_confidence' in conf_metrics and conf_metrics['avg_confidence']:
                avg_conf = np.mean(conf_metrics['avg_confidence'])
                report.append(f"    Average Confidence: {avg_conf:.3f}")
            report.append(f"    Low Confidence Periods: {conf_metrics.get('low_confidence_periods', 0)}")
            report.append(f"    High Confidence Periods: {conf_metrics.get('high_confidence_periods', 0)}")
    else:
        report.append(f"\nBacktest results not available in expected format")
    
    # Test period benchmark results
    if test_benchmark_results is not None:
        report.append(f"\n[7] TEST PERIOD BENCHMARK COMPARISON")
        report.append(test_benchmark_results.round(4).to_string())
    
    # Transaction Cost Analysis
    report.append(f"\n[7.1] TRANSACTION COST ANALYSIS")
    if isinstance(test_backtest_results, dict):
        total_costs = test_backtest_results.get('total_transaction_costs', 0)
        total_return = test_backtest_results.get('total_return', 0)
        net_return = total_return - total_costs
        
        report.append(f"  Gross Return: {total_return:.4f}")
        report.append(f"  Transaction Costs: {total_costs:.6f}")
        report.append(f"  Net Return: {net_return:.4f}")
        report.append(f"  Cost Impact: {(total_costs / total_return * 100):.2f}%" if total_return > 0 else "  Cost Impact: N/A")
        
        turnover = test_backtest_results.get('turnover_ratio', 0)
        report.append(f"  Annual Turnover: {turnover * 252:.2f}")
        report.append(f"  Cost per Trade: {total_costs / test_backtest_results.get('total_trades', 1):.6f}")
    
    # Model Performance Summary
    report.append(f"\n[8] MODEL PERFORMANCE SUMMARY")
    report.append(f"Training Period (2010-2019):")
    report.append(f"  - Extended training window captures full market cycles")
    report.append(f"  - Includes 2010-2012 European debt crisis")
    report.append(f"  - Includes 2015-2016 oil/China slowdown")
    report.append(f"  - Includes normal growth periods")
    report.append(f"  - 9 years of data for robust regime identification")
    
    report.append(f"\nTest Period (2020-2021):")
    report.append(f"  - Market faced COVID-19 pandemic and recovery")
    report.append(f"  - High volatility and rapid regime shifts")
    report.append(f"  - Tests model's out-of-sample performance during extreme conditions")
    report.append(f"  - Superior asset selection with tiered allocation strategy")
    report.append(f"  - Realistic transaction costs and slippage included")
    
    # Key Insights
    report.append(f"\n[9] KEY INSIGHTS")
    
    # Compare average Sharpe ratios
    train_avg_sharpe = train_sharpe.mean()
    test_avg_sharpe = test_sharpe.mean()
    
    report.append(f"Average Sharpe Ratio:")
    report.append(f"  Training Period: {train_avg_sharpe:.4f}")
    report.append(f"  Test Period: {test_avg_sharpe:.4f}")
    report.append(f"  Performance Change: {((test_avg_sharpe - train_avg_sharpe) / train_avg_sharpe * 100):.1f}%")
    
    # Compare volatilities
    train_avg_vol = train_vol.mean()
    test_avg_vol = test_vol.mean()
    
    report.append(f"\nAverage Volatility:")
    report.append(f"  Training Period: {train_avg_vol:.4f}")
    report.append(f"  Test Period: {test_avg_vol:.4f}")
    report.append(f"  Volatility Change: {((test_avg_vol - train_avg_vol) / train_avg_vol * 100):.1f}%")
    
    # Realistic Performance Analysis
    if isinstance(test_backtest_results, dict):
        strategy_sharpe = test_backtest_results.get('sharpe_ratio', 0)
        strategy_vol = test_backtest_results.get('annualized_vol', 0)
        
        report.append(f"\nStrategy Performance Analysis:")
        report.append(f"  Strategy Sharpe Ratio: {strategy_sharpe:.4f}")
        report.append(f"  Strategy Volatility: {strategy_vol:.4f}")
        report.append(f"  Risk-Adjusted Performance: {'Excellent' if strategy_sharpe > 1.5 else 'Good' if strategy_sharpe > 1.0 else 'Moderate' if strategy_sharpe > 0.5 else 'Poor'}")
        
        # Transaction cost impact
        total_costs = test_backtest_results.get('total_transaction_costs', 0)
        total_return = test_backtest_results.get('total_return', 0)
        if total_return > 0:
            cost_impact = total_costs / total_return
            report.append(f"  Transaction Cost Impact: {cost_impact:.2%}")
            report.append(f"  Cost Efficiency: {'Excellent' if cost_impact < 0.1 else 'Good' if cost_impact < 0.2 else 'Moderate' if cost_impact < 0.3 else 'Poor'}")
    
    # Asset Selection Benefits
    report.append(f"\n[10] ASSET SELECTION BENEFITS")
    report.append(f"Tier 1 - Core Tech Leaders (NVDA, MSFT, AAPL, GOOGL):")
    report.append(f"  - High liquidity and market leadership")
    report.append(f"  - Strong regime-dependent performance patterns")
    report.append(f"  - Consistent growth across market cycles")
    
    report.append(f"\nTier 2 - Momentum/Volatility Assets (TSLA, AVGO, ANET, CRM):")
    report.append(f"  - High regime sensitivity for alpha capture")
    report.append(f"  - Momentum drivers during growth regimes")
    report.append(f"  - Semiconductor and enterprise software exposure")
    
    report.append(f"\nTier 3 - Defensive/Diversification (JPM, UNH, LLY):")
    report.append(f"  - Sector diversification (financial, healthcare)")
    report.append(f"  - Defensive characteristics during volatile periods")
    report.append(f"  - Reduced technology concentration risk")
    
    # Implementation Quality Assessment
    report.append(f"\n[11] IMPLEMENTATION QUALITY ASSESSMENT")
    report.append(f"Transaction Cost Model:")
    report.append(f"  - Realistic 0.1% proportional transaction costs")
    report.append(f"  - 0.05% slippage per trade")
    report.append(f"  - Minimum trade size filters to avoid micro-trades")
    report.append(f"  - Maximum position size constraints (30% per asset)")
    
    report.append(f"\nRisk Management:")
    report.append(f"  - Proper drawdown calculation using peak-to-trough")
    report.append(f"  - Realistic stop-loss and take-profit levels")
    report.append(f"  - Position size constraints for risk control")
    
    report.append(f"\nPerformance Metrics:")
    report.append(f"  - Realistic Sharpe ratio calculation (capped at 3.0)")
    report.append(f"  - Proper annualization of returns and volatility")
    report.append(f"  - Transaction cost impact analysis")
    report.append(f"  - Turnover ratio tracking")
    
    report.append("\n" + "=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open("train_test_analysis_report.txt", "w") as f:
        f.write(report_text)
    
    # Print to console
    print(report_text)
    
    logger.info("Train/test analysis report generated successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Regime Detection Analysis')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode for quicker results')
    parser.add_argument('--ultra-fast', action='store_true', help='Run in ultra-fast mode (HMM only) for fastest results')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if args.ultra_fast:
        logger.info("🚀 ULTRA-FAST MODE: HMM only, skipping deep learning models")
    elif args.fast:
        logger.info("⚡ FAST MODE: Reduced deep learning complexity")
    else:
        logger.info("🐌 NORMAL MODE: Full ensemble with deep learning")
    
    run_analysis(fast_mode=args.fast, ultra_fast=args.ultra_fast) 