# main.py
"""
Main execution script for HMM-LSTM Hybrid Market Regime Detection System
Demonstrates the complete quantitative finance pipeline including:
- Data loading and preprocessing
- Market regime detection (HMM + LSTM/Transformer)
- Signal generation with regime awareness
- Risk management and portfolio optimization
- Backtesting with realistic constraints
- Monte Carlo simulation and stress testing
"""

import os
import sys
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import yaml

# Import your modules
from src.data import DataLoader, PortfolioConfig, get_portfolio_features_with_macro
from src.signals import SignalGenerator
from src.regime import MarketRegimeDetector, RegimeConfig
from src.risk import RiskManager, RiskConfig
from src.backtest import BacktestEngine
from src.monte_carlo import MonteCarlo, SimConfig
from src.deep_learning import DeepLearningConfig, LSTMRegimeDetector, TransformerRegimeDetector
from src.features import calculate_enhanced_features  # <-- Add this import if you need to use features directly
from src.statistical_validation import StatisticalValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('regime_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


def load_config(path="config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def debug_signals(signals, returns):
    print("Signal Diagnostics:")
    print(f"Signal mean: {signals.mean().mean():.6f}")
    print(f"Signal std: {signals.std().mean():.6f}")
    print(f"Non-zero signals: {(signals != 0).sum().sum()}")
    print(f"Signal range: [{signals.min().min():.4f}, {signals.max().max():.4f}]")
    # Check signal-return correlation
    correlations = []
    for col in signals.columns:
        if col in returns.columns:
            corr = signals[col].corr(returns[col].shift(-1))
            correlations.append(corr)
    print(f"Average signal-return correlation: {np.mean(correlations):.4f}")

def debug_risk_metrics(portfolio_returns):
    print("Risk Debug:")
    print(f"Portfolio returns mean: {portfolio_returns.mean():.6f}")
    print(f"Portfolio returns std: {portfolio_returns.std():.6f}")
    print(f"Non-zero returns: {(portfolio_returns != 0).sum()}")
    print(f"Return range: [{portfolio_returns.min():.6f}, {portfolio_returns.max():.6f}]")
    # Manual volatility calculation
    manual_vol = portfolio_returns.std() * np.sqrt(252)
    print(f"Manual volatility calc: {manual_vol:.6f}")

def debug_positions(backtest_results):
    positions = backtest_results['positions']
    print("Position Diagnostics:")
    print(f"Position mean: {positions.mean().mean():.6f}")
    print(f"Position std: {positions.std().mean():.6f}")
    print(f"Max position: {positions.max().max():.6f}")
    print(f"Active positions: {(positions.abs() > 0.001).sum().sum()}")


class RegimeDetectionPipeline:
    """Main pipeline for running the complete regime detection system"""
    
    def __init__(self, walk_forward_window: int = None, walk_forward_step: int = None):
        """Initialize the pipeline with default configurations"""
        self.results = {}
        self.market_data = None
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.walk_forward_window = walk_forward_window
        self.walk_forward_step = walk_forward_step
        # Load config from YAML
        self.config = load_config()
        self.setup_configurations()
        # No direct feature engineering here; handled by DataLoader and RegimeDetector
    
    def setup_configurations(self):
        """Setup all configuration objects for the pipeline from config.yml"""
        cfg = self.config
        # Portfolio Configuration
        self.portfolio_config = PortfolioConfig(**cfg.get('portfolio', {}))
        # Deep Learning Config
        dl_cfg = cfg.get('deep_learning', {})
        self.deep_learning_config = DeepLearningConfig(**dl_cfg)
        # Regime Config
        regime_cfg = cfg.get('regime', {})
        regime_cfg['deep_learning_config'] = self.deep_learning_config if regime_cfg.get('use_deep_learning', False) else None
        self.regime_config = RegimeConfig(**regime_cfg)
        # Risk Config
        self.risk_config = RiskConfig(**cfg.get('risk', {}))
        # Monte Carlo Config
        self.sim_config = SimConfig(**cfg.get('monte_carlo', {}))
        # Backtest Config (store for use in backtest step)
        self.backtest_config = cfg.get('backtest', {})
        
        logger.info("Configurations initialized successfully")
    
    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess market data"""
        logger.info("=" * 60)
        logger.info("STEP 1: LOADING MARKET DATA")
        logger.info("=" * 60)
        
        try:
            data_loader = DataLoader(self.portfolio_config)
            self.market_data = data_loader.load_data()
            # Features are now included in self.market_data['features'] if needed
            
            # Display data summary
            returns = self.market_data['returns']
            prices = self.market_data['prices']
            
            logger.info(f"Loaded data for {len(returns.columns)} assets")
            logger.info(f"Date range: {returns.index[0]} to {returns.index[-1]}")
            logger.info(f"Total observations: {len(returns)}")
            logger.info(f"Assets: {list(returns.columns)}")
            
            # Data quality summary
            logger.info("\nData Quality Summary:")
            logger.info(f"Missing values: {returns.isnull().sum().sum()}")
            logger.info(f"Zero returns: {(returns == 0).sum().sum()}")
            
            # Basic statistics
            logger.info("\nAnnualized Return Statistics:")
            annual_returns = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            sharpe_ratios = annual_returns / annual_vol
            
            stats_df = pd.DataFrame({
                'Annual Return': annual_returns,
                'Annual Volatility': annual_vol, 
                'Sharpe Ratio': sharpe_ratios
            })
            logger.info(f"\n{stats_df.round(3)}")
            
            return self.market_data
            
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            raise
    
    def detect_market_regimes(self) -> Dict[str, Any]:
        """Run regime detection using HMM and deep learning models"""
        logger.info("=" * 60)
        logger.info("STEP 2: MARKET REGIME DETECTION")
        logger.info("=" * 60)
        
        try:
            regime_results = {}
            returns = self.market_data['returns']
            # --- NEW: Compute portfolio features with macro ---
            portfolio_returns = returns.mean(axis=1)
            macro_data = self.market_data.get('macro', {})
            portfolio_features = get_portfolio_features_with_macro(portfolio_returns, macro_data)
            # Diagnostics for shape/type issues
            print("portfolio_returns type:", type(portfolio_returns))
            print("portfolio_returns shape:", getattr(portfolio_returns, 'shape', None))
            print("portfolio_returns columns (if DataFrame):", getattr(portfolio_returns, 'columns', None))
            print("portfolio_features['returns'] type:", type(portfolio_features['returns']))
            print("portfolio_features['returns'] shape:", getattr(portfolio_features['returns'], 'shape', None))
            # Force conversion to Series if needed
            if isinstance(portfolio_returns, pd.DataFrame):
                portfolio_returns = portfolio_returns.iloc[:, 0]
            if isinstance(portfolio_features['returns'], pd.DataFrame):
                portfolio_features['returns'] = portfolio_features['returns'].iloc[:, 0]
            print('DEBUG: Available portfolio features:', portfolio_features.columns.tolist())
            regime_detector = MarketRegimeDetector(self.regime_config)
            logger.info("Running HMM + LSTM hybrid regime detection...")
            regimes = regime_detector.fit_predict(portfolio_features, portfolio_returns)
            regime_stats = regime_detector.get_regime_stats(portfolio_features['returns'], regimes)
            validation = regime_detector.validate_model(portfolio_features, regimes)
            transition_matrix = regime_detector.get_transition_matrix()
            regime_results = {
                'regimes': regimes,
                'regime_stats': regime_stats,
                'validation': validation,
                'transition_matrix': transition_matrix,
                'detector': regime_detector
            }
            logger.info(f"Detected regimes: {regimes.value_counts().sort_index()}")
            logger.info(f"Model AIC: {validation['aic']:.2f}")
            logger.info(f"Model BIC: {validation['bic']:.2f}")
            logger.info(f"Regime persistence: {validation['regime_persistence']:.3f}")
            logger.info("\nRegime Statistics:")
            for regime, stats in regime_stats.items():
                logger.info(f"{regime}:")
                logger.info(f"  Mean Return: {stats['mean_return']:.2%}")
                logger.info(f"  Volatility: {stats['volatility']:.2%}")
                logger.info(f"  Frequency: {stats['frequency']:.2%}")
                logger.info(f"  Sharpe: {stats['sharpe']:.2f}")
            # --- Statistical Validation ---
            logger.info("\nPerforming statistical significance testing...")
            print('DEBUG: Features passed to validate_model_statistically:', portfolio_features.columns.tolist())
            stat_validation = regime_detector.validate_model_statistically(portfolio_features, regimes)
            logger.info(f"Statistical significance (cross-validation): {stat_validation.get('statistical_significance', False)}")
            self.results['statistical_validation'] = stat_validation
            # --- Benchmark Comparison ---
            logger.info("\nComparing Bayesian ensemble to HMM and random baselines...")
            benchmark_hmm = regime_detector.compare_to_benchmark(portfolio_features, benchmark='hmm')
            benchmark_random = regime_detector.compare_to_benchmark(portfolio_features, benchmark='random')
            self.results['benchmark_comparison'] = {
                'hmm': benchmark_hmm,
                'random': benchmark_random
            }
            logger.info(f"Ensemble vs HMM: Mean diff = {benchmark_hmm['test_result']['mean_difference']:.4f}, p = {benchmark_hmm['test_result']['p_value']:.4g}")
            logger.info(f"Ensemble vs Random: Mean diff = {benchmark_random['test_result']['mean_difference']:.4f}, p = {benchmark_random['test_result']['p_value']:.4g}")
            self.results['regimes'] = regime_results
            return regime_results
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            raise
    
    def generate_trading_signals(self) -> pd.DataFrame:
        """Generate regime-aware trading signals"""
        logger.info("=" * 60)
        logger.info("STEP 3: SIGNAL GENERATION")
        logger.info("=" * 60)
        
        try:
            # Initialize signal generator with regime detector
            regime_detector = self.results['regimes']['detector']
            
            signal_generator = SignalGenerator(
                lookback_fast=10,
                lookback_slow=21,
                normalize=False,
                use_regime=True,
                regime_detector=regime_detector,
                scaler_k=3.0,                # More aggressive scaling
                scaling_method='clip',       # Use 'clip' instead of 'tanh'
                vol_normalize=False          # Disable volatility normalization
            )
            print("SignalGenerator config:", signal_generator.scaler_k, signal_generator.scaling_method, signal_generator.vol_normalize)
            
            # Generate signals
            logger.info("Generating regime-aware trading signals...")
            signals = signal_generator.generate_signals(self.market_data)
            
            # Signal statistics
            logger.info(f"Generated signals for {len(signals.columns)} assets")
            logger.info(f"Signal range: [{signals.min().min():.2f}, {signals.max().max():.2f}]")
            logger.info(f"Non-zero signals: {(signals != 0).sum().sum()} / {signals.size}")
            
            # Signal correlation with returns (1-day forward)
            returns = self.market_data['returns']
            forward_returns = returns.shift(-1)
            
            signal_correlations = []
            for asset in signals.columns:
                if asset in forward_returns.columns:
                    corr = signals[asset].corr(forward_returns[asset])
                    signal_correlations.append(corr)
            
            logger.info(f"Average signal-return correlation: {np.mean(signal_correlations):.3f}")
            
            self.results['signals'] = signals
            # --- DEBUG: Signal Diagnostics ---
            debug_signals(signals, returns)
            # --- TEST: Signal Quality ---
            diagnostics = self.test_signal_quality()
            return signals
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            raise
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run comprehensive backtest with regime awareness"""
        logger.info("=" * 60)
        logger.info("STEP 4: BACKTESTING")
        logger.info("=" * 60)
        
        try:
            returns = self.market_data['returns']
            signals = self.results['signals']
            # Use numeric regime values for regime_confidence
            regime_detector = self.results['regimes']['detector']
            returns = self.market_data['returns']
            portfolio_returns = returns.mean(axis=1)
            numeric_regimes = regime_detector.predict(portfolio_returns, output_labels=False)
            # Map regime integer to a confidence value (example: higher for 'Low Vol', lower for 'High Vol')
            regime_to_conf = {0: 0.8, 1: 0.6, 2: 0.4}  # Adjust as needed for your regime mapping
            confidence_series = numeric_regimes.map(regime_to_conf)
            regime_confidence = pd.DataFrame(
                np.tile(confidence_series.values[:, None], (1, len(returns.columns))),
                index=returns.index,
                columns=returns.columns
            )
            
            # Initialize backtest engine
            backtest_engine = BacktestEngine(
                returns=returns,
                signals=signals,
                initial_cash=100000,
                transaction_cost=0.001,
                position_sizing='regime_confidence',
                regime_confidence=regime_confidence,
                confidence_threshold=0.8,  # Increase threshold to be more selective
                max_position_size=1.0,   # Increase max position for debugging
                base_position_size=0.5,   # Reduce from 2.0 to 0.5
                rebalance_freq='D',  # Change from 'M' to 'D' for daily rebalancing
                stop_loss=self.risk_config.stop_loss,
                take_profit=self.risk_config.take_profit,
                max_drawdown=self.risk_config.max_drawdown_limit,
                allow_short=True,  # Explicitly enable short selling
                min_trade_size=1e-6,  # Lower min trade size for debugging
                walk_forward_window=self.walk_forward_window,
                walk_forward_step=self.walk_forward_step
            )
            
            # Diagnostic: print first 20 rows of signals before backtest
            print("\n[DIAGNOSTIC] Signals head:\n", signals.head(20))

            # Run backtest
            if self.walk_forward_window is not None and self.walk_forward_step is not None:
                logger.info(f"Running WALK-FORWARD backtest (window={self.walk_forward_window}, step={self.walk_forward_step})...")
                backtest_results = backtest_engine._run_walk_forward()
            else:
                logger.info("Running standard (full-sample) backtest...")
                backtest_results = backtest_engine.run()
            
            # Display results
            metrics = backtest_results['metrics']
            logger.info("Backtest Results:")
            logger.info(f"Total Return: {metrics['total_return']:.2%}")
            logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
            logger.info(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
            
            # Transaction costs
            logger.info(f"Total Transaction Costs: ${metrics.get('total_transaction_costs', 0):,.2f}")
            logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
            
            self.results['backtest'] = backtest_results

            # Benchmark comparison
            try:
                from src.benchmarks import compare_with_benchmarks
                comparison_df = compare_with_benchmarks(backtest_results, returns)
                logger.info("\nBenchmark Comparison:")
                logger.info("\n" + str(comparison_df.to_string(index=False)))
                self.results['benchmark_comparison'] = comparison_df
            except Exception as bench_e:
                logger.warning(f"Benchmark comparison failed: {bench_e}")

            # --- DEBUG: Risk and Position Diagnostics ---
            portfolio_returns = backtest_results.get('daily_returns', None)
            if portfolio_returns is not None:
                debug_risk_metrics(portfolio_returns)
            debug_positions(backtest_results)

            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def analyze_risk(self) -> Dict[str, Any]:
        """Perform comprehensive risk analysis"""
        logger.info("=" * 60)
        logger.info("STEP 5: RISK ANALYSIS")
        logger.info("=" * 60)

        try:
            risk_manager = RiskManager(config=self.risk_config)
            if 'backtest' in self.results:
                portfolio_returns = self.results['backtest']['daily_returns']
            else:
                portfolio_returns = self.market_data['returns'].mean(axis=1)
            self.validate_risk_inputs(portfolio_returns)  # Enhanced diagnostics
            logger.info("Calculating comprehensive risk metrics and report...")
            risk_report = risk_manager.generate_risk_report(portfolio_returns)

            # Display summary and recommendations
            logger.info(risk_report.get('summary', 'No summary available.'))
            assessment = risk_report.get('risk_assessment', {})
            if assessment:
                logger.info(f"Risk Level: {assessment.get('risk_level', 'Unknown')}")
                logger.info(f"Position Recommendation: {assessment.get('position_recommendation', 1.0):.1%}")
                if assessment.get('key_risks'):
                    logger.info("Key Risks:")
                    for risk in assessment['key_risks']:
                        logger.info(f"  • {risk}")
            if risk_report.get('recommendations'):
                logger.info("Recommendations:")
                for rec in risk_report['recommendations']:
                    logger.info(f"  • {rec}")

            self.results['risk_analysis'] = risk_report
            return risk_report

        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            raise
    
    def run_monte_carlo(self) -> Dict[str, Any]:
        """Run Monte Carlo simulation for scenario analysis"""
        logger.info("=" * 60)
        logger.info("STEP 6: MONTE CARLO SIMULATION")
        logger.info("=" * 60)
        
        try:
            # Initialize Monte Carlo simulator
            mc_simulator = MonteCarlo(self.sim_config)
            
            # Run simulation
            logger.info("Running Monte Carlo simulation...")
            mc_results = mc_simulator.simulate(self.market_data)
            
            # Display results
            logger.info("Monte Carlo Results:")
            logger.info(f"Expected Return: {mc_results['annualized_return']:.2%}")
            logger.info(f"Annualized Volatility: {mc_results['annualized_vol']:.2%}")
            logger.info(f"Sharpe Ratio: {mc_results['sharpe_ratio']:.2f}")
            logger.info(f"VaR (95%): {mc_results['var_95']:.2%}")
            logger.info(f"VaR (99%): {mc_results['var_99']:.2%}")
            
            # Confidence intervals
            logger.info("Confidence Intervals:")
            for level, value in mc_results['confidence_intervals'].items():
                logger.info(f"  {level*100:.0f}%: {(value-1)*100:.1f}%")
            
            # Validation
            if mc_results['validation']['overall_valid']:
                logger.info("✓ Monte Carlo simulation validation passed")
            else:
                logger.warning("⚠ Monte Carlo simulation validation failed")
            # Print full validation details
            logger.info(f"Monte Carlo validation details: {mc_results['validation']}")
            
            self.results['monte_carlo'] = mc_results
            return mc_results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            # Don't raise - Monte Carlo is optional
            logger.info("Continuing without Monte Carlo results...")
            return {}
    
    def create_visualizations(self):
        """Create comprehensive visualizations of results"""
        logger.info("=" * 60)
        logger.info("STEP 7: CREATING VISUALIZATIONS")
        logger.info("=" * 60)
        
        try:
            pdf_path = os.path.join(self.output_dir, 'regime_detection_report.pdf')
            
            with PdfPages(pdf_path) as pdf:
                # 1. Portfolio Performance
                self._plot_portfolio_performance(pdf)
                
                # 2. Regime Analysis
                self._plot_regime_analysis(pdf)
                
                # 3. Signal Analysis
                self._plot_signal_analysis(pdf)
                
                # 4. Risk Analysis
                self._plot_risk_analysis(pdf)
                
                # 5. Monte Carlo Results
                if 'monte_carlo' in self.results:
                    self._plot_monte_carlo_results(pdf)
            
            logger.info(f"Visualizations saved to: {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            logger.info("Continuing without visualizations...")
    
    def _plot_portfolio_performance(self, pdf):
        """Plot portfolio performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16)
        
        if 'backtest' in self.results:
            equity_curve = self.results['backtest']['equity_curve']
            daily_returns = self.results['backtest']['daily_returns']
            
            # Equity curve
            axes[0, 0].plot(equity_curve.index, equity_curve.values / equity_curve.iloc[0])
            axes[0, 0].set_title('Cumulative Returns')
            axes[0, 0].set_ylabel('Cumulative Return')
            axes[0, 0].grid(True)
            
            # Drawdown
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown %')
            axes[0, 1].grid(True)
            
            # Return distribution
            axes[1, 0].hist(daily_returns.values, bins=50, alpha=0.7, density=True)
            axes[1, 0].set_title('Daily Return Distribution')
            axes[1, 0].set_xlabel('Daily Return')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].grid(True)
            
            # Rolling Sharpe
            rolling_sharpe = (daily_returns.rolling(63).mean() * 252) / (daily_returns.rolling(63).std() * np.sqrt(252))
            axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
            axes[1, 1].set_title('Rolling 3M Sharpe Ratio')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_regime_analysis(self, pdf):
        """Plot regime detection analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Market Regime Analysis', fontsize=16)
        
        if 'regimes' in self.results:
            regimes = self.results['regimes']['regimes']
            returns = self.market_data['returns'].mean(axis=1)
            
            # Regime time series
            axes[0, 0].plot(returns.index, returns.cumsum(), alpha=0.7, label='Cumulative Returns')
            
            # Color-code by regime
            unique_regimes = regimes.unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_regimes)))
            
            for i, regime in enumerate(unique_regimes):
                mask = regimes == regime
                axes[0, 0].scatter(returns.index[mask], returns.cumsum()[mask], 
                                 c=[colors[i]], label=f'Regime {regime}', alpha=0.6, s=10)
            
            axes[0, 0].set_title('Regime Evolution Over Time')
            axes[0, 0].set_ylabel('Cumulative Return')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Regime distribution
            regime_counts = regimes.value_counts().sort_index()
            axes[0, 1].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Regime Distribution')
            
            # Transition matrix heatmap
            if 'transition_matrix' in self.results['regimes']:
                trans_mat = self.results['regimes']['transition_matrix']
                im = axes[1, 0].imshow(trans_mat.values, cmap='Blues', aspect='auto')
                axes[1, 0].set_xticks(range(len(trans_mat.columns)))
                axes[1, 0].set_yticks(range(len(trans_mat.index)))
                axes[1, 0].set_xticklabels(trans_mat.columns)
                axes[1, 0].set_yticklabels(trans_mat.index)
                axes[1, 0].set_title('Regime Transition Matrix')
                plt.colorbar(im, ax=axes[1, 0])
            
            # Regime statistics
            if 'regime_stats' in self.results['regimes']:
                stats = self.results['regimes']['regime_stats']
                regime_names = list(stats.keys())
                returns_by_regime = [stats[r]['mean_return'] for r in regime_names]
                vol_by_regime = [stats[r]['volatility'] for r in regime_names]
                
                axes[1, 1].scatter(vol_by_regime, returns_by_regime)
                for i, regime in enumerate(regime_names):
                    axes[1, 1].annotate(regime, (vol_by_regime[i], returns_by_regime[i]))
                axes[1, 1].set_xlabel('Volatility')
                axes[1, 1].set_ylabel('Mean Return')
                axes[1, 1].set_title('Risk-Return by Regime')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_signal_analysis(self, pdf):
        """Plot signal analysis"""
        if 'signals' not in self.results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trading Signal Analysis', fontsize=16)
        
        signals = self.results['signals']
        returns = self.market_data['returns']
        
        # Signal time series for first asset
        asset = signals.columns[0]
        axes[0, 0].plot(signals.index, signals[asset], alpha=0.7, label='Signal')
        axes[0, 0].plot(returns.index, returns[asset].cumsum(), alpha=0.7, label='Cumulative Return')
        axes[0, 0].set_title(f'Signals vs Returns: {asset}')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Signal distribution
        all_signals = signals.values.flatten()
        axes[0, 1].hist(all_signals, bins=50, alpha=0.7, density=True)
        axes[0, 1].set_title('Signal Distribution (All Assets)')
        axes[0, 1].set_xlabel('Signal Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True)
        
        # Signal correlation with forward returns
        forward_returns = returns.shift(-1)
        correlations = []
        for asset in signals.columns:
            if asset in forward_returns.columns:
                corr = signals[asset].corr(forward_returns[asset])
                correlations.append(corr)
        
        axes[1, 0].bar(range(len(correlations)), correlations)
        axes[1, 0].set_title('Signal-Return Correlations')
        axes[1, 0].set_xlabel('Asset Index')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].grid(True)
        
        # Signal strength over time
        signal_strength = signals.abs().mean(axis=1)
        axes[1, 1].plot(signal_strength.index, signal_strength.values)
        axes[1, 1].set_title('Average Signal Strength Over Time')
        axes[1, 1].set_ylabel('Signal Strength')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_risk_analysis(self, pdf):
        """Plot risk analysis"""
        if 'risk_analysis' not in self.results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Risk Analysis', fontsize=16)
        
        risk_report = self.results['risk_analysis']
        
        if 'risk_metrics' in risk_report:
            metrics = risk_report['risk_metrics']
            
            # Risk metrics bar chart
            risk_metrics = ['volatility', 'var_95', 'expected_shortfall_95', 'max_drawdown']
            risk_values = [abs(metrics.get(m, 0)) for m in risk_metrics]
            risk_labels = ['Volatility', 'VaR 95%', 'ES 95%', 'Max DD']
            
            axes[0, 0].bar(risk_labels, risk_values)
            axes[0, 0].set_title('Key Risk Metrics')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True)
            
            # Risk score gauge (simplified)
            risk_score = metrics.get('risk_score', 50)
            axes[0, 1].pie([risk_score, 100-risk_score], labels=['Risk', 'Safe'], 
                          colors=['red', 'green'], startangle=90)
            axes[0, 1].set_title(f'Risk Score: {risk_score:.1f}/100')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_monte_carlo_results(self, pdf):
        """Plot Monte Carlo simulation results"""
        if 'monte_carlo' not in self.results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Monte Carlo Simulation Results', fontsize=16)
        
        mc_results = self.results['monte_carlo']
        
        # Distribution of final values
        final_values = mc_results.get('final_portfolio_values', mc_results.get('final_values'))
        axes[0, 0].hist(final_values, bins=50, alpha=0.7, density=True)
        expected_return = mc_results.get('mean_return', mc_results.get('expected_return'))
        if expected_return is not None:
            axes[0, 0].axvline(expected_return + 1, color='red', linestyle='--', label='Expected')
        axes[0, 0].axvline(mc_results['var_95'] + 1, color='orange', linestyle='--', label='VaR 95%')
        axes[0, 0].set_title('Distribution of Final Portfolio Values')
        axes[0, 0].set_xlabel('Portfolio Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Sample paths (subset)
        paths = mc_results.get('portfolio_values', mc_results.get('port_value', mc_results.get('paths')))
        if paths is not None:
            for i in range(min(100, paths.shape[0])):
                axes[0, 1].plot(paths[i, :], alpha=0.1, color='blue')
            axes[0, 1].plot(paths.mean(axis=0), color='red', linewidth=2, label='Mean Path')
            axes[0, 1].set_title('Sample Portfolio Paths')
            axes[0, 1].set_ylabel('Portfolio Value')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def save_results(self):
        """Save all results to files"""
        logger.info("=" * 60)
        logger.info("STEP 8: SAVING RESULTS")
        logger.info("=" * 60)
        
        try:
            # Save summary statistics
            summary = self.create_summary()
            
            # Save to JSON
            json_path = os.path.join(self.output_dir, 'results_summary.json')
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=4, default=str)
            
            # Save detailed results
            if 'backtest' in self.results:
                equity_curve = self.results['backtest']['equity_curve']
                equity_curve.to_csv(os.path.join(self.output_dir, 'equity_curve.csv'))
            
            if 'regimes' in self.results:
                regimes = self.results['regimes']['regimes']
                regimes.to_csv(os.path.join(self.output_dir, 'regimes.csv'))
            
            if 'signals' in self.results:
                signals = self.results['signals']
                signals.to_csv(os.path.join(self.output_dir, 'signals.csv'))
            
            logger.info(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def create_summary(self) -> Dict[str, Any]:
        """Create a comprehensive summary of all results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_config': {
                'tickers': self.portfolio_config.tickers,
                'start_date': self.portfolio_config.start_date,
                'end_date': self.portfolio_config.end_date,
                'frequency': self.portfolio_config.frequency
            }
        }
        
        # Add backtest summary
        if 'backtest' in self.results:
            metrics = self.results['backtest']['metrics']
            summary['backtest'] = {
                'total_return': metrics.get('total_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'annualized_volatility': metrics.get('annualized_volatility', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'win_rate': metrics.get('win_rate', 0)
            }
        
        # Add regime summary
        if 'regimes' in self.results:
            regimes = self.results['regimes']['regimes']
            validation = self.results['regimes']['validation']
            summary['regimes'] = {
                'n_regimes': len(regimes.unique()),
                'regime_distribution': regimes.value_counts().to_dict(),
                'aic': validation.get('aic', 0),
                'bic': validation.get('bic', 0),
                'regime_persistence': validation.get('regime_persistence', 0)
            }
        
        # Add risk summary
        if 'risk_analysis' in self.results and 'risk_metrics' in self.results['risk_analysis']:
            risk_metrics = self.results['risk_analysis']['risk_metrics']
            summary['risk'] = {
                'risk_score': risk_metrics.get('risk_score', 0),
                'var_95': risk_metrics.get('var_95', 0),
                'expected_shortfall_95': risk_metrics.get('expected_shortfall_95', 0),
                'volatility': risk_metrics.get('volatility', 0)
            }
        
        return summary
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from start to finish"""
        logger.info("🚀 Starting HMM-LSTM Market Regime Detection Pipeline")
        logger.info("Research Focus: Semiconductor Equity Markets")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load data
            self.load_market_data()
            
            # Step 2: Detect regimes
            self.detect_market_regimes()
            
            # Step 3: Generate signals
            self.generate_trading_signals()
            
            # Step 4: Run backtest
            # To enable walk-forward backtesting, pass walk_forward_window and walk_forward_step to the pipeline constructor.
            self.run_backtest()
            
            # Step 5: Analyze risk
            self.analyze_risk()
            
            # Step 6: Monte Carlo (optional)
            self.run_monte_carlo()
            
            # Step 7: Create visualizations
            self.create_visualizations()
            
            # Step 8: Save results
            self.save_results()
            
            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("=" * 80)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Total execution time: {duration}")
            logger.info(f"Results saved to: {self.output_dir}")
            
            # Print key results
            self.print_final_summary()
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def print_final_summary(self):
        """Print final summary of key results"""
        summary_lines = []
        summary_lines.append("\n📊 KEY RESULTS SUMMARY:")
        summary_lines.append("-" * 50)
        
        if 'backtest' in self.results:
            metrics = self.results['backtest']['metrics']
            summary_lines.append("🎯 PORTFOLIO PERFORMANCE:")
            summary_lines.append(f"   Total Return: {metrics.get('total_return', 0):.2%}")
            summary_lines.append(f"   Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            summary_lines.append(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            summary_lines.append(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            summary_lines.append(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        
        if 'regimes' in self.results:
            regimes = self.results['regimes']['regimes']
            validation = self.results['regimes']['validation']
            summary_lines.append(f"\n🔄 REGIME DETECTION:")
            summary_lines.append(f"   Number of regimes detected: {len(regimes.unique())}")
            summary_lines.append(f"   Model persistence: {validation.get('regime_persistence', 0):.3f}")
            summary_lines.append(f"   Model AIC: {validation.get('aic', 0):.1f}")
        
        if 'risk_analysis' in self.results:
            risk_metrics = self.results['risk_analysis'].get('risk_metrics', {})
            summary_lines.append(f"\n🛡️ RISK ASSESSMENT:")
            summary_lines.append(f"   Risk Score: {risk_metrics.get('risk_score', 0):.1f}/100")
            summary_lines.append(f"   VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
            summary_lines.append(f"   Portfolio Volatility: {risk_metrics.get('volatility', 0):.2%}")
        
        summary_lines.append("\n✅ All components executed successfully!")
        summary_lines.append("📄 Check the generated PDF report for detailed visualizations.")

        # Print to terminal
        print("\n".join(summary_lines))
        # Also log as before
        for line in summary_lines:
            logger.info(line)

    def validate_risk_inputs(self, returns):
        """Validate risk calculation inputs"""
        print(f"\n=== RISK CALCULATION VALIDATION ===")
        print(f"Returns type: {type(returns)}")
        print(f"Returns shape: {returns.shape if hasattr(returns, 'shape') else 'No shape'}")
        print(f"Returns length: {len(returns)}")
        print(f"Non-null returns: {returns.count() if hasattr(returns, 'count') else 'Cannot count'}")
        print(f"Returns range: [{returns.min():.6f}, {returns.max():.6f}]")
        print(f"Returns std: {returns.std():.6f}")
        if returns.std() == 0:
            print("⚠️  CRITICAL: Zero volatility detected!")
            print("Sample returns:", returns.head(10).values)

    def test_signal_quality(self):
        """Test signal generation quality"""
        print("\n=== SIGNAL QUALITY TEST ===")
        signals = self.results['signals']
        returns = self.market_data['returns']
        # Use the diagnostic function
        from src.signals import SignalGenerator
        signal_gen = SignalGenerator()  # Create instance for diagnostics
        diagnostics = signal_gen.diagnose_signals(signals, returns)
        print(f"Signal Statistics:")
        stats = diagnostics['signal_stats']
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print(f"\nCorrelation Analysis:")
        corr_stats = diagnostics['correlations']
        print(f"  Mean correlation: {corr_stats['mean']:.4f}")
        print(f"  Correlation std: {corr_stats['std']:.4f}")
        # Red flags
        if abs(corr_stats['mean']) < 0.01:
            print("  ⚠️  WARNING: Very weak signal-return correlation!")
        if stats['non_zero_count'] < stats['total_count'] * 0.5:
            print("  ⚠️  WARNING: Most signals are zero!")
        if stats['std'] < 0.1:
            print("  ⚠️  WARNING: Signals have very low variation!")
        return diagnostics


def main():
    """Main execution function"""
    try:
        # Run normal pipeline only (no placebo)
        logger.info("\n=== RUNNING PIPELINE ===")
        pipeline = RegimeDetectionPipeline(walk_forward_window=252, walk_forward_step=63)
        pipeline.run_complete_pipeline()
        logger.info("\n=== PIPELINE RESULTS ===")
        pipeline.print_final_summary()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
