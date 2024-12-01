from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.monte_carlo import MonteCarlo, SimConfig
from src.regime import MarketRegimeDetector, RegimeConfig
from src.risk import RiskManager, RiskConfig
from src.signals import SignalGenerator
from src.utils import plot_portfolio_analysis, calculate_metrics

@dataclass
class PortfolioAnalyzerConfig:
    """Configuration for portfolio analysis"""
    # Monte Carlo settings
    n_sims: int = 10000
    n_days: int = 252
    
    # Regime detection settings
    n_regimes: int = 3
    regime_window: int = 21
    
    # Risk management settings
    confidence_level: float = 0.95
    max_drawdown_limit: float = 0.20
    volatility_target: float = 0.15
    
    # Signal generation settings
    lookback_fast: int = 20
    lookback_slow: int = 50

class PortfolioAnalyzer:
    """Comprehensive portfolio analysis and risk management"""
    
    def __init__(self, config: PortfolioAnalyzerConfig):
        self.config = config
        
        # Initialize components
        self.monte_carlo = MonteCarlo(
            SimConfig(n_sims=config.n_sims, n_days=config.n_days)
        )
        
        self.regime_detector = MarketRegimeDetector(
            RegimeConfig(n_regimes=config.n_regimes, window_size=config.regime_window)
        )
        
        self.risk_manager = RiskManager(
            RiskConfig(
                confidence_level=config.confidence_level,
                max_drawdown_limit=config.max_drawdown_limit,
                volatility_target=config.volatility_target
            )
        )
        
        self.signal_generator = SignalGenerator(
            lookback_fast=config.lookback_fast,
            lookback_slow=config.lookback_slow
        )
    
    def analyze_portfolio(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Perform comprehensive portfolio analysis"""
        
        # 1. Monte Carlo Simulation
        mc_results = self.monte_carlo.simulate(market_data)
        
        # 2. Regime Detection
        regimes = self.regime_detector.fit_predict(market_data['returns'].mean(axis=1))
        regime_stats = self.regime_detector.get_regime_stats(
            market_data['returns'].mean(axis=1), 
            regimes
        )
        
        # 3. Risk Analysis
        risk_metrics = self.risk_manager.calculate_metrics(market_data['returns'])
        
        # 4. Signal Generation
        signals = self.signal_generator.generate_signals(market_data)
        
        # 5. Generate plots
        plot_portfolio_analysis(market_data, signals, mc_results)
        
        # Combine results
        analysis = {
            'monte_carlo': {
                'expected_return': mc_results['expected_return'],
                'var_95': mc_results['var_95'],
                'confidence_intervals': mc_results['confidence_intervals']
            },
            'regimes': {
                'current_regime': regimes.iloc[-1],
                'regime_stats': regime_stats,
                'transition_matrix': self.regime_detector.get_transition_matrix()
            },
            'risk': risk_metrics,
            'signals': signals.iloc[-1],  # Latest signals
            'validation': {
                'monte_carlo': mc_results['validation'],
                'regime': self.regime_detector.validate_model(
                    market_data['returns'].mean(axis=1), 
                    regimes
                )
            }
        }
        
        return analysis
    
    def get_portfolio_recommendations(self, analysis: Dict) -> Dict:
        """Generate portfolio recommendations based on analysis"""
        current_regime = analysis['regimes']['current_regime']
        risk_metrics = analysis['risk']
        signals = analysis['signals']
        
        # Adjust position sizes based on regime and risk
        position_scale = self._calculate_position_scale(
            current_regime,
            risk_metrics['portfolio_volatility'],
            risk_metrics['max_drawdown']
        )
        
        recommendations = {
            'position_sizes': signals * position_scale,
            'risk_budget': self._calculate_risk_budget(analysis),
            'regime_outlook': self._generate_regime_outlook(analysis['regimes']),
            'risk_warnings': self._generate_risk_warnings(risk_metrics)
        }
        
        return recommendations
    
    def _calculate_position_scale(self, regime: str, volatility: float, 
                                max_drawdown: float) -> float:
        """Calculate position scaling based on regime and risk metrics"""
        base_scale = {
            'Low Vol': 1.0,
            'Medium Vol': 0.8,
            'High Vol': 0.5
        }.get(regime, 0.8)
        
        vol_adjustment = min(1.0, self.config.volatility_target / volatility)
        dd_adjustment = min(1.0, self.config.max_drawdown_limit / abs(max_drawdown))
        
        return min(base_scale, vol_adjustment, dd_adjustment)