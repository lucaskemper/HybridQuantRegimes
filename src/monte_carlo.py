# src/monte_carlo.py

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from arch import arch_model


@dataclass
class SimConfig:
    n_sims: int = 20_000
    n_days: int = 1260                  # 5 years
    risk_free_rate: float = 0.05
    confidence_levels: List[float] = None
    simulation_mode: str = "block_bootstrap"
    block_size: int = 21                
    weights: Optional[np.ndarray] = None     # None = equal weight
    seed: Optional[int] = 42
    rebalance: str = "daily"            # "daily" or "none"
    garch_df: float = 6.0               # fat tails for garch mode

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


class MonteCarlo:
    def __init__(self, config: SimConfig):
        self.cfg = config
        if config.seed is not None:
            np.random.seed(config.seed)

    def simulate(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        returns = market_data["returns"]
        if returns.empty:
            raise ValueError("market_data['returns'] is empty.")

        log_returns = np.log1p(returns).dropna()
        n_assets = log_returns.shape[1]

        mode = self.cfg.simulation_mode.lower()

        # pick the engine
        if mode == "block_bootstrap":
            daily_log = self._block_bootstrap_fast(log_returns.values)
        elif mode == "garch_t":
            daily_log = self._garch_t(log_returns)
        elif mode == "regime_switching":
            if "regimes" not in market_data:
                raise ValueError("need regimes for this mode")
            daily_log = self._regime_switching(log_returns, market_data["regimes"])
        else:
            daily_log = self._gbm(log_returns)

        # weights
        w = self.cfg.weights
        if w is None:
            w = np.ones(n_assets) / n_assets
        w = np.array(w)

        # portfolio
        if self.cfg.rebalance == "daily":
            port_daily = (daily_log * w).sum(axis=2)
            port_value = np.exp(np.cumsum(port_daily, axis=1))
        else:
            prices = np.exp(np.cumsum(daily_log, axis=1))
            port_value = (prices * w).sum(axis=2)
            port_daily = np.diff(np.log(port_value), axis=1, prepend=0.0)

        total_ret = port_value[:, -1] - 1

        ann_ret = np.exp(port_daily.mean(axis=1) * 252) - 1
        ann_vol = port_daily.std(axis=1) * np.sqrt(252)
        sharpe = np.nan_to_num((ann_ret - self.cfg.risk_free_rate) / ann_vol)

        peak = np.maximum.accumulate(port_value, axis=1)
        max_dd = ((port_value - peak) / peak).min(axis=1)

        # proper cvar on losses
        losses = -total_ret
        var95_loss = np.percentile(losses, 95)
        var99_loss = np.percentile(losses, 99)

        tail95 = losses[losses > var95_loss]
        tail99 = losses[losses > var99_loss]
        cvar95 = tail95.mean() if len(tail95) > 0 else losses.max()
        cvar99 = tail99.mean() if len(tail99) > 0 else losses.max()

        var95 = -var95_loss
        var99 = -var99_loss
        cvar95 = -cvar95
        cvar99 = -cvar99

        ci = np.percentile(total_ret, np.array(self.cfg.confidence_levels) * 100)

        return {
            "final_values": port_value[:, -1],
            "port_value": port_value,
            "total_return": total_ret,
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": float(sharpe.mean()),
            "max_dd": max_dd,
            "win_rate": (total_ret > 0).mean() * 100,
            "var95": float(var95),
            "cvar95": float(cvar95),
            "var99": float(var99),
            "cvar99": float(cvar99),
            "ci": dict(zip(self.cfg.confidence_levels, ci)),
            "paths": port_value,
            "expected": float(total_ret.mean()),
            "annualized_ret": float(ann_ret.mean()),
            "annualized_vol": float(ann_vol.mean()),
            "validation": self._validate(daily_log, log_returns),
            "stats": {
                "mean": float(port_value[:, -1].mean()),
                "median": float(np.median(port_value[:, -1])),
                "std": float(port_value[:, -1].std()),
                "skew": float(pd.Series(port_value[:, -1]).skew()),
                "kurt": float(pd.Series(port_value[:, -1]).kurt()),
                "sharpe": float(sharpe.mean()),
            },
        }

    # block bootstrap
    def _block_bootstrap_fast(self, data):
        n, n_assets = data.shape
        block = self.cfg.block_size
        n_sims, n_days = self.cfg.n_sims, self.cfg.n_days

        n_full = n_days // block
        rem = n_days % block

        starts = np.random.randint(0, n - block + 1, size=(n_sims, n_full))
        idx = starts[..., None] + np.arange(block)
        full = data[idx.reshape(-1)].reshape(n_sims, n_full * block, n_assets)

        if rem:
            starts_rem = np.random.randint(0, n - block + 1, size=n_sims)
            rem_part = data[starts_rem[:, None] + np.arange(rem)]
            return np.concatenate([full, rem_part], axis=1)
        return full

    # garch-t 
    def _garch_t(self, logret):
        mu = logret.mean().values
        n_sims, n_days, n_assets = self.cfg.n_sims, self.cfg.n_days, logret.shape[1]
        df = self.cfg.garch_df

        models = {}
        for col in logret.columns:
            try:
                m = arch_model(logret[col]*100, vol='Garch', p=1, q=1, dist='t').fit(disp='off')
                models[col] = m
            except:
                models[col] = None

        L = np.linalg.cholesky(logret.corr().values + 1e-8 * np.eye(n_assets))

        z = np.random.standard_t(df, (n_sims, n_days, n_assets))
        z /= np.sqrt(df / (df-2))
        z_corr = z @ L.T  # correlate the shocks

        out = np.zeros((n_sims, n_days, n_assets))
        for s in range(n_sims):
            for a, col in enumerate(logret.columns):
                m = models[col]
                if m is None:
                    out[s,:,a] = mu[a] + logret[col].std() * z_corr[s,:,a]
                else:
                    sim = m.simulate(m.params, nobs=n_days)
                    vol = np.sqrt(sim['variance'].values) / 100
                    out[s,:,a] = mu[a] + vol * z_corr[s,:,a]
        return out

    # regime switching with per-regime corr (completed based on paper's 3.7)
    def _regime_switching(self, logret, regimes):
        regimes = regimes.reindex(logret.index).ffill().dropna()
        logret = logret.loc[regimes.index]

        labels, idx = np.unique(regimes, return_inverse=True)
        n_regimes = len(labels)

        mu_reg = []
        sigma_reg = []
        L_reg = []
        for k in range(n_regimes):
            mask = (idx == k)
            reg_data = logret.iloc[mask]  # use iloc for efficiency
            mu_reg.append(reg_data.mean().values)
            sigma_reg.append(reg_data.std().values)
            corr = reg_data.corr().values + 1e-8 * np.eye(n_assets)  # regularize
            L_reg.append(np.linalg.cholesky(corr))

        # transition matrix P (rows: from i to j)
        trans = np.zeros((n_regimes, n_regimes))
        for i in range(len(idx) - 1):
            trans[idx[i], idx[i + 1]] += 1
        trans /= np.maximum(trans.sum(axis=1, keepdims=True), 1e-8)  # normalize rows

        # initial regime probabilities (empirical frequency)
        initial_p = np.bincount(idx) / len(idx)

        # simulate regime paths (vectorized where possible)
        n_sims, n_days = self.cfg.n_sims, self.cfg.n_days
        sim_reg = np.zeros((n_sims, n_days), dtype=int)
        sim_reg[:, 0] = np.random.choice(n_regimes, size=n_sims, p=initial_p)
        cum_trans = np.cumsum(trans, axis=1)
        for t in range(1, n_days):
            u = np.random.rand(n_sims)
            sim_reg[:, t] = (u[:, None] < cum_trans[sim_reg[:, t - 1]]).sum(axis=1)

        # generate log returns
        mu_reg = np.array(mu_reg)  # (n_regimes, n_assets)
        sigma_reg = np.array(sigma_reg)
        L_reg = np.array(L_reg)  # (n_regimes, n_assets, n_assets)

        z = np.random.randn(n_sims, n_days, n_assets)
        out = np.zeros((n_sims, n_days, n_assets))
        for s in range(n_sims):
            for t in range(n_days):
                k = sim_reg[s, t]
                correlated = L_reg[k] @ z[s, t]
                scaled = sigma_reg[k] * correlated
                out[s, t] = mu_reg[k] - 0.5 * (sigma_reg[k] ** 2) + scaled

        return out

    # basic geometric brownian motion (constant params)
    def _gbm(self, logret):
        mu = logret.mean().values
        sigma = logret.std().values
        corr = logret.corr().values + 1e-8 * np.eye(logret.shape[1])
        L = np.linalg.cholesky(corr)

        n_sims, n_days, n_assets = self.cfg.n_sims, self.cfg.n_days, logret.shape[1]
        z = np.random.randn(n_sims, n_days, n_assets)

        correlated = np.einsum('ijk,kj->ij', z, L)  # vectorized correlate
        scaled = sigma * correlated  # broadcast sigma
        out = mu - 0.5 * (sigma ** 2) + scaled  # broadcast mu and variance adj

        return out

    # validate sims vs historical (moment matching)
    def _validate(self, sim_log, hist_log):
        # flatten sims to (n_sims * n_days, n_assets)
        sim_flat = sim_log.reshape(-1, sim_log.shape[-1])
        sim_mean = np.mean(sim_flat, axis=0)
        hist_mean = hist_log.mean().values
        sim_std = np.std(sim_flat, axis=0)
        hist_std = hist_log.std().values
        sim_corr = pd.DataFrame(sim_flat).corr().values
        hist_corr = hist_log.corr().values

        return {
            'mean_diff': float(np.abs(sim_mean - hist_mean).mean()),
            'std_diff': float(np.abs(sim_std - hist_std).mean()),
            'corr_diff': float(np.abs(sim_corr - hist_corr).mean()),
        }
