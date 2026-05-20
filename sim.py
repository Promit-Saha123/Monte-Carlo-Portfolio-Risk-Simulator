# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 23:10:19 2025

@author: sahap
"""
import numpy as np #imports linear algebra tools as well

def mu_cov(hist_returns):
    """
    Parameters
    ----------
    hist_returns : pd.DataFrame of log returns, shape(time points, number of assets)

    Returns
    -------
    Mean Vector (Drift) and Covariance Matrix (Volatility and Correlation)
    Both of these are needed for Monte carlo Return = N(mu, cov)

    """
    mean = hist_returns.mean().values # mean log return per asset. .values strips asset names and returns a matrix of just the means
    cov = hist_returns.cov().values # computes covariance matrix
    return mean, cov
    
def cholesky_L(cov, jitter = 1e-10, max_attempts = 6):
    """
    Takes in cov and returns Cholesky factor.
    If it isn't PSD then jitter is added until it is.
    If it isn't after 6 attempts then an error is raised
    
    Returns
    -------
    Cholesky factor

    """
    cov = np.asarray(cov, dtype = float)
    n = cov.shape[0] # .shape gives rows, columns so shape[0] gives rows
    I = np.eye(n) # makes a n x n matrix so we can add jitter to the diagonal which doesn't change anything to much
    
    for i in range(max_attempts):
        try:
            return np.linalg.cholesky(cov + (jitter * (10 ** i)) * I) # 10 ** i increases the jitter 10x each run. exponential backoff then applies to diagonal values
        except np.linalg.LinAlgError:
            continue # if cholesky fails try with more jitter continue to the next number
            
    raise np.linalg.LinAlgError("Cholesky failed: covariance matrix is still not PSD after jitter")

def sim_log_returns(mean, cov, sim=10000, horizon_days=1, seed=None, path_based=False):
    """
    Parameters
    ----------
    mean : array-like, shape (n_assets,)
        Mean (drift) of log returns per day.
    cov : array-like, shape (n_assets, n_assets)
        Covariance matrix of daily log returns.
    sim : int
        Number of Monte Carlo simulations.
    horizon_days : int
        Number of days to simulate.
    seed : int or None
        RNG seed for reproducibility.
    path_based : bool
        If False: returns aggregated log returns over the horizon, shape (sim, n_assets).
        If True:  returns daily log return paths, shape (sim, horizon_days, n_assets).

    Returns
    -------
    np.ndarray
        Simulated log returns (aggregated or path-based depending on path_based).
    """
    rng = np.random.default_rng(seed)

    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if mean.ndim != 1:
        raise ValueError("mean must be a 1D array of shape (n_assets,)")
    n_assets = mean.shape[0]

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square (n_assets, n_assets) matrix")
    if cov.shape[0] != n_assets:
        raise ValueError("mean and cov dimensions do not match")

    if not isinstance(sim, (int, np.integer)) or sim < 1:
        raise ValueError("sim must be an integer >= 1")
    if not isinstance(horizon_days, (int, np.integer)) or horizon_days < 1:
        raise ValueError("horizon_days must be an integer >= 1")

    # -------- Single-period (aggregate over horizon) --------
    if not path_based:
        mean_h = mean * horizon_days
        cov_h = cov * horizon_days

        # Use our robust Cholesky for numerical safety
        L = cholesky_L(cov_h)

        Z = rng.standard_normal(size=(sim, n_assets))  # iid N(0,1)
        shocks = Z @ L.T                               # correlated
        sims = mean_h + shocks                         # add drift
        return sims  # (sim, n_assets)

    # -------- Path-based (daily correlated draws) --------
    L = cholesky_L(cov)

    Z = rng.standard_normal(size=(sim, horizon_days, n_assets))  # iid N(0,1)
    shocks = Z @ L.T                                             # correlate last dim
    sims_path = mean + shocks                                    # broadcast mean to each day
    return sims_path  # (sim, horizon_days, n_assets)

def sim_log_returns_bootstrap(hist_returns, sim: int = 10_000, horizon_days: int = 252, seed: int | None = None, path_based: bool = False,):
    """
    Bootstrap Monte Carlo simulation using a FIXED internal block size.

    - Resamples historical log returns (rows) with replacement
    - Preserves cross-asset correlation
    - Captures fat tails + realistic crashes
    - Uses a fixed block size of 5 steps (≈ one trading week)

    Returns
    -------
    np.ndarray
        Simulated log returns
    """

    BLOCK_SIZE = 5  # 🔒 fixed, non-configurable

    # Convert input to numpy
    R = hist_returns.values if hasattr(hist_returns, "values") else np.asarray(hist_returns, dtype=float)
    R = np.asarray(R, dtype=float)

    T, n_assets = R.shape
    if T < BLOCK_SIZE + 1:
        raise ValueError("Not enough historical data for bootstrap simulation.")
        
    #Trim extreme left and right values so that doesn't skew results
    low = np.quantile(R, 0.005, axis=0)
    high = np.quantile(R, 0.995, axis=0)
    R = np.clip(R, low, high)

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # NON-path-based: total log return over horizon
    # --------------------------------------------------
    if not path_based:
        totals = np.zeros((sim, n_assets), dtype=float)
        steps_done = 0

        while steps_done < horizon_days:
            b = min(BLOCK_SIZE, horizon_days - steps_done)
            start_idx = rng.integers(0, T - b + 1, size=sim)

            # Vectorized: extract all blocks at once
            for offset in range(b):
                totals += R[start_idx + offset]

            steps_done += b

        return totals  # (sim, n_assets)

    # --------------------------------------------------
    # PATH-based: full return paths
    # --------------------------------------------------
    paths = np.zeros((sim, horizon_days, n_assets), dtype=float)
    t = 0

    while t < horizon_days:
        b = min(BLOCK_SIZE, horizon_days - t)
        start_idx = rng.integers(0, T - b + 1, size=sim)

        # Vectorized: use advanced indexing to fill all simulations at once
        for offset in range(b):
            paths[:, t + offset, :] = R[start_idx + offset]

        t += b

    return paths  # (sim, horizon_days, n_assets)