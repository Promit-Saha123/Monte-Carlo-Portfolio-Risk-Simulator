# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 15:29:41 2025

@author: sahap
"""

import numpy as np 
import pandas as pd

# input/ux

def parse_tickers(tickers_str: str): # turns input string into a clean ticker list
    
    # split string input into pieces wherever there's a comma
    parts = tickers_str.split(",")
    
    # Clean each piece (remove spacing and convert to uppercase) and keep non-empty parts
    tickers = []
    
    for ticker in parts:
        cleaned = ticker.strip() # remove leading and trailing spaces
        if cleaned != "": # keep only if not blank
            tickers.append(cleaned.upper()) # converts to uppercase
    
    if len(tickers) == 0:
        raise ValueError("You must input at least one ticker")
    return tickers

# Validation / Normalization

def validate_alpha(alpha: float): # handles cases where alpha isn't between 0 and 1
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    return alpha

def validate_date_range(start_date: str, end_date: str | None): # makes sure end date is after start
    start = pd.to_datetime(start_date)

    if end_date is not None:
        end = pd.to_datetime(end_date)
        if end <= start:
            raise ValueError("End date must be later than start date.")

    return start_date, end_date

#validates dollar holdings, computes total portfolio value and converts holdings into normalized weights

def weights_from_holdings(holdings: np.ndarray): # holding is dollar amounts per asset. [5000, 3000, 2000] np.ndarray is just a type hint
    holdings = np.asarray(holdings, dtype = float) #convert to np array and force everything to be a float
    
    if np.any(holdings < 0): # negative dollar holdings don't make sense
        raise ValueError("Dollar holdings must be non-negative")
        
    total = holdings.sum() # compute total portfolio value. Total holdings.
    
    if total <= 0:
        raise ValueError("Total Portfolio value must be greater than 0")
    
    weights = holdings / total
    return weights, total # returns weight array and total portfolio value

def normalize_weights(weight: np.ndarray, tol: float = 1e-6):
    # use when user already knows weight or percentage. Other is used when dollar amount is known
    weight = np.asarray(weight, dtype=float)
    
    if np.any(weight < 0):
        raise ValueError("Weights cannot be negative")
    
    s = weight.sum()
    
    if s <= 0:
        raise ValueError("Sum of weights must be greater than 0")

    # check if weights already sum to 1 (within numerical tolerance)
    if not np.isclose(s, 1.0, atol=tol):
        weight = weight / s

    return weight
        
def validate_positive_val(x: float, name: str = "value"):
    x = float(x)
    if x <= 0:
        raise ValueError(f"{name} must be > 0")
    return x

# data prep

def clean_prices(prices: pd.DataFrame): # expected to be a pandas dataframe
    prices = prices.dropna(how = "all") # removes rows where all columns are NaN. This could be bc market was closed, hiccups, etc.
    prices = prices.dropna() # keeps rows where all assets have prices, removes any with NaN
    return prices


def compute_log_returns(prices: pd.DataFrame):
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan) # handles infinities. Rare but still possible
    return returns.dropna()

# Monte Carlo
    
def portfolio_returns(sim_returns, weights):
    """
    Converts simulated asset log returns into portfolio log returns.
    """
    weights = np.asarray(weights, dtype=float)
    return sim_returns @ weights

def portfolio_return_path(sim_returns, weights):
    """
    sim_returns: (sim, horizon, n_assets)
    returns:     (sim, horizon)
    """
    weights = np.asarray(weights, dtype=float)
    return sim_returns @ weights

def max_drawdown(value_path):
    peak = np.maximum.accumulate(value_path)
    drawdown = (value_path - peak) / peak
    return drawdown.min()

def time_to_failure(value_path, threshold=0.7):
    for t, v in enumerate(value_path):
        if v <= threshold * value_path[0]:
            return t
    return None 