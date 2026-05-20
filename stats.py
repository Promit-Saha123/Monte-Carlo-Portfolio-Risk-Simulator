# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 23:10:03 2025

@author: sahap
"""

import numpy as np

def var_cvar(losses, alpha):
    if not (0 < alpha < 1): # selected alpha must be between 0 and 1
        raise ValueError("alpha must be between 0 and 1")

    losses = np.asarray(losses, dtype=float) # converts losses into an array
    losses = losses[~np.isnan(losses)] # drops NaN values using NumPy
    
    if losses.size == 0: # if we have 0 losses
        raise ValueError("losses is empty after cleaning")

    var = np.quantile(losses, alpha) # find var by giving us the tail cutoff

    tail = losses[losses >= var] # boolean masking to filter only true values
    
    if tail.size > 0: 
        cvar = tail.mean()
    else:
        cvar = var

    return var, cvar

