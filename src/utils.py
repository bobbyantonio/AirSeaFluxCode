import numpy as np

def safe_power(x, y):
    if np.abs(int(y) - y) > 1e-6:
        x = np.clip(x, a_min=1e-6, a_max=None)
    return np.power(x, y)

def safe_exp(exponent):
    
    exponent = np.clip(exponent, a_min=-700, a_max=700)
    
    return np.exp(exponent)

def safe_multiply(x, y):
    
    assert x.dtype == y.dtype
    
    max_val = np.finfo(x.dtype).max
    
    result = np.multiply(x, y)
    
    result[result == np.inf] = max_val
    
    return result