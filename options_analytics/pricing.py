"""
Black-Scholes option pricing functions for European calls and puts.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(
    s: float, k: float, r: float, t: float, sigma: float
) -> tuple[float, float]:
    """
    Compute d_1 and d_2, defined as:

    d_1 = (ln(s/k) + (r + sigma^2/2) * t) / (sigma * sqrt(t))
    d_2 = d_1 - sigma * sqrt(t)

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    d1 = (np.log(s / k) + (r + (sigma**2 / 2)) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def black_scholes_call(
    s: float, k: float, r: float, t: float, sigma: float
) -> float:
    """
    Black-Scholes price for a European call option, c, defined as:

    c = s * N(d_1) - k * e^(-r*t)*N(d_2)

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    d1, d2 = _d1_d2(s, k, r, t, sigma)
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def black_scholes_put(
    s: float, k: float, r: float, t: float, sigma: float
) -> float:
    """
    Black-Scholes price for a European put option, defined as:

    p = k*e^(-r*t) * N(-d_2) - s * N(-d_1)

    Computed directly rather than via put-call parity to avoid
    catastrophic cancellation for deep in-the-money puts.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    d1, d2 = _d1_d2(s, k, r, t, sigma)
    return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


def black_scholes_put_from_parity(
    s: float, k: float, r: float, t: float, sigma: float
) -> float:
    """
    Black-Scholes price for a European put option.
    Derived from put-call parity: p = c - s + k*e^(-r*t)

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    return black_scholes_call(s, k, r, t, sigma) - s + k * np.exp(-r * t)
