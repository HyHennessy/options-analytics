"""
Black-Scholes Greeks for European call and put options.
"""

import numpy as np
from scipy.stats import norm

from options_analytics.pricing import _d1_d2

# ---------------------------------------------------------------------------
# Private core functions — take d1/d2 directly to avoid redundant computation
# ---------------------------------------------------------------------------


def _delta_call(d1: float) -> float:
    """Core delta computation for calls."""
    return norm.cdf(d1)


def _delta_put(d1: float) -> float:
    """Core delta computation for puts."""
    return norm.cdf(d1) - 1


def _gamma(d1: float, s: float, sigma: float, t: float) -> float:
    """Core gamma computation, identical for calls and puts."""
    return norm.pdf(d1) / (s * sigma * np.sqrt(t))


def _vega(d1: float, s: float, t: float) -> float:
    """Core vega computation, per 1% move in volatility."""
    return s * np.sqrt(t) * norm.pdf(d1) / 100


def _theta_call(
    d1: float,
    d2: float,
    s: float,
    k: float,
    r: float,
    t: float,
    sigma: float,
    days: int = 365,
) -> float:
    """
    Core theta computation for calls per calendar day (days=365) or
    trading day (days=252) depending on convention required.
    """
    return (
        -(norm.pdf(d1) * s * sigma) / (2 * np.sqrt(t))
        - (r * k * np.exp(-r * t) * norm.cdf(d2))
    ) / days


def _theta_put(
    d1: float,
    d2: float,
    s: float,
    k: float,
    r: float,
    t: float,
    sigma: float,
    days: int = 365,
) -> float:
    """
    Core theta computation for puts per calendar day (days=365) or
    trading day (days=252) depending on convention required.
    """
    return (
        -(norm.pdf(d1) * s * sigma) / (2 * np.sqrt(t))
        + (r * k * np.exp(-r * t) * norm.cdf(-d2))
    ) / days


def _rho_call(d2: float, k: float, r: float, t: float) -> float:
    """Core rho computation for calls, per 1% move in the risk-free rate."""
    return k * t * np.exp(-r * t) * norm.cdf(d2) / 100


def _rho_put(d2: float, k: float, r: float, t: float) -> float:
    """Core rho computation for puts, per 1% move in the risk-free rate."""
    return -k * t * np.exp(-r * t) * norm.cdf(-d2) / 100


# ---------------------------------------------------------------------------
# Public API — each computes d1/d2 once and delegates to private functions
# ---------------------------------------------------------------------------


def delta_call(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """
    Delta for a European call option, defined as:

    delta_call = N(d_1)

    Measures sensitivity of the call price to a $1 move in the stock price.
    Bounded between 0 and 1. At-the-money calls have delta ≈ 0.5.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    d1, _ = _d1_d2(s, k, r, t, sigma)
    return _delta_call(d1)


def delta_put(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """
    Delta for a European put option, defined as:

    delta_put = N(d_1) - 1

    Measures sensitivity of the put price to a $1 move in the stock price.
    Bounded between -1 and 0. At-the-money puts have delta ≈ -0.5.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    d1, _ = _d1_d2(s, k, r, t, sigma)
    return _delta_put(d1)


def gamma(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """
    Gamma for a European option, defined as:

    gamma = N'(d_1) / (s * sigma * sqrt(t))

    Measures the rate of change of delta with respect to the stock price.
    Identical for calls and puts. High gamma means delta changes rapidly,
    requiring frequent rebalancing of a delta hedge.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    d1, _ = _d1_d2(s, k, r, t, sigma)
    return _gamma(d1, s, sigma, t)


def vega(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """
    Vega for a European option, defined as:

    vega = s * sqrt(t) * N'(d_1) / 100

    Measures sensitivity of the option price to a 1% move in volatility.
    Identical for calls and puts. Divided by 100 so the output represents
    the price change per 1 percentage point move in sigma.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    d1, _ = _d1_d2(s, k, r, t, sigma)
    return _vega(d1, s, t)


def theta_call(
    s: float, k: float, r: float, t: float, sigma: float, days: int = 365
) -> float:
    """
    Theta for a European call option, defined as:

    theta_call = (-(N'(d_1) * s * sigma) / (2 * sqrt(t)) - r * k * e^(-rT) * N(d_2)) / days

    Measures the daily time decay of the call price.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    days  : day convention — 365 for calendar days (default), 252 for trading days
    """
    d1, d2 = _d1_d2(s, k, r, t, sigma)
    return _theta_call(d1, d2, s, k, r, t, sigma, days)


def theta_put(
    s: float, k: float, r: float, t: float, sigma: float, days: int = 365
) -> float:
    """
    Theta for a European put option, defined as:

    theta_put = (-(N'(d_1) * s * sigma) / (2 * sqrt(t)) + r * k * e^(-rT) * N(-d_2)) / days

    Measures the daily time decay of the put price. Deep in-the-money puts
    can have positive theta due to interest earned on the strike.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    days  : day convention — 365 for calendar days (default), 252 for trading days
    """
    d1, d2 = _d1_d2(s, k, r, t, sigma)
    return _theta_put(d1, d2, s, k, r, t, sigma, days)


def rho_call(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """
    Rho for a European call option, defined as:

    rho_call = k * t * e^(-rT) * N(d_2)

    Measures sensitivity of the call price to a 1% move in the risk-free
    rate. Generally the smallest of the five Greeks in practice —
    option prices are relatively insensitive to small rate moves.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    _, d2 = _d1_d2(s, k, r, t, sigma)
    return _rho_call(d2, k, r, t)


def rho_put(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """
    Rho for a European put option, defined as:

    rho_put = -k * t * e^(-rT) * N(-d_2)

    Measures sensitivity of the put price to a 1% move in the risk-free
    rate. Negative for puts — rising rates reduce put value as the
    present value of the strike falls.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    """
    _, d2 = _d1_d2(s, k, r, t, sigma)
    return _rho_put(d2, k, r, t)


def all_greeks(
    s: float, k: float, r: float, t: float, sigma: float, days: int = 365
) -> dict:
    """
    Compute all Greeks for both calls and puts in a single pass.

    More efficient than calling each function individually as d1 and d2
    are computed once and shared across all calculations. This is the
    realistic use case in production — risk systems compute all Greeks
    simultaneously for each position.

    Parameters
    ----------
    s     : stock price
    k     : strike price
    r     : risk-free interest rate
    t     : time to expiry in years
    sigma : annualised volatility
    days  : day convention for theta — 365 for calendar days (default), 252 for trading days

    Returns
    -------
    dict with keys: delta_call, delta_put, gamma, vega,
                    theta_call, theta_put, rho_call, rho_put
    """
    d1, d2 = _d1_d2(s, k, r, t, sigma)
    return {
        "delta_call": _delta_call(d1),
        "delta_put": _delta_put(d1),
        "gamma": _gamma(d1, s, sigma, t),
        "vega": _vega(d1, s, t),
        "theta_call": _theta_call(d1, d2, s, k, r, t, sigma, days),
        "theta_put": _theta_put(d1, d2, s, k, r, t, sigma, days),
        "rho_call": _rho_call(d2, k, r, t),
        "rho_put": _rho_put(d2, k, r, t),
    }
