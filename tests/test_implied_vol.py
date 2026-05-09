"""
Tests for options_analytics.implied_vol — Newton-Raphson implied volatility solver.

Covers: roundtrip accuracy (ATM, OTM, ITM, high vol), input validation error
messages, convergence under tighter tolerance, and the exception hierarchy.
"""

import numpy as np
import pytest

from options_analytics.implied_vol import (
    ImpliedVolatilityError,
    implied_volatility,
)
from options_analytics.pricing import black_scholes_call

S, K, R, T = 100, 100, 0.05, 1


def test_roundtrip():
    """
    IV of a BS price should recover the volatility used to generate it.
    """
    sigma = 0.2
    price = black_scholes_call(S, K, R, T, sigma)
    assert abs(implied_volatility(price, S, K, R, T) - sigma) < 1e-6


def test_roundtrip_high_vol():
    """IV solver should recover high volatility inputs accurately."""
    sigma = 0.6
    price = black_scholes_call(S, K, R, T, sigma)
    assert abs(implied_volatility(price, S, K, R, T) - sigma) < 1e-6


def test_roundtrip_otm():
    """Out-of-the-money call."""
    sigma = 0.25
    price = black_scholes_call(S, 120, R, T, sigma)
    assert abs(implied_volatility(price, S, 120, R, T) - sigma) < 1e-6


def test_roundtrip_itm():
    """In-the-money call."""
    sigma = 0.25
    price = black_scholes_call(S, 80, R, T, sigma)
    assert abs(implied_volatility(price, S, 80, R, T) - sigma) < 1e-6


def test_error_message_stock_price():
    """Error message should identify the invalid stock price."""
    with pytest.raises(ImpliedVolatilityError) as exc_info:
        implied_volatility(5.0, s=0, k=K, r=R, t=T)
    assert "Stock price" in str(exc_info.value)


def test_error_message_strike():
    """Error message should identify the invalid strike price."""
    with pytest.raises(ImpliedVolatilityError) as exc_info:
        implied_volatility(5.0, s=S, k=0, r=R, t=T)
    assert "Strike price" in str(exc_info.value)


def test_error_message_time():
    """Error message should identify the invalid time to expiry."""
    with pytest.raises(ImpliedVolatilityError) as exc_info:
        implied_volatility(5.0, s=S, k=K, r=R, t=0)
    assert "Time to expiry" in str(exc_info.value)


def test_error_message_market_price():
    """Error message should identify the invalid market price."""
    with pytest.raises(ImpliedVolatilityError) as exc_info:
        implied_volatility(0.0, s=S, k=K, r=R, t=T)
    assert "Market price" in str(exc_info.value)


def test_error_message_price_above_stock():
    """Error message should identify that price exceeds stock price."""
    with pytest.raises(ImpliedVolatilityError) as exc_info:
        implied_volatility(S + 1, s=S, k=K, r=R, t=T)
    assert "less than stock price" in str(exc_info.value)


def test_error_message_below_intrinsic():
    """Error message should identify that price is below intrinsic value."""
    intrinsic = max(S - K * np.exp(-R * T), 0)
    with pytest.raises(ImpliedVolatilityError) as exc_info:
        implied_volatility(intrinsic - 0.01, s=S, k=K, r=R, t=T)
    assert "intrinsic value" in str(exc_info.value)


def test_convergence_tolerance():
    """Tighter tolerance."""
    sigma = 0.2
    price = black_scholes_call(S, K, R, T, sigma)
    assert (
        abs(implied_volatility(price, S, K, R, T, tolerance=1e-10) - sigma)
        < 1e-9
    )


def test_implied_volatility_error_is_exception():
    """
    ImpliedVolatilityError must be a subclass of Exception so it can
    be caught by broad except clauses in calling code.
    """
    assert issubclass(ImpliedVolatilityError, Exception)
    err = ImpliedVolatilityError("test message")
    assert str(err) == "test message"
