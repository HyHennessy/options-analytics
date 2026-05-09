import numpy as np
import pytest

from options_analytics.pricing import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_put_from_parity,
)

# Standard test parameters used across multiple tests.
S, K, R, T, SIGMA = 49, 50, 0.05, 20 / 52, 0.2


def test_call_known_value():
    """
    Call with standard parameters should be ~2.43.
    """
    assert abs(black_scholes_call(S, K, R, T, SIGMA) - 2.4) < 1e-2


def test_put_known_value():
    """
    Put with standard parameters should be ~2.45.
    """
    assert abs(black_scholes_put(S, K, R, T, SIGMA) - 2.45) < 1e-2


def test_put_call_parity():
    """
    Put-call parity must hold exactly: c - p = s - k*e^(-r*t)
    Tests consistency between the direct and parity implementations.
    """
    c = black_scholes_call(S, K, R, T, SIGMA)
    p = black_scholes_put(S, K, R, T, SIGMA)
    lhs = c - p
    rhs = S - K * np.exp(-R * T)
    assert abs(lhs - rhs) < 1e-10


def test_direct_and_parity_put_agree():
    """
    Both put implementations must return the same price.
    """
    p_direct = black_scholes_put(S, K, R, T, SIGMA)
    p_parity = black_scholes_put_from_parity(S, K, R, T, SIGMA)
    assert abs(p_direct - p_parity) < 1e-10


def test_call_increases_with_stock_price():
    """
    A call option becomes more valuable as the stock price rises.
    """
    c_low = black_scholes_call(90, K, R, T, SIGMA)
    c_mid = black_scholes_call(100, K, R, T, SIGMA)
    c_high = black_scholes_call(110, K, R, T, SIGMA)
    assert c_low < c_mid < c_high


def test_put_decreases_with_stock_price():
    """
    A put option becomes less valuable as the stock price rises.
    """
    p_low = black_scholes_put(90, K, R, T, SIGMA)
    p_mid = black_scholes_put(100, K, R, T, SIGMA)
    p_high = black_scholes_put(110, K, R, T, SIGMA)
    assert p_low > p_mid > p_high


def test_call_never_negative():
    """
    An option price can never be negative.
    """
    assert black_scholes_call(50, 100, R, T, SIGMA) >= 0
    assert black_scholes_call(100, 100, R, T, SIGMA) >= 0
    assert black_scholes_call(200, 100, R, T, SIGMA) >= 0


def test_zero_volatility_call():
    """
    With zero volatility the call price collapses to max(s - k*e^(-rT), 0)
    — the present value of the intrinsic value only.

    Test an in-the-money call.
    """
    intrinsic = max(55 - K * np.exp(-R * T), 0)
    assert abs(black_scholes_call(55, K, R, T, sigma=1e-10) - intrinsic) < 1e-4
