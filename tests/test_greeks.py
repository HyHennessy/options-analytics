"""
Tests for options_analytics.greeks — Black-Scholes Greeks for European options.

Covers: known values, signs, bounds, boundary conditions, call/put symmetry,
the Black-Scholes PDE identity, and consistency of all_greeks() against the
individual public functions.
"""

from options_analytics.greeks import (
    all_greeks,
    delta_call,
    delta_put,
    gamma,
    rho_call,
    rho_put,
    theta_call,
    theta_put,
    vega,
)
from options_analytics.pricing import black_scholes_call, black_scholes_put

# Standard test parameters used across multiple tests
S, K, R, T, SIGMA = 100, 100, 0.05, 1, 0.2


# ---------------------------------------------------------------------------
# Known values
# ---------------------------------------------------------------------------


def test_delta_call_known_value():
    assert abs(delta_call(S, K, R, T, SIGMA) - 0.6368) < 1e-3


def test_delta_put_known_value():
    assert abs(delta_put(S, K, R, T, SIGMA) - (-0.3632)) < 1e-3


def test_gamma_known_value():
    assert abs(gamma(S, K, R, T, SIGMA) - 0.01876) < 1e-3


def test_vega_known_value():
    # Per 1% move in vol
    assert abs(vega(S, K, R, T, SIGMA) - 0.3752) < 1e-3


def test_theta_call_known_value():
    # Per calendar day
    assert abs(theta_call(S, K, R, T, SIGMA) - (-0.01710)) < 1e-3


def test_theta_put_known_value():
    # Per calendar day
    assert abs(theta_put(S, K, R, T, SIGMA) - (-0.00477)) < 1e-3


def test_rho_call_known_value():
    # Per 1% move in rate
    assert abs(rho_call(S, K, R, T, SIGMA) - 0.5323) < 1e-3


def test_rho_put_known_value():
    # Per 1% move in rate
    assert abs(rho_put(S, K, R, T, SIGMA) - (-0.4189)) < 1e-3


# ---------------------------------------------------------------------------
# Signs
# ---------------------------------------------------------------------------


def test_delta_call_always_positive():
    for s in [50, 100, 150]:
        assert delta_call(s, K, R, T, SIGMA) > 0


def test_delta_put_always_negative():
    for s in [50, 100, 150]:
        assert delta_put(s, K, R, T, SIGMA) < 0


def test_gamma_always_positive():
    for s in [50, 100, 150]:
        assert gamma(s, K, R, T, SIGMA) > 0


def test_vega_always_positive():
    for s in [50, 100, 150]:
        assert vega(s, K, R, T, SIGMA) > 0


def test_theta_call_always_negative():
    for s in [50, 100, 150]:
        assert theta_call(s, K, R, T, SIGMA) < 0


def test_rho_call_always_positive():
    for s in [50, 100, 150]:
        assert rho_call(s, K, R, T, SIGMA) > 0


def test_rho_put_always_negative():
    for s in [50, 100, 150]:
        assert rho_put(s, K, R, T, SIGMA) < 0


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_delta_call_bounded():
    for s in [50, 100, 150]:
        assert 0 < delta_call(s, K, R, T, SIGMA) < 1


def test_delta_put_bounded():
    for s in [50, 100, 150]:
        assert -1 < delta_put(s, K, R, T, SIGMA) < 0


def test_delta_call_put_sum_to_one():
    """
    Delta call + |delta put| = 1 for same inputs.
    Follows from N(d1) + (1 - N(d1)) = 1.
    """
    for s in [50, 100, 150]:
        assert (
            delta_call(s, K, R, T, SIGMA)
            + abs(delta_put(s, K, R, T, SIGMA))
            - 1
        ) < 1e-10


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def test_delta_call_approaches_one_deep_itm():
    """Deep in-the-money call — almost certain to be exercised, delta → 1."""
    assert delta_call(500, K, R, T, SIGMA) > 0.999


def test_delta_call_approaches_zero_deep_otm():
    """Deep out-of-the-money call — almost certain to expire worthless, delta → 0."""
    assert delta_call(1, K, R, T, SIGMA) < 0.001


def test_delta_put_approaches_minus_one_deep_itm():
    """Deep in-the-money put — delta → -1."""
    assert delta_put(1, K, R, T, SIGMA) < -0.999


def test_delta_put_approaches_zero_deep_otm():
    """Deep out-of-the-money put — delta → 0."""
    assert delta_put(500, K, R, T, SIGMA) > -0.001


# ---------------------------------------------------------------------------
# Symmetry between calls and puts
# ---------------------------------------------------------------------------


def test_gamma_same_for_calls_and_puts():
    """
    Gamma must equal the numerical derivative of delta for both calls and puts.
    Since delta_put = delta_call - 1, the constant drops out and both give the
    same finite-difference gamma — confirming the single gamma() is consistent
    with both sides.
    """
    eps = 0.01
    num_gamma_call = (
        delta_call(S + eps, K, R, T, SIGMA)
        - delta_call(S - eps, K, R, T, SIGMA)
    ) / (2 * eps)
    num_gamma_put = (
        delta_put(S + eps, K, R, T, SIGMA) - delta_put(S - eps, K, R, T, SIGMA)
    ) / (2 * eps)
    assert abs(gamma(S, K, R, T, SIGMA) - num_gamma_call) < 1e-4
    assert abs(gamma(S, K, R, T, SIGMA) - num_gamma_put) < 1e-4


def test_vega_same_for_calls_and_puts():
    """
    Vega must equal the numerical derivative of price w.r.t. sigma for both
    calls and puts. Put-call parity guarantees they are equal; this confirms
    the single vega() is consistent with both option prices.
    """
    eps = 0.001
    num_vega_call = (
        (
            black_scholes_call(S, K, R, T, SIGMA + eps)
            - black_scholes_call(S, K, R, T, SIGMA - eps)
        )
        / (2 * eps)
        / 100
    )
    num_vega_put = (
        (
            black_scholes_put(S, K, R, T, SIGMA + eps)
            - black_scholes_put(S, K, R, T, SIGMA - eps)
        )
        / (2 * eps)
        / 100
    )
    assert abs(vega(S, K, R, T, SIGMA) - num_vega_call) < 1e-4
    assert abs(vega(S, K, R, T, SIGMA) - num_vega_put) < 1e-4


# ---------------------------------------------------------------------------
# Black-Scholes PDE identity
# ---------------------------------------------------------------------------


def test_bs_pde_identity_call():
    """
    All Greeks must jointly satisfy the Black-Scholes PDE:

    r*C = theta_annual + r*S*delta + 0.5*sigma^2*S^2*gamma

    Note: theta is divided by 365 to annualise it before use here.
    """
    c = black_scholes_call(S, K, R, T, SIGMA)
    d = delta_call(S, K, R, T, SIGMA)
    g = gamma(S, K, R, T, SIGMA)
    th = theta_call(S, K, R, T, SIGMA) * 365  # annualise

    lhs = R * c
    rhs = th + R * S * d + 0.5 * SIGMA**2 * S**2 * g

    assert abs(lhs - rhs) < 1e-5


def test_bs_pde_identity_put():
    """Same PDE identity must hold for puts."""
    p = black_scholes_put(S, K, R, T, SIGMA)
    d = delta_put(S, K, R, T, SIGMA)
    g = gamma(S, K, R, T, SIGMA)
    th = theta_put(S, K, R, T, SIGMA) * 365  # annualise

    lhs = R * p
    rhs = th + R * S * d + 0.5 * SIGMA**2 * S**2 * g

    assert abs(lhs - rhs) < 1e-5


# ---------------------------------------------------------------------------
# all_greeks consistency
# ---------------------------------------------------------------------------


def test_all_greeks_matches_individual_functions():
    """
    all_greeks() must return identical values to calling each
    function individually — verifies the single-pass optimisation
    hasn't introduced any errors.
    """
    greeks = all_greeks(S, K, R, T, SIGMA)

    assert abs(greeks["delta_call"] - delta_call(S, K, R, T, SIGMA)) < 1e-10
    assert abs(greeks["delta_put"] - delta_put(S, K, R, T, SIGMA)) < 1e-10
    assert abs(greeks["gamma"] - gamma(S, K, R, T, SIGMA)) < 1e-10
    assert abs(greeks["vega"] - vega(S, K, R, T, SIGMA)) < 1e-10
    assert abs(greeks["theta_call"] - theta_call(S, K, R, T, SIGMA)) < 1e-10
    assert abs(greeks["theta_put"] - theta_put(S, K, R, T, SIGMA)) < 1e-10
    assert abs(greeks["rho_call"] - rho_call(S, K, R, T, SIGMA)) < 1e-10
    assert abs(greeks["rho_put"] - rho_put(S, K, R, T, SIGMA)) < 1e-10
