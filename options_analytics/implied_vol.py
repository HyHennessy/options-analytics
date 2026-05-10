"""
Implied volatility solver for European call options.

Uses Newton-Raphson iteration with a Brenner-Subrahmanyam initial guess.
Raises ImpliedVolatilityError for invalid inputs or convergence failures.
"""

import numpy as np

from options_analytics.greeks import vega
from options_analytics.pricing import black_scholes_call


class ImpliedVolatilityError(Exception):
    """
    Raised when the implied volatility solver fails to converge or receives invalid inputs.

    This includes cases where the market price violates no-arbitrage bounds,
    vega collapses to zero, or the maximum number of iterations is exceeded.
    """


def implied_volatility(
    market_price: float,
    s: float,
    k: float,
    r: float,
    t: float,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """
    Compute the implied volatility of a European call option using Newton-Raphson.

    Solves for sigma in: Black-Scholes(sigma) - market_price = 0

    The Newton-Raphson update rule is:
        sigma_n+1 = sigma_n - (BS(sigma_n) - market_price) / Vega(sigma_n)

    The initial guess uses the Brenner-Subrahmanyam approximation:
        sigma_0 = sqrt(2*pi/T) * (market_price / S)

    Parameters
    ----------
    market_price   : observed market price of the call option
    s              : stock price
    k              : strike price
    r              : risk-free interest rate
    t              : time to expiry in years
    max_iterations : maximum number of Newton-Raphson iterations (default 100)
    tolerance      : convergence threshold on price error (default 1e-6)

    Raises
    ------
    ImpliedVolatilityError
        If inputs are invalid, the market price violates no-arbitrage bounds,
        vega collapses to zero, or the solver fails to converge.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if s <= 0:
        raise ImpliedVolatilityError(f"Stock price must be positive, got {s}")
    if k <= 0:
        raise ImpliedVolatilityError(f"Strike price must be positive, got {k}")
    if t <= 0:
        raise ImpliedVolatilityError(
            f"Time to expiry must be positive, got {t}"
        )
    if market_price <= 0:
        raise ImpliedVolatilityError(
            f"Market price must be positive, got {market_price}"
        )
    if market_price >= s:
        raise ImpliedVolatilityError(
            f"Market price {market_price} must be less than stock price {s} — "
            f"a call cannot be worth more than the underlying"
        )

    intrinsic = max(s - k * np.exp(-r * t), 0)
    if market_price <= intrinsic:
        raise ImpliedVolatilityError(
            f"Market price {market_price:.4f} is below intrinsic value {intrinsic:.4f} — "
            f"no real implied volatility exists"
        )

    # ------------------------------------------------------------------
    # Initial guess — Brenner-Subrahmanyam approximation
    # ------------------------------------------------------------------
    sigma = np.sqrt(2 * np.pi / t) * (market_price / s)

    # ------------------------------------------------------------------
    # Newton-Raphson loop
    # ------------------------------------------------------------------
    for i in range(max_iterations):

        price = black_scholes_call(s, k, r, t, sigma)
        price_error = price - market_price

        # Check convergence
        if abs(price_error) < tolerance:
            return sigma

        # vega() returns sensitivity per 1% move in vol (divided by 100).
        # Newton-Raphson needs the raw dC/dsigma, so multiply back.
        v = vega(s, k, r, t, sigma) * 100

        # Guard against vega collapse — deep ITM/OTM options
        if abs(v) < 1e-10:
            raise ImpliedVolatilityError(
                f"Vega collapsed to zero at iteration {i} — "
                f"option may be too deep in or out of the money"
            )

        # Newton-Raphson update
        sigma = sigma - price_error / v

        # Keep sigma in a sensible range
        sigma = max(1e-6, min(sigma, 10.0))

    raise ImpliedVolatilityError(
        f"Newton-Raphson failed to converge after {max_iterations} iterations — "
        f"final price error: {price_error:.2e}"
    )
