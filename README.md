# options-analytics

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-143%20passed-brightgreen)
![Code style](https://img.shields.io/badge/code%20style-black-000000)
![CI](https://github.com/HyHennessy/options-analytics/actions/workflows/ci.yml/badge.svg)

A self-contained Python library for pricing, analysing, and backtesting European options strategies using the Black-Scholes model. Built as a portfolio project to demonstrate quantitative finance implementation in Python — covering closed-form pricing, all five Greeks, a Newton-Raphson implied volatility solver, and a strategy backtester with historical market data.

---

## Modules

| Module | Description |
|---|---|
| `pricing.py` | Black-Scholes call and put pricing |
| `greeks.py` | Delta, Gamma, Vega, Theta, Rho |
| `implied_vol.py` | Newton-Raphson IV solver with Brenner-Subrahmanyam initial guess |
| `backtester.py` | Five strategies with daily P&L tracking and bust detection |
| `visualisations.py` | Plotly charts for backtester results and volatility smile |

---

## Installation

Clone the repository and install dependencies into a virtual environment:

```bash
git clone https://github.com/HyHennessy/options-analytics.git
cd options-analytics
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

### Pricing

```python
from options_analytics.pricing import black_scholes_call, black_scholes_put

call = black_scholes_call(s=49, k=50, r=0.05, t=20/52, sigma=0.20)
put  = black_scholes_put(s=49,  k=50, r=0.05, t=20/52, sigma=0.20)

print(f"Call: ${call:.4f}")  # Call: $2.4005
print(f"Put:  ${put:.4f}")   # Put:  $2.4482
```

### Greeks

```python
from options_analytics.greeks import delta_call, gamma, vega, theta_call, rho_call

print(f"Delta: {delta_call(s=49, k=50, r=0.05, t=20/52, sigma=0.20):.4f}")  # 0.5216
print(f"Gamma: {gamma(s=49, k=50, r=0.05, t=20/52, sigma=0.20):.4f}")       # 0.0665
print(f"Vega:  {vega(s=49, k=50, r=0.05, t=20/52, sigma=0.20):.4f}")        # 0.1211
```

### Implied Volatility

```python
from options_analytics.implied_vol import implied_volatility
from options_analytics.pricing import black_scholes_call

price = black_scholes_call(s=100, k=100, r=0.05, t=1.0, sigma=0.25)
iv    = implied_volatility(price, s=100, k=100, r=0.05, t=1.0)

print(f"Implied vol: {iv:.4f}")  # 0.2500
```

The solver converges to a tolerance of 1e-6 using Newton-Raphson, with a Brenner-Subrahmanyam closed-form estimate as the starting point.

### Backtester

```python
from options_analytics.backtester import Backtester

bt = Backtester(
    ticker="AAPL",
    start="2025-01-01",
    end="2026-01-01",
    strategy="covered_call",
    initial_cash=100_000,
)

results = bt.run()
bt.summary()
```

---

## Strategies

| Strategy | Description |
|---|---|
| `long_call` | Buy a call, pay premium upfront. Profits if spot rises above strike + premium. |
| `long_put` | Buy a put, pay premium upfront. Profits if spot falls below strike − premium. |
| `covered_call` | Hold 100 shares and sell a call against them. Income strategy; caps upside. |
| `cash_secured_put` | Sell a put holding cash collateral. Profits if spot stays above strike - premium, at expiry. |
| `wheel` | Cycle between cash-secured puts and covered calls to harvest premium continuously. |

---

## Visualisations

```python
from options_analytics.visualisations import (
    plot_portfolio_value,
    plot_cumulative_pnl,
    plot_daily_pnl,
    plot_portfolio_breakdown,
    plot_volatility_smile,
)

# Backtester charts — requires results dict from running all strategies
strategies = ["long_call", "long_put", "covered_call", "cash_secured_put", "wheel"]
results = {
    s: Backtester(ticker="AAPL", start="2025-01-01", end="2026-01-01", strategy=s, initial_cash=100_000).run()
    for s in strategies
}

plot_portfolio_value(results, ticker="AAPL", start="2025-01-01", end="2026-01-01").show()
plot_cumulative_pnl(results, ticker="AAPL", start="2025-01-01", end="2026-01-01").show()

# Volatility smile
strikes   = [88, 91, 94, 97, 100, 103, 106, 109, 112]
true_vols = [0.30, 0.27, 0.25, 0.22, 0.20, 0.21, 0.22, 0.24, 0.26]
prices    = [black_scholes_call(s=100, k=k, r=0.05, t=0.25, sigma=sv) for k, sv in zip(strikes, true_vols)]
ivs       = [implied_volatility(p, s=100, k=k, r=0.05, t=0.25) for p, k in zip(prices, strikes)]
 
plot_volatility_smile(strikes, ivs, spot=100, expiry_label="3-month").show()
```

---

## Demo Notebooks

| Notebook | Contents |
|---|---|
| `options_analytics_core_demo.ipynb` | Pricing, Greeks, implied vol, volatility smile |
| `options_analytics_backtester_demo.ipynb` | All five strategies on real market data with charts and summary table |

---

## Tests

```bash
pytest tests/ -v
```

143 tests covering all modules. Backtester tests mock `yfinance.download` so no live network calls are made.

---

## Project Structure

```
options-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml
├── options_analytics/
│   ├── __init__.py
│   ├── pricing.py
│   ├── greeks.py
│   ├── implied_vol.py
│   ├── backtester.py
│   └── visualisations.py
├── tests/
│   ├── __init__.py
│   ├── test_pricing.py
│   ├── test_greeks.py
│   ├── test_implied_vol.py
│   └── test_backtester.py
├── options_analytics_core_demo.ipynb
├── options_analytics_backtester_demo.ipynb
├── .gitattributes
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Dependencies

- `numpy` — numerical computation
- `pandas` — data handling and time series
- `yfinance` — market data download
- `plotly` — interactive visualisations

---

## Implementation Notes

- Black-Scholes pricing follows the standard GBM framework; log-returns are assumed normally distributed under the risk-neutral measure.
- The implied volatility solver is implemented from scratch using Newton-Raphson — no `scipy.optimize` — converging to 1e-6 tolerance with vega-based step control.
- Historical volatility is computed as rolling annualised standard deviation of log-returns, back-filled to handle the initial window.
- The backtester uses calendar days for option expiry and 252 trading days for volatility annualisation.
- Bust detection handles both day-zero insolvency (insufficient capital to open a position) and mid-backtest insolvency (portfolio value falls to zero or below).

---

## Limitations

This library prices European options under the assumptions of the Black-Scholes model. It does not account for:

- **Volatility smile/skew** — BS assumes constant volatility; the vol smile observed in real markets reflects this model's limitations
- **American exercise** — early exercise premium is not modelled
- **Dividends** — the backtester does not adjust for discrete dividend payments
- **Transaction costs and slippage** — fills are assumed at the theoretical BS price
