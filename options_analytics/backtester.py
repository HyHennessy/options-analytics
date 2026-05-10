"""
Backtester for different options strategies.

Strategies
----------
Long Call:
Buy a call option, paying the premium upfront.
Bet:  The stock rises above the strike by more than the premium paid.
Risk: Maximum loss is the premium paid. The option expires worthless if the
      stock finishes below the strike.

Long Put:
Buy a put option, paying the premium upfront.
Bet:  The stock falls below the strike by more than the premium paid.
Risk: Maximum loss is the premium paid. The option expires worthless if the
      stock finishes above the strike.

Covered Call:
Buy 100 shares and sell a call option against them, collecting the premium
upfront.
Bet:  The stock stays flat or rises modestly — you keep the premium and the
      shares.
Risk: A large rally above the strike caps your upside.
      A significant drop in the stock hurts you, only partially cushioned by
      the premium received.

Cash-Secured Put:
Sell a put option and hold enough cash to buy the shares if assigned.
Bet:  The stock stays above the strike at expiry — the put expires worthless
      and you keep the premium.
Risk: If the stock falls below the strike you are obligated to buy shares at
      above-market price. In a sustained downturn your losses can significantly
      exceed the premium collected.

Wheel:
A sequenced strategy that cycles between selling cash-secured puts and covered
calls. Sell puts until assigned, take the shares and sell calls, then return to
selling puts when the shares are called away.
Bet:  The stock trades sideways or within a range — you continuously harvest
      premium through the cycle.
Risk: Getting stuck holding shares in a sustained downtrend, where the stock
      falls faster than the premium income accumulates. The wheel does not
      protect against directional risk — it just generates income while it
      persists.
"""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from options_analytics.pricing import black_scholes_call, black_scholes_put

TRADING_DAYS = 252
CALENDAR_DAYS = 365
SHARES_PER_CONTRACT = 100
MIN_SIGMA = 0.01
VALID_STRATEGIES = frozenset(
    {"long_call", "long_put", "covered_call", "cash_secured_put", "wheel"}
)


class Backtester:
    def __init__(
        self,
        ticker: str,
        start: str,
        end: str,
        strategy: str,
        strike_pct: float = 1.0,
        expiry_days: int = 30,
        r: float = 0.05,
        vol_window: int = 30,
        initial_cash: float = 100_000.0,
    ):
        """
        Parameters
        ----------
        ticker       : Yahoo Finance ticker symbol
        start        : backtest start date, 'YYYY-MM-DD'
        end          : backtest end date, 'YYYY-MM-DD'
        strategy     : one of VALID_STRATEGIES — see module-level constant
        strike_pct   : strike price weighted by stock price (1.0 = ATM, 1.05 = 5% OTM call)
        expiry_days  : calendar days until option expiry
        r            : annualised risk-free rate
        vol_window   : rolling window in trading days for historical volatility
        initial_cash : starting cash balance in dollars
        """

        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------
        if pd.Timestamp(start) >= pd.Timestamp(end):
            raise ValueError(f"start ({start}) must be before end ({end})")
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"{strategy} is not a recognised strategy. "
                f"The recognised strategies are {sorted(VALID_STRATEGIES)}"
            )
        if strike_pct <= 0:
            raise ValueError(f"strike_pct ({strike_pct}) must be positive.")
        if initial_cash <= 0:
            raise ValueError(f"initial_cash ({initial_cash}) must be positive.")
        if not isinstance(expiry_days, int):
            raise TypeError(f"expiry_days ({expiry_days}) must be an integer.")
        if expiry_days <= 0:
            raise ValueError(f"expiry_days ({expiry_days}) must be positive.")
        if not isinstance(vol_window, int):
            raise TypeError(f"vol_window ({vol_window}) must be an integer.")
        if vol_window <= 0:
            raise ValueError(f"vol_window ({vol_window}) must be positive.")
        if r < 0:
            raise ValueError(f"r ({r}) must be non-negative.")

        self.ticker = ticker
        self.start = start
        self.end = end
        self.strategy = strategy
        self.strike_pct = strike_pct
        self.expiry_days = expiry_days
        self.r = r
        self.vol_window = vol_window
        self.initial_cash = initial_cash
        self._results: Optional[pd.DataFrame] = None

        # Back tester state variables — initialised (and therefore updated) in run()
        self.cash: float = 0.0
        self.shares: int = 0
        self.open_option: Optional[dict] = None
        self._wheel_state: str = "put"
        self._bust_date: Optional[pd.Timestamp] = None

    def run(self) -> pd.DataFrame:
        """
        Executes the backtest and returns a daily P&L DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: date, spot, cash, stock_value, option_value,
                     total_value, daily_pnl, cumulative_pnl
        """
        data = self._download_data()
        close = data["Close"]
        sigmas = self._compute_sigma(close)

        self.cash = self.initial_cash
        self.shares = 0
        self.open_option = None
        self._wheel_state = "put"
        self._bust_date = None

        records = []

        for i, (date, spot) in enumerate(close.items()):
            spot = float(spot)
            sigma = max(float(sigmas[date]), MIN_SIGMA)

            if i == 0:
                self._open_initial_position(spot, sigma, date)
                if self.open_option is None and self.shares == 0:
                    self._bust_date = date
                    break
            elif (
                self.open_option is not None
                and date >= self.open_option["expiry"]
            ):
                self._handle_expiry(spot, sigma, date)
                if (
                    self.open_option is None
                    and self.shares == 0
                    and self.strategy in ("long_call", "long_put")
                ):
                    self._bust_date = date
                    records.append(
                        {
                            "date": date,
                            "spot": spot,
                            "cash": self.cash,
                            "stock_value": 0.0,
                            "option_value": 0.0,
                            "total_value": self.cash,
                        }
                    )
                    break

            stock_value = self.shares * spot
            option_value = self._mark_option(spot, sigma, date)
            total_value = self.cash + stock_value + option_value

            if (
                self.open_option is not None
                and self.open_option["direction"] == "short"
            ):
                bust = self.cash + stock_value < self._close_cost(spot)
                if bust:
                    self.cash += stock_value
                    self.cash -= (
                        self._close_cost(spot)
                        if self.open_option["type"] == "call"
                        else 0
                    )
                    self.shares = 0
                    self._bust_date = date
                    records.append(
                        {
                            "date": date,
                            "spot": spot,
                            "cash": 0.0,
                            "stock_value": 0.0,
                            "option_value": 0.0,
                            "total_value": 0.0,
                        }
                    )
                    break

            records.append(
                {
                    "date": date,
                    "spot": spot,
                    "cash": self.cash,
                    "stock_value": stock_value,
                    "option_value": option_value,
                    "total_value": total_value,
                }
            )

        df = pd.DataFrame(records)

        # Catch the case where we don't have enough cash to start the strategy.
        if df.empty:
            self._results = df
            return df

        df["daily_pnl"] = df["total_value"].diff().fillna(0.0)
        df["cumulative_pnl"] = df["total_value"] - df["total_value"].iloc[0]
        self._results = df
        return df

    def summary(self) -> None:
        """
        Print performance statistics for the completed backtest.

        Raises
        ------
        RuntimeError
            If run() has not been called yet.
        """
        if self._results is None:
            raise RuntimeError("Call run() before summary()")

        if self._results.empty:
            print(
                f"*** BUST: portfolio insolvent on "
                f"{self._bust_date.date()}\n"
                f" — not enough initial cash to open position ***"
            )
            print(f"Strategy:           {self.strategy}")
            print(f"Ticker:             {self.ticker}")
            print(f"Period:             {self.start} to {self.end}")
            print(f"Cash remaining:     ${self.cash:,.2f}")
            print(f"Shares held:        {self.shares}")
            print(f"Stock value:        $0.00")
            print(f"Open contracts:     0")
            print(f"Option value:       $0.00")
            print(f"Portfolio value:    ${self.cash:,.2f}")
            return

        if self._bust_date is not None:
            print(
                f"*** BUST: portfolio insolvent on {self._bust_date.date()} ***"
            )

        df = self._results
        n_days = len(df)
        initial = df["total_value"].iloc[0]
        final = df["total_value"].iloc[-1]

        total_return = (final - initial) / initial
        years = n_days / TRADING_DAYS
        annualised_return = (1 + total_return) ** (1 / years) - 1

        daily_rf = self.r / TRADING_DAYS
        daily_returns = df["total_value"].pct_change().dropna()
        excess_returns = daily_returns - daily_rf
        sharpe = (excess_returns.mean() * TRADING_DAYS) / (
            excess_returns.std() * np.sqrt(TRADING_DAYS)
        )

        rolling_max = df["total_value"].cummax()
        max_drawdown = ((df["total_value"] - rolling_max) / rolling_max).min()

        win_rate = (df["daily_pnl"] > 0).mean()

        print(f"Strategy:           {self.strategy}")
        print(f"Ticker:             {self.ticker}")
        print(f"Period:             {self.start} to {self.end}")
        if self._bust_date is not None:
            print(f"Terminated:         {self._bust_date.date()} (bust)")
        print(f"Cash remaining:     ${self.cash:,.2f}")
        print(f"Shares held:        {self.shares}")
        print(f"Stock value:        ${df['stock_value'].iloc[-1]:,.2f}")
        print(
            f"Open contracts:     {self.open_option['contracts'] if self.open_option else 0}"
        )
        print(f"Option value:       ${df['option_value'].iloc[-1]:,.2f}")
        print(f"Portfolio value:    ${df['total_value'].iloc[-1]:,.2f}")
        print(f"Total return:       {total_return:.2%}")
        print(f"Annualised return:  {annualised_return:.2%}")
        print(f"Sharpe ratio:       {sharpe:.2f}")
        print(f"Max drawdown:       {max_drawdown:.2%}")
        print(f"Win rate:           {win_rate:.2%}")

    def _download_data(self) -> pd.DataFrame:
        """Download and validate OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance."""
        data = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )

        # Guard against the MultiIndex column returned by yfinance sometimes. (It is meant
        # to be there for multiple tickers, ie ("Close", "Ticker")).
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty:
            raise ValueError(
                f"No data returned for ticker '{self.ticker}' between "
                f"{self.start} and {self.end}. Check the ticker is valid and "
                f"the date range contains trading days."
            )
        return data

    def _compute_sigma(self, prices: pd.Series) -> pd.Series:
        """
        Compute rolling annualised historical volatility (AHV) from log returns.

        AHV = std(ln(p_t / p_{t-1}))

        This is more useful than simple returns as they're symmetric,
        time-additive and closer to normally distributed.

        (Black-Scholes is also built on the assumption that log returns are
        normally distributed)
        """

        log_returns = np.log(prices / prices.shift(1))

        # Multiply by the square root of the number of trading days to annualise.
        sigma = log_returns.rolling(window=self.vol_window).std() * np.sqrt(
            TRADING_DAYS
        )

        # Replace the NaNs, created from the first vol_window - 1 log_returns, with the closest
        # not NaN results.
        return sigma.bfill()

    def _buy_call(
        self, spot: float, sigma: float, date: pd.Timestamp
    ) -> Optional[dict]:
        """Price and open a long call, debiting premium from cash."""
        strike = spot * self.strike_pct
        expiry = date + timedelta(days=self.expiry_days)
        t = self.expiry_days / CALENDAR_DAYS
        premium = black_scholes_call(spot, strike, self.r, t, sigma)
        if self.cash < premium * SHARES_PER_CONTRACT:
            return None  # insufficient cash to open position
        self.cash -= premium * SHARES_PER_CONTRACT
        return {
            "type": "call",
            "direction": "long",
            "strike": strike,
            "expiry": expiry,
            "premium_paid": premium * SHARES_PER_CONTRACT,
            "contracts": 1,
        }

    def _buy_put(
        self, spot: float, sigma: float, date: pd.Timestamp
    ) -> Optional[dict]:
        """Price and open a long put, debiting premium from cash."""
        strike = spot * self.strike_pct
        expiry = date + timedelta(days=self.expiry_days)
        t = self.expiry_days / CALENDAR_DAYS
        premium = black_scholes_put(spot, strike, self.r, t, sigma)
        if self.cash < premium * SHARES_PER_CONTRACT:
            return None  # insufficient cash to open position
        self.cash -= premium * SHARES_PER_CONTRACT
        return {
            "type": "put",
            "direction": "long",
            "strike": strike,
            "expiry": expiry,
            "premium_paid": premium * SHARES_PER_CONTRACT,
            "contracts": 1,
        }

    def _sell_call(self, spot: float, sigma: float, date: pd.Timestamp) -> dict:
        """Price and open a short call, crediting premium to cash."""
        strike = spot * self.strike_pct
        expiry = date + timedelta(days=self.expiry_days)
        t = self.expiry_days / CALENDAR_DAYS
        premium = black_scholes_call(spot, strike, self.r, t, sigma)
        self.cash += premium * SHARES_PER_CONTRACT
        return {
            "type": "call",
            "direction": "short",
            "strike": strike,
            "expiry": expiry,
            "premium_received": premium * SHARES_PER_CONTRACT,
            "contracts": 1,
        }

    def _sell_put(self, spot: float, sigma: float, date: pd.Timestamp) -> dict:
        """Price and open a short put, crediting premium to cash."""
        strike = spot * self.strike_pct
        expiry = date + timedelta(days=self.expiry_days)
        t = self.expiry_days / CALENDAR_DAYS
        premium = black_scholes_put(spot, strike, self.r, t, sigma)
        self.cash += premium * SHARES_PER_CONTRACT
        return {
            "type": "put",
            "direction": "short",
            "strike": strike,
            "expiry": expiry,
            "premium_received": premium * SHARES_PER_CONTRACT,
            "contracts": 1,
        }

    def _check_itm(self, spot: float) -> bool:
        """Return True if the open option is in-the-money at expiry."""
        if self.open_option is None:
            return False
        if self.open_option["type"] == "call":
            return spot > self.open_option["strike"]
        return spot < self.open_option["strike"]

    def _mark_option(
        self, spot: float, sigma: float, date: pd.Timestamp
    ) -> float:
        """Return the mark-to-market value of the open option position."""
        if self.open_option is None:
            return 0.0
        t = max((self.open_option["expiry"] - date).days / CALENDAR_DAYS, 1e-6)
        try:
            if self.open_option["type"] == "call":
                price = black_scholes_call(
                    spot, self.open_option["strike"], self.r, t, sigma
                )
            else:
                price = black_scholes_put(
                    spot, self.open_option["strike"], self.r, t, sigma
                )
        except (ValueError, ZeroDivisionError):
            price = 0.0
        sign = 1.0 if self.open_option["direction"] == "long" else -1.0
        return sign * price * SHARES_PER_CONTRACT

    def _open_initial_position(
        self, spot: float, sigma: float, date: pd.Timestamp
    ) -> None:
        """Set up the initial portfolio position on day zero."""
        if self.strategy == "long_call":
            self.open_option = self._buy_call(spot, sigma, date)
        elif self.strategy == "long_put":
            self.open_option = self._buy_put(spot, sigma, date)
        elif self.strategy == "covered_call":
            # Check we have enough cash for the shares.
            if self.cash < spot * SHARES_PER_CONTRACT:
                return  # bust — insufficient capital check in run() will detect this
            self.cash -= spot * SHARES_PER_CONTRACT
            self.shares = SHARES_PER_CONTRACT
            self.open_option = self._sell_call(spot, sigma, date)
        elif self.strategy == "cash_secured_put":
            # Must hold strike * 100 in cash as collateral.
            if self.cash < spot * self.strike_pct * SHARES_PER_CONTRACT:
                return  # bust — collateral check in run() will detect this
            self.open_option = self._sell_put(spot, sigma, date)
        elif self.strategy == "wheel":
            # The first part of the wheel is a cash secured put —
            # must hold strike * 100 in cash as collateral.
            if self.cash < spot * self.strike_pct * SHARES_PER_CONTRACT:
                return  # bust — collateral check in run() will detect this
            self._wheel_state = "put"
            self.open_option = self._sell_put(spot, sigma, date)

    def _handle_expiry(
        self, spot: float, sigma: float, date: pd.Timestamp
    ) -> None:
        """Resolve the expired option, handle assignment, and open the next position."""
        itm = self._check_itm(spot)
        strike = self.open_option["strike"]
        self.open_option = None

        if self.strategy == "long_call":
            if itm:
                self.cash += (spot - strike) * SHARES_PER_CONTRACT
            self.open_option = self._buy_call(spot, sigma, date)

        elif self.strategy == "long_put":
            if itm:
                self.cash += (strike - spot) * SHARES_PER_CONTRACT
            self.open_option = self._buy_put(spot, sigma, date)

        elif self.strategy == "covered_call":
            if itm:
                self.cash += strike * SHARES_PER_CONTRACT
                self.shares = 0
                if self.cash < spot * SHARES_PER_CONTRACT:
                    return  # bust — close_cost check in run() will detect this
                self.cash -= spot * SHARES_PER_CONTRACT
                self.shares = SHARES_PER_CONTRACT
            self.open_option = self._sell_call(spot, sigma, date)

        elif self.strategy == "cash_secured_put":
            if itm:
                if self.cash < strike * SHARES_PER_CONTRACT:
                    return  # bust — collateral check in run() will detect this
                self.cash -= strike * SHARES_PER_CONTRACT
                self.shares = SHARES_PER_CONTRACT
                self.cash += spot * SHARES_PER_CONTRACT
                self.shares = 0
            self.open_option = self._sell_put(spot, sigma, date)

        elif self.strategy == "wheel":
            if self._wheel_state == "put":
                if itm:
                    if self.cash < strike * SHARES_PER_CONTRACT:
                        return  # bust — collateral check in run() will detect this
                    self.cash -= strike * SHARES_PER_CONTRACT
                    self.shares = SHARES_PER_CONTRACT
                    self._wheel_state = "call"
                    self.open_option = self._sell_call(spot, sigma, date)
                else:
                    self.open_option = self._sell_put(spot, sigma, date)
            else:  # 'call'
                if itm:
                    self.cash += strike * SHARES_PER_CONTRACT
                    self.shares = 0
                    self._wheel_state = "put"
                    self.open_option = self._sell_put(spot, sigma, date)
                else:
                    self.open_option = self._sell_call(spot, sigma, date)

    def _close_cost(self, spot: float) -> float:
        """Return the cost to close the open short option at intrinsic value."""
        if self.open_option is None or self.open_option["direction"] != "short":
            return 0.0
        if self.open_option["type"] == "call":
            return (
                max(spot - self.open_option["strike"], 0.0)
                * SHARES_PER_CONTRACT
            )
        return max(self.open_option["strike"] - spot, 0.0) * SHARES_PER_CONTRACT
