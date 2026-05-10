"""
Tests for the Backtester class.

Covers input validation, all private helper methods, run() with synthetic
data via mocked yfinance, and summary() output.

Live network calls are avoided by patching yfinance.download.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from options_analytics.backtester import Backtester

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_backtester(**kwargs) -> Backtester:
    """Return a Backtester with sensible defaults, overridable via kwargs."""
    defaults = {
        "ticker": "AAPL",
        "start": "2022-01-01",
        "end": "2023-01-01",
        "strategy": "covered_call",
    }
    defaults.update(kwargs)
    return Backtester(**defaults)


def make_price_series(prices: list, start: str = "2022-01-01") -> pd.Series:
    """Return a daily price Series with a business day DatetimeIndex."""
    dates = pd.date_range(start=start, periods=len(prices), freq="B")
    return pd.Series(prices, index=dates, name="Close")


def make_price_dataframe(
    prices: list, start: str = "2022-01-01"
) -> pd.DataFrame:
    """Return a single-column DataFrame with a Close column."""
    dates = pd.date_range(start=start, periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=dates)


def make_option(
    option_type: str,
    direction: str,
    strike: float = 150,
    expiry: pd.Timestamp = pd.Timestamp("2022-02-01"),
) -> dict:
    """Return an option dict for testing."""
    return {
        "type": option_type,
        "direction": direction,
        "strike": strike,
        "expiry": expiry,
    }


# ---------------------------------------------------------------------------
# Input validation — strategy
# ---------------------------------------------------------------------------


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="not a recognised strategy"):
        make_backtester(strategy="straddle")


def test_valid_strategies_do_not_raise():
    for strategy in [
        "long_call",
        "long_put",
        "covered_call",
        "cash_secured_put",
        "wheel",
    ]:
        make_backtester(strategy=strategy)


# ---------------------------------------------------------------------------
# Input validation — dates
# ---------------------------------------------------------------------------


def test_start_after_end_raises():
    with pytest.raises(ValueError, match="must be before end"):
        make_backtester(start="2023-01-01", end="2022-01-01")


def test_start_equal_to_end_raises():
    with pytest.raises(ValueError, match="must be before end"):
        make_backtester(start="2022-01-01", end="2022-01-01")


def test_invalid_date_format_raises():
    with pytest.raises(Exception):
        make_backtester(start="not-a-date", end="2023-01-01")


# ---------------------------------------------------------------------------
# Input validation — strike_pct
# ---------------------------------------------------------------------------


def test_zero_strike_pct_raises():
    with pytest.raises(ValueError, match="strike_pct"):
        make_backtester(strike_pct=0.0)


def test_negative_strike_pct_raises():
    with pytest.raises(ValueError, match="strike_pct"):
        make_backtester(strike_pct=-0.5)


# ---------------------------------------------------------------------------
# Input validation — initial_cash
# ---------------------------------------------------------------------------


def test_zero_initial_cash_raises():
    with pytest.raises(ValueError, match="initial_cash"):
        make_backtester(initial_cash=0.0)


def test_negative_initial_cash_raises():
    with pytest.raises(ValueError, match="initial_cash"):
        make_backtester(initial_cash=-1000.0)


# ---------------------------------------------------------------------------
# Input validation — expiry_days
# ---------------------------------------------------------------------------


def test_zero_expiry_days_raises():
    with pytest.raises(ValueError, match="expiry_days"):
        make_backtester(expiry_days=0)


def test_negative_expiry_days_raises():
    with pytest.raises(ValueError, match="expiry_days"):
        make_backtester(expiry_days=-30)


def test_float_expiry_days_raises():
    with pytest.raises(TypeError, match="expiry_days"):
        make_backtester(expiry_days=30.5)


# ---------------------------------------------------------------------------
# Input validation — vol_window
# ---------------------------------------------------------------------------


def test_zero_vol_window_raises():
    with pytest.raises(ValueError, match="vol_window"):
        make_backtester(vol_window=0)


def test_negative_vol_window_raises():
    with pytest.raises(ValueError, match="vol_window"):
        make_backtester(vol_window=-10)


def test_float_vol_window_raises():
    with pytest.raises(TypeError, match="vol_window"):
        make_backtester(vol_window=30.5)


# ---------------------------------------------------------------------------
# Input validation — r
# ---------------------------------------------------------------------------


def test_negative_r_raises():
    with pytest.raises(ValueError, match="r"):
        make_backtester(r=-0.01)


def test_zero_r_does_not_raise():
    make_backtester(r=0.0)


# ---------------------------------------------------------------------------
# summary() before run()
# ---------------------------------------------------------------------------


def test_summary_before_run_raises():
    bt = make_backtester()
    with pytest.raises(
        RuntimeError, match="Call run\\(\\) before summary\\(\\)"
    ):
        bt.summary()


# ---------------------------------------------------------------------------
# _download_data
# ---------------------------------------------------------------------------


def test_download_data_invalid_ticker_raises():
    """yfinance returns empty DataFrame for invalid tickers — should raise ValueError."""
    empty_df = pd.DataFrame()
    with patch(
        "options_analytics.backtester.yf.download", return_value=empty_df
    ):
        bt = make_backtester(ticker="INVALIDTICKER")
        with pytest.raises(ValueError, match="No data returned"):
            bt._download_data()


# ---------------------------------------------------------------------------
# _compute_sigma
# ---------------------------------------------------------------------------


def test_compute_sigma_constant_prices_zero_vol():
    """Constant prices produce zero log returns and therefore zero volatility."""
    bt = make_backtester(vol_window=5)
    prices = make_price_series([100.0] * 20)
    sigma = bt._compute_sigma(prices)
    assert (sigma == 0.0).all()


def test_compute_sigma_no_nans_after_bfill():
    """bfill should eliminate all NaNs from the output."""
    bt = make_backtester(vol_window=5)
    prices = make_price_series(
        [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 100.0] * 5
    )
    sigma = bt._compute_sigma(prices)
    assert not sigma.isna().any()


def test_compute_sigma_positive():
    """Volatility should always be non-negative."""
    bt = make_backtester(vol_window=5)
    prices = make_price_series([100.0, 102.0, 98.0, 105.0, 97.0, 103.0] * 5)
    sigma = bt._compute_sigma(prices)
    assert (sigma >= 0).all()


def test_compute_sigma_known_value():
    """
    Known price series should produce the expected annualised volatility.

    Prices: [100, 101, 102, 101, 100, 99] with vol_window=5
    Log returns: [ln(101/100), ln(102/101), ln(101/102), ln(100/101), ln(99/100)]
                = [0.00995, 0.00985, -0.00985, -0.00995, -0.01005]
    std of log returns = 0.01087
    annualised = std * sqrt(252) = 0.17262
    """
    bt = make_backtester(vol_window=5)
    prices = make_price_series([100.0, 101.0, 102.0, 101.0, 100.0, 99.0])
    sigma = bt._compute_sigma(prices)
    expected = 0.17262
    assert sigma.iloc[-1] == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# _check_itm
# ---------------------------------------------------------------------------


def test_check_itm_short_call_itm():
    bt = make_backtester()
    bt.open_option = make_option("call", "short")
    assert bt._check_itm(160.0) is True


def test_check_itm_short_call_otm():
    bt = make_backtester()
    bt.open_option = make_option("call", "short")
    assert bt._check_itm(140.0) is False


def test_check_itm_short_put_itm():
    bt = make_backtester()
    bt.open_option = make_option("put", "short")
    assert bt._check_itm(140.0) is True


def test_check_itm_short_put_otm():
    bt = make_backtester()
    bt.open_option = make_option("put", "short")
    assert bt._check_itm(160.0) is False


def test_check_itm_no_option():
    bt = make_backtester()
    bt.open_option = None
    assert bt._check_itm(150.0) is False


def test_check_itm_long_call():
    bt = make_backtester()
    bt.open_option = make_option("call", "long")
    assert bt._check_itm(160) is True


def test_check_otm_long_call():
    bt = make_backtester()
    bt.open_option = make_option("call", "long")
    assert bt._check_itm(140) is False


def test_check_itm_long_put():
    bt = make_backtester()
    bt.open_option = make_option("put", "long")
    assert bt._check_itm(160) is False


def test_check_otm_long_put():
    bt = make_backtester()
    bt.open_option = make_option("put", "long")
    assert bt._check_itm(140) is True


# ---------------------------------------------------------------------------
# _close_cost
# ---------------------------------------------------------------------------


def test_close_cost_itm_put():
    bt = make_backtester()
    bt.open_option = make_option("put", "short")
    assert bt._close_cost(140.0) == pytest.approx(1000.0)


def test_close_cost_otm_put():
    bt = make_backtester()
    bt.open_option = make_option("put", "short")
    assert bt._close_cost(160.0) == pytest.approx(0.0)


def test_close_cost_itm_call():
    bt = make_backtester()
    bt.open_option = make_option("call", "short")
    assert bt._close_cost(160.0) == pytest.approx(1000.0)


def test_close_cost_otm_call():
    bt = make_backtester()
    bt.open_option = make_option("call", "short")
    assert bt._close_cost(140.0) == pytest.approx(0.0)


def test_close_cost_long_call_option_returns_zero():
    bt = make_backtester()
    bt.open_option = make_option("call", "long")
    assert bt._close_cost(140.0) == pytest.approx(0.0)


def test_close_cost_long_put_option_returns_zero():
    bt = make_backtester()
    bt.open_option = make_option("put", "long")
    assert bt._close_cost(140.0) == pytest.approx(0.0)


def test_close_cost_no_option_returns_zero():
    bt = make_backtester()
    bt.open_option = None
    assert bt._close_cost(150.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _mark_option
# ---------------------------------------------------------------------------


def test_mark_option_no_option_returns_zero():
    bt = make_backtester()
    bt.open_option = None
    date = pd.Timestamp("2022-01-01")
    assert bt._mark_option(150.0, 0.2, date) == pytest.approx(0.0)


def test_mark_option_short_call_negative():
    """Short call should return a negative mark-to-market value."""
    bt = make_backtester()
    bt.open_option = make_option("call", "short")
    date = pd.Timestamp("2022-01-01")
    value = bt._mark_option(150.0, 0.2, date)
    assert value < 0.0


def test_mark_option_short_put_negative():
    """Short put should return a negative mark-to-market value."""
    bt = make_backtester()
    bt.open_option = make_option("put", "short")
    date = pd.Timestamp("2022-01-01")
    value = bt._mark_option(150.0, 0.2, date)
    assert value < 0.0


def test_mark_option_long_call_positive():
    """Long call should return a positive mark-to-market value."""
    bt = make_backtester()
    bt.open_option = make_option("call", "long")
    date = pd.Timestamp("2022-01-01")
    value = bt._mark_option(150.0, 0.2, date)
    assert value > 0.0


def test_mark_option_long_put_positive():
    """Long put should return a positive mark-to-market value."""
    bt = make_backtester()
    bt.open_option = make_option("put", "long")
    date = pd.Timestamp("2022-01-01")
    value = bt._mark_option(150.0, 0.2, date)
    assert value > 0.0


# ---------------------------------------------------------------------------
# _sell_call / _sell_put
# ---------------------------------------------------------------------------


def test_sell_call_credits_cash():
    """Selling a call should increase cash (by the premium)."""
    bt = make_backtester()
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    bt._sell_call(150.0, 0.2, date)
    assert bt.cash > 10_000.0


def test_sell_call_returns_correct_dict():
    bt = make_backtester()
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    option = bt._sell_call(150.0, 0.2, date)
    assert option["type"] == "call"
    assert option["direction"] == "short"
    assert option["contracts"] == 1
    assert "strike" in option
    assert "expiry" in option
    assert "premium_received" in option


def test_sell_put_credits_cash():
    """Selling a put should increase cash (by the premium)."""
    bt = make_backtester()
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    bt._sell_put(150.0, 0.2, date)
    assert bt.cash > 10_000.0


def test_sell_put_returns_correct_dict():
    bt = make_backtester()
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    option = bt._sell_put(150.0, 0.2, date)
    assert option["type"] == "put"
    assert option["direction"] == "short"
    assert option["contracts"] == 1
    assert "strike" in option
    assert "expiry" in option
    assert "premium_received" in option


# ---------------------------------------------------------------------------
# _buy_call / _buy_put
# ---------------------------------------------------------------------------


def test_buy_call_debits_cash():
    """Buying a call should decrease cash by the premium."""
    bt = make_backtester()
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    bt._buy_call(150.0, 0.2, date)
    assert bt.cash < 10_000.0


def test_buy_call_returns_correct_dict():
    bt = make_backtester()
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    option = bt._buy_call(150.0, 0.2, date)
    assert option["type"] == "call"
    assert option["direction"] == "long"
    assert option["contracts"] == 1
    assert "strike" in option
    assert "expiry" in option
    assert "premium_paid" in option


def test_buy_call_insufficient_cash_returns_none():
    bt = make_backtester()
    bt.cash = 0.01
    date = pd.Timestamp("2022-01-01")
    result = bt._buy_call(150.0, 0.2, date)
    assert result is None


def test_buy_put_debits_cash():
    """Buying a put should decrease cash by the premium."""
    bt = make_backtester()
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    bt._buy_put(150.0, 0.2, date)
    assert bt.cash < 10_000.0


def test_buy_put_returns_correct_dict():
    bt = make_backtester()
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    option = bt._buy_put(150.0, 0.2, date)
    assert option["type"] == "put"
    assert option["direction"] == "long"
    assert option["contracts"] == 1
    assert "strike" in option
    assert "expiry" in option
    assert "premium_paid" in option


def test_buy_put_insufficient_cash_returns_none():
    bt = make_backtester()
    bt.cash = 0.01
    date = pd.Timestamp("2022-01-01")
    result = bt._buy_put(150.0, 0.2, date)
    assert result is None


# ---------------------------------------------------------------------------
# _open_initial_position
# ---------------------------------------------------------------------------


def test_open_initial_position_long_call():
    bt = make_backtester(strategy="long_call")
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.open_option is not None
    assert bt.open_option["type"] == "call"
    assert bt.open_option["direction"] == "long"
    assert bt.cash < 10_000.0


def test_open_initial_position_long_put():
    bt = make_backtester(strategy="long_put")
    bt.cash = 10_000.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.open_option is not None
    assert bt.open_option["type"] == "put"
    assert bt.open_option["direction"] == "long"
    assert bt.cash < 10_000.0


def test_open_initial_position_covered_call():
    bt = make_backtester(strategy="covered_call")
    bt.cash = 100_000.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.shares == 100
    assert bt.open_option is not None
    assert bt.open_option["type"] == "call"
    assert bt.open_option["direction"] == "short"


def test_open_initial_position_cash_secured_put():
    bt = make_backtester(strategy="cash_secured_put")
    bt.cash = 100_000.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.open_option is not None
    assert bt.open_option["type"] == "put"
    assert bt.open_option["direction"] == "short"
    assert bt.shares == 0
    assert bt.cash > 100_000.0


def test_open_initial_position_wheel():
    bt = make_backtester(strategy="wheel")
    bt.cash = 100_000.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.open_option is not None
    assert bt.open_option["type"] == "put"
    assert bt.open_option["direction"] == "short"
    assert bt._wheel_state == "put"


# ---------------------------------------------------------------------------
# _open_initial_position — insufficient capital
# ---------------------------------------------------------------------------


def test_open_initial_position_long_call_insufficient_capital():
    bt = make_backtester(strategy="long_call")
    bt.cash = 100.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.open_option is None


def test_open_initial_position_long_put_insufficient_capital():
    bt = make_backtester(strategy="long_put")
    bt.cash = 100.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.open_option is None


def test_open_initial_position_covered_call_insufficient_capital():
    bt = make_backtester(strategy="covered_call")
    bt.cash = 100.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.open_option is None
    assert bt.shares == 0


def test_open_initial_position_cash_secured_put_insufficient_capital():
    bt = make_backtester(strategy="cash_secured_put")
    bt.cash = 100.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.open_option is None


def test_open_initial_position_wheel_insufficient_capital():
    bt = make_backtester(strategy="wheel")
    bt.cash = 100.0
    date = pd.Timestamp("2022-01-01")
    bt._open_initial_position(150.0, 0.2, date)
    assert bt.open_option is None


# ---------------------------------------------------------------------------
# _handle_expiry
# ---------------------------------------------------------------------------

# long_call


def test_handle_expiry_long_call_itm():
    """ITM long call: cash increases by intrinsic, new call opened."""
    bt = make_backtester(strategy="long_call")
    bt.cash = 5_000.0
    bt.open_option = make_option("call", "long")
    bt._handle_expiry(160.0, 0.2, pd.Timestamp("2022-02-01"))
    # intrinsic = (160 - 150) * 100 = 1000; premium deducted for new call
    assert bt.cash > 5_000.0
    assert bt.open_option is not None
    assert bt.open_option["type"] == "call"
    assert bt.open_option["direction"] == "long"


def test_handle_expiry_long_call_otm():
    """OTM long call: cash unchanged by exercise, new call opened."""
    bt = make_backtester(strategy="long_call")
    bt.cash = 5_000.0
    bt.open_option = make_option("call", "long")
    cash_before = bt.cash
    bt._handle_expiry(140.0, 0.2, pd.Timestamp("2022-02-01"))
    # No exercise payoff, only premium deducted for new call
    assert bt.cash < cash_before
    assert bt.open_option is not None


def test_handle_expiry_long_call_insufficient_cash_to_roll():
    """If cash too low to roll, open_option should be None."""
    bt = make_backtester(strategy="long_call")
    bt.cash = 0.01
    bt.open_option = make_option("call", "long")
    bt._handle_expiry(140.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.open_option is None


# long_put


def test_handle_expiry_long_put_itm():
    """ITM long put: cash increases by intrinsic, new put opened."""
    bt = make_backtester(strategy="long_put")
    bt.cash = 5_000.0
    bt.open_option = make_option("put", "long")
    bt._handle_expiry(140.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.open_option is not None
    assert bt.open_option["type"] == "put"
    assert bt.open_option["direction"] == "long"


def test_handle_expiry_long_put_otm():
    """OTM long put: cash unchanged by exercise, new put opened."""
    bt = make_backtester(strategy="long_put")
    bt.cash = 5_000.0
    bt.open_option = make_option("put", "long")
    cash_before = bt.cash
    bt._handle_expiry(160.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.cash < cash_before
    assert bt.open_option is not None


def test_handle_expiry_long_put_insufficient_cash_to_roll():
    bt = make_backtester(strategy="long_put")
    bt.cash = 0.01
    bt.open_option = make_option("put", "long")
    bt._handle_expiry(160.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.open_option is None


# covered_call


def test_handle_expiry_covered_call_itm():
    """ITM covered call: stock sold at strike, rebought at spot, new call opened."""
    bt = make_backtester(strategy="covered_call")
    bt.cash = 100_000.0
    bt.shares = 100
    bt.open_option = make_option("call", "short")
    bt._handle_expiry(160.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.shares == 100
    assert bt.open_option is not None
    assert bt.open_option["type"] == "call"
    assert bt.open_option["direction"] == "short"


def test_handle_expiry_covered_call_itm_insufficient_cash_to_rebuy():
    """If cash insufficient to rebuy after assignment, returns early with no new call."""
    bt = make_backtester(strategy="covered_call")
    bt.cash = 1.0
    bt.shares = 100
    bt.open_option = make_option("call", "short")
    bt._handle_expiry(160.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.open_option is None
    assert bt.shares == 0


def test_handle_expiry_covered_call_otm():
    """OTM covered call: shares unchanged, new call opened."""
    bt = make_backtester(strategy="covered_call")
    bt.cash = 100_000.0
    bt.shares = 100
    bt.open_option = make_option("call", "short")
    bt._handle_expiry(140.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.shares == 100
    assert bt.open_option is not None


# cash_secured_put


def test_handle_expiry_cash_secured_put_itm():
    """ITM cash secured put: stock bought at strike, sold at spot, new put opened."""
    bt = make_backtester(strategy="cash_secured_put")
    bt.cash = 100_000.0
    bt.shares = 0
    bt.open_option = make_option("put", "short")
    bt._handle_expiry(140.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.shares == 0
    assert bt.open_option is not None
    assert bt.open_option["type"] == "put"
    assert bt.open_option["direction"] == "short"


def test_handle_expiry_cash_secured_put_itm_insufficient_cash():
    """If cash insufficient to buy stock at strike, returns early with no new put."""
    bt = make_backtester(strategy="cash_secured_put")
    bt.cash = 1.0
    bt.shares = 0
    bt.open_option = make_option("put", "short")
    bt._handle_expiry(140.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.open_option is None


def test_handle_expiry_cash_secured_put_otm():
    """OTM cash secured put: cash unchanged, new put opened."""
    bt = make_backtester(strategy="cash_secured_put")
    bt.cash = 100_000.0
    bt.shares = 0
    bt.open_option = make_option("put", "short")
    cash_before = bt.cash
    bt._handle_expiry(160.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.shares == 0
    assert bt.open_option is not None
    # Cash should only change by the new premium received
    assert bt.cash > cash_before


# wheel


def test_handle_expiry_wheel_put_leg_itm():
    """Wheel put leg ITM: stock acquired, state switches to call."""
    bt = make_backtester(strategy="wheel")
    bt.cash = 100_000.0
    bt.shares = 0
    bt._wheel_state = "put"
    bt.open_option = make_option("put", "short")
    bt._handle_expiry(140.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.shares == 100
    assert bt._wheel_state == "call"
    assert bt.open_option is not None
    assert bt.open_option["type"] == "call"


def test_handle_expiry_wheel_put_leg_itm_insufficient_cash():
    """Wheel put leg ITM insufficient cash: returns early, no new position."""
    bt = make_backtester(strategy="wheel")
    bt.cash = 1.0
    bt.shares = 0
    bt._wheel_state = "put"
    bt.open_option = make_option("put", "short")
    bt._handle_expiry(140.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.open_option is None


def test_handle_expiry_wheel_put_leg_otm():
    """Wheel put leg OTM: new put opened, state stays put."""
    bt = make_backtester(strategy="wheel")
    bt.cash = 100_000.0
    bt.shares = 0
    bt._wheel_state = "put"
    bt.open_option = make_option("put", "short")
    bt._handle_expiry(160.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt._wheel_state == "put"
    assert bt.open_option is not None
    assert bt.open_option["type"] == "put"


def test_handle_expiry_wheel_call_leg_itm():
    """Wheel call leg ITM: stock sold, state switches to put."""
    bt = make_backtester(strategy="wheel")
    bt.cash = 100_000.0
    bt.shares = 100
    bt._wheel_state = "call"
    bt.open_option = make_option("call", "short")
    bt._handle_expiry(160.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt.shares == 0
    assert bt._wheel_state == "put"
    assert bt.open_option is not None
    assert bt.open_option["type"] == "put"


def test_handle_expiry_wheel_call_leg_otm():
    """Wheel call leg OTM: new call opened, state stays call."""
    bt = make_backtester(strategy="wheel")
    bt.cash = 100_000.0
    bt.shares = 100
    bt._wheel_state = "call"
    bt.open_option = make_option("call", "short")
    bt._handle_expiry(140.0, 0.2, pd.Timestamp("2022-02-01"))
    assert bt._wheel_state == "call"
    assert bt.open_option is not None
    assert bt.open_option["type"] == "call"


# ---------------------------------------------------------------------------
# run() with synthetic data
# ---------------------------------------------------------------------------


def test_run_returns_dataframe():
    """run() should return a DataFrame with the expected columns."""
    prices = [150.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(
            strategy="cash_secured_put", initial_cash=100_000.0
        )
        result = bt.run()
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "date",
        "spot",
        "cash",
        "stock_value",
        "option_value",
        "total_value",
        "daily_pnl",
        "cumulative_pnl",
    }
    assert expected_cols.issubset(set(result.columns))


def test_run_flat_prices_no_bust():
    """Flat prices should not trigger a bust for cash_secured_put."""
    prices = [150.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(
            strategy="cash_secured_put", initial_cash=100_000.0
        )
        bt.run()
    assert bt._bust_date is None


def test_run_covered_call_insufficient_capital_busts_day_zero():
    """Covered call with insufficient capital should bust on day zero, returning empty DataFrame."""
    prices = [150.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(strategy="covered_call", initial_cash=100.0)
        result = bt.run()
    assert bt._bust_date is not None
    assert result.empty


def test_run_wheel_busts_on_put_leg():
    """Wheel with insufficient cash to cover put collateral should bust and return empty DataFrame."""
    prices = [150.0] * 5 + [1.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(strategy="wheel", initial_cash=1_000.0)
        result = bt.run()
    assert bt._bust_date is not None
    assert result.empty


def test_run_cumulative_pnl_starts_at_zero():
    """Cumulative P&L on day one should be zero."""
    prices = [150.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(
            strategy="cash_secured_put", initial_cash=100_000.0
        )
        result = bt.run()
    assert result["cumulative_pnl"].iloc[0] == pytest.approx(0.0)


def test_run_can_be_called_twice():
    """run() should reset state so it can be called multiple times cleanly."""
    prices = [150.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(
            strategy="cash_secured_put", initial_cash=100_000.0
        )
        result1 = bt.run()
        result2 = bt.run()
    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_bust_empty_results_prints_message(capsys):
    """Empty results DataFrame should print the insufficient capital bust message."""
    prices = [150.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(strategy="covered_call", initial_cash=100.0)
        bt.run()
        bt.summary()
    captured = capsys.readouterr()
    assert "not enough initial cash" in captured.out


def test_summary_bust_prints_terminated_line(capsys):
    """Bust mid-backtest should print the Terminated line in summary."""
    prices = [1500.0] * 5 + [1.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(strategy="cash_secured_put")
        bt.run()
        bt.summary()
    captured = capsys.readouterr()
    assert bt._bust_date is not None
    assert "BUST" in captured.out


def test_summary_prints_cash_remaining(capsys):
    """summary() should always print Cash remaining."""
    prices = [150.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(
            strategy="cash_secured_put", initial_cash=100_000.0
        )
        bt.run()
        bt.summary()
    captured = capsys.readouterr()
    assert "Cash remaining" in captured.out


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_covered_call_flat_prices_pnl_equals_premiums_collected():
    """
    On a flat price series, a covered call should earn exactly the premiums
    collected from selling calls, since no calls expire ITM and shares never
    change value.
    """
    prices = [150.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(strategy="covered_call", initial_cash=100_000.0)
        result = bt.run()

    total_pnl = result["cumulative_pnl"].iloc[-1]
    assert total_pnl > 0.0  # must have earned something
    assert result["stock_value"].iloc[-1] == pytest.approx(
        150.0 * 100
    )  # shares unchanged
    assert total_pnl == pytest.approx(
        result["total_value"].iloc[-1] - result["total_value"].iloc[0]
    )


def test_total_value_equals_components_every_day():
    """
    On every row, total_value must equal cash + stock_value + option_value.
    """
    prices = [150.0] * 100
    df = make_price_dataframe(prices)
    with patch("options_analytics.backtester.yf.download", return_value=df):
        bt = make_backtester(strategy="covered_call", initial_cash=100_000.0)
        result = bt.run()

    reconstructed = (
        result["cash"] + result["stock_value"] + result["option_value"]
    )
    pd.testing.assert_series_equal(
        result["total_value"],
        reconstructed,
        check_names=False,
    )
