"""Unit tests for Phase 1 input validation and Phase 2 monthly aggregation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Ensure project root is importable when running via unittest discover.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_pipeline import (
    EXPECTED_TICKERS,
    OUTPUT_COLUMNS,
    AssignmentValidationError,
    InputValidationError,
    MonthlyValidationError,
    OutputValidationError,
    aggregate_monthly,
    calculate_indicators,
    load_and_validate_input,
    main,
    validate_assignment_dataset,
    validate_indicators,
    validate_monthly_data,
    validate_output_files,
    write_output_files,
)

REQUIRED_COLUMNS = [
    "date",
    "volume",
    "open",
    "high",
    "low",
    "close",
    "adjclose",
    "ticker",
]


def _base_row(
    date: str,
    ticker: str = "AAPL",
    *,
    volume: float = 1_000_000,
    open_: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 105.0,
    adjclose: float = 105.0,
) -> dict:
    return {
        "date": date,
        "volume": volume,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adjclose": adjclose,
        "ticker": ticker,
    }


def _write_csv(rows: list[dict], path: Path) -> Path:
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(path, index=False)
    return path


def _all_weekdays_2018_2019() -> list[pd.Timestamp]:
    return list(pd.bdate_range("2018-01-02", "2019-12-31"))


def _dates_covering_24_months(count: int = 503) -> list[pd.Timestamp]:
    """Build ``count`` unique weekdays spanning Jan 2018–Dec 2019."""
    dates = _all_weekdays_2018_2019()
    if len(dates) < count:
        raise RuntimeError(
            f"Not enough weekdays in 2018-2019 to build {count} rows."
        )
    selected = dates[:count]
    month_keys = {(ts.year, ts.month) for ts in selected}
    if len(month_keys) != 24:
        raise RuntimeError("Selected dates do not cover all 24 calendar months.")
    return selected


def _assignment_frame(tickers: list[str]) -> pd.DataFrame:
    dates = _dates_covering_24_months(503)
    frames = []
    for ticker in tickers:
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "volume": 1_000_000,
                    "open": 100.0,
                    "high": 110.0,
                    "low": 90.0,
                    "close": 105.0,
                    "adjclose": 105.0,
                    "ticker": ticker,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class LoadAndValidateInputTests(unittest.TestCase):
    def test_valid_synthetic_dataset(self) -> None:
        rows = [
            _base_row("2018-01-02", "AAPL"),
            _base_row("2018-01-03", "AAPL", open_=101, high=111, low=91, close=106),
            _base_row("2018-01-02", "MSFT", open_=50, high=55, low=45, close=52),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_csv(rows, Path(tmp) / "valid.csv")
            df = load_and_validate_input(csv_path)

        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), REQUIRED_COLUMNS)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))
        self.assertEqual(sorted(df["ticker"].unique().tolist()), ["AAPL", "MSFT"])

    def test_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing_cols.csv"
            pd.DataFrame(
                [{"date": "2018-01-02", "close": 100, "ticker": "AAPL"}]
            ).to_csv(path, index=False)

            with self.assertRaises(InputValidationError) as ctx:
                load_and_validate_input(path)

        self.assertIn("Missing required columns", str(ctx.exception))

    def test_invalid_dates(self) -> None:
        rows = [_base_row("not-a-date", "AAPL")]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(rows, Path(tmp) / "bad_dates.csv")
            with self.assertRaises(InputValidationError) as ctx:
                load_and_validate_input(path)

        self.assertIn("Invalid dates", str(ctx.exception))

    def test_invalid_numeric_and_missing_values(self) -> None:
        rows = [_base_row("2018-01-02", "AAPL", close=float("nan"))]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(rows, Path(tmp) / "missing_close.csv")
            with self.assertRaises(InputValidationError) as ctx:
                load_and_validate_input(path)
        self.assertIn("Missing required values", str(ctx.exception))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "non_numeric.csv"
            path.write_text(
                "date,volume,open,high,low,close,adjclose,ticker\n"
                "2018-01-02,abc,100,110,90,105,105,AAPL\n",
                encoding="utf-8",
            )
            with self.assertRaises(InputValidationError) as ctx:
                load_and_validate_input(path)
        self.assertIn("Missing required values", str(ctx.exception))

    def test_duplicate_ticker_date(self) -> None:
        rows = [
            _base_row("2018-01-02", "AAPL"),
            _base_row("2018-01-02", "AAPL", open_=101, high=111, low=91, close=106),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(rows, Path(tmp) / "dupes.csv")
            with self.assertRaises(InputValidationError) as ctx:
                load_and_validate_input(path)

        self.assertIn("Duplicate (ticker, date)", str(ctx.exception))

    def test_invalid_ohlc_relationships(self) -> None:
        rows = [
            _base_row(
                "2018-01-02",
                "AAPL",
                open_=100,
                high=90,
                low=95,
                close=96,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(rows, Path(tmp) / "bad_ohlc.csv")
            with self.assertRaises(InputValidationError) as ctx:
                load_and_validate_input(path)

        self.assertIn("OHLC prices are internally inconsistent", str(ctx.exception))


class AssignmentValidationTests(unittest.TestCase):
    def test_missing_or_unexpected_tickers(self) -> None:
        df = _assignment_frame(EXPECTED_TICKERS[:-1])
        with self.assertRaises(AssignmentValidationError) as ctx:
            validate_assignment_dataset(df)
        self.assertIn("Missing expected tickers", str(ctx.exception))
        self.assertIn("TSLA", str(ctx.exception))

        df = _assignment_frame(EXPECTED_TICKERS + ["FAKE"])
        with self.assertRaises(AssignmentValidationError) as ctx:
            validate_assignment_dataset(df)
        self.assertIn("Unexpected tickers present", str(ctx.exception))
        self.assertIn("FAKE", str(ctx.exception))

    def test_missing_calendar_months(self) -> None:
        dates = _dates_covering_24_months(503)
        # AAPL omits June 2018 entirely. Weekdays alone are not enough to keep
        # 503 rows, so pad with weekend dates from other months in-range.
        no_june = [
            ts
            for ts in _all_weekdays_2018_2019()
            if not (ts.year == 2018 and ts.month == 6)
        ]
        used = set(no_june)
        filler = []
        for ts in pd.date_range("2018-01-01", "2019-12-31", freq="D"):
            if ts in used:
                continue
            if ts.year == 2018 and ts.month == 6:
                continue
            filler.append(ts)
            used.add(ts)
            if len(no_june) + len(filler) >= 503:
                break
        aapl_dates = sorted(no_june + filler)[:503]
        self.assertEqual(len(aapl_dates), 503)
        self.assertFalse(
            any(ts.year == 2018 and ts.month == 6 for ts in aapl_dates)
        )

        frames = []
        for ticker in EXPECTED_TICKERS:
            ticker_dates = aapl_dates if ticker == "AAPL" else dates
            frames.append(
                pd.DataFrame(
                    {
                        "date": ticker_dates,
                        "volume": 1_000_000,
                        "open": 100.0,
                        "high": 110.0,
                        "low": 90.0,
                        "close": 105.0,
                        "adjclose": 105.0,
                        "ticker": ticker,
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)

        with self.assertRaises(AssignmentValidationError) as ctx:
            validate_assignment_dataset(df)

        message = str(ctx.exception)
        self.assertIn("AAPL", message)
        self.assertIn("missing calendar months", message)
        self.assertIn("2018-06", message)


def _daily_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _full_monthly_frame(
    *,
    drop_month_for: tuple[str, str] | None = None,
    duplicate_row: bool = False,
) -> pd.DataFrame:
    """Build a synthetic valid 240-row monthly frame for validator tests."""
    month_ends = list(pd.date_range("2018-01-31", "2019-12-31", freq="ME"))
    rows = []
    for ticker in EXPECTED_TICKERS:
        for month_end in month_ends:
            if drop_month_for and drop_month_for == (
                ticker,
                month_end.strftime("%Y-%m"),
            ):
                continue
            rows.append(
                {
                    "date": month_end,
                    "ticker": ticker,
                    "open": 100.0,
                    "high": 110.0,
                    "low": 90.0,
                    "close": 105.0,
                    "volume": 1_000_000,
                    "adjclose": 105.0,
                }
            )
    if duplicate_row and rows:
        rows.append(dict(rows[0]))
    return pd.DataFrame(rows)


class MonthlyAggregationTests(unittest.TestCase):
    def test_ohlc_correctness(self) -> None:
        # Intentionally unsorted within the month.
        rows = [
            _base_row("2018-01-15", "AAPL", open_=120, high=125, low=115, close=118, volume=300, adjclose=118),
            _base_row("2018-01-03", "AAPL", open_=100, high=105, low=95, close=102, volume=100, adjclose=102),
            _base_row("2018-01-31", "AAPL", open_=130, high=140, low=128, close=135, volume=400, adjclose=134),
            _base_row("2018-01-10", "AAPL", open_=110, high=150, low=90, close=112, volume=200, adjclose=112),
        ]
        monthly = aggregate_monthly(_daily_frame(rows))
        self.assertEqual(len(monthly), 1)
        row = monthly.iloc[0]
        self.assertEqual(row["open"], 100.0)   # first trading day (Jan 3)
        self.assertEqual(row["high"], 150.0)   # max daily high
        self.assertEqual(row["low"], 90.0)     # min daily low
        self.assertEqual(row["close"], 135.0)  # last trading day (Jan 31)
        self.assertEqual(row["date"], pd.Timestamp("2018-01-31"))

    def test_unsorted_input(self) -> None:
        rows = [
            _base_row("2018-02-28", "MSFT", open_=60, high=62, low=58, close=61, volume=50, adjclose=61),
            _base_row("2018-02-01", "MSFT", open_=40, high=45, low=39, close=44, volume=10, adjclose=44),
            _base_row("2018-02-15", "MSFT", open_=50, high=70, low=35, close=55, volume=20, adjclose=55),
        ]
        shuffled = _daily_frame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)
        monthly = aggregate_monthly(shuffled)
        row = monthly.iloc[0]
        self.assertEqual(row["open"], 40.0)
        self.assertEqual(row["close"], 61.0)
        self.assertEqual(row["adjclose"], 61.0)
        self.assertEqual(row["date"], pd.Timestamp("2018-02-28"))

    def test_ticker_isolation(self) -> None:
        rows = [
            _base_row("2018-03-01", "AAPL", open_=10, high=20, low=5, close=15, volume=1, adjclose=15),
            _base_row("2018-03-29", "AAPL", open_=16, high=25, low=8, close=22, volume=2, adjclose=21),
            _base_row("2018-03-01", "MSFT", open_=100, high=110, low=90, close=105, volume=10, adjclose=105),
            _base_row("2018-03-29", "MSFT", open_=106, high=200, low=80, close=150, volume=20, adjclose=149),
        ]
        monthly = aggregate_monthly(_daily_frame(rows))
        self.assertEqual(len(monthly), 2)
        by_ticker = monthly.set_index("ticker")
        self.assertEqual(by_ticker.loc["AAPL", "open"], 10.0)
        self.assertEqual(by_ticker.loc["AAPL", "high"], 25.0)
        self.assertEqual(by_ticker.loc["AAPL", "low"], 5.0)
        self.assertEqual(by_ticker.loc["AAPL", "close"], 22.0)
        self.assertEqual(by_ticker.loc["MSFT", "open"], 100.0)
        self.assertEqual(by_ticker.loc["MSFT", "high"], 200.0)
        self.assertEqual(by_ticker.loc["MSFT", "low"], 80.0)
        self.assertEqual(by_ticker.loc["MSFT", "close"], 150.0)

    def test_volume_and_adjusted_close(self) -> None:
        rows = [
            _base_row("2018-04-02", "PEP", open_=50, high=55, low=49, close=54, volume=100, adjclose=53),
            _base_row("2018-04-16", "PEP", open_=54, high=60, low=50, close=58, volume=250, adjclose=57),
            _base_row("2018-04-30", "PEP", open_=58, high=65, low=57, close=64, volume=150, adjclose=63),
        ]
        monthly = aggregate_monthly(_daily_frame(rows))
        row = monthly.iloc[0]
        self.assertEqual(row["volume"], 500)
        self.assertEqual(row["adjclose"], 63.0)
        self.assertEqual(row["close"], 64.0)

    def test_month_end_dates(self) -> None:
        rows = [
            _base_row("2018-02-01", "TSLA", open_=1, high=2, low=1, close=2, volume=1, adjclose=2),
            _base_row("2018-02-28", "TSLA", open_=2, high=3, low=2, close=3, volume=1, adjclose=3),
            _base_row("2018-12-03", "TSLA", open_=4, high=5, low=4, close=5, volume=1, adjclose=5),
            _base_row("2018-12-31", "TSLA", open_=5, high=6, low=5, close=6, volume=1, adjclose=6),
        ]
        monthly = aggregate_monthly(_daily_frame(rows))
        dates = list(monthly["date"])
        self.assertEqual(dates, [pd.Timestamp("2018-02-28"), pd.Timestamp("2018-12-31")])


class MonthlyValidationTests(unittest.TestCase):
    def test_rejects_duplicate_ticker_month(self) -> None:
        df = _full_monthly_frame(duplicate_row=True)
        with self.assertRaises(MonthlyValidationError) as ctx:
            validate_monthly_data(df)
        message = str(ctx.exception)
        self.assertTrue(
            "Duplicate ticker-month" in message
            or "240 monthly records" in message
            or "exactly 24 monthly rows" in message,
            msg=message,
        )

    def test_rejects_missing_months_or_incorrect_row_count(self) -> None:
        df = _full_monthly_frame(drop_month_for=("AAPL", "2018-06"))
        with self.assertRaises(MonthlyValidationError) as ctx:
            validate_monthly_data(df)
        message = str(ctx.exception)
        self.assertTrue(
            "240 monthly records" in message
            or "exactly 24 monthly rows" in message
            or "must cover January 2018" in message,
            msg=message,
        )


def _monthly_from_closes(
    closes: list[float],
    ticker: str = "AAPL",
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Build N month-end rows from a close series (N >= 1)."""
    n = len(closes)
    dates = list(pd.date_range("2018-01-31", periods=n, freq="ME"))
    highs = highs if highs is not None else [c + 5 for c in closes]
    lows = lows if lows is not None else [c - 5 for c in closes]
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000_000] * n,
            "adjclose": closes,
        }
    )


class IndicatorCalculationTests(unittest.TestCase):
    def test_sma_correctness(self) -> None:
        closes = [float(i) for i in range(1, 25)]  # 1..24
        monthly = _monthly_from_closes(closes)
        result = calculate_indicators(monthly)
        # Manual SMA-10 at index 9 (10th row): mean(1..10) = 5.5
        self.assertTrue(pd.isna(result.loc[8, "sma_10"]))
        self.assertAlmostEqual(result.loc[9, "sma_10"], 5.5)
        # SMA-10 at index 10: mean(2..11) = 6.5
        self.assertAlmostEqual(result.loc[10, "sma_10"], 6.5)
        # Manual SMA-20 at index 19: mean(1..20) = 10.5
        self.assertTrue(pd.isna(result.loc[18, "sma_20"]))
        self.assertAlmostEqual(result.loc[19, "sma_20"], 10.5)
        # SMA-20 at index 20: mean(2..21) = 11.5
        self.assertAlmostEqual(result.loc[20, "sma_20"], 11.5)

    def test_ema_initialization_and_recursion(self) -> None:
        closes = [float(i) for i in range(1, 25)]
        monthly = _monthly_from_closes(closes)
        result = calculate_indicators(monthly)

        alpha_10 = 2 / (10 + 1)
        sma_10_seed = sum(closes[:10]) / 10
        self.assertTrue(pd.isna(result.loc[8, "ema_10"]))
        self.assertAlmostEqual(result.loc[9, "ema_10"], sma_10_seed)

        expected_next = alpha_10 * closes[10] + (1 - alpha_10) * sma_10_seed
        self.assertAlmostEqual(result.loc[10, "ema_10"], expected_next)

        alpha_20 = 2 / (20 + 1)
        sma_20_seed = sum(closes[:20]) / 20
        self.assertTrue(pd.isna(result.loc[18, "ema_20"]))
        self.assertAlmostEqual(result.loc[19, "ema_20"], sma_20_seed)
        expected_ema20_next = alpha_20 * closes[20] + (1 - alpha_20) * sma_20_seed
        self.assertAlmostEqual(result.loc[20, "ema_20"], expected_ema20_next)

    def test_zscore_correctness(self) -> None:
        closes = [float(i) for i in range(1, 25)]
        monthly = _monthly_from_closes(closes)
        result = calculate_indicators(monthly)

        window = pd.Series(closes[:20])
        mean_20 = window.mean()
        std_20 = window.std(ddof=1)  # sample std, Pandas default
        expected_z = (closes[19] - mean_20) / std_20
        self.assertTrue(pd.isna(result.loc[18, "zscore_20"]))
        self.assertAlmostEqual(result.loc[19, "zscore_20"], expected_z)

    def test_zero_standard_deviation_zscore(self) -> None:
        closes = [100.0] * 24
        monthly = _monthly_from_closes(closes)
        result = calculate_indicators(monthly)
        z = result["zscore_20"]
        self.assertTrue(z.isna().all())
        self.assertFalse(((z == float("inf")) | (z == float("-inf"))).any())

    def test_donchian_uses_high_low_not_close(self) -> None:
        closes = [50.0] * 24
        highs = [100.0 + i for i in range(24)]
        lows = [10.0 - (i * 0.1) for i in range(24)]
        monthly = _monthly_from_closes(closes, highs=highs, lows=lows)
        result = calculate_indicators(monthly)

        expected_high = max(highs[:20])
        expected_low = min(lows[:20])
        self.assertAlmostEqual(result.loc[19, "donchian_high_20"], expected_high)
        self.assertAlmostEqual(result.loc[19, "donchian_low_20"], expected_low)
        # Channels must not collapse to the constant close.
        self.assertNotAlmostEqual(result.loc[19, "donchian_high_20"], 50.0)
        self.assertNotAlmostEqual(result.loc[19, "donchian_low_20"], 50.0)

    def test_bollinger_bands(self) -> None:
        closes = [float(i) for i in range(1, 25)]
        monthly = _monthly_from_closes(closes)
        result = calculate_indicators(monthly)

        window = pd.Series(closes[:20])
        mid = window.mean()
        sd = window.std(ddof=1)
        self.assertAlmostEqual(result.loc[19, "bollinger_mid_20"], mid)
        self.assertAlmostEqual(result.loc[19, "bollinger_upper_20"], mid + 2 * sd)
        self.assertAlmostEqual(result.loc[19, "bollinger_lower_20"], mid - 2 * sd)
        self.assertGreaterEqual(
            result.loc[19, "bollinger_upper_20"], result.loc[19, "bollinger_mid_20"]
        )
        self.assertGreaterEqual(
            result.loc[19, "bollinger_mid_20"], result.loc[19, "bollinger_lower_20"]
        )

    def test_ticker_isolation(self) -> None:
        aapl = _monthly_from_closes([float(i) for i in range(1, 25)], ticker="AAPL")
        msft = _monthly_from_closes(
            [100.0 + i for i in range(24)], ticker="MSFT"
        )
        monthly = pd.concat([aapl, msft], ignore_index=True)
        result = calculate_indicators(monthly)

        aapl_out = result[result["ticker"] == "AAPL"].reset_index(drop=True)
        msft_out = result[result["ticker"] == "MSFT"].reset_index(drop=True)
        self.assertAlmostEqual(aapl_out.loc[9, "sma_10"], 5.5)
        self.assertAlmostEqual(msft_out.loc[9, "sma_10"], sum(100.0 + i for i in range(10)) / 10)
        self.assertNotAlmostEqual(aapl_out.loc[9, "sma_10"], msft_out.loc[9, "sma_10"])

    def test_warmup_periods(self) -> None:
        monthly = _monthly_from_closes([float(i) for i in range(1, 25)])
        result = calculate_indicators(monthly)

        self.assertTrue(result.loc[:8, "sma_10"].isna().all())
        self.assertTrue(result.loc[:8, "ema_10"].isna().all())
        self.assertTrue(result.loc[9, "sma_10"] == result.loc[9, "sma_10"])  # not NaN
        self.assertTrue(result.loc[9, "ema_10"] == result.loc[9, "ema_10"])

        cols_20 = [
            "sma_20",
            "ema_20",
            "zscore_20",
            "donchian_high_20",
            "donchian_low_20",
            "bollinger_mid_20",
            "bollinger_upper_20",
            "bollinger_lower_20",
        ]
        self.assertTrue(result.loc[:18, cols_20].isna().all(axis=None))
        self.assertTrue(result.loc[19, "sma_20"] == result.loc[19, "sma_20"])
        self.assertTrue(result.loc[19, "ema_20"] == result.loc[19, "ema_20"])

    def test_row_preservation_and_no_mutation(self) -> None:
        monthly = _full_monthly_frame()
        # Vary closes so indicators are non-trivial and std is non-zero.
        monthly = monthly.copy()
        monthly["close"] = [
            100.0 + (i % 24) + (0 if ticker == "AAPL" else 50)
            for i, ticker in enumerate(monthly["ticker"])
        ]
        monthly["open"] = monthly["close"]
        monthly["high"] = monthly["close"] + 5
        monthly["low"] = monthly["close"] - 5
        monthly["adjclose"] = monthly["close"]

        original = monthly.copy(deep=True)
        result = calculate_indicators(monthly)

        self.assertEqual(len(result), 240)
        self.assertTrue((result.groupby("ticker").size() == 24).all())
        pd.testing.assert_frame_equal(
            monthly.reset_index(drop=True),
            original.reset_index(drop=True),
        )
        validate_indicators(result, original_monthly_df=original)


def _indicator_frame_for_outputs() -> pd.DataFrame:
    """Build a full 240-row indicator frame suitable for output tests."""
    monthly = _full_monthly_frame().copy()
    monthly["close"] = [
        100.0 + (i % 24) + (hash(ticker) % 17)
        for i, ticker in enumerate(monthly["ticker"])
    ]
    monthly["open"] = monthly["close"]
    monthly["high"] = monthly["close"] + 8
    monthly["low"] = monthly["close"] - 8
    monthly["adjclose"] = monthly["close"]
    return calculate_indicators(monthly)


class OutputPartitioningTests(unittest.TestCase):
    def test_file_generation(self) -> None:
        indicator_df = _indicator_frame_for_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            written = write_output_files(indicator_df, tmp)
            names = sorted(path.name for path in written)
            expected = sorted(f"result_{ticker}.csv" for ticker in EXPECTED_TICKERS)
            self.assertEqual(names, expected)
            self.assertEqual(
                sorted(p.name for p in Path(tmp).glob("result_*.csv")),
                expected,
            )

    def test_correct_row_counts(self) -> None:
        indicator_df = _indicator_frame_for_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            write_output_files(indicator_df, tmp)
            for ticker in EXPECTED_TICKERS:
                df = pd.read_csv(Path(tmp) / f"result_{ticker}.csv")
                self.assertEqual(len(df), 24, msg=ticker)

    def test_correct_schema(self) -> None:
        indicator_df = _indicator_frame_for_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            write_output_files(indicator_df, tmp)
            for ticker in EXPECTED_TICKERS:
                df = pd.read_csv(Path(tmp) / f"result_{ticker}.csv")
                self.assertEqual(list(df.columns), OUTPUT_COLUMNS)
                self.assertEqual(len(df.columns), 18)

    def test_ticker_isolation(self) -> None:
        indicator_df = _indicator_frame_for_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            write_output_files(indicator_df, tmp)
            for ticker in EXPECTED_TICKERS:
                df = pd.read_csv(Path(tmp) / f"result_{ticker}.csv")
                self.assertTrue((df["ticker"] == ticker).all())

    def test_date_formatting(self) -> None:
        indicator_df = _indicator_frame_for_outputs()
        expected_dates = [
            ts.strftime("%Y-%m-%d")
            for ts in pd.date_range("2018-01-31", "2019-12-31", freq="ME")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            write_output_files(indicator_df, tmp)
            text = (Path(tmp) / "result_AAPL.csv").read_text(encoding="utf-8")
            lines = text.strip().splitlines()
            self.assertEqual(len(lines), 25)  # header + 24
            dates = [line.split(",")[0] for line in lines[1:]]
            self.assertEqual(dates, expected_dates)
            for date in dates:
                self.assertRegex(date, r"^\d{4}-\d{2}-\d{2}$")

    def test_missing_indicator_values_are_empty_fields(self) -> None:
        indicator_df = _indicator_frame_for_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            write_output_files(indicator_df, tmp)
            lines = (
                Path(tmp) / "result_AAPL.csv"
            ).read_text(encoding="utf-8").splitlines()
            header = lines[0].split(",")
            sma10_idx = header.index("sma_10")
            ema10_idx = header.index("ema_10")
            sma20_idx = header.index("sma_20")
            # First data row should have empty SMA/EMA fields.
            first = lines[1].split(",")
            self.assertEqual(first[sma10_idx], "")
            self.assertEqual(first[ema10_idx], "")
            self.assertEqual(first[sma20_idx], "")
            joined = "\n".join(lines[:10])
            self.assertNotIn("NaN", joined)
            self.assertNotIn("None", joined)

    def test_index_exclusion(self) -> None:
        indicator_df = _indicator_frame_for_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            write_output_files(indicator_df, tmp)
            df = pd.read_csv(Path(tmp) / "result_MSFT.csv")
            self.assertFalse(any(str(c).startswith("Unnamed") for c in df.columns))
            self.assertEqual(list(df.columns), OUTPUT_COLUMNS)

    def test_invalid_input_does_not_replace_outputs(self) -> None:
        good = _indicator_frame_for_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            write_output_files(good, tmp)
            before = {
                path.name: path.read_text(encoding="utf-8")
                for path in Path(tmp).glob("result_*.csv")
            }
            bad = good.drop(columns=["sma_10"])
            with self.assertRaises(OutputValidationError):
                write_output_files(bad, tmp)
            after = {
                path.name: path.read_text(encoding="utf-8")
                for path in Path(tmp).glob("result_*.csv")
            }
            self.assertEqual(before, after)

    def test_output_validation_detects_corruption(self) -> None:
        indicator_df = _indicator_frame_for_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            write_output_files(indicator_df, tmp)
            path = Path(tmp) / "result_AAPL.csv"

            # Missing file
            path.unlink()
            with self.assertRaises(OutputValidationError):
                validate_output_files(tmp)
            write_output_files(indicator_df, tmp)

            # Incorrect row count
            df = pd.read_csv(path)
            df.head(10).to_csv(path, index=False)
            with self.assertRaises(OutputValidationError):
                validate_output_files(tmp)
            write_output_files(indicator_df, tmp)

            # Wrong column order
            df = pd.read_csv(path)
            df = df[OUTPUT_COLUMNS[::-1]]
            df.to_csv(path, index=False)
            with self.assertRaises(OutputValidationError):
                validate_output_files(tmp)
            write_output_files(indicator_df, tmp)

            # Extra column
            df = pd.read_csv(path)
            df["extra"] = 1
            df.to_csv(path, index=False)
            with self.assertRaises(OutputValidationError):
                validate_output_files(tmp)
            write_output_files(indicator_df, tmp)

            # Incorrect ticker identity
            df = pd.read_csv(path)
            df["ticker"] = "MSFT"
            df.to_csv(path, index=False)
            with self.assertRaises(OutputValidationError):
                validate_output_files(tmp)

    def test_stale_generated_files_removed(self) -> None:
        indicator_df = _indicator_frame_for_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stale = out / "result_FAKE.csv"
            keep = out / "notes.txt"
            stale.write_text("stale\n", encoding="utf-8")
            keep.write_text("keep me\n", encoding="utf-8")
            write_output_files(indicator_df, out)
            self.assertFalse(stale.exists())
            self.assertTrue(keep.exists())
            self.assertEqual(keep.read_text(encoding="utf-8"), "keep me\n")

    def test_end_to_end_integration(self) -> None:
        input_csv = PROJECT_ROOT / "data" / "output_file.csv"
        if not input_csv.is_file():
            self.skipTest("Supplied dataset data/output_file.csv is not present")

        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main(
                ["--input", str(input_csv), "--output-dir", tmp]
            )
            self.assertEqual(exit_code, 0)
            loaded = validate_output_files(tmp)
            self.assertEqual(len(loaded), 10)
            for ticker, df in loaded.items():
                self.assertEqual(len(df), 24)
                self.assertEqual(list(df.columns), OUTPUT_COLUMNS)
                self.assertEqual(len(df.columns), 18)
                self.assertTrue((df["ticker"] == ticker).all())


class FinalAuditTests(unittest.TestCase):
    """Independent numerical checks that do not trust production EMA helpers."""

    def test_insufficient_observations_keep_rows_with_missing_indicators(self) -> None:
        monthly = _monthly_from_closes([10.0, 11.0, 12.0, 13.0, 14.0], ticker="AAPL")
        result = calculate_indicators(monthly)
        self.assertEqual(len(result), 5)
        self.assertTrue(result["sma_10"].isna().all())
        self.assertTrue(result["ema_10"].isna().all())
        self.assertTrue(result["sma_20"].isna().all())

    def test_deterministic_repeated_runs(self) -> None:
        input_csv = PROJECT_ROOT / "data" / "output_file.csv"
        if not input_csv.is_file():
            self.skipTest("Supplied dataset data/output_file.csv is not present")

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            self.assertEqual(main(["--input", str(input_csv), "--output-dir", tmp1]), 0)
            self.assertEqual(main(["--input", str(input_csv), "--output-dir", tmp2]), 0)
            for ticker in EXPECTED_TICKERS:
                left = (Path(tmp1) / f"result_{ticker}.csv").read_bytes()
                right = (Path(tmp2) / f"result_{ticker}.csv").read_bytes()
                self.assertEqual(left, right, msg=ticker)

    def test_independent_recompute_against_real_aapl_output(self) -> None:
        """Recompute SMA/EMA/Z/Donchian/Bollinger without production helpers."""
        input_csv = PROJECT_ROOT / "data" / "output_file.csv"
        if not input_csv.is_file():
            self.skipTest("Supplied dataset data/output_file.csv is not present")

        daily = load_and_validate_input(input_csv)
        monthly = aggregate_monthly(daily)
        aapl_m = monthly[monthly["ticker"] == "AAPL"].sort_values("date").reset_index(
            drop=True
        )
        close = aapl_m["close"]
        high = aapl_m["high"]
        low = aapl_m["low"]

        def independent_ema(series: pd.Series, period: int) -> list[float]:
            alpha = 2 / (period + 1)
            values = [float("nan")] * len(series)
            values[period - 1] = float(series.iloc[:period].mean())
            for i in range(period, len(series)):
                values[i] = alpha * float(series.iloc[i]) + (1 - alpha) * values[i - 1]
            return values

        exp_sma10 = close.rolling(10, min_periods=10).mean()
        exp_sma20 = close.rolling(20, min_periods=20).mean()
        exp_ema10 = independent_ema(close, 10)
        exp_ema20 = independent_ema(close, 20)
        mean20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std()
        exp_z = ((close - mean20) / std20).where(std20 > 0)
        exp_dh = high.rolling(20, min_periods=20).max()
        exp_dl = low.rolling(20, min_periods=20).min()
        exp_mid = mean20
        exp_up = mean20 + 2 * std20
        exp_lo = mean20 - 2 * std20

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--input", str(input_csv), "--output-dir", tmp]), 0)
            out = pd.read_csv(Path(tmp) / "result_AAPL.csv")
            out["date"] = pd.to_datetime(out["date"])

        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(out), 24)
        pd.testing.assert_series_equal(
            out["sma_10"], exp_sma10, check_names=False, atol=1e-9, rtol=0
        )
        pd.testing.assert_series_equal(
            out["sma_20"], exp_sma20, check_names=False, atol=1e-9, rtol=0
        )
        for i, expected in enumerate(exp_ema10):
            if pd.isna(expected):
                self.assertTrue(pd.isna(out.loc[i, "ema_10"]))
            else:
                self.assertAlmostEqual(out.loc[i, "ema_10"], expected, places=9)
        for i, expected in enumerate(exp_ema20):
            if pd.isna(expected):
                self.assertTrue(pd.isna(out.loc[i, "ema_20"]))
            else:
                self.assertAlmostEqual(out.loc[i, "ema_20"], expected, places=9)

        # EMA seed equals SMA at first valid index.
        self.assertAlmostEqual(out.loc[9, "ema_10"], out.loc[9, "sma_10"], places=9)
        self.assertAlmostEqual(out.loc[19, "ema_20"], out.loc[19, "sma_20"], places=9)

        pd.testing.assert_series_equal(
            out["zscore_20"], exp_z, check_names=False, atol=1e-9, rtol=0
        )
        pd.testing.assert_series_equal(
            out["donchian_high_20"], exp_dh, check_names=False, atol=1e-9, rtol=0
        )
        pd.testing.assert_series_equal(
            out["donchian_low_20"], exp_dl, check_names=False, atol=1e-9, rtol=0
        )
        pd.testing.assert_series_equal(
            out["bollinger_mid_20"], exp_mid, check_names=False, atol=1e-9, rtol=0
        )
        pd.testing.assert_series_equal(
            out["bollinger_upper_20"], exp_up, check_names=False, atol=1e-9, rtol=0
        )
        pd.testing.assert_series_equal(
            out["bollinger_lower_20"], exp_lo, check_names=False, atol=1e-9, rtol=0
        )

        # High/low channels must track high/low extremes, not close-only extremes.
        self.assertAlmostEqual(
            float(out.loc[19, "donchian_high_20"]),
            float(high.iloc[:20].max()),
            places=9,
        )
        self.assertAlmostEqual(
            float(out.loc[19, "donchian_low_20"]),
            float(low.iloc[:20].min()),
            places=9,
        )
        self.assertGreaterEqual(
            float(out.loc[19, "donchian_high_20"]),
            float(close.iloc[:20].max()),
        )
        self.assertLessEqual(
            float(out.loc[19, "donchian_low_20"]),
            float(close.iloc[:20].min()),
        )


if __name__ == "__main__":
    unittest.main()
