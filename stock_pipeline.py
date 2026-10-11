"""Stock Monthly Analytics Pipeline — validation, aggregation, indicators, outputs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

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

NUMERIC_COLUMNS = [
    "volume",
    "open",
    "high",
    "low",
    "close",
    "adjclose",
]

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adjclose",
]

MONTHLY_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjclose",
]

INDICATOR_COLUMNS = [
    "sma_10",
    "sma_20",
    "ema_10",
    "ema_20",
    "zscore_20",
    "donchian_high_20",
    "donchian_low_20",
    "bollinger_mid_20",
    "bollinger_upper_20",
    "bollinger_lower_20",
]

OUTPUT_COLUMNS = MONTHLY_COLUMNS + INDICATOR_COLUMNS

EXPECTED_TICKERS = [
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "CSCO",
    "MSFT",
    "NFLX",
    "PEP",
    "TMUS",
    "TSLA",
]

EXPECTED_PERIOD_START = pd.Timestamp("2018-01-01")
EXPECTED_PERIOD_END = pd.Timestamp("2019-12-31")
EXPECTED_MONTHS = 24
EXPECTED_MONTHLY_ROWS_PER_TICKER = 24
EXPECTED_TOTAL_MONTHLY_RECORDS = 240
EXPECTED_RECORDS_PER_TICKER = 503
EXPECTED_MONTH_ENDS = pd.date_range(
    start="2018-01-31", end="2019-12-31", freq="ME"
)

DEFAULT_INPUT_PATH = "data/output_file.csv"
DEFAULT_OUTPUT_DIR = "output"


class InputValidationError(ValueError):
    """Raised when the input dataset fails schema or content validation."""


class AssignmentValidationError(ValueError):
    """Raised when the dataset fails assignment-specific coverage checks."""


class MonthlyValidationError(ValueError):
    """Raised when monthly aggregation results fail validation."""


class IndicatorValidationError(ValueError):
    """Raised when technical indicator results fail validation."""


class OutputValidationError(ValueError):
    """Raised when output CSV generation or on-disk validation fails."""


def load_and_validate_input(input_path: str | Path) -> pd.DataFrame:
    """Load a CSV and validate schema, types, and OHLC consistency.

    Returns a validated DataFrame with lowercase column names and a
    datetime ``date`` column. Raises ``InputValidationError`` on failure.
    """
    path = Path(input_path)
    if not path.is_file():
        raise InputValidationError(f"Input file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas error surface varies
        raise InputValidationError(f"Failed to read CSV '{path}': {exc}") from exc

    if df.empty:
        raise InputValidationError("Input dataset is empty.")

    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise InputValidationError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = df["date"].isna()
    if invalid_dates.any():
        bad_rows = df.index[invalid_dates].tolist()[:10]
        raise InputValidationError(
            f"Invalid dates found in {int(invalid_dates.sum())} row(s). "
            f"Example row indices: {bad_rows}"
        )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    null_mask = df[REQUIRED_COLUMNS].isna().any(axis=1)
    if null_mask.any():
        null_counts = df[REQUIRED_COLUMNS].isna().sum()
        null_cols = null_counts[null_counts > 0].to_dict()
        raise InputValidationError(
            "Missing required values detected. "
            f"Null counts by column: {null_cols}"
        )

    for col in NUMERIC_COLUMNS:
        # Finite check without importing NumPy: NaN != NaN; abs(inf) == inf.
        non_finite = ~((df[col] == df[col]) & (df[col].abs() != float("inf")))
        if non_finite.any():
            raise InputValidationError(
                f"Column '{col}' contains invalid or non-finite numeric values "
                f"in {int(non_finite.sum())} row(s)."
            )

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    if (df["ticker"] == "").any():
        raise InputValidationError("Ticker values must be non-empty strings.")

    duplicate_mask = df.duplicated(subset=["ticker", "date"], keep=False)
    if duplicate_mask.any():
        dupes = (
            df.loc[duplicate_mask, ["ticker", "date"]]
            .drop_duplicates()
            .head(10)
        )
        examples = [
            f"{row.ticker}@{row.date.date()}" for row in dupes.itertuples()
        ]
        raise InputValidationError(
            f"Duplicate (ticker, date) combinations found "
            f"({int(duplicate_mask.sum())} rows involved). "
            f"Examples: {examples}"
        )

    for col in PRICE_COLUMNS:
        non_positive = df[col] <= 0
        if non_positive.any():
            raise InputValidationError(
                f"Column '{col}' must contain positive prices; "
                f"found {int(non_positive.sum())} non-positive value(s)."
            )

    inconsistent = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    if inconsistent.any():
        bad_idx = df.index[inconsistent].tolist()[:10]
        raise InputValidationError(
            "OHLC prices are internally inconsistent "
            f"in {int(inconsistent.sum())} row(s). "
            "Expected low <= open/close <= high and high >= low. "
            f"Example row indices: {bad_idx}"
        )

    df = df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    return df


def validate_assignment_dataset(df: pd.DataFrame) -> None:
    """Validate assignment-specific ticker coverage and historical period.

    Raises ``AssignmentValidationError`` when a check fails.
    """
    if df.empty:
        raise AssignmentValidationError("Assignment dataset is empty.")

    present = sorted(df["ticker"].unique().tolist())
    expected = sorted(EXPECTED_TICKERS)

    missing_tickers = sorted(set(expected) - set(present))
    unexpected_tickers = sorted(set(present) - set(expected))

    if missing_tickers:
        raise AssignmentValidationError(
            f"Missing expected tickers: {missing_tickers}"
        )
    if unexpected_tickers:
        raise AssignmentValidationError(
            f"Unexpected tickers present: {unexpected_tickers}"
        )

    min_date = df["date"].min()
    max_date = df["date"].max()
    if min_date < EXPECTED_PERIOD_START or max_date > EXPECTED_PERIOD_END:
        raise AssignmentValidationError(
            "Historical period must fall within January 2018 through December 2019. "
            f"Found range: {min_date.date()} to {max_date.date()}"
        )

    # Require coverage across all 24 calendar months in the expected window.
    expected_month_starts = pd.date_range(
        start="2018-01-01", end="2019-12-01", freq="MS"
    )
    expected_month_keys = {
        (ts.year, ts.month) for ts in expected_month_starts
    }

    for ticker in EXPECTED_TICKERS:
        ticker_df = df[df["ticker"] == ticker]
        record_count = len(ticker_df)
        if record_count != EXPECTED_RECORDS_PER_TICKER:
            raise AssignmentValidationError(
                f"Ticker '{ticker}' has {record_count} daily records; "
                f"expected {EXPECTED_RECORDS_PER_TICKER}."
            )

        month_keys = {
            (ts.year, ts.month) for ts in ticker_df["date"]
        }
        missing_months = sorted(expected_month_keys - month_keys)
        if missing_months:
            formatted = [f"{year}-{month:02d}" for year, month in missing_months]
            raise AssignmentValidationError(
                f"Ticker '{ticker}' is missing calendar months: {formatted}"
            )

        covered = len(month_keys)
        if covered != EXPECTED_MONTHS:
            raise AssignmentValidationError(
                f"Ticker '{ticker}' covers {covered} calendar months; "
                f"expected {EXPECTED_MONTHS}."
            )

        if ticker_df.duplicated(subset=["date"]).any():
            raise AssignmentValidationError(
                f"Ticker '{ticker}' has duplicate daily observations."
            )


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validated daily OHLCV rows into calendar-month summaries.

    Rules:
    - ``open`` / ``close`` / ``adjclose`` are snapshots from the first and last
      trading days after chronological sorting within each ticker.
    - ``high`` / ``low`` are the monthly extremes across all daily rows.
    - ``volume`` is the sum of daily volume.
    - ``date`` is the calendar month-end (for example, 2018-01-31).
    """
    if df.empty:
        raise MonthlyValidationError("Cannot aggregate an empty daily DataFrame.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise MonthlyValidationError(
            f"Daily DataFrame missing required columns for aggregation: {missing}"
        )

    working = df[REQUIRED_COLUMNS].copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise")
    working["ticker"] = working["ticker"].astype(str).str.strip().str.upper()

    # Chronological order is required so first/last capture true month open/close.
    working = working.sort_values(["ticker", "date"], kind="mergesort")
    working["month"] = working["date"].dt.to_period("M")

    monthly = (
        working.groupby(["ticker", "month"], sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            adjclose=("adjclose", "last"),
        )
        .reset_index()
    )

    # Period.to_timestamp(how="end") yields month-end; normalize to midnight.
    monthly["date"] = monthly["month"].dt.to_timestamp(how="end").dt.normalize()
    monthly = monthly.drop(columns=["month"])
    monthly = monthly[MONTHLY_COLUMNS]
    monthly = monthly.sort_values(["ticker", "date"], kind="mergesort").reset_index(
        drop=True
    )
    return monthly


def validate_monthly_data(monthly_df: pd.DataFrame) -> None:
    """Validate monthly aggregation results for the assignment dataset."""
    if monthly_df is None or monthly_df.empty:
        raise MonthlyValidationError("Monthly dataset is empty.")

    missing_cols = [col for col in MONTHLY_COLUMNS if col not in monthly_df.columns]
    if missing_cols:
        raise MonthlyValidationError(
            f"Monthly dataset missing required columns: {missing_cols}"
        )

    df = monthly_df[MONTHLY_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise MonthlyValidationError("Monthly dataset contains invalid dates.")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    present = sorted(df["ticker"].unique().tolist())
    expected = sorted(EXPECTED_TICKERS)

    if len(present) != len(EXPECTED_TICKERS):
        raise MonthlyValidationError(
            f"Expected exactly {len(EXPECTED_TICKERS)} distinct tickers; "
            f"found {len(present)}: {present}"
        )

    missing_tickers = sorted(set(expected) - set(present))
    unexpected_tickers = sorted(set(present) - set(expected))
    if missing_tickers:
        raise MonthlyValidationError(
            f"Missing expected tickers in monthly data: {missing_tickers}"
        )
    if unexpected_tickers:
        raise MonthlyValidationError(
            f"Unexpected tickers in monthly data: {unexpected_tickers}"
        )

    if len(df) != EXPECTED_TOTAL_MONTHLY_RECORDS:
        raise MonthlyValidationError(
            f"Expected {EXPECTED_TOTAL_MONTHLY_RECORDS} monthly records; "
            f"found {len(df)}."
        )

    rows_per_ticker = df.groupby("ticker").size()
    bad_counts = rows_per_ticker[
        rows_per_ticker != EXPECTED_MONTHLY_ROWS_PER_TICKER
    ]
    if not bad_counts.empty:
        details = {str(k): int(v) for k, v in bad_counts.to_dict().items()}
        raise MonthlyValidationError(
            "Each ticker must have exactly "
            f"{EXPECTED_MONTHLY_ROWS_PER_TICKER} monthly rows. "
            f"Violations: {details}"
        )

    month_keys = df["date"].dt.to_period("M")
    duplicate_mask = df.assign(month=month_keys).duplicated(
        subset=["ticker", "month"], keep=False
    )
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, ["ticker", "date"]]
            .drop_duplicates()
            .head(10)
        )
        formatted = [
            f"{row.ticker}@{row.date.date()}" for row in examples.itertuples()
        ]
        raise MonthlyValidationError(
            "Duplicate ticker-month combinations found. "
            f"Examples: {formatted}"
        )

    expected_month_ends = set(EXPECTED_MONTH_ENDS)
    for ticker in EXPECTED_TICKERS:
        ticker_df = df[df["ticker"] == ticker].sort_values("date", kind="mergesort")
        if not ticker_df["date"].is_monotonic_increasing:
            raise MonthlyValidationError(
                f"Dates for ticker '{ticker}' are not chronological."
            )

        ticker_dates = set(ticker_df["date"])
        missing_months = sorted(expected_month_ends - ticker_dates)
        extra_months = sorted(ticker_dates - expected_month_ends)
        if missing_months or extra_months:
            raise MonthlyValidationError(
                f"Ticker '{ticker}' must cover January 2018 through December 2019 "
                f"with calendar month-end dates. "
                f"Missing: {[d.date() for d in missing_months]}; "
                f"Unexpected: {[d.date() for d in extra_months]}"
            )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        nulls = int(df[col].isna().sum())
        if nulls:
            raise MonthlyValidationError(
                f"Monthly column '{col}' has {nulls} missing/invalid value(s)."
            )
        non_finite = ~((df[col] == df[col]) & (df[col].abs() != float("inf")))
        if non_finite.any():
            raise MonthlyValidationError(
                f"Monthly column '{col}' contains non-finite values."
            )

    for col in PRICE_COLUMNS:
        non_positive = df[col] <= 0
        if non_positive.any():
            raise MonthlyValidationError(
                f"Monthly column '{col}' must be positive; "
                f"found {int(non_positive.sum())} non-positive value(s)."
            )

    if (df["volume"] < 0).any():
        raise MonthlyValidationError("Monthly volume must be non-negative.")

    inconsistent = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    if inconsistent.any():
        bad_idx = df.index[inconsistent].tolist()[:10]
        raise MonthlyValidationError(
            "Monthly OHLC prices are internally inconsistent "
            f"in {int(inconsistent.sum())} row(s). "
            f"Example row indices: {bad_idx}"
        )


def _sma_seeded_ema(close: pd.Series, period: int) -> pd.Series:
    """EMA with SMA seed at the first complete window, then recursive updates.

    Uses ``ewm(span=period, adjust=False)`` on a series whose first valid value
    is replaced by the SMA seed so initialization matches the assignment.
    """
    sma = close.rolling(window=period, min_periods=period).mean()
    result = pd.Series(float("nan"), index=close.index, dtype="float64")
    if sma.isna().all():
        return result

    seed_label = sma.first_valid_index()
    seed_pos = close.index.get_loc(seed_label)
    if isinstance(seed_pos, slice):  # pragma: no cover - non-unique index
        seed_pos = seed_pos.start

    tail = close.iloc[seed_pos:].copy()
    tail.iloc[0] = float(sma.loc[seed_label])
    ema_tail = tail.ewm(span=period, adjust=False).mean()
    result.iloc[seed_pos:] = ema_tail
    return result


def _indicators_for_ticker(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for one ticker's monthly rows."""
    frame = ticker_df.sort_values("date", kind="mergesort").copy()
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]

    frame["sma_10"] = close.rolling(window=10, min_periods=10).mean()
    frame["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    frame["ema_10"] = _sma_seeded_ema(close, 10)
    frame["ema_20"] = _sma_seeded_ema(close, 20)

    rolling_mean_20 = close.rolling(window=20, min_periods=20).mean()
    rolling_std_20 = close.rolling(window=20, min_periods=20).std()
    zscore = (close - rolling_mean_20) / rolling_std_20
    # Zero (or near-zero) std must not produce +/- infinity.
    frame["zscore_20"] = zscore.where(rolling_std_20 > 0)

    frame["donchian_high_20"] = high.rolling(window=20, min_periods=20).max()
    frame["donchian_low_20"] = low.rolling(window=20, min_periods=20).min()

    frame["bollinger_mid_20"] = rolling_mean_20
    frame["bollinger_upper_20"] = rolling_mean_20 + (2 * rolling_std_20)
    frame["bollinger_lower_20"] = rolling_mean_20 - (2 * rolling_std_20)

    return frame


def calculate_indicators(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly technical indicators independently for each ticker.

    Operates on a copy of ``monthly_df``. Preserves all input rows and columns
    and appends the required indicator columns.
    """
    if monthly_df is None or monthly_df.empty:
        raise IndicatorValidationError(
            "Cannot calculate indicators on an empty monthly DataFrame."
        )

    missing = [col for col in MONTHLY_COLUMNS if col not in monthly_df.columns]
    if missing:
        raise IndicatorValidationError(
            f"Monthly DataFrame missing required columns: {missing}"
        )

    working = monthly_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise")
    working["ticker"] = working["ticker"].astype(str).str.strip().str.upper()
    working = working.sort_values(["ticker", "date"], kind="mergesort")

    parts = [
        _indicators_for_ticker(group)
        for _, group in working.groupby("ticker", sort=True)
    ]
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["ticker", "date"], kind="mergesort").reset_index(
        drop=True
    )
    return result[OUTPUT_COLUMNS]


def validate_indicators(
    indicator_df: pd.DataFrame,
    original_monthly_df: pd.DataFrame | None = None,
) -> None:
    """Validate technical indicator outputs for the assignment dataset."""
    if indicator_df is None or indicator_df.empty:
        raise IndicatorValidationError("Indicator dataset is empty.")

    missing_cols = [
        col for col in OUTPUT_COLUMNS if col not in indicator_df.columns
    ]
    if missing_cols:
        raise IndicatorValidationError(
            f"Indicator dataset missing required columns: {missing_cols}"
        )

    df = indicator_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise IndicatorValidationError("Indicator dataset contains invalid dates.")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    if len(df) != EXPECTED_TOTAL_MONTHLY_RECORDS:
        raise IndicatorValidationError(
            f"Expected {EXPECTED_TOTAL_MONTHLY_RECORDS} monthly records after "
            f"indicator calculation; found {len(df)}."
        )

    rows_per_ticker = df.groupby("ticker").size()
    bad_counts = rows_per_ticker[
        rows_per_ticker != EXPECTED_MONTHLY_ROWS_PER_TICKER
    ]
    if not bad_counts.empty:
        details = {str(k): int(v) for k, v in bad_counts.to_dict().items()}
        raise IndicatorValidationError(
            "Each ticker must retain exactly "
            f"{EXPECTED_MONTHLY_ROWS_PER_TICKER} rows. Violations: {details}"
        )

    month_keys = df["date"].dt.to_period("M")
    duplicate_mask = df.assign(month=month_keys).duplicated(
        subset=["ticker", "month"], keep=False
    )
    if duplicate_mask.any():
        raise IndicatorValidationError(
            "Duplicate ticker-month combinations found after indicator calculation."
        )

    for ticker, ticker_df in df.groupby("ticker", sort=True):
        ticker_df = ticker_df.sort_values("date", kind="mergesort").reset_index(
            drop=True
        )
        if not ticker_df["date"].is_monotonic_increasing:
            raise IndicatorValidationError(
                f"Dates for ticker '{ticker}' are not chronological."
            )
        if len(ticker_df) < 20:
            raise IndicatorValidationError(
                f"Ticker '{ticker}' has fewer than 20 monthly rows."
            )

        # First 9 rows: SMA/EMA 10 must be missing.
        early_10 = ticker_df.loc[:8, ["sma_10", "ema_10"]]
        if not early_10.isna().all(axis=None):
            raise IndicatorValidationError(
                f"Ticker '{ticker}' has SMA/EMA 10 values during the 9-month "
                "warm-up period."
            )

        # First 19 rows: all 20-period indicators must be missing.
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
        early_20 = ticker_df.loc[:18, cols_20]
        if not early_20.isna().all(axis=None):
            raise IndicatorValidationError(
                f"Ticker '{ticker}' has 20-period indicator values during the "
                "19-month warm-up period."
            )

        # After warm-up, SMA/EMA should be present (finite).
        if ticker_df.loc[9:, "sma_10"].isna().any() or ticker_df.loc[
            9:, "ema_10"
        ].isna().any():
            raise IndicatorValidationError(
                f"Ticker '{ticker}' has missing SMA/EMA 10 after warm-up."
            )
        if ticker_df.loc[19:, "sma_20"].isna().any() or ticker_df.loc[
            19:, "ema_20"
        ].isna().any():
            raise IndicatorValidationError(
                f"Ticker '{ticker}' has missing SMA/EMA 20 after warm-up."
            )

    for col in INDICATOR_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        # Finite among non-null: reject +/- inf.
        non_null = series.notna()
        if non_null.any():
            values = series[non_null]
            non_finite = ~((values == values) & (values.abs() != float("inf")))
            if non_finite.any():
                raise IndicatorValidationError(
                    f"Indicator '{col}' contains non-finite values."
                )

    valid_bb = (
        df["bollinger_mid_20"].notna()
        & df["bollinger_upper_20"].notna()
        & df["bollinger_lower_20"].notna()
    )
    if valid_bb.any():
        ordered = (
            (df.loc[valid_bb, "bollinger_upper_20"] >= df.loc[valid_bb, "bollinger_mid_20"])
            & (
                df.loc[valid_bb, "bollinger_mid_20"]
                >= df.loc[valid_bb, "bollinger_lower_20"]
            )
        )
        if not bool(ordered.all()):
            raise IndicatorValidationError(
                "Bollinger Bands are not ordered as upper >= mid >= lower."
            )

    valid_dc = df["donchian_high_20"].notna() & df["donchian_low_20"].notna()
    if valid_dc.any():
        ordered_dc = (
            df.loc[valid_dc, "donchian_high_20"] >= df.loc[valid_dc, "donchian_low_20"]
        )
        if not bool(ordered_dc.all()):
            raise IndicatorValidationError(
                "Donchian high is less than Donchian low for one or more rows."
            )

    if original_monthly_df is not None:
        before = original_monthly_df[MONTHLY_COLUMNS].copy().reset_index(drop=True)
        after = df[MONTHLY_COLUMNS].copy().reset_index(drop=True)
        before["date"] = pd.to_datetime(before["date"])
        after["date"] = pd.to_datetime(after["date"])
        before["ticker"] = before["ticker"].astype(str).str.upper()
        after["ticker"] = after["ticker"].astype(str).str.upper()
        before = before.sort_values(["ticker", "date"], kind="mergesort").reset_index(
            drop=True
        )
        after = after.sort_values(["ticker", "date"], kind="mergesort").reset_index(
            drop=True
        )
        if not before.equals(after):
            # Allow float equality via compare on aligned frames.
            compared = before.compare(after, keep_shape=False)
            if not compared.empty:
                raise IndicatorValidationError(
                    "Original monthly OHLCV contents changed after indicator "
                    f"calculation. Diff preview:\n{compared.head(10)}"
                )


def _result_filename(ticker: str) -> str:
    return f"result_{ticker}.csv"


def _prepare_indicator_df_for_write(indicator_df: pd.DataFrame) -> pd.DataFrame:
    """Validate the in-memory indicator frame before any CSV is written."""
    if indicator_df is None or indicator_df.empty:
        raise OutputValidationError("Cannot write outputs from an empty DataFrame.")

    missing_cols = [col for col in OUTPUT_COLUMNS if col not in indicator_df.columns]
    if missing_cols:
        raise OutputValidationError(
            f"Cannot write outputs; missing required columns: {missing_cols}"
        )

    df = indicator_df[OUTPUT_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise OutputValidationError("Indicator DataFrame contains invalid dates.")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    present = sorted(df["ticker"].unique().tolist())
    expected = sorted(EXPECTED_TICKERS)
    missing_tickers = sorted(set(expected) - set(present))
    unexpected_tickers = sorted(set(present) - set(expected))
    if missing_tickers:
        raise OutputValidationError(
            f"Cannot write outputs; missing tickers: {missing_tickers}"
        )
    if unexpected_tickers:
        raise OutputValidationError(
            f"Cannot write outputs; unexpected tickers: {unexpected_tickers}"
        )

    rows_per_ticker = df.groupby("ticker").size()
    bad = rows_per_ticker[rows_per_ticker != EXPECTED_MONTHLY_ROWS_PER_TICKER]
    if not bad.empty:
        details = {str(k): int(v) for k, v in bad.to_dict().items()}
        raise OutputValidationError(
            "Cannot write outputs; each ticker must have exactly "
            f"{EXPECTED_MONTHLY_ROWS_PER_TICKER} rows. Violations: {details}"
        )

    if len(df) != EXPECTED_TOTAL_MONTHLY_RECORDS:
        raise OutputValidationError(
            f"Cannot write outputs; expected {EXPECTED_TOTAL_MONTHLY_RECORDS} "
            f"rows, found {len(df)}."
        )

    return df


def _write_ticker_csv(ticker_df: pd.DataFrame, path: Path) -> None:
    """Write one ticker CSV with YYYY-MM-DD dates and empty missing fields."""
    frame = ticker_df[OUTPUT_COLUMNS].sort_values("date", kind="mergesort").copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame.to_csv(path, index=False, na_rep="")


def validate_output_files(output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Validate on-disk ``result_{SYMBOL}.csv`` files in ``output_dir``."""
    output_path = Path(output_dir)
    if not output_path.is_dir():
        raise OutputValidationError(f"Output directory does not exist: {output_path}")

    expected_names = {_result_filename(ticker) for ticker in EXPECTED_TICKERS}
    found_paths = sorted(output_path.glob("result_*.csv"))
    found_names = {path.name for path in found_paths}

    missing = sorted(expected_names - found_names)
    unexpected = sorted(found_names - expected_names)
    if missing or unexpected:
        raise OutputValidationError(
            "Output directory does not contain exactly the 10 expected result "
            f"files. Missing: {missing}; Unexpected: {unexpected}"
        )

    loaded: dict[str, pd.DataFrame] = {}
    expected_month_ends = list(EXPECTED_MONTH_ENDS)

    for ticker in EXPECTED_TICKERS:
        path = output_path / _result_filename(ticker)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise OutputValidationError(
                f"Failed to read output file '{path}': {exc}"
            ) from exc

        if df.columns.duplicated().any():
            raise OutputValidationError(
                f"{path.name} has duplicate column names: {list(df.columns)}"
            )

        unnamed = [col for col in df.columns if str(col).startswith("Unnamed")]
        if unnamed:
            raise OutputValidationError(
                f"{path.name} contains an unwanted index/unnamed column: {unnamed}"
            )

        if list(df.columns) != OUTPUT_COLUMNS:
            raise OutputValidationError(
                f"{path.name} schema mismatch. "
                f"Expected {OUTPUT_COLUMNS}, found {list(df.columns)}"
            )

        if len(OUTPUT_COLUMNS) != 18 or len(df.columns) != 18:
            raise OutputValidationError(
                f"{path.name} must have exactly 18 columns; found {len(df.columns)}."
            )

        if len(df) != EXPECTED_MONTHLY_ROWS_PER_TICKER:
            raise OutputValidationError(
                f"{path.name} must contain exactly "
                f"{EXPECTED_MONTHLY_ROWS_PER_TICKER} data rows; found {len(df)}."
            )

        df = df.copy()
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        if not (df["ticker"] == ticker).all():
            bad = sorted(set(df["ticker"]) - {ticker})
            raise OutputValidationError(
                f"{path.name} contains rows for unexpected ticker(s): {bad}"
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise OutputValidationError(f"{path.name} contains invalid dates.")

        if not df["date"].is_monotonic_increasing:
            raise OutputValidationError(
                f"{path.name} dates are not in chronological order."
            )

        if df.duplicated(subset=["date"]).any():
            raise OutputValidationError(
                f"{path.name} has duplicate ticker-month/date values."
            )

        actual_dates = list(df["date"])
        if actual_dates != expected_month_ends:
            raise OutputValidationError(
                f"{path.name} must cover month-end dates from 2018-01-31 through "
                f"2019-12-31. First={actual_dates[0].date() if actual_dates else None}, "
                f"last={actual_dates[-1].date() if actual_dates else None}."
            )

        for col in MONTHLY_COLUMNS:
            if col in {"date", "ticker"}:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.isna().any():
                raise OutputValidationError(
                    f"{path.name} has missing/invalid market-data values in '{col}'."
                )
            non_finite = ~((series == series) & (series.abs() != float("inf")))
            if non_finite.any():
                raise OutputValidationError(
                    f"{path.name} has non-finite market-data values in '{col}'."
                )

        for col in INDICATOR_COLUMNS:
            series = pd.to_numeric(df[col], errors="coerce")
            non_null = series.notna()
            if non_null.any():
                values = series[non_null]
                non_finite = ~((values == values) & (values.abs() != float("inf")))
                if non_finite.any():
                    raise OutputValidationError(
                        f"{path.name} has non-finite indicator values in '{col}'."
                    )

        # Expected warm-up missing values must remain missing.
        if not df.loc[:8, ["sma_10", "ema_10"]].isna().all(axis=None):
            raise OutputValidationError(
                f"{path.name} has SMA/EMA 10 values during the warm-up period."
            )
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
        if not df.loc[:18, cols_20].isna().all(axis=None):
            raise OutputValidationError(
                f"{path.name} has 20-period indicator values during the warm-up period."
            )

        loaded[ticker] = df

    return loaded


def write_output_files(
    indicator_df: pd.DataFrame, output_dir: str | Path
) -> list[Path]:
    """Partition indicators by ticker and write ``result_{SYMBOL}.csv`` files.

    Writes the complete 10-file set into a staging directory under
    ``output_dir``, validates those staged files on disk, then promotes each
    file into place with ``os.replace`` (atomic per file on the same
    filesystem). Destination ``result_*.csv`` files are not replaced if staging
    validation fails. Obsolete unexpected ``result_*.csv`` files are removed
    afterward; unrelated non-matching files are left untouched.

    Note: true multi-file transactional replace is not available on typical
    filesystems. Staging prevents promoting an invalid set, but an interrupted
    promote mid-loop could leave a mixed generation of destination files.
    """
    prepared = _prepare_indicator_df_for_write(indicator_df)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stage_dir = Path(
        tempfile.mkdtemp(prefix=".stock_pipeline_stage_", dir=output_path)
    )
    written: list[Path] = []
    try:
        for ticker in EXPECTED_TICKERS:
            ticker_df = prepared[prepared["ticker"] == ticker]
            _write_ticker_csv(ticker_df, stage_dir / _result_filename(ticker))

        # Validate the complete staged set before replacing destination files.
        validate_output_files(stage_dir)

        for ticker in EXPECTED_TICKERS:
            src = stage_dir / _result_filename(ticker)
            dest = output_path / _result_filename(ticker)
            promoting = output_path / f".{_result_filename(ticker)}.promoting"
            shutil.copy2(src, promoting)
            os.replace(promoting, dest)
            written.append(dest)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        for leftover in output_path.glob(".result_*.csv.promoting"):
            try:
                leftover.unlink()
            except OSError:
                pass

    # Remove only obsolete result_*.csv files for unexpected tickers.
    for path in output_path.glob("result_*.csv"):
        symbol = path.name.removeprefix("result_").removesuffix(".csv")
        if symbol not in EXPECTED_TICKERS:
            path.unlink()

    validate_output_files(output_path)
    return written


def _months_covered_per_ticker(df: pd.DataFrame) -> dict[str, int]:
    months = (
        df.assign(year_month=df["date"].dt.to_period("M"))
        .groupby("ticker")["year_month"]
        .nunique()
        .to_dict()
    )
    return {str(ticker): int(count) for ticker, count in months.items()}


def _indicator_valid_missing_counts(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for col in INDICATOR_COLUMNS:
        valid = int(df[col].notna().sum())
        missing = int(df[col].isna().sum())
        counts[col] = (valid, missing)
    return counts


def print_summary(
    input_path: Path,
    output_dir: Path,
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    written_files: list[Path],
) -> None:
    """Print end-to-end pipeline summary from actual execution results."""
    monthly_rows_per_ticker = monthly_df.groupby("ticker").size().to_dict()
    valid_missing = _indicator_valid_missing_counts(indicator_df)
    date_min = indicator_df["date"].min()
    date_max = indicator_df["date"].max()

    print("=== Stock Monthly Analytics Pipeline Summary ===")
    print(f"Input file path              : {input_path}")
    print(f"Output directory             : {output_dir}")
    print(f"Daily input records          : {len(daily_df)}")
    print(f"Monthly records processed    : {len(monthly_df)}")
    print(f"Tickers processed            : {indicator_df['ticker'].nunique()}")
    print(f"CSV files generated/validated: {len(written_files)}")
    print("Row count per ticker         :")
    for ticker in sorted(monthly_rows_per_ticker):
        print(f"  - {ticker}: {monthly_rows_per_ticker[ticker]}")
    print(f"Columns per output file      : {len(OUTPUT_COLUMNS)}")
    print(f"Output date range            : {pd.Timestamp(date_min).date()} to {pd.Timestamp(date_max).date()}")
    print(f"Indicator columns ({len(INDICATOR_COLUMNS)}): {INDICATOR_COLUMNS}")
    print("Valid / missing indicator counts:")
    for col in INDICATOR_COLUMNS:
        valid, missing = valid_missing[col]
        print(f"  - {col}: valid={valid}, missing={missing}")
    print("Generated files:")
    for path in written_files:
        print(f"  - {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stock Monthly Analytics Pipeline: validate daily input, "
            "aggregate monthly OHLCV, calculate indicators, and write "
            "per-ticker result CSV files."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the input CSV (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for result_*.csv files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    try:
        daily_df = load_and_validate_input(input_path)
        validate_assignment_dataset(daily_df)
        monthly_df = aggregate_monthly(daily_df)
        validate_monthly_data(monthly_df)
        original_monthly = monthly_df.copy()
        indicator_df = calculate_indicators(monthly_df)
        validate_indicators(indicator_df, original_monthly_df=original_monthly)
        if not monthly_df.equals(original_monthly):
            raise IndicatorValidationError(
                "calculate_indicators mutated the input monthly DataFrame."
            )
        written_files = write_output_files(indicator_df, output_dir)
        validate_output_files(output_dir)
    except (
        InputValidationError,
        AssignmentValidationError,
        MonthlyValidationError,
        IndicatorValidationError,
        OutputValidationError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: Unexpected failure: {exc}", file=sys.stderr)
        return 1

    print_summary(
        input_path,
        output_dir,
        daily_df,
        monthly_df,
        indicator_df,
        written_files,
    )
    print(
        "\nSUCCESS: Full pipeline completed. All output CSV files were generated "
        "and validated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
