# PRD: Monthly Stock Data Aggregation and Technical Indicators

## 1. Project Overview

**Project Name:** Stock Monthly Analytics Pipeline

**Type:** Data Engineering Intern Hiring Assignment

**Technology:** Python and Pandas

**Objective:** Build a modular data-processing pipeline that transforms daily historical stock market data into monthly stock summaries, calculates technical indicators, and exports the results into individual CSV files for each stock ticker.

The pipeline must be accurate, reproducible, easy to run locally, and sufficiently documented for another developer to understand and execute independently.

## 2. Problem Statement

Financial datasets commonly contain daily stock prices, whereas analysts may need monthly summaries to evaluate broader price trends.

The application will read a master CSV containing daily OHLCV data for 10 stocks, aggregate the daily records into monthly observations, calculate moving averages and other technical indicators, and generate one output CSV for each ticker.

## 3. Target Dataset

**Source:** https://github.com/sandeep-tt/tt-intern-dataset

Expected input filename: `output_file.csv`

Expected columns:

`date, volume, open, high, low, close, adjclose, ticker`

Expected tickers:

`AAPL, AMD, AMZN, AVGO, CSCO, MSFT, NFLX, PEP, TMUS, TSLA`

Expected historical period: January 2018 through December 2019.

Expected input size: approximately 5,030 daily records.

## 4. Functional Requirements

### FR-1: Input Loading and Validation

- Load the CSV using Pandas.
- Parse `date` into a valid datetime column.
- Confirm that all required input columns exist.
- Validate that dates and numeric price/volume fields are valid.
- Confirm that the dataset contains the expected 10 tickers.
- Detect duplicate combinations of `ticker` and `date`.
- Sort records chronologically within each ticker.
- Fail with a clear error message when invalid input prevents reliable processing.

The original dataset must never be modified.

### FR-2: Monthly Aggregation

Aggregate each ticker independently into calendar months.

| Field | Monthly aggregation |
|---|---|
| Open | First trading day's open |
| High | Maximum daily high |
| Low | Minimum daily low |
| Close | Last trading day's close |
| Volume | Sum of daily trading volume |
| Adjclose | Last trading day's adjusted close |

Each monthly record must represent one calendar month.

The output date will use the calendar month-end date, such as `2018-01-31`.

### FR-3: Simple Moving Averages

Calculate the following indicators using monthly closing prices:

- `sma_10`: rolling mean over 10 monthly closing prices.
- `sma_20`: rolling mean over 20 monthly closing prices.

Require a complete rolling window before producing a value.

### FR-4: Exponential Moving Averages

Calculate:

- `ema_10`: 10-month exponential moving average.
- `ema_20`: 20-month exponential moving average.

Use the smoothing factor:

`alpha = 2 / (period + 1)`

Initialize each EMA with the corresponding SMA and then update it recursively using the monthly close and previous EMA.

This makes the initialization convention explicit and reproducible.

### FR-5: Z-Score

Calculate a 20-month rolling Z-score from monthly closing prices:

`zscore_20 = (close - rolling_mean_20) / rolling_std_20`

Use the sample standard deviation, consistent with Pandas' default `rolling().std()` behavior.

If the rolling standard deviation is zero, return a missing value rather than infinity.

### FR-6: Donchian Channels

Calculate 20-month Donchian Channels:

- `donchian_high_20`: rolling maximum of monthly High.
- `donchian_low_20`: rolling minimum of monthly Low.

The explicit High/Low formula in the assignment takes precedence over interpreting every indicator as close-only.

### FR-7: Bollinger Bands

Calculate 20-month Bollinger Bands using monthly closing prices and two standard deviations:

- `bollinger_mid_20`: 20-month rolling mean.
- `bollinger_upper_20`: rolling mean plus two rolling standard deviations.
- `bollinger_lower_20`: rolling mean minus two rolling standard deviations.

Use a complete 20-month window.

### FR-8: Output Partitioning

Generate exactly 10 CSV files:

- `result_AAPL.csv`
- `result_AMD.csv`
- `result_AMZN.csv`
- `result_AVGO.csv`
- `result_CSCO.csv`
- `result_MSFT.csv`
- `result_NFLX.csv`
- `result_PEP.csv`
- `result_TMUS.csv`
- `result_TSLA.csv`

Each file must contain exactly 24 data rows, ordered chronologically from January 2018 through December 2019.

Each output must include the monthly data, ticker identifier, and all required indicator columns.

Initial missing indicator values must remain as empty CSV fields. Rows must not be dropped merely because a rolling indicator has insufficient historical data.

## 5. Expected Output Schema

Each CSV should contain these columns in a consistent order:

1. `date`
2. `ticker`
3. `open`
4. `high`
5. `low`
6. `close`
7. `volume`
8. `adjclose`
9. `sma_10`
10. `sma_20`
11. `ema_10`
12. `ema_20`
13. `zscore_20`
14. `donchian_high_20`
15. `donchian_low_20`
16. `bollinger_mid_20`
17. `bollinger_upper_20`
18. `bollinger_lower_20`

The output filenames, column names, and column order must remain consistent across all tickers.

## 6. Technical Architecture

Separate responsibilities into modular functions:

- `load_and_validate_input()`: loads the source file and validates its structure.
- `aggregate_monthly()`: performs ticker-wise monthly aggregation.
- `calculate_indicators()`: calculates all technical indicators from monthly records.
- `process_ticker()`: coordinates monthly aggregation and indicator calculation for a ticker.
- `write_output_files()`: writes the 10 CSV files.
- `validate_outputs()`: checks the output files, ticker coverage, row counts, and required columns.
- `main()`: connects the pipeline stages and provides a runnable entry point.

Use Pandas vectorized aggregation, rolling windows, and exponential weighting wherever appropriate. Do not use third-party technical analysis libraries.

## 7. Non-Functional Requirements

- **Correctness:** Monthly OHLC values must follow the specified first, last, maximum, and minimum rules.
- **Modularity:** Aggregation, indicator calculations, and file writing must be separate functions.
- **Reproducibility:** Identical input files must produce identical outputs.
- **Maintainability:** Use meaningful names, type hints where useful, and focused functions.
- **Error handling:** Report missing columns, invalid values, duplicate ticker-date records, and invalid output conditions clearly.
- **Portability:** The script must run locally using Python and Pandas.
- **Efficiency:** Avoid unnecessary Python row-by-row loops for numerical calculations.
- **Data integrity:** Preserve the source CSV and do not silently discard records.

## 8. Acceptance Criteria

The assignment is complete when:

1. The input is read and validated successfully.
2. All 10 expected tickers are processed.
3. The monthly OHLC aggregation rules are correct.
4. Every required indicator is calculated with the documented window and initialization assumptions.
5. Exactly 10 output files are generated with the required filenames.
6. Every output contains exactly 24 rows and 18 columns.
7. All files cover January 2018 through December 2019 in chronological order.
8. Early rolling-window results are represented as missing values without removing rows.
9. The pipeline can be executed with a documented command.
10. The README explains setup, execution, indicator formulas, assumptions, and validation results.

## 9. Explicit Assumptions

- All indicator periods refer to months, not days, because calculations are performed after monthly resampling.
- The Z-score and Bollinger Bands use a 20-month close-based rolling window.
- Donchian Channels use monthly highs and lows, following the formula given in the assignment.
- Bollinger Bands use the conventional multiplier `k = 2`.
- The regular `close`, rather than `adjclose`, is the basis for all closing-price indicators.
- Volume is summed across daily records; adjusted close is taken from the last trading day of each month.
- Rolling indicators require a complete window. Early missing values are expected and valid.
- EMA initialization uses the first complete SMA window before recursive EMA updates.
- The expected date coverage is January 2018 through December 2019, inclusive.
- The assignment's exact 24-row requirement takes precedence over dropping months or incomplete indicator rows.

## 10. Out of Scope

- Stock-price predictions or buy/sell recommendations.
- Web dashboards, APIs, or databases.
- Cloud deployment, scheduled ingestion, or distributed processing.
- Third-party technical analysis libraries.
- Downloading live market data.
- Financial interpretation of the indicators.

## 11. Documentation

The README must include the project objective, prerequisites, setup instructions, input schema, execution command, output structure, indicator formulas, key assumptions, and validation procedure.

## 12. Delivery

A public GitHub repository containing the source code, README, dependency declaration, and appropriate tests, with generated output files available for review. The original dataset must remain unchanged.