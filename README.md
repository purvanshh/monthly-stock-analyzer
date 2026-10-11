# Monthly Stock Analyzer

Python and Pandas pipeline that converts daily OHLCV stock data into monthly summaries, calculates technical indicators, and writes one CSV file per ticker.

Built for the Data Engineering Intern hiring assignment.

## Objective

Given two years of daily prices for 10 tickers (January 2018 through December 2019):

1. Aggregate daily rows into monthly OHLCV records.
2. Calculate SMA, EMA, Z-score, Donchian Channels, and Bollinger Bands on the monthly series.
3. Write exactly 10 partitioned result files (`24` rows × `18` columns each).

## Dataset

Source repository: [sandeep-tt/tt-intern-dataset](https://github.com/sandeep-tt/tt-intern-dataset)

Place the downloaded file at:

```text
data/output_file.csv
```

Input columns:

```text
date, volume, open, high, low, close, adjclose, ticker
```

Tickers:

```text
AAPL, AMD, AMZN, AVGO, CSCO, MSFT, NFLX, PEP, TMUS, TSLA
```

The pipeline never modifies the source CSV.

## Prerequisites

- Python 3.10+
- Pandas (see `requirements.txt`)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with:

```bash
.venv\Scripts\activate
```

## Run the pipeline

```bash
python stock_pipeline.py --input data/output_file.csv --output-dir output
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--input` | `data/output_file.csv` | Path to the daily source CSV |
| `--output-dir` | `output` | Directory for `result_*.csv` files |

### Pipeline stages

1. Load and validate daily input.
2. Aggregate to monthly OHLCV per ticker.
3. Validate monthly coverage (`10` tickers × `24` months = `240` rows).
4. Calculate all 10 indicator columns per ticker.
5. Validate the indicator DataFrame.
6. Stage-write all 10 CSVs, validate on disk, then promote into `--output-dir`.
7. Remove obsolete unexpected `result_*.csv` files only (unrelated files are kept).

On any critical failure the process exits with a nonzero status and does not print a success message.

## Expected outputs

```text
output/
├── result_AAPL.csv
├── result_AMD.csv
├── result_AMZN.csv
├── result_AVGO.csv
├── result_CSCO.csv
├── result_MSFT.csv
├── result_NFLX.csv
├── result_PEP.csv
├── result_TMUS.csv
└── result_TSLA.csv
```

### Output schema (exact column order)

```text
date
ticker
open
high
low
close
volume
adjclose
sma_10
sma_20
ema_10
ema_20
zscore_20
donchian_high_20
donchian_low_20
bollinger_mid_20
bollinger_upper_20
bollinger_lower_20
```

- Dates use `YYYY-MM-DD` calendar month-ends from `2018-01-31` through `2019-12-31`.
- Missing warm-up indicators are empty CSV fields (not `0`, `NaN`, or `None` text).
- Numeric values are preserved without cosmetic two-decimal rounding.

## Monthly OHLCV aggregation

| Field | Rule |
| --- | --- |
| open | Opening price on the first trading day of the month |
| high | Maximum daily high during the month |
| low | Minimum daily low during the month |
| close | Closing price on the last trading day of the month |
| volume | Sum of daily volume |
| adjclose | Adjusted close on the last trading day of the month |
| date | Calendar month-end date |

Rows are sorted by ticker and date before first/last snapshots so file order cannot distort results. Each ticker is aggregated independently.

## Technical indicators

All windows are measured in months, not days. Indicators are calculated independently for each ticker after monthly aggregation.

### Simple Moving Averages

```text
sma_10 = close.rolling(window=10, min_periods=10).mean()
sma_20 = close.rolling(window=20, min_periods=20).mean()
```

### Exponential Moving Averages (SMA-seeded)

Smoothing factor:

```text
alpha = 2 / (period + 1)
```

Recurrence:

```text
EMA_t = alpha * close_t + (1 - alpha) * EMA_previous
```

Initialization:

1. Leave EMA missing until the first complete window exists.
2. Set the first valid EMA equal to the SMA of that window.
3. Apply the recurrence for every later month.

This matches the assignment guidance that the first EMA uses the SMA as its starting value. A default Pandas `ewm()` start from the first close would disagree with this convention.

### Z-score

```text
zscore_20 = (close - rolling_mean_20) / rolling_std_20
```

Uses Pandas sample standard deviation (`ddof=1`). If the rolling standard deviation is zero, the Z-score is left missing instead of producing infinity.

### Donchian Channels

```text
donchian_high_20 = high.rolling(window=20, min_periods=20).max()
donchian_low_20  = low.rolling(window=20, min_periods=20).min()
```

These use monthly high and low, not close.

### Bollinger Bands

```text
bollinger_mid_20   = close.rolling(window=20, min_periods=20).mean()
bollinger_upper_20 = mid + 2 * rolling_std_20
bollinger_lower_20 = mid - 2 * rolling_std_20
```

### Expected warm-up missing values

With 24 monthly rows per ticker:

| Indicator | Valid rows | Missing rows |
| --- | ---: | ---: |
| SMA 10 / EMA 10 | 15 | 9 |
| SMA 20 / EMA 20 | 5 | 19 |
| Donchian high / low | 5 each | 19 each |
| Bollinger mid / upper / lower | 5 each | 19 each |
| Z-score 20 | up to 5 | at least 19 |

Warm-up rows are retained. They are not dropped because an indicator is still missing.

## Assumptions

- Indicator periods refer to months because calculations run after monthly aggregation.
- Regular `close` (not `adjclose`) is used for close-based indicators.
- Donchian Channels use monthly `high` and `low`.
- Bollinger Band multiplier is `k = 2`.
- Z-score and Bollinger Bands use a 20-month window and sample standard deviation.
- EMA initialization uses the SMA of the first complete window, then recursive updates.
- Output dates are calendar month-ends even when the last trading day falls earlier in the month.
- Volume is summed; adjusted close is taken from the last trading day of the month.

## Validation behavior

- Invalid input, monthly, or indicator states raise clear errors.
- Output writing validates the in-memory frame, writes a staged 10-file set, validates staged files, then promotes each file with `os.replace`.
- On-disk checks cover filenames, schema order, 24 rows, ticker isolation, month-end coverage, finite numerics, and warm-up emptiness.
- Staging prevents promoting an invalid complete set. A hard interrupt during promotion could still leave a mixed destination set; rerun the pipeline to repair.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Example successful run

Illustrative summary shape using the supplied dataset:

```text
=== Stock Monthly Analytics Pipeline Summary ===
Input file path              : data/output_file.csv
Output directory             : output
Daily input records          : 5030
Monthly records processed    : 240
Tickers processed            : 10
CSV files generated/validated: 10
Columns per output file      : 18
Output date range            : 2018-01-31 to 2019-12-31

SUCCESS: Full pipeline completed. All output CSV files were generated and validated.
```

## Project layout

```text
.
├── data/
│   └── output_file.csv
├── output/
│   ├── result_AAPL.csv
│   ├── result_AMD.csv
│   ├── result_AMZN.csv
│   ├── result_AVGO.csv
│   ├── result_CSCO.csv
│   ├── result_MSFT.csv
│   ├── result_NFLX.csv
│   ├── result_PEP.csv
│   ├── result_TMUS.csv
│   └── result_TSLA.csv
├── tests/
│   └── test_stock_pipeline.py
├── stock_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```
