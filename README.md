# Stock Monthly Analytics Pipeline

Python + Pandas data-engineering pipeline that transforms daily OHLCV stock data into monthly summaries, calculates technical indicators, and writes one validated CSV per ticker.

This repository implements the Data Engineering Intern hiring assignment described in the project PRD.

## Objective

Read a master CSV of daily prices for 10 stocks (January 2018–December 2019), aggregate each ticker to calendar-month OHLCV records, compute SMA/EMA/Z-score/Donchian/Bollinger indicators on the monthly series, and export:

```text
output/result_{SYMBOL}.csv
```

Exactly 10 files × 24 rows × 18 columns.

## Dataset source

Public assignment dataset:

- Repository: [https://github.com/sandeep-tt/tt-intern-dataset](https://github.com/sandeep-tt/tt-intern-dataset)
- Expected local path after download: `data/output_file.csv`

Expected input columns:

```text
date, volume, open, high, low, close, adjclose, ticker
```

Expected tickers:

```text
AAPL, AMD, AMZN, AVGO, CSCO, MSFT, NFLX, PEP, TMUS, TSLA
```

The pipeline never modifies the source CSV. If you redistribute this project, confirm whether the dataset itself should be committed or fetched from the public source above.

## Prerequisites

- Python 3.10+
- Pandas only as the direct third-party dependency (`requirements.txt`)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `output_file.csv` under `data/` (see dataset source above).

## Run the complete pipeline

```bash
python stock_pipeline.py \
  --input data/output_file.csv \
  --output-dir output
```

| Argument | Default | Meaning |
|---|---|---|
| `--input` | `data/output_file.csv` | Daily source CSV |
| `--output-dir` | `output` | Destination for `result_*.csv` |

### Pipeline stages

1. Load and validate daily input.
2. Aggregate to monthly OHLCV per ticker.
3. Validate monthly coverage (10 × 24 = 240 rows).
4. Calculate all **10** indicator columns per ticker.
5. Validate the indicator DataFrame.
6. Stage-write all 10 CSVs, validate on disk, then promote into `--output-dir`.
7. Remove obsolete unexpected `result_*.csv` files only (unrelated files are kept).

On any critical failure the process exits nonzero and does **not** print a success message.

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

### Output schema (exact order, 18 columns)

```text
date, ticker, open, high, low, close, volume, adjclose,
sma_10, sma_20, ema_10, ema_20, zscore_20,
donchian_high_20, donchian_low_20,
bollinger_mid_20, bollinger_upper_20, bollinger_lower_20
```

- Dates are `YYYY-MM-DD` calendar month-ends from `2018-01-31` through `2019-12-31`.
- Missing warm-up indicators are empty CSV fields (not `0`, `"NaN"`, or `"None"`).
- Numeric values are written without cosmetic two-decimal rounding.

## Monthly OHLCV aggregation

| Field | Rule |
|---|---|
| `open` | First trading day's open |
| `high` | Max daily high |
| `low` | Min daily low |
| `close` | Last trading day's close |
| `volume` | Sum of daily volume |
| `adjclose` | Last trading day's adjusted close |
| `date` | Calendar month-end |

Rows are sorted by ticker/date before first/last snapshots so file order cannot distort results. Tickers are aggregated independently.

## Technical indicators (monthly windows)

| Column | Formula / convention |
|---|---|
| `sma_10` / `sma_20` | `close.rolling(n, min_periods=n).mean()` |
| `ema_10` / `ema_20` | `alpha = 2/(n+1)`; first EMA = SMA of first complete window; then `EMA_t = alpha*close_t + (1-alpha)*EMA_{t-1}` |
| `zscore_20` | `(close - mean_20) / std_20` with sample std (`ddof=1`); missing if std is 0 |
| `donchian_high_20` | 20-month rolling max of monthly `high` |
| `donchian_low_20` | 20-month rolling min of monthly `low` |
| `bollinger_mid_20` | 20-month SMA of `close` |
| `bollinger_upper_20` / `bollinger_lower_20` | mid ± `2 * std_20` |

### Expected warm-up missingness (24 months/ticker)

| Indicator | Valid / ticker | Missing / ticker |
|---|---:|---:|
| SMA/EMA 10 | 15 | 9 |
| SMA/EMA 20, Donchian, Bollinger | 5 | 19 |
| Z-score 20 | up to 5 | ≥ 19 |

Warm-up rows are retained.

## Validation behavior

- Invalid input/monthly/indicator states raise clear errors.
- Output writing validates the in-memory frame, writes a staged 10-file set, validates staged files, then promotes with per-file `os.replace`.
- On-disk checks cover filenames, 18-column schema/order, 24 rows, ticker isolation, month-end coverage, finite numerics, and warm-up emptiness.
- Staging prevents promoting an invalid complete set. A hard interrupt during promotion could still leave a mixed destination set (OS limitation); rerun the pipeline to repair.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Acceptance checklist

- [x] Required input columns validated; invalid/missing/duplicate/OHLC-broken inputs rejected
- [x] All 10 expected tickers processed for Jan 2018–Dec 2019
- [x] Monthly open/high/low/close/volume/adjclose follow first/max/min/last/sum/last rules
- [x] All 10 indicator columns implemented with documented formulas (SMA-seeded EMA)
- [x] Exactly 10 `result_{SYMBOL}.csv` files, each 24×18
- [x] Warm-up missing values preserved as empty fields
- [x] Source dataset left unchanged
- [x] Automated tests cover validation, aggregation, indicators, outputs, and independent recomputation
- [x] Assignment result CSVs are not gitignored (`output/result_*.csv` is trackable)

## Example successful run (illustrative shape; counts from the supplied dataset)

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
├── data/output_file.csv          # obtain from dataset source if absent
├── output/result_*.csv           # generated deliverables (trackable)
├── tests/test_stock_pipeline.py
├── stock_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```
