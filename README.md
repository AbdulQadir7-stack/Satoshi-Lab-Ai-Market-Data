# SatoshiLab AI

SatoshiLab AI is a lightweight, local research platform for evaluating whether simple machine-learning models can classify the direction of the next BTC/USDT candle. It compares a majority-class baseline with a Random Forest on a later, chronologically held-out period and offers a transparent hypothetical backtest.

> **Educational research only.** Results are historical experiments, not financial advice, trading recommendations, or evidence of future profitability. The application does not place trades.

## Features

- Public Binance spot OHLCV retrieval using CCXT, without API credentials.
- Selectable history windows of 3 days, 10 days, 25 days, or 4 months.
- Clearly identified synthetic candles for offline demonstrations.
- Validated, chronological OHLCV normalization and deterministic technical features.
- A chronological holdout, majority baseline, and seeded scikit-learn Random Forest.
- Accuracy, precision, recall, F1, confusion matrices, and held-out predictions.
- A fee-aware, long-or-flat backtest with equity curve, drawdown, and benchmark.
- Local model persistence using joblib; downloaded data and model artifacts are ignored by Git.

## Requirements and Windows setup

Python 3.10 is supported. In PowerShell, from the project directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If script activation is restricted, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell session and activate again. Dependency installation is performed by the user; no API keys, GPU, Docker, or paid services are needed.

## Run the dashboard

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

The dashboard opens on **Public Binance** and caches fetched candles for five minutes to avoid repeated network requests during dashboard interactions. Choose **Synthetic demo data** to explore offline. Network, exchange, or missing-CCXT errors are shown, with an explicit button to switch to synthetic data. Synthetic data is clearly labelled and is never silently substituted for exchange data.

Training requires at least 38 usable feature rows, with 30 chronological training rows and 8 later test rows. The dashboard explains when a selected short history/timeframe does not contain enough data.

## Run tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

The deterministic tests do not access the network.

## Architecture

```text
app.py
src/
  config.py
  data/          fetcher.py, processor.py
  features/      engineering.py
  models/        train.py, predict.py
  evaluation/    metrics.py, backtest.py
data/
  raw/           downloaded/raw files (ignored)
  processed/     processed files (ignored)
models/          local joblib artifacts (ignored)
tests/
docs/
```

See [docs/architecture.md](docs/architecture.md) for the data flow and [docs/methodology.md](docs/methodology.md) for target, split, and backtest definitions.

## How to interpret the results

The target at time *t* is 1 if candle *t+1* closes above its own open; otherwise it is 0. Features are computed using market information through the close of candle *t*. Initial rolling-window rows and the final row (whose target is not yet observable) are omitted.

The first 80% of usable rows train both classifiers; the final 20% form the unseen test period. Accuracy is the fraction classified correctly. Precision and recall describe positive-class predictions and hits; F1 is their harmonic mean. The confusion matrix uses actual classes as rows (down, up) and predicted classes as columns (down, up). Undefined positive-class metrics are reported as zero. Metrics are not hardcoded and may be uninformative on small or changing datasets.

## Backtest assumptions and limitations

A model prediction computed after candle *t* closes is applied to the next candle's open-to-close move. A positive prediction holds a long position for that candle; zero stays in cash. The fee is charged on position entry, exit, and final liquidation. The benchmark compounds close-to-next-close returns across the same signal periods. This is a simplified candle-level simulation: it omits slippage, spread, partial fills, funding, market impact, and execution constraints. It does not establish a profitable or deployable strategy.

## Known limitations and future work

- The dashboard uses a single chronological holdout, not walk-forward validation.
- Features, strategy assumptions, and models are intentionally simple.
- Binance availability and rate limits depend on network and region.
- The Random Forest artifact is saved locally and is not automatically reused as an evaluation set.
- Further research could add walk-forward evaluation, robust position sizing, slippage modeling, and additional baselines.
