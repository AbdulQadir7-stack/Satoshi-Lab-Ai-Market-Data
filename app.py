"""Streamlit dashboard for BTC/USDT machine-learning research."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import FEATURE_COLUMNS, OHLCV_COLUMNS
from src.data.fetcher import fetch_ohlcv, generate_sample_ohlcv
from src.data.processor import clean_ohlcv
from src.evaluation.backtest import run_backtest
from src.features.engineering import build_features
from src.models.train import MIN_DATASET_ROWS, TrainingResult, train_and_evaluate


@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(source: str, timeframe: str, history_range: str) -> pd.DataFrame:
    """Load, validate, and briefly cache the currently selected market data."""
    if source == "Public Binance":
        candles = fetch_ohlcv(timeframe=timeframe, history_range=history_range)
    elif source == "Synthetic demo data":
        candles = generate_sample_ohlcv(timeframe=timeframe, history_range=history_range)
    else:
        raise ValueError(f"Unsupported market data source: {source!r}.")

    missing_columns = set(OHLCV_COLUMNS).difference(candles.columns)
    if missing_columns:
        raise ValueError(f"Loaded market data is missing columns: {sorted(missing_columns)}.")
    return clean_ohlcv(candles)


def main() -> None:
    st.set_page_config(page_title="SatoshiLab AI", page_icon="₿", layout="wide")

    st.title("SatoshiLab AI")
    st.caption("BTC/USDT machine-learning research and evaluation platform")
    st.warning(
        "Educational research only. This is not financial advice, a trading signal, "
        "or a guarantee of future performance. No trades are executed."
    )

    with st.sidebar:
        st.header("Research setup")
        source = st.radio(
            "Market data",
            ["Public Binance", "Synthetic demo data"],
            index=0,
            key="market_data_source",
        )
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"])
        history_range = st.selectbox(
            "Historical range", ["3 days", "10 days", "25 days", "4 months"], index=2
        )
        fee_percent = st.number_input(
            "Trading fee per side (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.05,
        )

    data_selection = (source, timeframe, history_range)
    fallback_key = "demo_fallback_selection"
    if st.session_state.get("data_selection") != data_selection:
        st.session_state["training_result"] = None
        st.session_state["data_selection"] = data_selection
        st.session_state.pop(fallback_key, None)

    using_demo_fallback = (
        source == "Public Binance"
        and st.session_state.get(fallback_key) == data_selection
    )
    try:
        with st.spinner("Loading BTC/USDT candles..."):
            if using_demo_fallback:
                candles = load_market_data(
                    "Synthetic demo data", timeframe, history_range
                )
                source_status = (
                    "SYNTHETIC DEMO FALLBACK — Public Binance remains selected; "
                    "these are not real market data."
                )
            else:
                candles = load_market_data(source, timeframe, history_range)
                if source == "Public Binance":
                    source_status = "Live public historical candles from Binance (CCXT)"
                else:
                    source_status = "SYNTHETIC DEMO DATA — not real market data"
    except Exception as error:
        st.error(f"Could not load the selected data source: {error}")
        if source == "Public Binance":
            st.info(
                "Check the network and CCXT installation, or load synthetic demo data "
                "without changing the selected live source."
            )
            if st.button("Load synthetic demo fallback"):
                st.session_state[fallback_key] = data_selection
                st.rerun()
        else:
            st.info("Check the selected timeframe and history range, then try again.")
        return

    if using_demo_fallback:
        st.warning(
            "Synthetic demo fallback is active. The selected source remains Public "
            "Binance; use the button below to retry live data."
        )
        if st.button("Retry Public Binance"):
            st.session_state.pop(fallback_key, None)
            st.rerun()
    st.info(source_status)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("BTC/USDT market history")
        price_chart = go.Figure(
            data=[
                go.Candlestick(
                    x=candles["timestamp"],
                    open=candles["open"],
                    high=candles["high"],
                    low=candles["low"],
                    close=candles["close"],
                    name="BTC/USDT",
                )
            ]
        )
        price_chart.update_layout(
            xaxis_title="UTC timestamp",
            yaxis_title="Price (USDT)",
            height=440,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(price_chart, width="stretch")
    with right:
        st.subheader("Volume")
        volume_chart = go.Figure(
            go.Bar(x=candles["timestamp"], y=candles["volume"], name="Volume")
        )
        volume_chart.update_layout(
            xaxis_title="UTC timestamp", yaxis_title="BTC", height=440
        )
        st.plotly_chart(volume_chart, width="stretch")

    with st.expander("Data preview"):
        st.dataframe(candles.tail(20), width="stretch")
        st.caption(
            f"{len(candles)} validated candles | timeframe: {timeframe} | "
            f"requested history: {history_range}"
        )

    st.subheader("Feature overview")
    st.write(
        "Features are computed from candles through time **t**, when candle t closes. "
        "The target is whether candle **t+1** closes above its own open. "
        "Warm-up rows with unavailable rolling features are excluded."
    )
    st.code(", ".join(FEATURE_COLUMNS))

    try:
        features = build_features(candles)
    except Exception as error:
        st.error(f"Feature engineering failed: {error}")
        return

    st.caption(f"Usable feature rows: {len(features)}")
    can_train = len(features) >= MIN_DATASET_ROWS
    if not can_train:
        st.warning(
            f"Not enough usable rows to train and evaluate yet. At least "
            f"{MIN_DATASET_ROWS} are needed; this selection currently provides "
            f"{len(features)}. Choose a longer history or a shorter candle timeframe."
        )

    if "training_result" not in st.session_state:
        st.session_state["training_result"] = None

    if st.button("Train and evaluate", type="primary", disabled=not can_train):
        try:
            with st.spinner("Training chronological baseline and random forest..."):
                st.session_state["training_result"] = train_and_evaluate(features)
        except Exception as error:
            st.error(f"Training or evaluation failed: {error}")

    result: TrainingResult | None = st.session_state["training_result"]
    if result is None:
        st.info(
            "Train a model to see held-out test metrics, predictions, and the backtest. "
            "The selected history must contain enough usable feature rows."
        )
    else:
        st.subheader("Chronological holdout evaluation")
        st.write(
            f"Training samples: **{result.train_size}** | "
            f"Unseen test samples: **{result.test_size}**"
        )
        metric_rows = []
        for model_name, values in result.metrics.items():
            metric_rows.append(
                {
                    "Model": model_name,
                    "Accuracy": values["accuracy"],
                    "Precision": values["precision"],
                    "Recall": values["recall"],
                    "F1": values["f1"],
                }
            )
        st.dataframe(pd.DataFrame(metric_rows).set_index("Model"), width="stretch")

        model_choice = st.selectbox(
            "Confusion matrix model", list(result.metrics.keys()), key="confusion_model"
        )
        matrix = result.metrics[model_choice]["confusion_matrix"]
        st.write("Rows = actual (down, up); columns = predicted (down, up).")
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual down", "Actual up"],
                columns=["Predicted down", "Predicted up"],
            ),
            width="content",
        )

        predictions = result.random_forest_predictions
        st.subheader("Held-out test predictions")
        prediction_table = pd.DataFrame(
            {
                "Actual next candle up": result.y_test,
                "Predicted next candle up": predictions,
            }
        )
        st.dataframe(prediction_table.tail(30), width="stretch")

        st.subheader("Hypothetical long-or-flat backtest")
        st.caption(
            "A prediction made after candle t closes is applied to candle t+1's "
            "open-to-close return. Cash when predicted down; fee applied on entry, "
            "exit, and final liquidation. Buy-and-hold compounds close-to-next-close "
            "returns over the same evaluation periods. Simplified, hypothetical simulation."
        )
        try:
            backtest = run_backtest(candles, predictions, fee_rate=fee_percent / 100)
            equity_chart = go.Figure()
            equity_chart.add_scatter(
                x=backtest.equity.index, y=backtest.equity, name="Strategy"
            )
            equity_chart.add_scatter(
                x=backtest.benchmark_equity.index,
                y=backtest.benchmark_equity,
                name="Buy and hold",
            )
            equity_chart.update_layout(
                xaxis_title="UTC timestamp", yaxis_title="Growth of 1 unit"
            )
            st.plotly_chart(equity_chart, width="stretch")
            cols = st.columns(3)
            cols[0].metric("Strategy total return", f"{backtest.total_return:.2%}")
            cols[1].metric("Max drawdown", f"{backtest.max_drawdown:.2%}")
            cols[2].metric("Long entries", str(backtest.trade_count))
            st.metric("Benchmark total return", f"{backtest.benchmark_return:.2%}")
        except Exception as error:
            st.error(f"Backtest could not be calculated: {error}")


if __name__ == "__main__":
    main()
