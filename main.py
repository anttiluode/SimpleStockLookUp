import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from typing import List, Callable
from sklearn.metrics import mean_squared_error
from math import sqrt

# ---------------------------
# 1. Setup and Configuration
# ---------------------------

# Set page configuration at the very top
st.set_page_config(
    page_title="📈 Simple Stock LookUp With Future Price Estimates",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 2. Helper Functions
# ---------------------------

@st.cache_data
def get_sp500_tickers() -> List[str]:
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]  # For yfinance compatibility
        return tickers
    except Exception as e:
        st.error("Failed to fetch S&P 500 tickers. Please check your internet connection or try again later.")
        return []

def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetches historical stock data for the given ticker and period."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            st.warning(f"No data found for ticker `{ticker}`.")
            return pd.DataFrame()
        return hist
    except Exception as e:
        st.error(f"An error occurred while fetching data for `{ticker}`.")
        return pd.DataFrame()

def plot_candlestick(hist: pd.DataFrame, ticker: str):
    """Plots an interactive candlestick chart for the given historical data."""
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='Price'
        )])
        fig.update_layout(
            title=f"{ticker} Candlestick Chart",
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            template='plotly_dark',
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("An error occurred while plotting the candlestick chart.")

def display_performance_metrics(hist: pd.DataFrame):
    """Displays key performance metrics for the stock."""
    try:
        latest_close = hist['Close'][-1]
        previous_close = hist['Close'][-2]
        pct_change = ((latest_close - previous_close) / previous_close) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Latest Close Price", value=f"${latest_close:,.2f}")
        with col2:
            st.metric(label="Change (%)", value=f"{pct_change:.2f}%", delta=f"{latest_close - previous_close:.2f}")
    except Exception as e:
        st.error("An error occurred while calculating performance metrics.")

def display_company_overview(ticker: str):
    """Displays a brief overview of the company."""
    try:
        stock = yf.Ticker(ticker)
        company_info = stock.info
        st.subheader("🏢 Company Overview")
        st.write(f"**Name:** {company_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
        st.write(f"**Description:** {company_info.get('longBusinessSummary', 'N/A')}")
    except Exception as e:
        st.error("An error occurred while fetching company information.")

def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands for given data."""
    data['SMA'] = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Upper_BB'] = data['SMA'] + (rolling_std * num_std)
    data['Lower_BB'] = data['SMA'] - (rolling_std * num_std)
    return data

def calculate_stochastic_oscillator(data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator for given data."""
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    data['%K'] = (data['Close'] - low_min) / (high_max - low_min) * 100
    data['%D'] = data['%K'].rolling(window=d_window).mean()
    return data

def calculate_macd(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """Calculate MACD for given data."""
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate RSI for given data."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def predict_macd(hist: pd.DataFrame, days_ahead: int = 30, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.Series:
    """Predict future prices using MACD with customizable parameters."""
    hist = calculate_macd(hist, short_window, long_window, signal_window)
    last_price = hist['Close'].iloc[-1]
    last_macd = hist['MACD'].iloc[-1]
    last_signal = hist['Signal_Line'].iloc[-1]
    
    predictions = [last_price]
    for _ in range(days_ahead):
        if last_macd > last_signal:
            next_price = predictions[-1] * 1.001  # Slight increase
        else:
            next_price = predictions[-1] * 0.999  # Slight decrease
        predictions.append(next_price)
    
    future_dates = pd.date_range(start=hist.index[-1] + timedelta(days=1), periods=days_ahead)
    return pd.Series(predictions[1:], index=future_dates)

def predict_rsi(hist: pd.DataFrame, days_ahead: int = 30, window: int = 14) -> pd.Series:
    """Predict future prices using RSI with customizable parameters."""
    hist = calculate_rsi(hist, window)
    last_price = hist['Close'].iloc[-1]
    last_rsi = hist['RSI'].iloc[-1]
    
    predictions = [last_price]
    for _ in range(days_ahead):
        if last_rsi > 70:
            next_price = predictions[-1] * 0.998  # Overbought, expect decrease
        elif last_rsi < 30:
            next_price = predictions[-1] * 1.002  # Oversold, expect increase
        else:
            next_price = predictions[-1] * 1.0005  # Neutral, slight increase
        predictions.append(next_price)
    
    future_dates = pd.date_range(start=hist.index[-1] + timedelta(days=1), periods=days_ahead)
    return pd.Series(predictions[1:], index=future_dates)

def predict_momentum(hist: pd.DataFrame, days_ahead: int = 30) -> pd.Series:
    """Predict future prices using a momentum strategy."""
    if hist.empty:
        return pd.Series(dtype=float)

    momentum_period = 5  # Lookback period for momentum
    hist['Momentum'] = hist['Close'].diff(momentum_period)
    last_price = hist['Close'].iloc[-1]
    
    predictions = [last_price]
    for _ in range(days_ahead):
        if hist['Momentum'].iloc[-1] > 0:
            next_price = predictions[-1] * 1.001  # Slight increase
        else:
            next_price = predictions[-1] * 0.999  # Slight decrease
        predictions.append(next_price)

    future_dates = pd.date_range(start=hist.index[-1] + timedelta(days=1), periods=days_ahead)
    return pd.Series(predictions[1:], index=future_dates)

def predict_mean_reversion(hist: pd.DataFrame, days_ahead: int = 30) -> pd.Series:
    """Predict future prices using a mean reversion strategy."""
    if hist.empty:
        return pd.Series(dtype=float)
    
    mean_price = hist['Close'].mean()
    last_price = hist['Close'].iloc[-1]
    predictions = [last_price]

    for _ in range(days_ahead):
        next_price = mean_price  # Predict back to the mean
        predictions.append(next_price)

    future_dates = pd.date_range(start=hist.index[-1] + timedelta(days=1), periods=days_ahead)
    return pd.Series(predictions[1:], index=future_dates)

def backtest_model(hist: pd.DataFrame, prediction_func: Callable, window: int = 30, **kwargs) -> float:
    """Backtest the prediction model and return RMSE."""
    actual_prices = hist['Close']
    predictions = []
    
    for i in range(len(hist) - window):
        train_data = hist.iloc[:i + window]
        pred = prediction_func(train_data, days_ahead=1, **kwargs)
        if not pred.empty:  # Check if predictions are not empty
            predictions.append(pred.iloc[0])
        else:
            predictions.append(np.nan)  # Append NaN if prediction fails

    predictions = pd.Series(predictions, index=actual_prices.index[window:])
    predictions = predictions.dropna()  # Drop NaN values for RMSE calculation

    if predictions.empty:
        return np.nan  # Return NaN if there are no valid predictions
    
    rmse = sqrt(mean_squared_error(actual_prices[window:][predictions.index], predictions))
    return rmse

# ---------------------------
# 3. Streamlit App
# ---------------------------

def main():
    # Title
    st.title("📈 Simple Stock LookUp With Future Price Estimates")

    # Sidebar for user inputs
    st.sidebar.header("🔍 Stock Selection")

    # Fetch S&P 500 tickers for the dropdown
    sp500_tickers = get_sp500_tickers()

    # Stock Symbol Selection
    ticker_option = st.sidebar.selectbox(
        "Select a Stock from S&P 500",
        options=["--Select a Ticker--"] + sp500_tickers
    )

    # Alternatively, allow user to input any ticker symbol
    user_ticker = st.sidebar.text_input("Or Enter a Ticker Symbol (e.g., AAPL, MSFT)")

    # Time Range Selection
    time_ranges = {
           "5 Days": "5d",
        "1 Month": "1mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "5 Years": "5y",
        "10 Years": "10y",
        "All Time": "max"
    }

    selected_range = st.sidebar.selectbox(
        "Select Time Range",
        options=list(time_ranges.keys())
    )

    # Updated Prediction Model Selection with parameter customization
    prediction_models = {
        "Momentum": predict_momentum,
        "Mean Reversion": predict_mean_reversion,
        "MACD": predict_macd,
        "RSI": predict_rsi
    }
    
    selected_model = st.sidebar.selectbox("Select Prediction Model", options=list(prediction_models.keys()))

    # Model-specific parameter customization
    model_params = {}
    if selected_model == "MACD":
        short_window = st.sidebar.slider("MACD Short Window", min_value=5, max_value=20, value=12)
        long_window = st.sidebar.slider("MACD Long Window", min_value=20, max_value=50, value=26)
        signal_window = st.sidebar.slider("MACD Signal Window", min_value=5, max_value=20, value=9)
        model_params = {"short_window": short_window, "long_window": long_window, "signal_window": signal_window}
    elif selected_model == "RSI":
        rsi_window = st.sidebar.slider("RSI Window", min_value=5, max_value=30, value=14)
        model_params = {"window": rsi_window}

    # Days Ahead for Prediction
    days_ahead = st.sidebar.slider("Days Ahead for Prediction", min_value=1, max_value=60, value=30)

    # Fetch Data Button
    fetch_button = st.sidebar.button("Fetch Data")

    if ticker_option != "--Select a Ticker--":
        ticker = ticker_option
    elif user_ticker:
        ticker = user_ticker.upper()
    else:
        ticker = None

    if fetch_button and ticker:
        with st.spinner(f"Fetching data for `{ticker}`..."):
            hist = fetch_stock_data(ticker, time_ranges[selected_range])
        
        if not hist.empty:
            # Display Performance Metrics
            display_performance_metrics(hist)

            # Display Company Overview
            display_company_overview(ticker)

            # Plot Candlestick Chart
            plot_candlestick(hist, ticker)

            # Advanced Financial Indicators
            st.subheader("📊 Advanced Financial Indicators")
            
            # Bollinger Bands
            bb_window = st.slider("Bollinger Bands Window", min_value=5, max_value=50, value=20)
            bb_std = st.slider("Bollinger Bands Standard Deviation", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            hist_bb = calculate_bollinger_bands(hist, window=bb_window, num_std=bb_std)
            
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=hist_bb.index, y=hist_bb['Close'], name="Close"))
            fig_bb.add_trace(go.Scatter(x=hist_bb.index, y=hist_bb['Upper_BB'], name="Upper BB"))
            fig_bb.add_trace(go.Scatter(x=hist_bb.index, y=hist_bb['Lower_BB'], name="Lower BB"))
            fig_bb.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_bb, use_container_width=True)

            # Stochastic Oscillator
            k_window = st.slider("Stochastic %K Window", min_value=5, max_value=30, value=14)
            d_window = st.slider("Stochastic %D Window", min_value=1, max_value=10, value=3)
            hist_stoch = calculate_stochastic_oscillator(hist, k_window=k_window, d_window=d_window)
            
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=hist_stoch.index, y=hist_stoch['%K'], name="%K"))
            fig_stoch.add_trace(go.Scatter(x=hist_stoch.index, y=hist_stoch['%D'], name="%D"))
            fig_stoch.update_layout(title="Stochastic Oscillator", xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig_stoch, use_container_width=True)

            # Prediction and Backtesting
            st.subheader(f"📈 Future Price Estimates Using {selected_model}")
            future_predictions = prediction_models[selected_model](hist, days_ahead=days_ahead, **model_params)
            
            if not future_predictions.empty:
                st.line_chart(future_predictions, use_container_width=True)

            # Backtesting
            backtest_window = st.slider("Backtesting Window (days)", min_value=30, max_value=365, value=90)
            rmse = backtest_model(hist, prediction_models[selected_model], window=backtest_window, **model_params)
            if not np.isnan(rmse):
                st.write(f"Backtesting RMSE (Root Mean Square Error): {rmse:.2f}")
                st.write(f"This indicates the average prediction error over the last {backtest_window} days.")
            else:
                st.write("No predictions were generated for backtesting.")

            # Explanations for Financial Indicators
            st.write("### Explanations of Financial Indicators")
            st.write("**Bollinger Bands**: Bollinger Bands consist of a middle band (SMA) and two outer bands. The outer bands are standard deviations above and below the SMA. They indicate volatility; when the price touches the upper band, it might be overbought, while touching the lower band might indicate oversold conditions.")
            st.write("**Stochastic Oscillator**: This measures the level of the closing price relative to the high-low range over a given period. Values over 80 indicate overbought conditions, while values under 20 indicate oversold conditions.")
            st.write("**MACD**: Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. It consists of the MACD line and the Signal line. A bullish signal occurs when the MACD line crosses above the Signal line, indicating potential upward momentum, while a bearish signal occurs when the MACD line crosses below the Signal line.")
            st.write("**RSI**: The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. The RSI moves between 0 and 100 and is typically used to identify overbought or oversold conditions in a market. An RSI above 70 is typically considered overbought, while an RSI below 30 is considered oversold.")

            # Display Raw Data (optional)
            st.subheader("📊 Historical Data")
            st.dataframe(hist.tail(100))  # Show last 100 entries
    elif fetch_button:
        st.warning("Please select or enter a valid ticker symbol.")

    # Footer
    st.markdown("---")
    st.markdown("### Antti Luode / ChatGPT / Claude AI - For Entertainment purposes only.")
    st.markdown("**Data Source:** [Yahoo Finance](https://finance.yahoo.com/)")

if __name__ == "__main__":
    main()
