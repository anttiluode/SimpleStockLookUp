# 📈 Simple Stock LookUp With Future Price Estimates

This Streamlit application provides an interactive interface to explore stock data, perform technical analysis, and make future price predictions using various financial models. The app fetches data from Yahoo Finance and supports multiple prediction strategies, including MACD, RSI, Momentum, and Mean Reversion.

## Features

- **Stock Selection**: Choose from S&P 500 tickers or enter any valid ticker symbol.
- **Historical Data Visualization**: Display stock prices as candlestick charts.
- **Performance Metrics**: View key performance indicators for selected stocks.
- **Technical Indicators**: Calculate and visualize Bollinger Bands and Stochastic Oscillator.
- **Future Price Predictions**: Make predictions using different financial models with customizable parameters.
- **Backtesting**: Evaluate prediction accuracy using historical data through RMSE calculation.
- **Explanations**: Understand the various financial indicators and prediction strategies used.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anttiluode/SimpleStockLookUp.git
   cd SimpleStockLookUp
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run main.py
Usage
Select a stock ticker from the S&P 500 list or enter your own.
Choose the desired time range for historical data.
Select a prediction model and customize parameters as needed.
Fetch data to visualize stock performance and make future price estimates.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
This application utilizes data from Yahoo Finance.
Built with Streamlit, yfinance, pandas, numpy, and Plotly.

For entertainment purposes only. 
