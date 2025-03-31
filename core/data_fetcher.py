import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

# Set up logger
logger = logging.getLogger(__name__)

def fetch_market_data(tickers, start_date, end_date=None):
    """
    Fetch historical market data for a list of tickers
    
    Parameters:
    tickers (list): List of ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
    
    Returns:
    dict: Dictionary with tickers as keys and DataFrames as values
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            # Get historical data
            df = yf.download(ticker, start=start_date, end=end_date)
            
            # If df has MultiIndex columns, flatten them
            if isinstance(df.columns, pd.MultiIndex):
                # Select just the data for this ticker (needed for multiple tickers)
                if ticker in df.columns.levels[1]:
                    # Get the specific ticker data
                    ticker_data = pd.DataFrame()
                    for col in df.columns.levels[0]:
                        ticker_data[col] = df[(col, ticker)]
                    df = ticker_data
                else:
                    # If using a single ticker, column names might be different
                    # Flatten the columns directly
                    df.columns = [col[0] for col in df.columns]
            
            # In newer versions of yfinance, the column is 'Close' instead of 'Adj Close'
            # Let's handle both cases
            if 'Adj Close' not in df.columns:
                if 'Close' in df.columns:
                    df['Adj Close'] = df['Close']
                    print(f"Added 'Adj Close' column (copy of 'Close')")
            
            # Calculate returns
            df['Returns'] = df['Adj Close'].pct_change()
            
            # Calculate some basic indicators
            df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
            
            # Store in the dictionary
            data[ticker] = df
            
            # Create data directory if it doesn't exist
            os.makedirs('data/market', exist_ok=True)
            
            # Save to CSV
            csv_path = f"data/market/{ticker}_{start_date}_{end_date}.csv"
            df.to_csv(csv_path)
            
            print(f"Successfully fetched data for {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return data

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    
    # Make two series: one for gains and one for losses
    gains = delta.copy()
    losses = delta.copy()
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # First value is sum of gains
    first_avg_gain = gains.iloc[:period].mean()
    first_avg_loss = losses.iloc[:period].mean()
    
    # Initialize averages
    avg_gain = pd.Series([first_avg_gain], index=[prices.index[period]])
    avg_loss = pd.Series([first_avg_loss], index=[prices.index[period]])
    
    # Loop through data points
    for i in range(period + 1, len(prices)):
        avg_gain = pd.concat([avg_gain, pd.Series(
            [(avg_gain.iloc[-1] * (period - 1) + gains.iloc[i]) / period],
            index=[prices.index[i]])])
        avg_loss = pd.concat([avg_loss, pd.Series(
            [(avg_loss.iloc[-1] * (period - 1) + losses.iloc[i]) / period],
            index=[prices.index[i]])])
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def fetch_intraday_data(ticker, interval='1h', period='1d'):
    """
    Fetch intraday market data for a ticker
    
    Parameters:
    ticker (str): Ticker symbol
    interval (str): Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
    period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
    DataFrame: Intraday data
    """
    try:
        # Get intraday data
        ticker_data = yf.Ticker(ticker)
        df = ticker_data.history(period=period, interval=interval)
        
        return df
    except Exception as e:
        print(f"Error fetching intraday data for {ticker}: {e}")
        return pd.DataFrame()

def get_company_info(ticker):
    """
    Get company information for a ticker
    
    Parameters:
    ticker (str): Ticker symbol
    
    Returns:
    dict: Company information
    """
    try:
        # Get company information
        ticker_data = yf.Ticker(ticker)
        info = ticker_data.info
        
        return info
    except Exception as e:
        print(f"Error fetching company info for {ticker}: {e}")
        return {}

def get_earnings_data(ticker):
    """
    Get earnings data for a ticker
    
    Parameters:
    ticker (str): Ticker symbol
    
    Returns:
    DataFrame: Earnings data
    """
    try:
        # Get earnings data
        ticker_data = yf.Ticker(ticker)
        earnings = ticker_data.earnings
        
        return earnings
    except Exception as e:
        print(f"Error fetching earnings data for {ticker}: {e}")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Define parameters
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start = '2023-01-01'
    end = '2023-03-01'
    
    # Fetch data
    market_data = fetch_market_data(tickers, start, end)
    
    for ticker, data in market_data.items():
        print(f"\n{ticker} data shape: {data.shape}")
        print(f"{ticker} data columns: {data.columns.tolist()}")
        print(f"{ticker} first few rows:")
        print(data.head())