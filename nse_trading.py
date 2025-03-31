import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Import core modules
from core.ml_trading_model import MLTradingModel
from core.backtest_system import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nse_trading')

# NSE top traded stocks
NSE_TICKERS = [
    'SCOM',  # Safaricom
    'EQTY',  # Equity Group
    'KCB',   # KCB Group
    'EABL',  # East African Breweries
    'COOP',  # Co-operative Bank
    'BAT',   # British American Tobacco
    'SCAN',  # Scangroup
    'NMG',   # Nation Media Group
    'JUB',   # Jubilee Insurance
    'STBK'   # Stanbic Bank
]

def fetch_nse_data(tickers, start_date, end_date=None):
    """
    Fetch data for stocks listed on the Nairobi Securities Exchange
    
    Parameters:
    tickers (list): List of NSE ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
    
    Returns:
    dict: Dictionary with tickers as keys and DataFrames as values
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    data = {}
    
    # Create directories for NSE data
    os.makedirs('data/nse', exist_ok=True)
    
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}...")
        
        try:
            # OPTION 1: Load from local CSV files
            # If you have CSV files with NSE data, uncomment this section
            # csv_path = f"data/nse/{ticker}.csv"
            # if os.path.exists(csv_path):
            #     df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
            #     logger.info(f"Loaded data from {csv_path}")
            # else:
            #     logger.warning(f"No data file found for {ticker}. Creating sample data.")
            #     df = create_sample_nse_data(ticker, start_date, end_date)
            
            # OPTION 2: For demonstration, create sample data
            # In reality, you would fetch this from an API or other source
            df = create_sample_nse_data(ticker, start_date, end_date)
            
            # Make sure all required columns are present
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' missing, creating with Close values")
                    df[col] = df['Close'] if 'Close' in df.columns else 0
            
            # Add Adj Close if not present (same as Close for markets without adjustments)
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            # Calculate basic indicators
            df['Returns'] = df['Adj Close'].pct_change()
            df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
            
            # Store in data dictionary
            data[ticker] = df
            
            # Save to CSV for future use
            df.to_csv(f"data/nse/{ticker}_processed.csv")
            logger.info(f"Successfully fetched and processed data for {ticker}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
    
    return data

def create_sample_nse_data(ticker, start_date, end_date):
    """
    Create sample NSE data for demonstration
    
    Parameters:
    ticker (str): Ticker symbol
    start_date (str): Start date
    end_date (str): End date
    
    Returns:
    DataFrame: Sample stock data
    """
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business days (NSE is closed on weekends)
    date_range = pd.date_range(start=start, end=end, freq='B')
    
    # Base price depends on the ticker (realistic for NSE stocks)
    if ticker == 'SCOM':  # Safaricom
        base_price = 30.0  # KES
        volatility = 0.015
    elif ticker == 'EQTY':  # Equity Group
        base_price = 45.0
        volatility = 0.018
    elif ticker == 'KCB':  # KCB Group
        base_price = 40.0
        volatility = 0.017
    elif ticker == 'EABL':  # East African Breweries
        base_price = 170.0
        volatility = 0.020
    else:
        base_price = 50.0
        volatility = 0.019
    
    # Generate price data with a slight upward drift (realistic for long-term stock behavior)
    price_changes = np.random.normal(0.0002, volatility, size=len(date_range))
    prices = base_price * (1 + np.cumsum(price_changes))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Close': prices,
        'Open': prices * np.random.uniform(0.99, 1.01, size=len(date_range)),
        'High': prices * np.random.uniform(1.0, 1.03, size=len(date_range)),
        'Low': prices * np.random.uniform(0.97, 1.0, size=len(date_range)),
        'Volume': np.random.randint(10000, 500000, size=len(date_range))
    }, index=date_range)
    
    # NSE-specific characteristics
    # 1. Lower volume on certain days (e.g., around holidays)
    holidays = ['2023-01-01', '2023-04-07', '2023-04-10', '2023-05-01', 
               '2023-06-01', '2023-10-20', '2023-12-25', '2023-12-26']
    
    for holiday in holidays:
        # Reduce volume around holidays
        holiday_date = pd.to_datetime(holiday)
        nearby_dates = [holiday_date - timedelta(days=1), 
                       holiday_date + timedelta(days=1)]
        
        for date in nearby_dates:
            if date in df.index:
                df.loc[date, 'Volume'] = df.loc[date, 'Volume'] * 0.5
    
    # 2. Price limits (NSE has 10% daily price movement limits)
    # Ensure no daily price change exceeds 10%
    daily_returns = df['Close'].pct_change()
    limit_exceeded = abs(daily_returns) > 0.099
    
    if limit_exceeded.any():
        for date in df.index[limit_exceeded]:
            if date == df.index[0]:  # Skip first day
                continue
                
            previous_close = df.loc[df.index[df.index.get_loc(date) - 1], 'Close']
            
            if daily_returns[date] > 0.099:
                # Cap the upward move at 9.9%
                df.loc[date, 'Close'] = previous_close * 1.099
                df.loc[date, 'Open'] = previous_close * 1.05
                df.loc[date, 'High'] = previous_close * 1.099
                df.loc[date, 'Low'] = previous_close * 1.03
            elif daily_returns[date] < -0.099:
                # Cap the downward move at -9.9%
                df.loc[date, 'Close'] = previous_close * 0.901
                df.loc[date, 'Open'] = previous_close * 0.95
                df.loc[date, 'High'] = previous_close * 0.97
                df.loc[date, 'Low'] = previous_close * 0.901
    
    return df

def run_nse_analysis(tickers=NSE_TICKERS[:5], days=365):
    """
    Run analysis on NSE stocks
    
    Parameters:
    tickers (list): List of NSE ticker symbols
    days (int): Number of days of history to analyze
    
    Returns:
    dict: Analysis results by ticker
    """
    logger.info(f"Running analysis for NSE stocks: {tickers}")
    
    # Set up dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Create directories
    os.makedirs('results/nse', exist_ok=True)
    
    # Fetch NSE data
    market_data = fetch_nse_data(tickers, start_date, end_date)
    
    results = {}
    for ticker in tickers:
        if ticker not in market_data or market_data[ticker].empty:
            logger.warning(f"No data available for {ticker}, skipping...")
            continue
            
        logger.info(f"\nAnalyzing {ticker}...")
        data = market_data[ticker]
        
        # Check if we have enough data
        if len(data) < 100:  # NSE may have fewer trading days
            logger.warning(f"Not enough data for {ticker} ({len(data)} days), skipping...")
            continue
        
        # Prepare features and train model
        ml_model = MLTradingModel()
        processed_data = ml_model.prepare_features(data)
        
        # Use features suitable for potentially lower liquidity markets
        features = [
            'Returns', 'Price_to_SMA20', 'Volume_Change',
            'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3'
        ]
        
        # Train model
        try:
            training_results = ml_model.train(processed_data, features)
            
            # Generate signals
            signals = ml_model.predict(processed_data)
            
            # Run backtest
            backtester = Backtester()
            backtest_results = backtester.run_backtest(data, signals)
            
            # Plot results
            backtester.plot_results(save_path=f'results/nse/{ticker}_backtest.png')
            
            # Store results
            results[ticker] = {
                'total_return': backtest_results['total_return_pct'],
                'annual_return': backtest_results['annual_return_pct'],
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'max_drawdown': backtest_results['max_drawdown_pct']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
    
    # Create summary table
    if results:
        summary = pd.DataFrame(results).T
        summary.columns = ['Total Return (%)', 'Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
        summary.index.name = 'Ticker'
        
        # Save summary
        summary.to_csv(f'results/nse/nse_summary_{datetime.now().strftime("%Y%m%d")}.csv')
        logger.info("\nSummary of Results:")
        logger.info(summary.to_string())
        
        # Plot comparison chart
        plt.figure(figsize=(10, 6))
        summary['Annual Return (%)'].plot(kind='bar', color='blue')
        plt.title('Annual Returns by NSE Stock')
        plt.ylabel('Annual Return (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'results/nse/nse_comparison_{datetime.now().strftime("%Y%m%d")}.png')
        
    return results

def nse_market_adjustments(data, ticker):
    """
    Apply NSE-specific adjustments to data
    
    Parameters:
    data (DataFrame): Stock data
    ticker (str): Ticker symbol
    
    Returns:
    DataFrame: Adjusted data
    """
    # Make a copy to avoid modifying the original
    adjusted = data.copy()
    
    # 1. Handle missing data (NSE might have more gaps)
    # Forward fill up to 5 days of missing data
    adjusted = adjusted.resample('D').asfreq()
    adjusted = adjusted.fillna(method='ffill', limit=5)
    
    # 2. Handle lower liquidity
    # Identify and filter out days with suspiciously low volume
    if 'Volume' in adjusted.columns:
        avg_volume = adjusted['Volume'].median()
        min_volume = avg_volume * 0.1  # 10% of median volume
        
        # Create a flag for low volume days
        adjusted['Low_Volume'] = adjusted['Volume'] < min_volume
    
    # 3. Handle NSE-specific trading rules
    # NSE has daily price movement limits (e.g., 10% for most stocks)
    if 'Close' in adjusted.columns and 'Adj Close' in adjusted.columns:
        adjusted['Daily_Change'] = adjusted['Close'].pct_change().abs()
        adjusted['Price_Limit_Hit'] = adjusted['Daily_Change'] > 0.095  # Just under 10%
    
    return adjusted

if __name__ == "__main__":
    # Run NSE analysis with default settings
    run_nse_analysis()