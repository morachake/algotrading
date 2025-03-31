import pandas as pd
import numpy as np
import requests
import json
import re
import os
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AlternativeData')

class AlternativeDataProcessor:
    """
    Process and integrate alternative data sources for algorithmic trading
    """
    
    def __init__(self, data_dir='data/alternative'):
        """Initialize the data processor"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_economic_indicators(self, indicators=None, start_date=None, end_date=None):
        """
        Fetch economic indicators data 
        
        Parameters:
        indicators (list): List of indicator codes
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
        Returns:
        DataFrame: Economic indicator data
        """
        logger.info("Fetching economic indicators data")
        
        # Default indicators if none provided
        if indicators is None:
            indicators = ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'INDPRO']
            
        # In a real implementation, you would connect to an API
        # For example, using FRED API or similar
        # Here we'll create simulated data for demonstration
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
            
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Create simulated data
        data = {}
        
        for indicator in indicators:
            # Generate realistic values for each indicator
            if indicator == 'GDP':
                # Quarterly GDP growth rate (percent)
                values = np.random.normal(0.5, 0.3, size=len(date_range) // 3 + 1)
                # Replicate each value for 3 months to simulate quarterly data
                values = np.repeat(values, 3)[:len(date_range)]
            elif indicator == 'UNRATE':
                # Unemployment rate (percent)
                values = np.random.normal(4.0, 0.2, size=len(date_range))
            elif indicator == 'CPIAUCSL':
                # Consumer Price Index (percent change)
                values = np.random.normal(0.2, 0.1, size=len(date_range))
            elif indicator == 'FEDFUNDS':
                # Federal Funds Rate (percent)
                # Start with a random value and make small changes each month
                values = [np.random.uniform(1.5, 2.5)]
                for i in range(1, len(date_range)):
                    # Random walk with small changes
                    values.append(values[-1] + np.random.normal(0, 0.1))
            elif indicator == 'INDPRO':
                # Industrial Production Index (percent change)
                values = np.random.normal(0.3, 0.2, size=len(date_range))
            else:
                # Generic random values for other indicators
                values = np.random.normal(0, 1, size=len(date_range))
                
            data[indicator] = values
            
        # Create DataFrame
        df = pd.DataFrame(data, index=date_range)
        
        # Save to CSV
        csv_path = os.path.join(self.data_dir, 'economic_indicators.csv')
        df.to_csv(csv_path)
        logger.info(f"Saved economic indicators to {csv_path}")
        
        return df
    
    def fetch_social_media_sentiment(self, tickers, lookback_days=30):
        """
        Fetch social media sentiment data for tickers
        
        Parameters:
        tickers (list): List of ticker symbols
        lookback_days (int): Number of days to look back
        
        Returns:
        DataFrame: Social media sentiment data
        """
        logger.info(f"Fetching social media sentiment for {tickers}")
        
        # In a real implementation, you would connect to social media APIs
        # or use specialized data providers
        # Here we'll create simulated data for demonstration
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame with random sentiment scores
        data = {}
        
        for ticker in tickers:
            # Generate random sentiment scores between -1 and 1
            # with a slight positive bias (0.05) to simulate market optimism
            sentiment_scores = np.random.normal(0.05, 0.3, size=len(date_range))
            # Clip to range [-1, 1]
            sentiment_scores = np.clip(sentiment_scores, -1, 1)
            
            # Generate random volume of mentions
            mention_volume = np.random.randint(10, 1000, size=len(date_range))
            
            # Store in data dictionary
            data[f'{ticker}_sentiment'] = sentiment_scores
            data[f'{ticker}_volume'] = mention_volume
        
        # Create DataFrame
        df = pd.DataFrame(data, index=date_range)
        
        # Save to CSV
        csv_path = os.path.join(self.data_dir, 'social_media_sentiment.csv')
        df.to_csv(csv_path)
        logger.info(f"Saved social media sentiment to {csv_path}")
        
        return df
    
    def fetch_options_data(self, tickers, lookback_days=30):
        """
        Fetch options market data for tickers
        
        Parameters:
        tickers (list): List of ticker symbols
        lookback_days (int): Number of days to look back
        
        Returns:
        dict: Dictionary with options data by ticker
        """
        logger.info(f"Fetching options data for {tickers}")
        
        # In a real implementation, you would connect to options data APIs
        # or use specialized data providers
        # Here we'll create simulated data for demonstration
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        options_data = {}
        
        for ticker in tickers:
            # Generate realistic options data
            
            # Put-Call ratio (values around 1.0, with deviations)
            put_call_ratio = np.random.normal(1.0, 0.2, size=len(date_range))
            
            # Implied volatility (realistic ranges)
            implied_volatility = np.random.normal(0.2, 0.05, size=len(date_range))
            implied_volatility = np.clip(implied_volatility, 0.05, 0.5)
            
            # Options volume
            options_volume = np.random.randint(1000, 10000, size=len(date_range))
            
            # Open interest
            open_interest = np.random.randint(5000, 50000, size=len(date_range))
            
            # Skew (difference between OTM put and call IV)
            skew = np.random.normal(0.05, 0.02, size=len(date_range))
            
            # Create DataFrame for this ticker
            df = pd.DataFrame({
                'put_call_ratio': put_call_ratio,
                'implied_volatility': implied_volatility,
                'options_volume': options_volume,
                'open_interest': open_interest,
                'skew': skew
            }, index=date_range)
            
            options_data[ticker] = df
            
            # Save to CSV
            csv_path = os.path.join(self.data_dir, f'{ticker}_options_data.csv')
            df.to_csv(csv_path)
            logger.info(f"Saved {ticker} options data to {csv_path}")
        
        return options_data
    
    def fetch_insider_trading(self, tickers, lookback_days=90):
        """
        Fetch insider trading data for tickers
        
        Parameters:
        tickers (list): List of ticker symbols
        lookback_days (int): Number of days to look back
        
        Returns:
        dict: Dictionary with insider trading data by ticker
        """
        logger.info(f"Fetching insider trading data for {tickers}")
        
        # In a real implementation, you would connect to SEC data or APIs
        # Here we'll create simulated data for demonstration
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        insider_data = {}
        
        for ticker in tickers:
            # Simulate insider transactions
            # Number of transactions (random between 1 and 20)
            num_transactions = np.random.randint(1, 20)
            
            # Generate transaction data
            transactions = []
            
            for _ in range(num_transactions):
                # Transaction date
                days_back = np.random.randint(0, lookback_days)
                transaction_date = end_date - timedelta(days=days_back)
                
                # Transaction type (buy or sell)
                transaction_type = np.random.choice(['BUY', 'SELL'], p=[0.3, 0.7])
                
                # Transaction amount (shares)
                shares = np.random.randint(100, 10000)
                
                # Share price
                price = np.random.uniform(50, 200)
                
                # Insider role
                role = np.random.choice(['CEO', 'CFO', 'CTO', 'Director', 'VP'])
                
                transactions.append({
                    'date': transaction_date.strftime('%Y-%m-%d'),
                    'insider_name': f"Executive {np.random.randint(1, 10)}",
                    'role': role,
                    'transaction_type': transaction_type,
                    'shares': shares,
                    'price': price,
                    'value': shares * price
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            if not df.empty:
                # Sort by date
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # Save to CSV
                csv_path = os.path.join(self.data_dir, f'{ticker}_insider_trading.csv')
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {ticker} insider trading data to {csv_path}")
            
            insider_data[ticker] = df
        
        return insider_data
    
    def create_alternative_data_features(self, market_data, tickers):
        """
        Create features from alternative data sources
        
        Parameters:
        market_data (dict): Dictionary of market data by ticker
        tickers (list): List of ticker symbols
        
        Returns:
        dict: Updated market data with alternative data features
        """
        logger.info("Creating alternative data features")
        
        # Fetch alternative data
        economic_indicators = self.fetch_economic_indicators()
        social_sentiment = self.fetch_social_media_sentiment(tickers)
        options_data = self.fetch_options_data(tickers)
        insider_data = self.fetch_insider_trading(tickers)
        
        # Enhance market data with alternative data features
        enhanced_data = {}
        
        for ticker in tickers:
            if ticker not in market_data:
                logger.warning(f"No market data found for {ticker}. Skipping...")
                continue
                
            # Get the market data for this ticker
            data = market_data[ticker].copy()
            
            # Add economic indicators (resample to match market data frequency)
            if not economic_indicators.empty:
                for col in economic_indicators.columns:
                    # Resample and forward fill to match market data index
                    indicator = economic_indicators[col].resample('D').ffill()
                    indicator = indicator.reindex(data.index, method='ffill')
                    data[f'econ_{col}'] = indicator
            
            # Add social media sentiment
            if any(col.startswith(f'{ticker}_') for col in social_sentiment.columns):
                sentiment_col = f'{ticker}_sentiment'
                volume_col = f'{ticker}_volume'
                
                if sentiment_col in social_sentiment.columns and volume_col in social_sentiment.columns:
                    # Resample and match market data index
                    sentiment = social_sentiment[sentiment_col].resample('D').mean()
                    sentiment = sentiment.reindex(data.index, method='ffill')
                    data['social_sentiment'] = sentiment
                    
                    volume = social_sentiment[volume_col].resample('D').sum()
                    volume = volume.reindex(data.index, method='ffill')
                    data['social_volume'] = volume
                    
                    # Create sentiment momentum features
                    data['social_sentiment_3d'] = data['social_sentiment'].rolling(window=3).mean()
                    data['social_sentiment_7d'] = data['social_sentiment'].rolling(window=7).mean()
                    data['social_sentiment_change'] = data['social_sentiment'].diff()
            
            # Add options data
            if ticker in options_data and not options_data[ticker].empty:
                options_df = options_data[ticker]
                
                # Resample and match market data index
                for col in options_df.columns:
                    options_series = options_df[col].resample('D').mean() if col != 'options_volume' else options_df[col].resample('D').sum()
                    options_series = options_series.reindex(data.index, method='ffill')
                    data[f'options_{col}'] = options_series
                
                # Create derived features
                if 'options_put_call_ratio' in data.columns:
                    data['options_put_call_ratio_change'] = data['options_put_call_ratio'].diff()
                    data['options_put_call_ratio_5d'] = data['options_put_call_ratio'].rolling(window=5).mean()
                
                if 'options_implied_volatility' in data.columns:
                    data['options_iv_percentile'] = data['options_implied_volatility'].rolling(window=252).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else None
                    )
            
            # Add insider trading features
            if ticker in insider_data and not insider_data[ticker].empty:
                insider_df = insider_data[ticker]
                
                # Convert to time series of buy/sell volumes
                if not insider_df.empty:
                    insider_df['date'] = pd.to_datetime(insider_df['date'])
                    insider_df = insider_df.set_index('date')
                    
                    # Create buy and sell series
                    buy_mask = insider_df['transaction_type'] == 'BUY'
                    if any(buy_mask):
                        buys = insider_df[buy_mask]['value'].resample('D').sum()
                    else:
                        buys = pd.Series(0, index=pd.date_range(start=data.index[0], end=data.index[-1], freq='D'))
                        
                    sell_mask = insider_df['transaction_type'] == 'SELL'
                    if any(sell_mask):
                        sells = insider_df[sell_mask]['value'].resample('D').sum()
                    else:
                        sells = pd.Series(0, index=pd.date_range(start=data.index[0], end=data.index[-1], freq='D'))
                    
                    # Calculate net buying/selling
                    net_volume = buys.sub(sells, fill_value=0)
                    
                    # Reindex to match market data
                    net_volume = net_volume.reindex(data.index, fill_value=0)
                    
                    # Add feature to data
                    data['insider_net_volume'] = net_volume
                    
                    # Create cumulative and rolling features
                    data['insider_net_volume_30d'] = data['insider_net_volume'].rolling(window=30).sum()
                    data['insider_net_volume_90d'] = data['insider_net_volume'].rolling(window=90).sum()
            
            # Store the enhanced data
            enhanced_data[ticker] = data
            
            # Save to CSV
            csv_path = os.path.join(self.data_dir, f'{ticker}_enhanced_data.csv')
            data.to_csv(csv_path)
            logger.info(f"Saved enhanced data for {ticker} to {csv_path}")
        
        return enhanced_data


# Example usage
if __name__ == "__main__":
    # Initialize the alternative data processor
    alt_data = AlternativeDataProcessor()
    
    # Define tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Create sample market data
    market_data = {}
    for ticker in tickers:
        # Generate sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        prices = np.random.normal(100, 1, size=len(dates)).cumsum() + 100
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.99, 1.0, size=len(dates)),
            'High': prices * np.random.uniform(1.0, 1.02, size=len(dates)),
            'Low': prices * np.random.uniform(0.98, 0.99, size=len(dates)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, size=len(dates))
        }, index=dates)
        
        df['Adj Close'] = df['Close'] # Add Adj Close column
        
        market_data[ticker] = df
    
    # Create alternative data features
    enhanced_data = alt_data.create_alternative_data_features(market_data, tickers)
    
    # Print some features
    ticker = tickers[0]
    print(f"\nAlternative Data Features for {ticker}:")
    print(enhanced_data[ticker].columns.tolist())
    
    if 'social_sentiment' in enhanced_data[ticker].columns:
        print("\nSocial Sentiment Summary:")
        print(enhanced_data[ticker][['social_sentiment', 'social_volume']].describe())
    
    if 'options_put_call_ratio' in enhanced_data[ticker].columns:
        print("\nOptions Data Summary:")
        options_cols = [col for col in enhanced_data[ticker].columns if col.startswith('options_')]
        print(enhanced_data[ticker][options_cols].describe())
    
    if 'insider_net_volume' in enhanced_data[ticker].columns:
        print("\nInsider Trading Summary:")
        insider_cols = [col for col in enhanced_data[ticker].columns if col.startswith('insider_')]
        print(enhanced_data[ticker][insider_cols].describe())