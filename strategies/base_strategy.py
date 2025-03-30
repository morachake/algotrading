from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

# Get logger
logger = logging.getLogger('strategies')

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self, name='base'):
        """
        Initialize the strategy
        
        Parameters:
        name (str): Strategy name
        """
        self.name = name
        self.is_initialized = False
        self.is_trained = False
    
    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize the strategy with parameters
        
        Parameters:
        **kwargs: Strategy-specific parameters
        
        Returns:
        bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def train(self, data, **kwargs):
        """
        Train the strategy using historical data
        
        Parameters:
        data (DataFrame): Historical market data
        **kwargs: Training parameters
        
        Returns:
        dict: Training results
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data, **kwargs):
        """
        Generate trading signals
        
        Parameters:
        data (DataFrame): Market data
        **kwargs: Signal generation parameters
        
        Returns:
        DataFrame: Trading signals
        """
        pass
    
    def validate_data(self, data):
        """
        Validate that input data has required columns
        
        Parameters:
        data (DataFrame): Market data
        
        Returns:
        bool: True if data is valid, False otherwise
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Required column '{col}' not found in data")
                return False
        
        return True
    
    def prepare_data(self, data):
        """
        Prepare data for strategy (calculate basic indicators)
        
        Parameters:
        data (DataFrame): Market data
        
        Returns:
        DataFrame: Processed data
        """
        # Validate data
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        # Make a copy to avoid modifying the original
        processed = data.copy()
        
        # Calculate returns
        processed['Returns'] = processed['Adj Close'].pct_change()
        
        # Calculate simple moving averages
        processed['SMA_20'] = processed['Adj Close'].rolling(window=20).mean()
        processed['SMA_50'] = processed['Adj Close'].rolling(window=50).mean()
        
        # Calculate price to moving average ratio
        processed['Price_to_SMA20'] = processed['Adj Close'] / processed['SMA_20'] - 1
        processed['Price_to_SMA50'] = processed['Adj Close'] / processed['SMA_50'] - 1
        
        # SMA crossover
        processed['SMA_Cross'] = (processed['SMA_20'] > processed['SMA_50']).astype(int)
        
        # Volatility (standard deviation of returns)
        processed['Volatility'] = processed['Returns'].rolling(window=20).std()
        
        # Return lags
        for i in range(1, 6):
            processed[f'Return_Lag_{i}'] = processed['Returns'].shift(i)
        
        # Volume features
        processed['Volume_Change'] = processed['Volume'].pct_change()
        processed['Volume_SMA_5'] = processed['Volume'].rolling(window=5).mean()
        processed['Volume_Ratio'] = processed['Volume'] / processed['Volume_SMA_5']
        
        # RSI (Relative Strength Index)
        delta = processed['Adj Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        processed['RSI'] = 100 - (100 / (1 + rs))
        
        return processed
    
    def evaluate_signals(self, signals, data):
        """
        Evaluate the performance of signals
        
        Parameters:
        signals (DataFrame): Generated signals
        data (DataFrame): Market data with actual prices
        
        Returns:
        dict: Performance metrics
        """
        if 'Signal' not in signals.columns:
            logger.error("No 'Signal' column found in signals DataFrame")
            return {}
        
        # Align signals with data
        signals = signals.reindex(data.index).fillna(0)
        
        # Calculate returns
        data['Returns'] = data['Adj Close'].pct_change()
        
        # Calculate strategy returns (assuming positions are held until next signal)
        data['Strategy_Returns'] = signals['Signal'].shift(1) * data['Returns']
        
        # Remove NaN values
        returns = data['Returns'].dropna()
        strategy_returns = data['Strategy_Returns'].dropna()
        
        # Calculate performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        
        # Calculate volatility
        volatility = strategy_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = (1 + strategy_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        trades = strategy_returns[strategy_returns != 0]
        wins = (trades > 0).sum()
        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
    
    def calculate_signal_strength(self, probability):
        """
        Calculate signal strength based on probability
        
        Parameters:
        probability (float): Signal probability
        
        Returns:
        int: Signal strength (-1, 0, or 1)
        """
        if probability > 0.7:
            return 1  # Strong buy
        elif probability < 0.3:
            return -1  # Strong sell
        else:
            return 0  # Hold
    
    def __str__(self):
        """String representation of the strategy"""
        return f"{self.name} Strategy"