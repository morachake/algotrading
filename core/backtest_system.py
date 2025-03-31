import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Set up logger
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self):
        """Initialize the backtester"""
        self.results = None
    
    def run_backtest(self, price_data, signals, initial_cash=100000.0):
        """
        Run a simple backtest using price data and signals
        
        Parameters:
        price_data (DataFrame): Historical price data
        signals (DataFrame): Trading signals
        initial_cash (float): Initial cash amount
        
        Returns:
        dict: Backtest results and performance metrics
        """
        # Create a copy of price data
        backtest_data = price_data.copy()
        
        # Check if we have the required columns
        if 'Adj Close' not in backtest_data.columns:
            if 'Close' in backtest_data.columns:
                backtest_data['Adj Close'] = backtest_data['Close']
                print("Added 'Adj Close' column (copy of 'Close')")
            else:
                raise ValueError("Neither 'Adj Close' nor 'Close' column found in data")
        
        # Align signals with price data
        aligned_signals = signals.reindex(backtest_data.index).fillna(0)
        backtest_data = backtest_data.assign(Signal=aligned_signals['Signal'])
        
        # Initialize portfolio metrics columns
        backtest_data = backtest_data.assign(
            Position=0,
            Cash=initial_cash,
            Holdings=0,
            Portfolio_Value=initial_cash
        )
        
        # Calculate daily returns
        backtest_data = backtest_data.assign(Returns=backtest_data['Adj Close'].pct_change())
        
        # Iterate through data and apply trading logic
        for i in range(1, len(backtest_data)):
            # Get previous and current signals
            prev_signal = backtest_data['Signal'].iloc[i-1]
            curr_signal = backtest_data['Signal'].iloc[i]
            
            # Get previous position, cash, and holdings
            prev_position = backtest_data['Position'].iloc[i-1]
            prev_cash = backtest_data['Cash'].iloc[i-1]
            prev_holdings = backtest_data['Holdings'].iloc[i-1]
            
            # Default to previous values
            position = prev_position
            cash = prev_cash
            holdings = prev_holdings
            
            # Get current price
            price = backtest_data['Adj Close'].iloc[i]
            
            # Check for changes in signal
            if prev_signal != curr_signal:
                # Buy signal
                if curr_signal == 1 and prev_position == 0:
                    # Calculate shares to buy (use 90% of cash)
                    cash_to_use = prev_cash * 0.9
                    shares = int(cash_to_use / price)
                    
                    if shares > 0:
                        cost = shares * price
                        cash = prev_cash - cost
                        holdings = shares * price
                        position = shares
                
                # Sell signal
                elif curr_signal == -1 and prev_position > 0:
                    # Sell all shares
                    shares = prev_position
                    proceeds = shares * price
                    cash = prev_cash + proceeds
                    holdings = 0
                    position = 0
            else:
                # Update holdings value if position is unchanged
                if prev_position > 0:
                    holdings = prev_position * price
            
            # Update backtest data using .loc
            backtest_data.loc[backtest_data.index[i], 'Position'] = position
            backtest_data.loc[backtest_data.index[i], 'Cash'] = cash
            backtest_data.loc[backtest_data.index[i], 'Holdings'] = holdings
        
        # Calculate portfolio value
        backtest_data = backtest_data.assign(Portfolio_Value=backtest_data['Cash'] + backtest_data['Holdings'])
        
        # Calculate strategy returns
        backtest_data = backtest_data.assign(Strategy_Returns=backtest_data['Portfolio_Value'].pct_change())
        
        # Calculate cumulative returns
        backtest_data = backtest_data.assign(Cumulative_Returns=(1 + backtest_data['Strategy_Returns']).cumprod() - 1)
        
        # Calculate drawdowns
        backtest_data = backtest_data.assign(Cumulative_Max=backtest_data['Cumulative_Returns'].cummax())
        backtest_data = backtest_data.assign(Drawdown=backtest_data['Cumulative_Returns'] - backtest_data['Cumulative_Max'])
        
        # Calculate performance metrics
        total_return = (backtest_data['Portfolio_Value'].iloc[-1] / initial_cash) - 1
        annual_return = ((1 + total_return) ** (252 / len(backtest_data))) - 1
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = backtest_data['Strategy_Returns'].mean() / backtest_data['Strategy_Returns'].std() * np.sqrt(252)
        
        # Calculate maximum drawdown
        max_drawdown = backtest_data['Drawdown'].min()
        
        # Print results
        print("\nBacktest Results:")
        print(f"Starting Portfolio Value: ${initial_cash:.2f}")
        print(f"Final Portfolio Value: ${backtest_data['Portfolio_Value'].iloc[-1]:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        
        # Save backtest data
        self.results = backtest_data
        
        # Return performance metrics
        return {
            'initial_cash': initial_cash,
            'final_value': backtest_data['Portfolio_Value'].iloc[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annual_return': annual_return,
            'annual_return_pct': annual_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'backtest_data': backtest_data
        }
    
    def plot_results(self, save_path=None):
        """
        Plot backtest results
        
        Parameters:
        save_path (str): Path to save the plot, if None, the plot will be displayed
        
        Returns:
        None
        """
        if self.results is None:
            print("No backtest results available. Run a backtest first.")
            return
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot portfolio value
        self.results['Portfolio_Value'].plot(ax=axes[0], title='Portfolio Value')
        axes[0].set_ylabel('Value ($)')
        axes[0].grid(True)
        
        # Plot drawdown
        self.results['Drawdown'].plot(ax=axes[1], title='Drawdown', color='red')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True)
        
        # Plot position
        self.results['Position'].plot(ax=axes[2], title='Position Size', color='green')
        axes[2].set_ylabel('Shares')
        axes[2].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def run_pyfolio_analysis(self, price_data, signals, initial_cash=100000.0):
        """
        Run a detailed backtest with PyFolio-style analysis
        
        Parameters:
        price_data (DataFrame): Historical price data
        signals (DataFrame): Trading signals
        initial_cash (float): Initial cash amount
        
        Returns:
        dict: Performance metrics and tearsheet
        """
        # Create a simple portfolio DataFrame
        portfolio = price_data.copy()
        
        # Initialize position and holdings columns
        portfolio = portfolio.assign(
            position=0,
            holdings=0,
            cash=initial_cash,
            total=initial_cash
        )
        
        # Process signals
        for date, row in signals.iterrows():
            if date in portfolio.index:
                signal = row['Signal']
                
                # Get the price for this date
                if 'Close' in portfolio.columns:
                    price = portfolio.loc[date, 'Close']
                else:
                    price = portfolio.loc[date, 'Adj Close']
                
                # Update position based on signal
                if signal == 1:  # Buy signal
                    portfolio.loc[date:, 'position'] = 1
                elif signal == -1:  # Sell signal
                    portfolio.loc[date:, 'position'] = 0
        
        # Calculate holdings and portfolio value
        for i, (idx, row) in enumerate(portfolio.iterrows()):
            if 'Close' in portfolio.columns:
                portfolio.at[idx, 'holdings'] = row['position'] * row['Close']
            else:
                portfolio.at[idx, 'holdings'] = row['position'] * row['Adj Close']
        
        # Calculate cash (assumes we buy/sell 1 share at a time for simplicity)
        cash = initial_cash
        for i in range(1, len(portfolio)):
            prev_idx = portfolio.index[i-1]
            curr_idx = portfolio.index[i]
            
            prev_pos = portfolio.at[prev_idx, 'position']
            curr_pos = portfolio.at[curr_idx, 'position']
            
            # If position changed, update cash
            if prev_pos != curr_pos:
                if 'Close' in portfolio.columns:
                    price = portfolio.at[curr_idx, 'Close']
                else:
                    price = portfolio.at[curr_idx, 'Adj Close']
                    
                if curr_pos > prev_pos:  # Bought
                    cash -= price
                else:  # Sold
                    cash += price
            
            portfolio.at[curr_idx, 'cash'] = cash
        
        # Calculate total portfolio value
        portfolio = portfolio.assign(total=portfolio['holdings'] + portfolio['cash'])
        
        # Calculate returns
        portfolio = portfolio.assign(returns=portfolio['total'].pct_change())
        
        # Extract daily returns for analysis
        returns = portfolio['returns'].dropna()
        
        # Calculate performance metrics
        total_return = (portfolio['total'].iloc[-1] - initial_cash) / initial_cash
        annual_return = ((1 + total_return) ** (252 / len(portfolio)) - 1)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe
        
        # Maximum drawdown
        portfolio = portfolio.assign(peak=portfolio['total'].cummax())
        portfolio = portfolio.assign(drawdown=(portfolio['total'] - portfolio['peak']) / portfolio['peak'])
        max_drawdown = portfolio['drawdown'].min()
        
        # Plot performance
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        if 'Close' in portfolio.columns:
            portfolio['Close'].plot(title='Price')
        else:
            portfolio['Adj Close'].plot(title='Price')
        
        # Plot buy and sell signals
        buys = portfolio[portfolio['position'] > portfolio['position'].shift(1)].index
        sells = portfolio[portfolio['position'] < portfolio['position'].shift(1)].index
        
        if 'Close' in portfolio.columns:
            plt.plot(buys, portfolio.loc[buys, 'Close'], '^', markersize=10, color='g', label='Buy')
            plt.plot(sells, portfolio.loc[sells, 'Close'], 'v', markersize=10, color='r', label='Sell')
        else:
            plt.plot(buys, portfolio.loc[buys, 'Adj Close'], '^', markersize=10, color='g', label='Buy')
            plt.plot(sells, portfolio.loc[sells, 'Adj Close'], 'v', markersize=10, color='r', label='Sell')
            
        plt.legend()
        
        plt.subplot(3, 1, 2)
        portfolio['total'].plot(title='Portfolio Value')
        
        plt.subplot(3, 1, 3)
        portfolio['drawdown'].plot(title='Drawdown')
        
        plt.tight_layout()
        plt.savefig('backtest_performance.png')
        plt.close()
        
        return {
            'initial_cash': initial_cash,
            'final_value': portfolio['total'].iloc[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annual_return': annual_return,
            'annual_return_pct': annual_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100
        }