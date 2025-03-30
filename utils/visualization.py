import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

def plot_backtest_results(returns, results, ticker, strategy_name):
    """
    Plot backtest results
    
    Parameters:
    returns (Series): Daily returns
    results (dict): Backtest results
    ticker (str): Ticker symbol
    strategy_name (str): Strategy name
    
    Returns:
    Figure: Matplotlib figure
    """
    # Calculate cumulative returns
    if isinstance(returns, pd.Series):
        cum_returns = (1 + returns).cumprod() - 1
    else:
        cum_returns = pd.Series(index=returns.index)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot equity curve
    cum_returns.plot(ax=axes[0], color='blue', linewidth=2)
    axes[0].set_title(f'{ticker} - {strategy_name.capitalize()} Strategy Performance', fontsize=16)
    axes[0].set_ylabel('Cumulative Return', fontsize=12)
    axes[0].grid(True)
    
    # Add key performance metrics as text
    metrics_text = (
        f"Total Return: {results.get('total_return_pct', 0):.2f}%\n"
        f"Annual Return: {results.get('annual_return_pct', 0):.2f}%\n"
        f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n"
        f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%"
    )
    
    # Position the text in the upper left with a light background
    axes[0].text(0.02, 0.95, metrics_text, transform=axes[0].transAxes,
                fontsize=12, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Plot drawdowns
    if 'drawdown' in results:
        drawdowns = results['drawdown']
        if isinstance(drawdowns, pd.Series):
            drawdowns.plot(ax=axes[1], color='red', linewidth=1.5)
            axes[1].fill_between(drawdowns.index, 0, drawdowns, color='red', alpha=0.3)
    else:
        # Calculate drawdowns from returns
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max) - 1
        drawdowns.plot(ax=axes[1], color='red', linewidth=1.5)
        axes[1].fill_between(drawdowns.index, 0, drawdowns, color='red', alpha=0.3)
    
    axes[1].set_title('Drawdowns', fontsize=14)
    axes[1].set_ylabel('Drawdown', fontsize=12)
    axes[1].grid(True)
    
    # Plot monthly returns
    if isinstance(returns, pd.Series) and len(returns) > 30:
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.plot(kind='bar', ax=axes[2], color='green', alpha=0.7)
        axes[2].set_title('Monthly Returns', fontsize=14)
        axes[2].set_ylabel('Return', fontsize=12)
        axes[2].grid(True)
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
    else:
        # If not enough data for monthly returns, plot rolling returns
        rolling_returns = returns.rolling(window=20).mean()
        rolling_returns.plot(ax=axes[2], color='green', linewidth=1.5)
        axes[2].set_title('20-Day Rolling Returns', fontsize=14)
        axes[2].set_ylabel('Return', fontsize=12)
        axes[2].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_signal_distribution(signals, ticker):
    """
    Plot distribution of trading signals
    
    Parameters:
    signals (DataFrame): Trading signals
    ticker (str): Ticker symbol
    
    Returns:
    Figure: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot signal distribution
    if 'Signal' in signals.columns:
        signal_counts = signals['Signal'].value_counts()
        signal_counts.plot(kind='bar', ax=axes[0, 0], color='blue', alpha=0.7)
        axes[0, 0].set_title('Signal Distribution', fontsize=14)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].grid(True)
    
            # Plot signal over time
    if 'Signal' in signals.columns:
        signals['Signal'].plot(ax=axes[0, 1], color='purple', marker='o', linestyle='None')
        axes[0, 1].set_title('Signals Over Time', fontsize=14)
        axes[0, 1].set_ylabel('Signal', fontsize=12)
        axes[0, 1].grid(True)
    
    # Plot price and signals
    if 'Adj Close' in signals.columns and 'Signal' in signals.columns:
        ax = axes[1, 0]
        signals['Adj Close'].plot(ax=ax, color='blue', linewidth=1.5)
        
        # Plot buy signals
        buy_signals = signals[signals['Signal'] == 1]
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, buy_signals['Adj Close'], 
                      color='green', marker='^', s=100, label='Buy')
        
        # Plot sell signals
        sell_signals = signals[signals['Signal'] == -1]
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, sell_signals['Adj Close'], 
                      color='red', marker='v', s=100, label='Sell')
        
        ax.set_title(f'{ticker} Price and Signals', fontsize=14)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend()
        ax.grid(True)
    
    # Plot signal probability if available
    if 'Probability' in signals.columns:
        signals['Probability'].plot(ax=axes[1, 1], color='orange', linewidth=1.5)
        axes[1, 1].set_title('Signal Probability', fontsize=14)
        axes[1, 1].set_ylabel('Probability', fontsize=12)
        axes[1, 1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_portfolio_performance(portfolio_history):
    """
    Plot portfolio performance over time
    
    Parameters:
    portfolio_history (DataFrame): Portfolio history dataframe
    
    Returns:
    Figure: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot portfolio value
    if 'total_value' in portfolio_history.columns:
        portfolio_history['total_value'].plot(ax=axes[0, 0], color='blue', linewidth=2)
        axes[0, 0].set_title('Portfolio Value', fontsize=14)
        axes[0, 0].set_ylabel('Value ($)', fontsize=12)
        axes[0, 0].grid(True)
    
    # Plot portfolio returns
    if 'total_value' in portfolio_history.columns:
        portfolio_returns = portfolio_history['total_value'].pct_change()
        portfolio_returns.plot(ax=axes[0, 1], color='green', linewidth=1.5)
        axes[0, 1].set_title('Daily Returns', fontsize=14)
        axes[0, 1].set_ylabel('Return (%)', fontsize=12)
        axes[0, 1].grid(True)
    
    # Plot cash vs. invested
    if 'cash' in portfolio_history.columns and 'invested_value' in portfolio_history.columns:
        portfolio_history[['cash', 'invested_value']].plot(ax=axes[1, 0], linewidth=2)
        axes[1, 0].set_title('Cash vs. Invested', fontsize=14)
        axes[1, 0].set_ylabel('Value ($)', fontsize=12)
        axes[1, 0].grid(True)
    
    # Plot profit/loss
    if 'profit_loss' in portfolio_history.columns:
        portfolio_history['profit_loss'].plot(ax=axes[1, 1], color='purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Profit/Loss', fontsize=14)
        axes[1, 1].set_ylabel('P/L ($)', fontsize=12)
        axes[1, 1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from a trained model
    
    Parameters:
    model: Trained model with feature_importances_ attribute
    feature_names (list): List of feature names
    
    Returns:
    Figure: Matplotlib figure
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort feature importance
    indices = np.argsort(importance)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bar chart
    ax.barh(range(len(sorted_importance)), sorted_importance, align='center', color='skyblue')
    ax.set_yticks(range(len(sorted_importance)))
    ax.set_yticklabels(sorted_feature_names)
    ax.set_title('Feature Importance', fontsize=16)
    ax.set_xlabel('Importance', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_correlation_heatmap(data, title='Feature Correlation'):
    """
    Create a correlation heatmap for features
    
    Parameters:
    data (DataFrame): Data with features
    title (str): Plot title
    
    Returns:
    Figure: Matplotlib figure
    """
    # Calculate correlation matrix
    corr = data.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(corr.values, cmap='coolwarm')
    
    # Set ticks
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    
    # Add colorbar
    cbar = fig.colorbar(im)
    
    # Add title
    plt.title(title, fontsize=16)
    
    # Loop over data and create text annotations
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f"{corr.values[i, j]:.2f}",
                           ha="center", va="center", color="black")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_trading_dashboard(ticker, prices, signals, backtest_results):
    """
    Create a comprehensive trading dashboard
    
    Parameters:
    ticker (str): Ticker symbol
    prices (DataFrame): Price data
    signals (DataFrame): Trading signals
    backtest_results (dict): Backtest results
    
    Returns:
    Figure: Matplotlib figure
    """
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Trading Dashboard - {ticker}", fontsize=20)
    
    # Plot 1: Price chart with signals
    ax1 = axes[0, 0]
    prices['Adj Close'].plot(ax=ax1, color='blue', linewidth=1.5)
    
    # Add buy signals
    buy_signals = signals[signals['Signal'] == 1]
    if not buy_signals.empty:
        ax1.scatter(buy_signals.index, buy_signals['Adj Close'], 
                   color='green', marker='^', s=100, label='Buy')
    
    # Add sell signals
    sell_signals = signals[signals['Signal'] == -1]
    if not sell_signals.empty:
        ax1.scatter(sell_signals.index, sell_signals['Adj Close'], 
                   color='red', marker='v', s=100, label='Sell')
    
    ax1.set_title('Price Chart with Signals', fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Performance metrics
    ax2 = axes[0, 1]
    metrics = {
        'Total Return': backtest_results.get('total_return_pct', 0),
        'Annual Return': backtest_results.get('annual_return_pct', 0),
        'Sharpe Ratio': backtest_results.get('sharpe_ratio', 0),
        'Max Drawdown': abs(backtest_results.get('max_drawdown_pct', 0))
    }
    
    # Create bar chart for metrics
    bars = ax2.bar(range(len(metrics)), list(metrics.values()), color=['blue', 'green', 'purple', 'red'])
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(list(metrics.keys()))
    ax2.set_title('Performance Metrics', fontsize=14)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Plot 3: Cumulative returns
    ax3 = axes[1, 0]
    returns = prices['Adj Close'].pct_change().dropna()
    cum_returns = (1 + returns).cumprod() - 1
    cum_returns.plot(ax=ax3, color='green', linewidth=2)
    ax3.set_title('Cumulative Returns', fontsize=14)
    ax3.set_ylabel('Return', fontsize=12)
    ax3.grid(True)
    
    # Plot 4: Drawdowns
    ax4 = axes[1, 1]
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    drawdowns.plot(ax=ax4, color='red', linewidth=1.5)
    ax4.fill_between(drawdowns.index, 0, drawdowns, color='red', alpha=0.3)
    ax4.set_title('Drawdowns', fontsize=14)
    ax4.set_ylabel('Drawdown', fontsize=12)
    ax4.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for the suptitle
    
    return fig

# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='B')
    
    # Create price data
    prices = pd.DataFrame({
        'Open': np.random.normal(100, 1, size=len(dates)).cumsum() + 100,
        'High': np.random.normal(101, 1, size=len(dates)).cumsum() + 100,
        'Low': np.random.normal(99, 1, size=len(dates)).cumsum() + 100,
        'Close': np.random.normal(100, 1, size=len(dates)).cumsum() + 100,
        'Adj Close': np.random.normal(100, 1, size=len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, size=len(dates))
    }, index=dates)
    
    # Create returns
    returns = prices['Adj Close'].pct_change().dropna()
    
    # Create signals
    signals = pd.DataFrame({
        'Adj Close': prices['Adj Close'],
        'Signal': np.random.choice([-1, 0, 1], size=len(dates), p=[0.1, 0.8, 0.1]),
        'Probability': np.random.uniform(0, 1, size=len(dates))
    }, index=dates)
    
    # Create backtest results
    backtest_results = {
        'total_return_pct': 15.5,
        'annual_return_pct': 12.2,
        'sharpe_ratio': 1.8,
        'max_drawdown_pct': -8.5,
        'win_rate': 0.65
    }
    
    # Plot backtest results
    fig1 = plot_backtest_results(returns, backtest_results, 'AAPL', 'ml')
    plt.savefig('backtest_results.png')
    
    # Plot signal distribution
    fig2 = plot_signal_distribution(signals, 'AAPL')
    plt.savefig('signal_distribution.png')
    
    # Create trading dashboard
    fig3 = create_trading_dashboard('AAPL', prices, signals, backtest_results)
    plt.savefig('trading_dashboard.png')
    
    print("Visualizations created and saved.")