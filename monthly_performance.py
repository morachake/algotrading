import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import calendar

# Import core modules
from core.data_fetcher import fetch_market_data
from core.ml_trading_model import MLTradingModel
from core.backtest_system import Backtester

def generate_monthly_performance_report(ticker='AAPL', years=1):
    """
    Generate a monthly performance report for a trading strategy
    
    Parameters:
    ticker (str): Ticker symbol
    years (int): Number of years of history to use
    
    Returns:
    DataFrame: Monthly performance report
    """
    print(f"Generating monthly performance report for {ticker} ({years} years)...")
    
    # Set up dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    
    # Create directories
    os.makedirs('data/market', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    # Step 1: Fetch market data
    print("Fetching market data...")
    market_data = fetch_market_data([ticker], start_date, end_date)
    data = market_data.get(ticker)
    
    if data is None or data.empty:
        print(f"Error: No data available for {ticker}")
        return None
    
    # Step 2: Generate trading signals
    print("Generating trading signals...")
    ml_model = MLTradingModel()
    processed_data = ml_model.prepare_features(data)
    
    # Use features that don't need as much history
    features = [
        'Returns', 'Price_to_SMA20', 'Volume_Change',
        'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3'
    ]
    
    # Train and generate signals
    ml_model.train(processed_data, features)
    signals = ml_model.predict(processed_data)
    
    # Step 3: Run backtest
    print("Running backtest...")
    backtester = Backtester()
    results = backtester.run_backtest(data, signals)
    backtest_data = results['backtest_data']
    
    # Step 4: Calculate monthly performance
    print("Calculating monthly performance...")
    
    # Calculate monthly returns for the strategy
    strategy_monthly = backtest_data['Strategy_Returns'].resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Calculate monthly returns for buy & hold
    backtest_data['Stock_Returns'] = backtest_data['Adj Close'].pct_change()
    stock_monthly = backtest_data['Stock_Returns'].resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Create monthly comparison DataFrame
    monthly_data = pd.DataFrame({
        'Strategy': strategy_monthly * 100,  # Convert to percentage
        'Buy&Hold': stock_monthly * 100,     # Convert to percentage
        'Outperformance': (strategy_monthly - stock_monthly) * 100  # Difference in percentage
    })
    
    # Add month and year columns
    monthly_data['Year'] = monthly_data.index.year
    monthly_data['Month'] = monthly_data.index.month
    monthly_data['MonthName'] = monthly_data.index.strftime('%b')
    
    # Create a pivot table for the heatmap
    pivot_strategy = pd.pivot_table(
        monthly_data, 
        values='Strategy', 
        index='Year', 
        columns='MonthName',
        aggfunc='first'
    )
    
    pivot_outperformance = pd.pivot_table(
        monthly_data, 
        values='Outperformance', 
        index='Year', 
        columns='MonthName',
        aggfunc='first'
    )
    
    # Order the months correctly
    month_order = [calendar.month_abbr[i] for i in range(1, 13)]
    pivot_strategy = pivot_strategy.reindex(columns=month_order)
    pivot_outperformance = pivot_outperformance.reindex(columns=month_order)
    
    # Calculate statistics
    monthly_stats = pd.DataFrame({
        'Strategy Avg': monthly_data.groupby('MonthName')['Strategy'].mean(),
        'Strategy Win%': monthly_data.groupby('MonthName')['Strategy'].apply(lambda x: (x > 0).mean() * 100),
        'Outperformance Avg': monthly_data.groupby('MonthName')['Outperformance'].mean(),
        'Outperformance Win%': monthly_data.groupby('MonthName')['Outperformance'].apply(lambda x: (x > 0).mean() * 100)
    })
    
    # Reindex to correct month order
    monthly_stats = monthly_stats.reindex(month_order)
    
    # Calculate year performance
    yearly_data = pd.DataFrame({
        'Strategy': monthly_data.groupby('Year')['Strategy'].apply(
            lambda x: (1 + x/100).prod() - 1
        ) * 100,
        'Buy&Hold': monthly_data.groupby('Year')['Buy&Hold'].apply(
            lambda x: (1 + x/100).prod() - 1
        ) * 100
    })
    yearly_data['Outperformance'] = yearly_data['Strategy'] - yearly_data['Buy&Hold']
    
    # Plot monthly heatmap
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.title(f"{ticker} - Monthly Strategy Returns (%)", fontsize=14)
    
    # Create heatmap for strategy returns
    ax = plt.gca()
    cmap = plt.cm.RdYlGn  # Red for negative, Yellow for neutral, Green for positive
    
    # Create heatmap with a centered colormap
    vmin = min(-10, pivot_strategy.min().min())
    vmax = max(10, pivot_strategy.max().max())
    heatmap = ax.pcolor(pivot_strategy, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Returns (%)')
    
    # Set labels
    ax.set_yticks(np.arange(0.5, len(pivot_strategy.index), 1))
    ax.set_yticklabels(pivot_strategy.index)
    ax.set_xticks(np.arange(0.5, len(pivot_strategy.columns), 1))
    ax.set_xticklabels(pivot_strategy.columns)
    
    # Add text annotations
    for i in range(len(pivot_strategy.index)):
        for j in range(len(pivot_strategy.columns)):
            value = pivot_strategy.iloc[i, j]
            if pd.notna(value):
                ax.text(j + 0.5, i + 0.5, f"{value:.1f}%",
                       ha="center", va="center",
                       color="black" if abs(value) < 20 else "white")
    
    plt.subplot(2, 1, 2)
    plt.title(f"{ticker} - Monthly Outperformance vs Buy & Hold (%)", fontsize=14)
    
    # Create heatmap for outperformance
    ax = plt.gca()
    
    # Create heatmap with a centered colormap
    vmin = min(-5, pivot_outperformance.min().min())
    vmax = max(5, pivot_outperformance.max().max())
    heatmap = ax.pcolor(pivot_outperformance, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Outperformance (%)')
    
    # Set labels
    ax.set_yticks(np.arange(0.5, len(pivot_outperformance.index), 1))
    ax.set_yticklabels(pivot_outperformance.index)
    ax.set_xticks(np.arange(0.5, len(pivot_outperformance.columns), 1))
    ax.set_xticklabels(pivot_outperformance.columns)
    
    # Add text annotations
    for i in range(len(pivot_outperformance.index)):
        for j in range(len(pivot_outperformance.columns)):
            value = pivot_outperformance.iloc[i, j]
            if pd.notna(value):
                ax.text(j + 0.5, i + 0.5, f"{value:.1f}%",
                       ha="center", va="center",
                       color="black" if abs(value) < 10 else "white")
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"results/reports/{ticker}_monthly_performance_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Monthly performance chart saved to {plot_path}")
    
    # Save monthly data to CSV
    csv_path = f"results/reports/{ticker}_monthly_performance_{timestamp}.csv"
    monthly_data.to_csv(csv_path)
    print(f"Monthly performance data saved to {csv_path}")
    
    # Save monthly stats to CSV
    stats_path = f"results/reports/{ticker}_monthly_stats_{timestamp}.csv"
    monthly_stats.to_csv(stats_path)
    print(f"Monthly statistics saved to {stats_path}")
    
    # Save yearly performance to CSV
    yearly_path = f"results/reports/{ticker}_yearly_performance_{timestamp}.csv"
    yearly_data.to_csv(yearly_path)
    print(f"Yearly performance data saved to {yearly_path}")
    
    return monthly_data, pivot_strategy, monthly_stats, yearly_data

if __name__ == "__main__":
    # Generate monthly performance report for AAPL
    generate_monthly_performance_report('AAPL', years=1)