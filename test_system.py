import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import core modules
from core.data_fetcher import fetch_market_data
from core.ml_trading_model import MLTradingModel
from core.sentiment_analyzer import FinancialSentimentAnalyzer
from core.backtest_system import Backtester

def run_quick_test():
    """Run a quick test of the system with minimal data"""
    print("Running quick test of the algorithmic trading system...")
    
    # Create directories
    os.makedirs('data/market', exist_ok=True)
    os.makedirs('results/backtest', exist_ok=True)
    
    # Define parameters - using a short date range for quick testing
    ticker = 'AAPL'  # Just one ticker for speed
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"Testing period: {start_date} to {end_date}")
    
    # Step 1: Fetch market data
    print("\nStep 1: Fetching market data...")
    market_data = fetch_market_data([ticker], start_date, end_date)
    
    if ticker not in market_data:
        print(f"Error: Could not fetch data for {ticker}")
        return
    
    data = market_data[ticker]
    print(f"Fetched {len(data)} rows of data for {ticker}")
    print(f"Data columns: {data.columns.tolist()}")
    print(data.head())
    
    # Step 2: Prepare features and train model
    print("\nStep 2: Training model...")
    ml_model = MLTradingModel()
    
    # Check if data has the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Warning: Data is missing these columns: {missing_columns}")
        
    processed_data = ml_model.prepare_features(data)
    print(f"Prepared data with {len(processed_data)} rows")
    
    if len(processed_data) < 20:
        print("Warning: Not enough data to train a model effectively. Need at least 20 rows.")
        return
    
    # Train the model
    train_results = ml_model.train(processed_data)
    print(f"Model trained with test accuracy: {train_results['test_accuracy']:.4f}")
    
    # Step 3: Generate trading signals
    print("\nStep 3: Generating trading signals...")
    signals = ml_model.predict(processed_data)
    print(f"Generated {len(signals)} signals")
    print(signals[['Adj Close', 'Probability', 'Signal']].tail())
    
    # Count signal types
    buy_signals = (signals['Signal'] == 1).sum()
    sell_signals = (signals['Signal'] == -1).sum()
    hold_signals = (signals['Signal'] == 0).sum()
    print(f"Signal breakdown: Buy={buy_signals}, Sell={sell_signals}, Hold={hold_signals}")
    
    # Step 4: Run backtest
    print("\nStep 4: Running backtest...")
    backtester = Backtester()
    results = backtester.run_backtest(data, signals)
    
    # Step 5: Plot backtest results
    print("\nStep 5: Plotting results...")
    backtester.plot_results(save_path=f'results/backtest/{ticker}_test_backtest.png')
    
    print(f"\nBacktest results saved to results/backtest/{ticker}_test_backtest.png")
    print("\nQuick test completed successfully!")
    
    return results

if __name__ == "__main__":
    # Run the quick test
    run_quick_test()