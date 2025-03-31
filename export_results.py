import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Import core modules
from core.data_fetcher import fetch_market_data
from core.ml_trading_model import MLTradingModel
from core.backtest_system import Backtester

def export_backtest_to_excel(ticker='AAPL', days=365):
    """
    Run a backtest and export the results to Excel
    
    Parameters:
    ticker (str): Ticker symbol
    days (int): Number of days of history to use
    """
    print(f"Running backtest for {ticker} with {days} days of data...")
    
    # Set up dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Create directories
    os.makedirs('data/market', exist_ok=True)
    os.makedirs('results/excel', exist_ok=True)
    
    # Step 1: Fetch market data
    print("Fetching market data...")
    market_data = fetch_market_data([ticker], start_date, end_date)
    data = market_data.get(ticker)
    
    if data is None or data.empty:
        print(f"Error: No data available for {ticker}")
        return
    
    # Step 2: Generate trading signals
    print("Generating trading signals...")
    ml_model = MLTradingModel()
    processed_data = ml_model.prepare_features(data)
    
    # Use features that don't need as much history
    features = [
        'Returns', 'Price_to_SMA20', 'Volume_Change',
        'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3'
    ]
    
    ml_model.train(processed_data, features)
    signals = ml_model.predict(processed_data)
    
    # Step 3: Run backtest
    print("Running backtest...")
    backtester = Backtester()
    results = backtester.run_backtest(data, signals)
    backtest_data = results['backtest_data']
    
    # Step 4: Export to Excel
    print("Exporting results to Excel...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_path = f"results/excel/{ticker}_backtest_{timestamp}.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Summary sheet
        summary = pd.DataFrame({
            'Metric': [
                'Ticker',
                'Start Date',
                'End Date',
                'Initial Capital',
                'Final Value',
                'Total Return (%)',
                'Annual Return (%)',
                'Sharpe Ratio',
                'Max Drawdown (%)',
                'Buy Signals',
                'Sell Signals',
                'Hold Signals'
            ],
            'Value': [
                ticker,
                start_date,
                end_date,
                f"${results['initial_cash']:.2f}",
                f"${results['final_value']:.2f}",
                f"{results['total_return_pct']:.2f}%",
                f"{results['annual_return_pct']:.2f}%",
                f"{results['sharpe_ratio']:.2f}",
                f"{results['max_drawdown_pct']:.2f}%",
                (signals['Signal'] == 1).sum(),
                (signals['Signal'] == -1).sum(),
                (signals['Signal'] == 0).sum()
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format columns for different data types
        worksheet = writer.sheets['Summary']
        worksheet.column_dimensions['A'].width = 20
        worksheet.column_dimensions['B'].width = 25
        
        # Trading Signals sheet
        signals_df = signals[['Adj Close', 'Probability', 'Signal']].copy()
        signals_df['Signal Text'] = signals_df['Signal'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
        signals_df.to_excel(writer, sheet_name='Trading Signals')
        
        # Format signal sheet
        worksheet = writer.sheets['Trading Signals']
        worksheet.column_dimensions['A'].width = 20
        worksheet.column_dimensions['B'].width = 15
        worksheet.column_dimensions['C'].width = 15
        worksheet.column_dimensions['D'].width = 10
        worksheet.column_dimensions['E'].width = 15
        
        # Backtest Results sheet
        backtest_columns = [
            'Adj Close', 'Signal', 'Position', 'Cash', 'Holdings', 
            'Portfolio_Value', 'Returns', 'Strategy_Returns', 'Cumulative_Returns', 'Drawdown'
        ]
        backtest_display = backtest_data[backtest_columns].copy()
        backtest_display.to_excel(writer, sheet_name='Backtest Results')
        
        # Add trade list sheet (entries and exits)
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for date, row in backtest_data.iterrows():
            # Entry
            if row['Position'] > 0 and position == 0:
                position = row['Position']
                entry_price = row['Adj Close']
                entry_date = date
            # Exit
            elif row['Position'] == 0 and position > 0:
                exit_price = row['Adj Close']
                profit = (exit_price - entry_price) * position
                profit_pct = (exit_price / entry_price - 1) * 100
                
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': date,
                    'Holding Period': (date - entry_date).days,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Shares': position,
                    'Profit/Loss': profit,
                    'Return (%)': profit_pct,
                })
                
                position = 0
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df.to_excel(writer, sheet_name='Trades')
            
            # Format trades sheet
            worksheet = writer.sheets['Trades']
            for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                worksheet.column_dimensions[col].width = 15
    
    print(f"Results exported to {excel_path}")
    return excel_path

if __name__ == "__main__":
    # Export AAPL backtest to Excel
    export_backtest_to_excel('AAPL', days=365)