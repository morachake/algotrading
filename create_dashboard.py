import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

# Import core modules
from core.data_fetcher import fetch_market_data
from core.ml_trading_model import MLTradingModel
from core.backtest_system import Backtester

def create_interactive_dashboard(ticker='AAPL', days=365, save_html=True):
    """
    Create an interactive dashboard for backtest results
    
    Parameters:
    ticker (str): Ticker symbol
    days (int): Number of days of history to use
    save_html (bool): Whether to save the dashboard as HTML
    
    Returns:
    str: Path to the saved HTML file (if save_html=True)
    """
    print(f"Creating dashboard for {ticker} with {days} days of data...")
    
    # Set up dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Create directories
    os.makedirs('data/market', exist_ok=True)
    os.makedirs('results/dashboard', exist_ok=True)
    
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
    
    # Train and generate signals
    training_results = ml_model.train(processed_data, features)
    signals = ml_model.predict(processed_data)
    
    # Step 3: Run backtest
    print("Running backtest...")
    backtester = Backtester()
    results = backtester.run_backtest(data, signals)
    backtest_data = results['backtest_data']
    
    # Step 4: Create interactive dashboard
    print("Creating interactive dashboard...")
    
    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"type": "pie"}, {"type": "bar"}],
            [{"colspan": 2}, None]
        ],
        subplot_titles=(
            f"{ticker} Stock Price with Trading Signals",
            "Portfolio Value vs Buy & Hold",
            "Signal Distribution", "Monthly Returns",
            "Drawdown"
        ),
        vertical_spacing=0.1,
        row_heights=[0.3, 0.25, 0.2, 0.25]
    )
    
    # Add stock price with trading signals
    fig.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=backtest_data['Adj Close'],
            mode='lines',
            name='Stock Price',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    
    # Add buy signals
    buy_signals = backtest_data[backtest_data['Signal'] == 1]
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Adj Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up'
            )
        ),
        row=1, col=1
    )
    
    # Add sell signals
    sell_signals = backtest_data[backtest_data['Signal'] == -1]
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Adj Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down'
            )
        ),
        row=1, col=1
    )
    
    # Create buy & hold baseline for comparison
    initial_capital = results['initial_cash']
    initial_price = backtest_data['Adj Close'].iloc[0]
    shares = initial_capital / initial_price
    buy_hold_value = shares * backtest_data['Adj Close']
    
    # Add portfolio value vs buy & hold comparison
    fig.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=backtest_data['Portfolio_Value'],
            mode='lines',
            name='Strategy',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=buy_hold_value,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='gray', width=1.5, dash='dot')
        ),
        row=2, col=1
    )
    
    # Add signal distribution pie chart
    signal_counts = backtest_data['Signal'].value_counts()
    signal_labels = {1: 'Buy', 0: 'Hold', -1: 'Sell'}
    signal_names = [signal_labels.get(s, str(s)) for s in signal_counts.index]
    
    fig.add_trace(
        go.Pie(
            labels=signal_names,
            values=signal_counts.values,
            name='Signal Distribution',
            marker_colors=['green', 'gray', 'red'],
            textinfo='percent+label'
        ),
        row=3, col=1
    )
    
    # Add monthly returns bar chart
    monthly_returns = backtest_data['Strategy_Returns'].resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    fig.add_trace(
        go.Bar(
            x=monthly_returns.index,
            y=monthly_returns.values * 100,  # Convert to percentage
            name='Monthly Returns',
            marker_color=np.where(monthly_returns > 0, 'green', 'red')
        ),
        row=3, col=2
    )
    
    # Add drawdown chart
    fig.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=backtest_data['Drawdown'] * 100,  # Convert to percentage
            mode='lines',
            name='Drawdown',
            line=dict(color='red')
        ),
        row=4, col=1
    )
    
    # Add horizontal line at zero for drawdown
    fig.add_trace(
        go.Scatter(
            x=[backtest_data.index[0], backtest_data.index[-1]],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False
        ),
        row=4, col=1
    )
    
    # Add summary statistics as annotations
    summary_text = (
        f"<b>Performance Summary:</b><br>"
        f"Total Return: {results['total_return_pct']:.2f}%<br>"
        f"Annual Return: {results['annual_return_pct']:.2f}%<br>"
        f"Sharpe Ratio: {results['sharpe_ratio']:.2f}<br>"
        f"Max Drawdown: {results['max_drawdown_pct']:.2f}%<br>"
        f"Period: {start_date} to {end_date}"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=summary_text,
        showarrow=False,
        font=dict(size=14),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title=f"<b>{ticker} Algorithmic Trading Dashboard</b>",
        height=1200,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Update yaxis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=3, col=2)
    fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)
    
    # Save the dashboard
    if save_html:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = f"results/dashboard/{ticker}_dashboard_{timestamp}.html"
        
        pio.write_html(fig, file=html_path, auto_open=False)
        print(f"Dashboard saved to {html_path}")
        return html_path
    else:
        # Display the dashboard
        fig.show()
        return None

if __name__ == "__main__":
    # Create dashboard for AAPL
    create_interactive_dashboard('AAPL', days=365)