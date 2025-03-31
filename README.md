# Algorithmic Trading System: Comprehensive Guide

## Overview

This algorithmic trading system provides a complete framework for developing, testing, and analyzing trading strategies using machine learning models and technical indicators. It supports multiple markets including US stocks and the Nairobi Securities Exchange (NSE).

## System Architecture

The system consists of the following key components:

1. **Data Acquisition** - Fetches historical market data from sources like Yahoo Finance or custom data sources
2. **Feature Engineering** - Creates technical indicators and other features for ML models
3. **ML Modeling** - Trains machine learning models to predict price movements
4. **Signal Generation** - Generates buy/sell signals based on model predictions
5. **Backtesting** - Tests strategies on historical data and calculates performance metrics
6. **Visualization** - Creates reports, charts, and dashboards to analyze results

## Directory Structure

```
algotrading/
├── core/
│   ├── __init__.py
│   ├── data_fetcher.py             # Market data acquisition
│   ├── ml_trading_model.py         # ML model for signals
│   ├── sentiment_analyzer.py       # News/social media analysis
│   ├── backtest_system.py          # Backtesting framework
│   ├── performance_metrics.py      # Financial metrics calculation
│   └── alternative_data.py         # Alternative data integration
├── utils/
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── logger.py                   # Logging utilities
│   └── visualization.py            # Charts and visualizations
├── data/
│   ├── market/                     # Market price data
│   ├── nse/                        # NSE-specific data
│   └── alternative/                # Alternative data storage
├── results/
│   ├── backtest/                   # Backtest results
│   ├── reports/                    # Performance reports
│   ├── dashboard/                  # Interactive dashboards
│   ├── excel/                      # Excel exports
│   └── nse/                        # NSE-specific results
├── main.py                         # Main script for US markets
├── train_models.py                 # Model training script
├── test_system.py                  # Basic system testing
├── test_longer_period.py           # Extended testing
├── export_results.py               # Export results to Excel
├── create_dashboard.py             # Create interactive dashboards
├── monthly_performance.py          # Monthly performance analysis
├── nse_trading.py                  # NSE-specific trading script
└── requirements.txt                # Dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/algotrading.git
   cd algotrading
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Python 3.10 is recommended for best compatibility
   python -m venv env_310
   
   # On Windows:
   env_310\Scripts\activate
   
   # On macOS/Linux:
   source env_310/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create required directories**:
   ```bash
   mkdir -p data/{market,nse,alternative} results/{backtest,reports,dashboard,excel,nse}
   ```

## Usage Guide

### 1. Basic Testing

Test the system with a quick run to ensure everything is set up correctly:

```bash
python test_system.py
```

This will:
- Fetch 90 days of Apple (AAPL) stock data
- Create technical indicators and features
- Train a machine learning model
- Generate trading signals
- Run a backtest
- Save results to `results/backtest/AAPL_test_backtest.png`

### 2. Extended Testing

For a more comprehensive test with longer data period:

```bash
python test_longer_period.py
```

This will:
- Fetch 1 year of Apple (AAPL) stock data
- Create a more robust feature set
- Train and evaluate the model
- Run a more comprehensive backtest
- Save results to `results/backtest/AAPL_longer_backtest.png`

### 3. Training Models

Train models for multiple stocks:

```bash
python train_models.py --tickers AAPL MSFT GOOGL --sentiment --optimize
```

Parameters:
- `--tickers`: Specify stock symbols (default: AAPL, MSFT, GOOGL)
- `--sentiment`: Include sentiment analysis (optional)
- `--alternative_data`: Include alternative data (optional)
- `--optimize`: Perform hyperparameter optimization (optional)
- `--start_date`: Specify start date (format: YYYY-MM-DD)
- `--end_date`: Specify end date (format: YYYY-MM-DD)

### 4. Backtesting

Run a backtest to evaluate strategy performance:

```bash
python backtest.py --tickers AAPL MSFT GOOGL --strategy ml --sentiment
```

Parameters:
- `--tickers`: Specify stock symbols
- `--strategy`: Choose strategy type (ml, technical, sentiment, combined)
- `--sentiment`: Include sentiment analysis
- `--start_date`/`--end_date`: Specify date range

### 5. Exporting Results to Excel

Create Excel reports with detailed performance metrics:

```bash
python export_results.py
```

This generates an Excel file with multiple sheets:
- Summary of performance metrics
- Trading signals with dates and probabilities
- Complete backtest results
- List of all trades with profit/loss

### 6. Creating Interactive Dashboards

Generate interactive HTML dashboards for visual analysis:

```bash
python create_dashboard.py
```

The dashboard includes:
- Price chart with buy/sell signals
- Performance comparison to buy & hold
- Signal distribution
- Monthly returns
- Drawdown analysis

### 7. Monthly Performance Analysis

Analyze performance patterns by month:

```bash
python monthly_performance.py
```

This creates:
- Monthly returns heatmap
- Performance by calendar month
- Comparison to benchmark
- Yearly performance summary

### 8. NSE Market Analysis

Analyze Nairobi Securities Exchange stocks:

```bash
python nse_trading.py
```

This will:
- Generate sample NSE stock data (or use your own data files)
- Apply NSE-specific adjustments (price limits, holiday calendar)
- Train models adapted for NSE characteristics
- Backtest strategies and generate performance reports
- Create comparison charts across NSE stocks

## How It Works - Technical Details

### Data Fetching (`data_fetcher.py`)

1. **Yahoo Finance Integration**: Fetches data using the `yfinance` library
2. **NSE Data Handling**: Custom functions for NSE data sources
3. **Data Cleaning**: Handles missing values, adjusts for stock splits/dividends
4. **Data Structure**: Returns dictionary with tickers as keys and DataFrames as values

### ML Model (`ml_trading_model.py`)

1. **Feature Engineering**:
   - Technical indicators (SMA, volatility, price momentum)
   - Volume indicators (volume change, relative volume)
   - Return lag features (previous day returns)

2. **Model Training**:
   - Uses RandomForest classifier by default
   - Target: Binary classification (price up/down in next N days)
   - Train/test split with cross-validation
   - Feature importance analysis

3. **Signal Generation**:
   - Probability thresholds for buy/sell signals
   - Signal strength based on prediction confidence

### Backtesting System (`backtest_system.py`)

1. **Portfolio Simulation**:
   - Tracks positions, cash, and holdings
   - Implements trade execution logic
   - Calculates portfolio value over time

2. **Performance Metrics**:
   - Total and annualized returns
   - Sharpe ratio for risk-adjusted performance
   - Maximum drawdown analysis
   - Win rate and profit factor

3. **Visualization**:
   - Equity curve charts
   - Drawdown visualization
   - Position size tracking

### Sentiment Analysis (`sentiment_analyzer.py`)

1. **Text Processing**:
   - Cleans and normalizes news text
   - Entity recognition for stock mentions

2. **Sentiment Scoring**:
   - VADER sentiment analysis with finance adaptations
   - TextBlob for secondary sentiment scoring
   - Combined sentiment score calculation

3. **Feature Creation**:
   - Daily sentiment scores
   - Rolling sentiment indicators
   - Sentiment change signals

### Alternative Data Integration (`alternative_data.py`)

1. **Data Sources**:
   - Economic indicators
   - Social media sentiment
   - Options market data
   - Insider trading activity

2. **Feature Fusion**:
   - Combines alternative data with market data
   - Creates derived features
   - Aligns timestamps across different data sources

### NSE Specific Adaptations (`nse_trading.py`)

1. **Market Characteristics**:
   - Realistic price levels for NSE stocks
   - 10% daily price movement limits
   - Holiday calendars for Kenya
   - Lower liquidity adjustments

2. **Sample Data Generation**:
   - Creates realistic NSE data with proper characteristics
   - Includes top traded NSE stocks (Safaricom, Equity, KCB, etc.)

3. **Analysis Adjustments**:
   - Features suitable for NSE market behavior
   - Risk management adapted for NSE volatility and liquidity

## Performance Analysis

The system provides comprehensive performance analysis:

1. **Basic Metrics**:
   - Total and annualized returns
   - Sharpe and Sortino ratios
   - Maximum drawdown
   - Win rate and profit factor

2. **Advanced Analysis**:
   - Monthly returns heatmap
   - Performance by market regime
   - Sensitivity to parameter changes
   - Transaction cost impact

3. **Visualization**:
   - Interactive HTML dashboards
   - Excel reports with multiple sheets
   - Performance charts and signal analysis
   - Comparison across multiple stocks

## Customization Options

### 1. Data Sources

To use different data sources:
1. Modify `data_fetcher.py` to implement your custom data acquisition function
2. For NSE, update `fetch_nse_data()` in `nse_trading.py` to use your data source

Example for custom NSE data:
```python
def fetch_nse_data(tickers, start_date, end_date=None):
    # Replace with your NSE data source
    # Example: Load from your broker's CSV exports
    data = {}
    for ticker in tickers:
        file_path = f"data/nse/{ticker}.csv"
        if os.path.exists(file_path):
            data[ticker] = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data
```

### 2. ML Models

To use different machine learning algorithms:
1. Modify the `MLTradingModel` class in `ml_trading_model.py`
2. Example for XGBoost:

```python
from xgboost import XGBClassifier

# In MLTradingModel.__init__:
self.model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

### 3. Trading Strategies

Create different strategies by:
1. Modifying signal generation in `ml_trading_model.py`
2. Adjusting thresholds for buy/sell signals
3. Creating custom indicators and features

### 4. Risk Management

Enhance risk management by:
1. Adding position sizing logic based on volatility
2. Implementing stop-loss and take-profit logic
3. Adding portfolio-level risk constraints

## Testing With Real Data

For real production use:

1. **Data Collection**:
   - Set up reliable data feeds for your target markets
   - Ensure data quality and consistency
   - Store historical data in a consistent format

2. **Model Training Workflow**:
   - Train models on historical data
   - Validate on out-of-sample periods
   - Retrain periodically as new data becomes available

3. **Performance Monitoring**:
   - Track key performance metrics
   - Compare actual vs. expected performance
   - Adjust strategies based on market regime changes

## Troubleshooting

### Common Issues:

1. **Data Unavailability**:
   - Check your internet connection
   - Verify ticker symbols are correct
   - Ensure date range is valid

2. **Model Performance Issues**:
   - Try different feature sets
   - Adjust training parameters
   - Use more historical data

3. **Memory Errors**:
   - Process fewer tickers at once
   - Use shorter date ranges
   - Optimize data structures

4. **Import Errors**:
   - Ensure all dependencies are installed
   - Check file paths and module naming
   - Verify Python version compatibility (3.8-3.10 recommended)

## Development Roadmap

Future enhancements could include:

1. Live trading integration with brokers
2. Real-time data feeds and signal alerts
3. Advanced portfolio optimization
4. Deep learning models for price prediction
5. More sophisticated sentiment analysis
6. Web interface for monitoring and control

## Conclusion

This algorithmic trading system provides a flexible framework for developing, testing, and analyzing trading strategies across different markets. While the default configuration focuses on US stocks and the NSE, it can be adapted for any market with appropriate data sources.

For best results, focus on:
1. High-quality data
2. Robust feature engineering
3. Proper validation methodology
4. Realistic transaction cost modeling
5. Appropriate risk management