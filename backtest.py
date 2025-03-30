import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import core modules
from core.data_fetcher import fetch_market_data
from core.ml_trading_model import MLTradingModel
from core.sentiment_analyzer import FinancialSentimentAnalyzer
from core.backtest_system import Backtester
from core.performance_metrics import PerformanceMetrics
from utils.logger import setup_logger
from utils.config import load_config
from utils.visualization import plot_backtest_results

# Set up logger
logger = setup_logger('backtest')

def run_backtest(tickers, start_date, end_date, config_file, 
                 strategy_type='ml', use_sentiment=True, save_results=True):
    """
    Run backtest for selected tickers
    
    Parameters:
    tickers (list): List of ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    config_file (str): Path to configuration file
    strategy_type (str): Strategy type ('ml', 'technical', 'sentiment', 'combined')
    use_sentiment (bool): Whether to include sentiment analysis
    save_results (bool): Whether to save results to file
    
    Returns:
    dict: Backtest results by ticker
    """
    logger.info(f"Starting backtest for {tickers} from {start_date} to {end_date}")
    
    # Load configuration
    config = load_config(config_file)
    
    # Get parameters from config
    initial_capital = config.getfloat('BACKTEST', 'initial_capital', fallback=100000.0)
    commission = config.getfloat('BACKTEST', 'commission', fallback=0.001)  # 0.1%
    results_dir = config.get('BACKTEST', 'results_dir', fallback='results/backtest')
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize components
    ml_model = MLTradingModel()
    sentiment_analyzer = FinancialSentimentAnalyzer() if use_sentiment else None
    backtester = Backtester()
    
    # Fetch market data
    market_data = fetch_market_data(tickers, start_date, end_date)
    
    # Store results
    backtest_results = {}
    
    # Process each ticker
    for ticker in tickers:
        logger.info(f"Backtesting {ticker}")
        
        if ticker not in market_data:
            logger.warning(f"No market data for {ticker}. Skipping...")
            continue
        
        try:
            # Get ticker data
            data = market_data[ticker].copy()
            
            # Prepare features based on strategy type
            if strategy_type == 'ml':
                # ML features with technical indicators
                processed_data = ml_model.prepare_features(data)
                
                # Add sentiment if requested
                if use_sentiment:
                    try:
                        processed_data = sentiment_analyzer.generate_sentiment_features(processed_data, ticker)
                    except Exception as e:
                        logger.error(f"Error adding sentiment features: {e}")
            
            elif strategy_type == 'technical':
                # Only use technical indicators
                processed_data = data.copy()
                processed_data['Returns'] = processed_data['Adj Close'].pct_change()
                processed_data['SMA_20'] = processed_data['Adj Close'].rolling(window=20).mean()
                processed_data['SMA_50'] = processed_data['Adj Close'].rolling(window=50).mean()
                processed_data['SMA_Cross'] = (processed_data['SMA_20'] > processed_data['SMA_50']).astype(int)
                
                # Generate simple signals based on SMA cross
                processed_data['Signal'] = 0
                processed_data.loc[processed_data['SMA_Cross'] == 1, 'Signal'] = 1
                processed_data.loc[processed_data['SMA_Cross'] == 0, 'Signal'] = -1
                
            elif strategy_type == 'sentiment':
                # Only use sentiment-based signals
                processed_data = data.copy()
                processed_data = sentiment_analyzer.generate_sentiment_features(processed_data, ticker)
                
                # Generate simple signals based on sentiment
                processed_data['Signal'] = 0
                processed_data.loc[processed_data['sentiment_score'] > 0.2, 'Signal'] = 1
                processed_data.loc[processed_data['sentiment_score'] < -0.2, 'Signal'] = -1
                
            elif strategy_type == 'combined':
                # Use both technical and sentiment
                processed_data = ml_model.prepare_features(data)
                processed_data = sentiment_analyzer.generate_sentiment_features(processed_data, ticker)
                
                # Will train model with both types of features
            
            # Remove NaN values
            processed_data = processed_data.dropna()
            
            # Train model if using ML strategy
            if strategy_type in ['ml', 'combined']:
                # Define features based on strategy type
                if strategy_type == 'ml':
                    features = [
                        'Returns', 'Price_to_SMA20', 'Price_to_SMA50', 
                        'Volatility', 'Return_Lag_1', 'Return_Lag_2',
                        'Return_Lag_3', 'Volume_Change', 'Volume_Ratio'
                    ]
                else:  # combined
                    features = [
                        'Returns', 'Price_to_SMA20', 'Price_to_SMA50', 
                        'Volatility', 'Return_Lag_1', 'Volume_Change',
                        'sentiment_score', 'sentiment_score_3d'
                    ]
                
                # Train the model
                ml_model.train(processed_data, features)
                
                # Generate signals
                signals = ml_model.predict(processed_data)
            else:
                # Technical or sentiment strategy already has signals
                signals = processed_data[['Adj Close', 'Signal']].copy()
                signals['Probability'] = 0.5  # Dummy probability
            
            # Run backtest
            results = backtester.run_pyfolio_analysis(processed_data, signals, initial_capital)
            
            # Store results
            backtest_results[ticker] = results
            
            # Save results if requested
            if save_results:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_file = os.path.join(results_dir, f"backtest_{ticker}_{strategy_type}_{timestamp}.json")
                
                # Convert to JSON serializable format
                json_results = {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
                
                # Save to file
                import json
                with open(results_file, 'w') as f:
                    json.dump(json_results, f, indent=2)
                
                logger.info(f"Saved backtest results to {results_file}")
                
                # Save performance chart
                chart_file = os.path.join(results_dir, f"backtest_{ticker}_{strategy_type}_{timestamp}.png")
                
                # Generate returns series
                returns = processed_data['Returns']
                
                # Plot and save
                fig = plot_backtest_results(returns, results, ticker, strategy_type)
                plt.savefig(chart_file)
                plt.close(fig)
                
                logger.info(f"Saved performance chart to {chart_file}")
            
        except Exception as e:
            logger.error(f"Error backtesting {ticker}: {e}")
    
    return backtest_results

def main():
    """Main function for backtesting"""
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                      help='Ticker symbols to backtest')
    parser.add_argument('--start_date', default=None,
                      help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--end_date', default=None,
                      help='End date (YYYY-MM-DD format)')
    parser.add_argument('--strategy', default='ml',
                      choices=['ml', 'technical', 'sentiment', 'combined'],
                      help='Strategy type to backtest')
    parser.add_argument('--sentiment', action='store_true',
                      help='Include sentiment analysis')
    parser.add_argument('--config', default='config.ini',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Determine date range
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    if args.start_date:
        start_date = args.start_date
    else:
        # Default to 1 year of data
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Run backtest
    results = run_backtest(
        args.tickers, 
        start_date, 
        end_date, 
        args.config, 
        strategy_type=args.strategy, 
        use_sentiment=args.sentiment
    )
    
    # Print summary
    logger.info("Backtest Summary:")
    
    for ticker, result in results.items():
        logger.info(f"{ticker}:")
        logger.info(f"  Total Return: {result['total_return_pct']:.2f}%")
        logger.info(f"  Annual Return: {result['annual_return_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {result['max_drawdown_pct']:.2f}%")

if __name__ == "__main__":
    main()