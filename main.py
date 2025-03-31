import os
import argparse
from datetime import datetime, timedelta
import logging

# Import core modules
from core.data_fetcher import fetch_market_data
from core.ml_trading_model import MLTradingModel
from core.sentiment_analyzer import FinancialSentimentAnalyzer
from core.backtest_system import Backtester

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

def main():
    """Main entry point for the algorithmic trading system"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Algorithmic Trading System')
    parser.add_argument('action', choices=['fetch', 'train', 'backtest', 'live'], 
                      help='Action to perform')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                      help='Ticker symbols to process')
    parser.add_argument('--start_date', default=None,
                      help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--end_date', default=None,
                      help='End date (YYYY-MM-DD format)')
    parser.add_argument('--use_sentiment', action='store_true',
                      help='Whether to use sentiment analysis')
    
    args = parser.parse_args()
    
    # Determine date range
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    if args.start_date:
        start_date = args.start_date
    else:
        # Default to 1 year of data
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    
    logger.info(f"Processing data for {args.tickers} from {start_date} to {end_date}")
    
    # Create directories
    os.makedirs('data/market', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/backtest', exist_ok=True)
    
    # Perform requested action
    if args.action == 'fetch':
        # Fetch and process market data
        logger.info("Fetching market data...")
        market_data = fetch_market_data(args.tickers, start_date, end_date)
        
        # Save some basic stats
        for ticker, data in market_data.items():
            logger.info(f"Fetched {len(data)} rows of data for {ticker}")
    
    elif args.action == 'train':
        # Fetch market data if needed
        logger.info("Fetching market data for training...")
        market_data = fetch_market_data(args.tickers, start_date, end_date)
        
        # Initialize ML model
        ml_model = MLTradingModel()
        
        # Train model for each ticker
        for ticker, data in market_data.items():
            logger.info(f"Training model for {ticker}...")
            
            # Prepare features
            processed_data = ml_model.prepare_features(data)
            
            # Add sentiment if requested
            if args.use_sentiment:
                logger.info(f"Adding sentiment analysis for {ticker}...")
                sentiment_analyzer = FinancialSentimentAnalyzer()
                processed_data = sentiment_analyzer.generate_sentiment_features(processed_data, ticker)
            
            # Train the model
            train_results = ml_model.train(processed_data)
            
            logger.info(f"Model for {ticker} trained with test accuracy: {train_results['test_accuracy']:.4f}")
            
    elif args.action == 'backtest':
        # Fetch market data if needed
        logger.info("Fetching market data for backtesting...")
        market_data = fetch_market_data(args.tickers, start_date, end_date)
        
        # Initialize components
        ml_model = MLTradingModel()
        backtester = Backtester()
        
        # Run backtest for each ticker
        for ticker, data in market_data.items():
            logger.info(f"Running backtest for {ticker}...")
            
            # Prepare features
            processed_data = ml_model.prepare_features(data)
            
            # Add sentiment if requested
            if args.use_sentiment:
                logger.info(f"Adding sentiment analysis for {ticker}...")
                sentiment_analyzer = FinancialSentimentAnalyzer()
                processed_data = sentiment_analyzer.generate_sentiment_features(processed_data, ticker)
            
            # Train the model
            ml_model.train(processed_data)
            
            # Generate signals
            signals = ml_model.predict(processed_data)
            
            # Run backtest
            results = backtester.run_backtest(data, signals)
            
            # Plot results
            backtester.plot_results(save_path=f'results/backtest/{ticker}_backtest.png')
            
            logger.info(f"Backtest for {ticker} completed with total return: {results['total_return_pct']:.2f}%")
            
    elif args.action == 'live':
        logger.info("Live trading mode not implemented in this version.")
        logger.info("Use 'python live_demo.py' for live trading demonstration.")
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()