import os
import argparse
from datetime import datetime, timedelta

# Import core modules
from core.data_fetcher import fetch_market_data
from core.ml_trading_model import MLTradingModel
from core.sentiment_analyzer import FinancialSentimentAnalyzer
from core.alternative_data import AlternativeDataProcessor
from utils.logger import setup_logger
from utils.config import load_config

# Set up logger
logger = setup_logger('main')

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
    parser.add_argument('--config', default='config.ini',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine date range
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    if args.start_date:
        start_date = args.start_date
    else:
        # Default to 1 year of data
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    
    logger.info(f"Processing data for {args.tickers} from {start_date} to {end_date}")
    
    # Perform requested action
    if args.action == 'fetch':
        # Fetch and process market data
        logger.info("Fetching market data...")
        market_data = fetch_market_data(args.tickers, start_date, end_date)
        
        # Fetch alternative data if enabled
        if config.getboolean('DATA', 'use_alternative_data', fallback=False):
            logger.info("Fetching alternative data...")
            alt_data = AlternativeDataProcessor()
            enhanced_data = alt_data.create_alternative_data_features(market_data, args.tickers)
            
            for ticker in args.tickers:
                if ticker in enhanced_data:
                    logger.info(f"Enhanced data for {ticker} with {len(enhanced_data[ticker].columns)} features")
    
    elif args.action == 'train':
        # Redirect to training script
        logger.info("Training models - this should be done using train_models.py")
        
    elif args.action == 'backtest':
        # Redirect to backtesting script
        logger.info("Backtesting - this should be done using backtest.py")
        
    elif args.action == 'live':
        # Redirect to live trading script
        logger.info("Live trading - this should be done using live_demo.py")
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()