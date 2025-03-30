import os
import argparse
import pandas as pd
import numpy as np
import pickle
import time
import json
from datetime import datetime, timedelta

# Import core modules
from core.data_fetcher import fetch_market_data
from core.ml_trading_model import MLTradingModel
from core.sentiment_analyzer import FinancialSentimentAnalyzer
from core.alternative_data import AlternativeDataProcessor
from utils.logger import setup_logger, TradeLogger
from utils.config import load_config
from utils.visualization import plot_portfolio_performance, create_trading_dashboard

# Set up loggers
logger = setup_logger('live_demo')
trade_logger = TradeLogger('live_demo')

class LiveTradingDemo:
    """
    Live trading demonstration for the algorithmic trading system
    """
    
    def __init__(self, config_file='config.ini'):
        """
        Initialize the live trading demo
        
        Parameters:
        config_file (str): Path to configuration file
        """
        logger.info("Initializing live trading demo")
        
        # Load configuration
        self.config = load_config(config_file)
        self.config_file = config_file
        
        # Initialize components
        self.ml_model = MLTradingModel()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.alt_data_processor = AlternativeDataProcessor()
        
        # Dictionary to store market data
        self.market_data = {}
        
        # Dictionary to store trained models
        self.models = {}
        
        # Portfolio tracking
        self.portfolio = {
            'cash': self.config.getfloat('LIVE', 'initial_capital', fallback=100000.0),
            'positions': {},
            'trades': [],
            'history': []
        }
        
        # Create directories
        os.makedirs('results/live', exist_ok=True)
        os.makedirs('data/live', exist_ok=True)
        
        # Load trained models if available
        self._load_models()
        
        # Load portfolio state if available
        self._load_portfolio()
        
        logger.info("Live trading demo initialized")
    
    def _load_models(self):
        """Load trained models from model directory"""
        model_dir = self.config.get('MODEL', 'model_dir', fallback='models')
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory {model_dir} not found. No models loaded.")
            return
        
        # Get list of model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'model' in f]
        
        if not model_files:
            logger.warning("No model files found.")
            return
        
        # Load the most recent model for each ticker
        tickers = set()
        for model_file in model_files:
            ticker = model_file.split('_')[0]
            tickers.add(ticker)
        
        for ticker in tickers:
            # Find the most recent model file for this ticker
            ticker_models = [f for f in model_files if f.startswith(f"{ticker}_model")]
            if not ticker_models:
                continue
                
            # Sort by timestamp
            ticker_models.sort(reverse=True)
            latest_model_file = ticker_models[0]
            
            # Load the model
            try:
                with open(os.path.join(model_dir, latest_model_file), 'rb') as f:
                    model_info = pickle.load(f)
                
                self.models[ticker] = model_info
                logger.info(f"Loaded model for {ticker} from {latest_model_file}")
            except Exception as e:
                logger.error(f"Error loading model for {ticker}: {e}")
    
    def _load_portfolio(self):
        """Load portfolio state from file"""
        portfolio_file = 'data/live/portfolio.json'
        
        if os.path.exists(portfolio_file):
            try:
                with open(portfolio_file, 'r') as f:
                    self.portfolio = json.load(f)
                logger.info("Loaded portfolio state")
            except Exception as e:
                logger.error(f"Error loading portfolio: {e}")
    
    def _save_portfolio(self):
        """Save portfolio state to file"""
        portfolio_file = 'data/live/portfolio.json'
        
        try:
            with open(portfolio_file, 'w') as f:
                json.dump(self.portfolio, f, indent=2)
            logger.info("Saved portfolio state")
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
    
    def setup(self, tickers, days_of_history=30):
        """
        Set up the live trading demo
        
        Parameters:
        tickers (list): List of ticker symbols to trade
        days_of_history (int): Number of days of historical data to fetch
        """
        logger.info(f"Setting up live trading demo for {tickers}")
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_of_history)).strftime('%Y-%m-%d')
        
        # Fetch market data
        self.market_data = fetch_market_data(tickers, start_date, end_date)
        
        # Process data for each ticker
        for ticker in tickers:
            if ticker not in self.market_data:
                logger.warning(f"No market data found for {ticker}. Skipping...")
                continue
            
            # Prepare features
            data = self.market_data[ticker].copy()
            processed_data = self.ml_model.prepare_features(data)
            
            # Add sentiment features
            if self.config.getboolean('DATA', 'use_sentiment_analysis', fallback=True):
                try:
                    processed_data = self.sentiment_analyzer.generate_sentiment_features(processed_data, ticker)
                    logger.info(f"Added sentiment features for {ticker}")
                except Exception as e:
                    logger.error(f"Error adding sentiment features: {e}")
            
            # Add alternative data features
            if self.config.getboolean('DATA', 'use_alternative_data', fallback=False):
                try:
                    enhanced_data = self.alt_data_processor.create_alternative_data_features(
                        {ticker: processed_data}, [ticker]
                    )
                    processed_data = enhanced_data[ticker]
                    logger.info(f"Added alternative data features for {ticker}")
                except Exception as e:
                    logger.error(f"Error adding alternative data features: {e}")
            
            # Update market data with processed data
            self.market_data[ticker] = processed_data
            
            # Train model if not already loaded
            if ticker not in self.models:
                logger.info(f"No pretrained model found for {ticker}. Training new model...")
                
                # Define features
                features = [
                    'Returns', 'Price_to_SMA20', 'Price_to_SMA50', 
                    'Volatility', 'Return_Lag_1', 'Return_Lag_2',
                    'Return_Lag_3', 'Volume_Change', 'Volume_Ratio'
                ]
                
                # Add sentiment features if available
                sentiment_features = [
                    'sentiment_score', 'sentiment_score_3d', 'sentiment_score_7d',
                    'sentiment_positive', 'sentiment_negative'
                ]
                
                features.extend([f for f in sentiment_features if f in processed_data.columns])
                
                # Train the model
                try:
                    train_result = self.ml_model.train(processed_data.dropna(), features)
                    
                    # Store model information
                    self.models[ticker] = {
                        'model': self.ml_model.model,
                        'features': features,
                        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'performance': train_result
                    }
                    
                    logger.info(f"Trained model for {ticker}")
                except Exception as e:
                    logger.error(f"Error training model for {ticker}: {e}")
        
        logger.info("Live trading demo setup complete")
    
    def generate_signals(self, tickers):
        """
        Generate trading signals for selected tickers
        
        Parameters:
        tickers (list): List of ticker symbols
        
        Returns:
        dict: Dictionary of signals by ticker
        """
        logger.info(f"Generating signals for {tickers}")
        
        signals = {}
        
        for ticker in tickers:
            if ticker not in self.market_data:
                logger.warning(f"No market data for {ticker}. Skipping...")
                continue
            
            if ticker not in self.models:
                logger.warning(f"No trained model for {ticker}. Skipping...")
                continue
            
            try:
                # Get data and model
                data = self.market_data[ticker].copy().dropna()
                model_info = self.models[ticker]
                
                # Set up ML model with pretrained model
                self.ml_model.model = model_info['model']
                self.ml_model.features = model_info['features']
                self.ml_model.is_trained = True
                
                # Generate signals
                ticker_signals = self.ml_model.predict(data)
                
                # Get the most recent signal
                if not ticker_signals.empty:
                    latest_signal = ticker_signals.iloc[-1]
                    
                    signals[ticker] = {
                        'date': latest_signal.name.strftime('%Y-%m-%d'),
                        'price': latest_signal['Adj Close'],
                        'signal': int(latest_signal['Signal']),  # Convert to int for JSON serialization
                        'probability': float(latest_signal['Probability']),
                        'prediction': int(latest_signal['Prediction'])
                    }
                    
                    # Add sentiment if available
                    if 'sentiment_score' in latest_signal:
                        signals[ticker]['sentiment_score'] = float(latest_signal['sentiment_score'])
                    
                    logger.info(f"Generated signal for {ticker}: {signals[ticker]['signal']}")
                else:
                    logger.warning(f"No signals generated for {ticker}")
            
            except Exception as e:
                logger.error(f"Error generating signals for {ticker}: {e}")
        
        # Save signals to file
        signals_file = f"data/live/signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(signals_file, 'w') as f:
                json.dump(signals, f, indent=2)
            logger.info(f"Saved signals to {signals_file}")
        except Exception as e:
            logger.error(f"Error saving signals: {e}")
        
        return signals
    
    def execute_trades(self, signals):
        """
        Execute trades based on signals
        
        Parameters:
        signals (dict): Dictionary of signals by ticker
        
        Returns:
        list: List of executed trades
        """
        logger.info("Executing trades based on signals")
        
        # Get trading parameters
        position_size = self.config.getfloat('LIVE', 'position_size', fallback=0.1)
        max_positions = self.config.getint('LIVE', 'max_positions', fallback=5)
        
        executed_trades = []
        
        # Process each signal
        for ticker, signal_info in signals.items():
            signal = signal_info['signal']
            price = signal_info['price']
            probability = signal_info['probability']
            
            # Check if we have a position in this ticker
            has_position = ticker in self.portfolio['positions']
            
            # Buy signal
            if signal == 1 and not has_position:
                # Check if we've reached max positions
                if len(self.portfolio['positions']) >= max_positions:
                    logger.info(f"Maximum positions reached ({max_positions}). Not buying {ticker}.")
                    continue
                
                # Calculate position size
                cash = self.portfolio['cash']
                position_value = cash * position_size
                shares = int(position_value / price)
                
                if shares > 0:
                    # Execute buy trade
                    cost = shares * price
                    
                    # Check if we have enough cash
                    if cost <= cash:
                        # Update portfolio
                        self.portfolio['cash'] -= cost
                        self.portfolio['positions'][ticker] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'cost': cost,
                            'current_price': price,
                            'current_value': shares * price
                        }
                        
                        # Log trade
                        trade_info = {
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'value': cost,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'reason': f"Buy signal with probability {probability:.2f}"
                        }
                        
                        self.portfolio['trades'].append(trade_info)
                        executed_trades.append(trade_info)
                        
                        trade_logger.log_trade(
                            ticker, 'BUY', price, shares, 
                            f"Signal: {signal}, Probability: {probability:.2f}"
                        )
                        
                        logger.info(f"Bought {shares} shares of {ticker} at ${price:.2f} for ${cost:.2f}")
                    else:
                        logger.warning(f"Not enough cash to buy {ticker}. Need ${cost:.2f}, have ${cash:.2f}")
                else:
                    logger.warning(f"Not enough cash to buy at least 1 share of {ticker}")
            
            # Sell signal
            elif signal == -1 and has_position:
                # Get position details
                position = self.portfolio['positions'][ticker]
                shares = position['shares']
                entry_price = position['entry_price']
                cost = position['cost']
                
                # Calculate profit/loss
                value = shares * price
                profit = value - cost
                profit_pct = (profit / cost) * 100
                
                # Update portfolio
                self.portfolio['cash'] += value
                del self.portfolio['positions'][ticker]
                
                # Log trade
                trade_info = {
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': value,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'reason': f"Sell signal with probability {1-probability:.2f}"
                }
                
                self.portfolio['trades'].append(trade_info)
                executed_trades.append(trade_info)
                
                trade_logger.log_trade(
                    ticker, 'SELL', price, shares, 
                    f"Signal: {signal}, Profit: ${profit:.2f} ({profit_pct:.2f}%)"
                )
                
                logger.info(f"Sold {shares} shares of {ticker} at ${price:.2f} for ${value:.2f}, Profit: ${profit:.2f} ({profit_pct:.2f}%)")
        
        # Save portfolio state
        self._save_portfolio()
        
        return executed_trades
    
    def update_portfolio(self):
        """
        Update portfolio with current market prices
        
        Returns:
        dict: Updated portfolio information
        """
        logger.info("Updating portfolio with current market prices")
        
        # Get current market data for positions
        tickers = list(self.portfolio['positions'].keys())
        
        if not tickers:
            logger.info("No positions to update")
            return self.portfolio
        
        # Fetch latest market data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            market_data = fetch_market_data(tickers, start_date, end_date)
            
            total_value = self.portfolio['cash']
            
            # Update each position
            for ticker, position in self.portfolio['positions'].items():
                if ticker in market_data:
                    data = market_data[ticker]
                    if not data.empty:
                        # Get the latest price
                        latest_price = data['Adj Close'].iloc[-1]
                        
                        # Update position
                        shares = position['shares']
                        entry_price = position['entry_price']
                        cost = position['cost']
                        
                        current_value = shares * latest_price
                        profit = current_value - cost
                        profit_pct = (profit / cost) * 100
                        
                        # Update position information
                        self.portfolio['positions'][ticker].update({
                            'current_price': latest_price,
                            'current_value': current_value,
                            'profit': profit,
                            'profit_pct': profit_pct
                        })
                        
                        # Add to total value
                        total_value += current_value
                        
                        logger.info(f"Updated {ticker}: {shares} shares at ${latest_price:.2f}, Value: ${current_value:.2f}, Profit: ${profit:.2f} ({profit_pct:.2f}%)")
            
            # Calculate portfolio metrics
            invested_value = sum(pos['current_value'] for pos in self.portfolio['positions'].values())
            
            # Create portfolio snapshot
            snapshot = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cash': self.portfolio['cash'],
                'invested_value': invested_value,
                'total_value': total_value,
                'positions': len(self.portfolio['positions'])
            }
            
            # Add snapshot to history
            self.portfolio['history'].append(snapshot)
            
            # Save portfolio state
            self._save_portfolio()
            
            # Save portfolio history to CSV
            history_file = 'results/live/portfolio_history.csv'
            
            try:
                # Convert history to DataFrame
                history_df = pd.DataFrame(self.portfolio['history'])
                history_df.to_csv(history_file, index=False)
                logger.info(f"Saved portfolio history to {history_file}")
            except Exception as e:
                logger.error(f"Error saving portfolio history: {e}")
            
            # Log portfolio status
            portfolio_info = {
                'cash': self.portfolio['cash'],
                'value': total_value,
                'positions': []
            }
            
            for ticker, position in self.portfolio['positions'].items():
                portfolio_info['positions'].append({
                    'ticker': ticker,
                    'quantity': position['shares'],
                    'cost_basis': position['cost'],
                    'current_value': position['current_value'],
                    'unrealized_pnl': position.get('profit', 0)
                })
            
            trade_logger.log_portfolio(portfolio_info)
            
            return self.portfolio
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            return self.portfolio
    
    def generate_reports(self):
        """
        Generate performance reports
        
        Returns:
        dict: Report information
        """
        logger.info("Generating performance reports")
        
        reports = {}
        
        try:
            # Create portfolio performance chart
            if self.portfolio['history']:
                history_df = pd.DataFrame(self.portfolio['history'])
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                history_df = history_df.set_index('timestamp')
                
                # Create performance chart
                fig = plot_portfolio_performance(history_df)
                
                # Save chart
                chart_file = f"results/live/portfolio_performance_{datetime.now().strftime('%Y%m%d')}.png"
                fig.savefig(chart_file)
                plt.close(fig)
                
                logger.info(f"Saved portfolio performance chart to {chart_file}")
                reports['portfolio_chart'] = chart_file
            
            # Create individual ticker dashboards
            for ticker, position in self.portfolio['positions'].items():
                if ticker in self.market_data:
                    # Create dashboard
                    data = self.market_data[ticker]
                    
                    # Generate signals for dashboard
                    if ticker in self.models:
                        model_info = self.models[ticker]
                        
                        # Set up ML model with pretrained model
                        self.ml_model.model = model_info['model']
                        self.ml_model.features = model_info['features']
                        self.ml_model.is_trained = True
                        
                        # Generate signals
                        signals = self.ml_model.predict(data.dropna())
                        
                        # Create dashboard
                        dashboard_fig = create_trading_dashboard(
                            ticker, 
                            data, 
                            signals,
                            {'entry_price': position['entry_price'], 'current_price': position['current_price']}
                        )
                        
                        # Save dashboard
                        dashboard_file = f"results/live/{ticker}_dashboard_{datetime.now().strftime('%Y%m%d')}.png"
                        dashboard_fig.savefig(dashboard_file)
                        plt.close(dashboard_fig)
                        
                        logger.info(f"Saved dashboard for {ticker} to {dashboard_file}")
                        reports[f'{ticker}_dashboard'] = dashboard_file
            
            return reports
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            return reports
    
    def run_demo_cycle(self, tickers):
        """
        Run a complete demo cycle
        
        Parameters:
        tickers (list): List of ticker symbols
        
        Returns:
        dict: Cycle results
        """
        logger.info(f"Running demo cycle for {tickers}")
        
        try:
            # 1. Generate signals
            signals = self.generate_signals(tickers)
            
            # 2. Execute trades
            trades = self.execute_trades(signals)
            
            # 3. Update portfolio
            portfolio = self.update_portfolio()
            
            # 4. Generate reports
            reports = self.generate_reports()
            
            # Return cycle results
            results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signals': signals,
                'trades': trades,
                'portfolio': {
                    'cash': portfolio['cash'],
                    'positions': len(portfolio['positions']),
                    'value': sum(pos['current_value'] for pos in portfolio['positions'].values()) + portfolio['cash']
                },
                'reports': reports
            }
            
            logger.info("Demo cycle completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error running demo cycle: {e}")
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
    
    def run_continuous(self, tickers, interval_seconds=3600, max_cycles=None):
        """
        Run the demo continuously
        
        Parameters:
        tickers (list): List of ticker symbols
        interval_seconds (int): Interval between cycles in seconds
        max_cycles (int): Maximum number of cycles to run, None for unlimited
        """
        logger.info(f"Starting continuous demo with interval {interval_seconds} seconds")
        
        cycle_count = 0
        
        try:
            while True:
                cycle_start = time.time()
                
                logger.info(f"Running cycle {cycle_count + 1}")
                
                # Run a demo cycle
                self.run_demo_cycle(tickers)
                
                cycle_count += 1
                
                # Check if we've reached the maximum number of cycles
                if max_cycles is not None and cycle_count >= max_cycles:
                    logger.info(f"Reached maximum number of cycles ({max_cycles}). Stopping.")
                    break
                
                # Calculate time to sleep
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, interval_seconds - cycle_duration)
                
                if sleep_time > 0:
                    logger.info(f"Waiting {sleep_time:.1f} seconds until next cycle")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Demo stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous demo: {e}")
        finally:
            logger.info("Continuous demo ended")


def main():
    """Main function for live trading demo"""
    parser = argparse.ArgumentParser(description='Live Trading Demo')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                      help='Ticker symbols to trade')
    parser.add_argument('--config', default='config.ini',
                      help='Path to configuration file')
    parser.add_argument('--history', type=int, default=30,
                      help='Days of historical data to use')
    parser.add_argument('--interval', type=int, default=3600,
                      help='Interval between cycles in seconds')
    parser.add_argument('--cycles', type=int, default=None,
                      help='Maximum number of cycles to run')
    parser.add_argument('--continuous', action='store_true',
                      help='Run continuously')
    
    args = parser.parse_args()
    
    # Initialize the live trading demo
    demo = LiveTradingDemo(args.config)
    
    # Set up the demo
    demo.setup(args.tickers, args.history)
    
    if args.continuous:
        # Run continuously
        demo.run_continuous(args.tickers, args.interval, args.cycles)
    else:
        # Run a single cycle
        demo.run_demo_cycle(args.tickers)

if __name__ == "__main__":
    main()