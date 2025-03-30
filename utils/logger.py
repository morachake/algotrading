import logging
import os
from datetime import datetime

def setup_logger(name, log_level=logging.INFO, log_dir='results/logs'):
    """
    Set up a logger with file and console handlers
    
    Parameters:
    name (str): Logger name
    log_level (int): Logging level
    log_dir (str): Directory to save log files
    
    Returns:
    logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


class TradeLogger:
    """
    Specialized logger for trading activities
    """
    
    def __init__(self, name, log_dir='results/logs'):
        """
        Initialize trade logger
        
        Parameters:
        name (str): Logger name
        log_dir (str): Directory to save log files
        """
        self.logger = setup_logger(f"trade_{name}", log_dir=log_dir)
        self.trade_log_file = os.path.join(log_dir, f"trades_{name}_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # Create trade log file with header if it doesn't exist
        if not os.path.exists(self.trade_log_file):
            os.makedirs(os.path.dirname(self.trade_log_file), exist_ok=True)
            with open(self.trade_log_file, 'w') as f:
                f.write("timestamp,ticker,action,price,quantity,value,reason\n")
    
    def log_trade(self, ticker, action, price, quantity, reason=None):
        """
        Log a trade
        
        Parameters:
        ticker (str): Ticker symbol
        action (str): Trade action (BUY, SELL)
        price (float): Execution price
        quantity (int/float): Number of shares/contracts
        reason (str): Reason for the trade
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        value = price * quantity
        
        # Log to main logger
        self.logger.info(f"TRADE: {action} {quantity} {ticker} @ ${price:.2f} (${value:.2f}) - {reason or 'N/A'}")
        
        # Log to CSV file
        with open(self.trade_log_file, 'a') as f:
            f.write(f"{timestamp},{ticker},{action},{price:.2f},{quantity},{value:.2f},{reason or 'N/A'}\n")
    
    def log_portfolio(self, portfolio):
        """
        Log portfolio status
        
        Parameters:
        portfolio (dict): Portfolio information
        """
        self.logger.info(f"PORTFOLIO: Cash=${portfolio.get('cash', 0):.2f}, Value=${portfolio.get('value', 0):.2f}")
        
        for position in portfolio.get('positions', []):
            ticker = position.get('ticker')
            quantity = position.get('quantity')
            cost_basis = position.get('cost_basis')
            current_value = position.get('current_value')
            unrealized_pnl = position.get('unrealized_pnl')
            
            self.logger.info(f"  POSITION: {ticker} - {quantity} shares, Cost=${cost_basis:.2f}, Value=${current_value:.2f}, P/L=${unrealized_pnl:.2f}")


# Example usage
if __name__ == "__main__":
    # Set up a regular logger
    logger = setup_logger("test")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Set up a trade logger
    trade_logger = TradeLogger("test")
    
    # Log some trades
    trade_logger.log_trade("AAPL", "BUY", 150.25, 10, "ML Signal")
    trade_logger.log_trade("MSFT", "SELL", 300.50, 5, "Stop Loss Triggered")
    
    # Log portfolio
    portfolio = {
        'cash': 10000.50,
        'value': 25000.75,
        'positions': [
            {'ticker': 'AAPL', 'quantity': 10, 'cost_basis': 1502.50, 'current_value': 1550.00, 'unrealized_pnl': 47.50},
            {'ticker': 'GOOGL', 'quantity': 5, 'cost_basis': 6500.00, 'current_value': 6700.25, 'unrealized_pnl': 200.25}
        ]
    }
    
    trade_logger.log_portfolio(portfolio)