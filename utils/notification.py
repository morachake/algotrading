import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime

# Setup logger
logger = logging.getLogger('notification')

class NotificationManager:
    """
    Manage notifications for the trading system
    """
    
    def __init__(self, config):
        """
        Initialize notification manager
        
        Parameters:
        config: Configuration object
        """
        self.config = config
        
        # Email settings
        self.email_enabled = config.getboolean('NOTIFICATION', 'enable_email', fallback=False)
        if self.email_enabled:
            self.smtp_server = config.get('NOTIFICATION', 'smtp_server', fallback='')
            self.smtp_port = config.getint('NOTIFICATION', 'smtp_port', fallback=587)
            self.email_from = config.get('NOTIFICATION', 'email_from', fallback='')
            self.email_to = config.get('NOTIFICATION', 'email_to', fallback='')
            self.email_password = config.get('NOTIFICATION', 'email_password', fallback='')
            
            if not all([self.smtp_server, self.email_from, self.email_to, self.email_password]):
                logger.warning("Email notifications enabled but some settings are missing")
                self.email_enabled = False
        
        # SMS/other notification settings could be added here
    
    def send_email(self, subject, message):
        """
        Send email notification
        
        Parameters:
        subject (str): Email subject
        message (str): Email message
        
        Returns:
        bool: True if sent successfully, False otherwise
        """
        if not self.email_enabled:
            logger.warning("Email notifications not enabled")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_from, self.email_password)
            
            # Send email
            text = msg.as_string()
            server.sendmail(self.email_from, self.email_to, text)
            
            # Disconnect
            server.quit()
            
            logger.info(f"Email sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def send_trade_notification(self, trade_info):
        """
        Send notification about a trade
        
        Parameters:
        trade_info (dict): Trade information
        
        Returns:
        bool: True if sent successfully, False otherwise
        """
        if not self.email_enabled:
            return False
        
        ticker = trade_info.get('ticker', '')
        action = trade_info.get('action', '')
        shares = trade_info.get('shares', 0)
        price = trade_info.get('price', 0.0)
        value = trade_info.get('value', 0.0)
        reason = trade_info.get('reason', '')
        
        subject = f"Trade Alert: {action} {ticker}"
        
        message = f"""
Trade Executed: {action} {shares} shares of {ticker}

Details:
- Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Price: ${price:.2f}
- Total Value: ${value:.2f}
- Reason: {reason}

This is an automated notification from your trading system.
        """
        
        return self.send_email(subject, message)
    
    def send_portfolio_summary(self, portfolio):
        """
        Send notification with portfolio summary
        
        Parameters:
        portfolio (dict): Portfolio information
        
        Returns:
        bool: True if sent successfully, False otherwise
        """
        if not self.email_enabled:
            return False
        
        cash = portfolio.get('cash', 0.0)
        positions = portfolio.get('positions', {})
        total_value = cash + sum(pos.get('current_value', 0.0) for pos in positions.values())
        
        subject = f"Portfolio Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        message = f"""
Trading System Portfolio Summary

Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Cash: ${cash:.2f}
- Invested: ${(total_value - cash):.2f}
- Total Value: ${total_value:.2f}
- Number of Positions: {len(positions)}

Current Positions:
"""
        
        # Add position details
        for ticker, position in positions.items():
            shares = position.get('shares', 0)
            entry_price = position.get('entry_price', 0.0)
            current_price = position.get('current_price', 0.0)
            current_value = position.get('current_value', 0.0)
            profit = position.get('profit', 0.0)
            profit_pct = position.get('profit_pct', 0.0)
            
            message += f"""
{ticker}: {shares} shares
  Entry: ${entry_price:.2f}
  Current: ${current_price:.2f}
  Value: ${current_value:.2f}
  P/L: ${profit:.2f} ({profit_pct:.2f}%)
"""
        
        message += """
This is an automated notification from your trading system.
        """
        
        return self.send_email(subject, message)
    
    def send_signal_alert(self, ticker, signal_type, price, probability, sentiment=None):
        """
        Send notification about a trading signal
        
        Parameters:
        ticker (str): Ticker symbol
        signal_type (int): Signal type (1=buy, -1=sell, 0=hold)
        price (float): Current price
        probability (float): Signal probability
        sentiment (float, optional): Sentiment score
        
        Returns:
        bool: True if sent successfully, False otherwise
        """
        if not self.email_enabled:
            return False
        
        # Convert signal type to text
        signal_text = "BUY" if signal_type == 1 else "SELL" if signal_type == -1 else "HOLD"
        
        subject = f"Signal Alert: {signal_text} {ticker}"
        
        message = f"""
Trading Signal Generated: {signal_text} {ticker}

Details:
- Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Price: ${price:.2f}
- Signal Probability: {probability:.2f}
"""
        
        if sentiment is not None:
            message += f"- Sentiment Score: {sentiment:.2f}\n"
        
        message += """
This is an automated notification from your trading system.
        """
        
        return self.send_email(subject, message)

# Example usage
if __name__ == "__main__":
    import configparser
    
    # Create sample config
    config = configparser.ConfigParser()
    config['NOTIFICATION'] = {
        'enable_email': 'True',
        'smtp_server': 'smtp.example.com',
        'smtp_port': '587',
        'email_from': 'trading@example.com',
        'email_to': 'user@example.com',
        'email_password': 'password123'
    }
    
    # Initialize notification manager
    notifier = NotificationManager(config)
    
    # Test trade notification
    trade_info = {
        'ticker': 'AAPL',
        'action': 'BUY',
        'shares': 10,
        'price': 150.25,
        'value': 1502.50,
        'reason': 'ML Signal with 0.85 probability'
    }
    
    notifier.send_trade_notification(trade_info)
    
    # Test portfolio summary
    portfolio = {
        'cash': 10000.50,
        'positions': {
            'AAPL': {
                'shares': 10,
                'entry_price': 150.25,
                'current_price': 155.75,
                'current_value': 1557.50,
                'profit': 55.00,
                'profit_pct': 3.66
            },
            'MSFT': {
                'shares': 5,
                'entry_price': 270.50,
                'current_price': 280.25,
                'current_value': 1401.25,
                'profit': 48.75,
                'profit_pct': 3.60
            }
        }
    }
    
    notifier.send_portfolio_summary(portfolio)
    
    # Test signal alert
    notifier.send_signal_alert('GOOGL', 1, 2250.75, 0.92, 0.65)