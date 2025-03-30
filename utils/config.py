import configparser
import os
import logging

def create_default_config(config_file):
    """Create a default configuration file"""
    config = configparser.ConfigParser()
    
    # General configuration
    config['GENERAL'] = {
        'project_name': 'Algorithmic Trading System',
        'log_level': 'INFO'
    }
    
    # Data configuration
    config['DATA'] = {
        'data_dir': 'data',
        'use_alternative_data': 'False',
        'use_sentiment_analysis': 'True'
    }
    
    # Model configuration
    config['MODEL'] = {
        'model_dir': 'models',
        'model_type': 'random_forest',
        'n_estimators': '100',
        'max_depth': '10',
        'prediction_horizon': '5'  # days
    }
    
    # Backtesting configuration
    config['BACKTEST'] = {
        'initial_capital': '100000',
        'commission': '0.001',  # 0.1%
        'slippage': '0.001',    # 0.1%
        'results_dir': 'results/backtest'
    }
    
    # Live trading configuration
    config['LIVE'] = {
        'paper_trading': 'True',
        'position_size': '0.1',  # 10% of capital per position
        'max_positions': '5',
        'update_interval': '3600',  # seconds
        'stop_loss': '0.05',     # 5% stop loss
        'take_profit': '0.1'     # 10% take profit
    }
    
    # API configuration (no real keys)
    config['API'] = {
        'use_api': 'False',
        'api_key': '',
        'api_secret': '',
        'base_url': ''
    }
    
    # Notification configuration
    config['NOTIFICATION'] = {
        'enable_email': 'False',
        'smtp_server': '',
        'smtp_port': '587',
        'email_from': '',
        'email_to': '',
        'email_password': ''
    }
    
    # Write the configuration to file
    with open(config_file, 'w') as f:
        config.write(f)
    
    return config

def load_config(config_file):
    """Load configuration from file, create default if not exists"""
    if not os.path.exists(config_file):
        logging.warning(f"Configuration file {config_file} not found. Creating default configuration.")
        return create_default_config(config_file)
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    return config

def update_config(config_file, section, option, value):
    """Update a configuration option"""
    config = load_config(config_file)
    
    if section not in config:
        config.add_section(section)
    
    config.set(section, option, str(value))
    
    with open(config_file, 'w') as f:
        config.write(f)
    
    return config

def get_config_value(config, section, option, fallback=None, as_type=str):
    """Get a configuration value with type conversion"""
    if section not in config:
        return fallback
    
    if option not in config[section]:
        return fallback
    
    value = config[section][option]
    
    if as_type == bool:
        return config.getboolean(section, option, fallback=fallback)
    elif as_type == int:
        return config.getint(section, option, fallback=fallback)
    elif as_type == float:
        return config.getfloat(section, option, fallback=fallback)
    else:
        return value

# Example usage
if __name__ == "__main__":
    config_file = "config.ini"
    
    # Load or create configuration
    config = load_config(config_file)
    
    # Print configuration
    for section in config.sections():
        print(f"[{section}]")
        for option in config[section]:
            print(f"  {option} = {config[section][option]}")
        print()
    
    # Update a configuration option
    update_config(config_file, "MODEL", "n_estimators", "200")
    
    # Get a configuration value with type conversion
    initial_capital = get_config_value(config, "BACKTEST", "initial_capital", fallback=100000, as_type=float)
    print(f"Initial capital: ${initial_capital:.2f}")