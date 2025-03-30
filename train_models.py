import os
import argparse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import core modules
from core.data_fetcher import fetch_market_data
from core.ml_trading_model import MLTradingModel
from core.sentiment_analyzer import FinancialSentimentAnalyzer
from core.alternative_data import AlternativeDataProcessor
from utils.logger import setup_logger
from utils.config import load_config
from utils.visualization import plot_feature_importance, create_correlation_heatmap

# Set up logger
logger = setup_logger('train_models')

def train_models(tickers, start_date, end_date, config_file, 
                model_type='random_forest', use_sentiment=True, 
                use_alternative_data=False, optimize_hyperparams=False,
                save_models=True):
    """
    Train machine learning models for each ticker
    
    Parameters:
    tickers (list): List of ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    config_file (str): Path to configuration file
    model_type (str): Type of model to train
    use_sentiment (bool): Whether to include sentiment features
    use_alternative_data (bool): Whether to include alternative data features
    optimize_hyperparams (bool): Whether to optimize hyperparameters
    save_models (bool): Whether to save trained models
    
    Returns:
    dict: Trained models by ticker
    """
    logger.info(f"Training models for {tickers} from {start_date} to {end_date}")
    
    # Load configuration
    config = load_config(config_file)
    
    # Get parameters from config
    model_dir = config.get('MODEL', 'model_dir', fallback='models')
    prediction_horizon = config.getint('MODEL', 'prediction_horizon', fallback=5)
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize components
    ml_model = MLTradingModel()
    sentiment_analyzer = FinancialSentimentAnalyzer() if use_sentiment else None
    alt_data_processor = AlternativeDataProcessor() if use_alternative_data else None
    
    # Fetch market data
    market_data = fetch_market_data(tickers, start_date, end_date)
    
    # Store trained models
    trained_models = {}
    
    # Process each ticker
    for ticker in tickers:
        logger.info(f"Training model for {ticker}")
        
        if ticker not in market_data:
            logger.warning(f"No market data for {ticker}. Skipping...")
            continue
        
        try:
            # Get ticker data
            data = market_data[ticker].copy()
            
            # Prepare features
            processed_data = ml_model.prepare_features(data)
            
            # Add sentiment features if requested
            if use_sentiment:
                try:
                    processed_data = sentiment_analyzer.generate_sentiment_features(processed_data, ticker)
                    logger.info(f"Added sentiment features for {ticker}")
                except Exception as e:
                    logger.error(f"Error adding sentiment features: {e}")
            
            # Add alternative data features if requested
            if use_alternative_data:
                try:
                    enhanced_data = alt_data_processor.create_alternative_data_features({ticker: processed_data}, [ticker])
                    processed_data = enhanced_data[ticker]
                    logger.info(f"Added alternative data features for {ticker}")
                except Exception as e:
                    logger.error(f"Error adding alternative data features: {e}")
            
            # Remove NaN values
            processed_data = processed_data.dropna()
            
            # Define features
            technical_features = [
                'Returns', 'Price_to_SMA20', 'Price_to_SMA50', 
                'Volatility', 'Return_Lag_1', 'Return_Lag_2',
                'Return_Lag_3', 'Volume_Change', 'Volume_Ratio'
            ]
            
            sentiment_features = [
                'sentiment_score', 'sentiment_score_3d', 'sentiment_score_7d',
                'sentiment_positive', 'sentiment_negative'
            ]
            
            alternative_features = [
                'social_sentiment', 'social_volume', 'social_sentiment_change',
                'options_put_call_ratio', 'options_implied_volatility',
                'insider_net_volume_30d'
            ]
            
            # Select features based on what's available and requested
            features = technical_features.copy()
            
            if use_sentiment:
                features.extend([f for f in sentiment_features if f in processed_data.columns])
            
            if use_alternative_data:
                features.extend([f for f in alternative_features if f in processed_data.columns])
            
            # Create feature correlation heatmap
            if len(features) > 1:
                correlation_fig = create_correlation_heatmap(
                    processed_data[features].dropna(), 
                    title=f"{ticker} Feature Correlation"
                )
                
                # Save correlation heatmap
                correlation_file = os.path.join(model_dir, f"{ticker}_feature_correlation.png")
                correlation_fig.savefig(correlation_file)
                plt.close(correlation_fig)
                logger.info(f"Saved feature correlation heatmap to {correlation_file}")
            
            # Train the model
            if optimize_hyperparams:
                # Create and optimize model
                if model_type == 'random_forest':
                    base_model = RandomForestClassifier(random_state=42)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                    
                    # Extract features and target
                    X = processed_data[features]
                    y = processed_data['Target']
                    
                    # Perform grid search
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=5, 
                        scoring='accuracy', n_jobs=-1
                    )
                    
                    grid_search.fit(X, y)
                    
                    # Get best model
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
                    logger.info(f"Best parameters for {ticker}: {best_params}")
                    logger.info(f"Best score: {grid_search.best_score_:.4f}")
                    
                    # Set the optimized model
                    ml_model.model = best_model
                    ml_model.is_trained = True
                    ml_model.features = features
                else:
                    # Train with default parameters
                    train_result = ml_model.train(processed_data, features)
                    logger.info(f"Training accuracy: {train_result['train_accuracy']:.4f}")
                    logger.info(f"Testing accuracy: {train_result['test_accuracy']:.4f}")
            else:
                # Train with default parameters
                train_result = ml_model.train(processed_data, features)
                logger.info(f"Training accuracy: {train_result['train_accuracy']:.4f}")
                logger.info(f"Testing accuracy: {train_result['test_accuracy']:.4f}")
            
            # Generate feature importance plot
            if hasattr(ml_model.model, 'feature_importances_'):
                importance_fig = plot_feature_importance(ml_model.model, features)
                
                # Save feature importance plot
                importance_file = os.path.join(model_dir, f"{ticker}_feature_importance.png")
                importance_fig.savefig(importance_file)
                plt.close(importance_fig)
                logger.info(f"Saved feature importance plot to {importance_file}")
            
            # Save model if requested
            if save_models:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_file = os.path.join(model_dir, f"{ticker}_model_{timestamp}.pkl")
                
                # Create model info dictionary
                model_info = {
                    'model': ml_model.model,
                    'features': features,
                    'train_date': timestamp,
                    'training_data_start': start_date,
                    'training_data_end': end_date,
                    'model_type': model_type,
                    'prediction_horizon': prediction_horizon,
                    'performance': train_result
                }
                
                # Save to file
                with open(model_file, 'wb') as f:
                    pickle.dump(model_info, f)
                
                logger.info(f"Saved trained model to {model_file}")
            
            # Store trained model
            trained_models[ticker] = {
                'model': ml_model.model,
                'features': features,
                'performance': train_result
            }
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
    
    return trained_models

def main():
    """Main function for training models"""
    parser = argparse.ArgumentParser(description='Train trading models')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                      help='Ticker symbols to train models for')
    parser.add_argument('--start_date', default=None,
                      help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--end_date', default=None,
                      help='End date (YYYY-MM-DD format)')
    parser.add_argument('--model_type', default='random_forest',
                      choices=['random_forest', 'gradient_boosting', 'neural_network'],
                      help='Type of model to train')
    parser.add_argument('--sentiment', action='store_true',
                      help='Include sentiment analysis')
    parser.add_argument('--alternative_data', action='store_true',
                      help='Include alternative data')
    parser.add_argument('--optimize', action='store_true',
                      help='Optimize hyperparameters')
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
    
    # Train models
    trained_models = train_models(
        args.tickers, 
        start_date, 
        end_date, 
        args.config, 
        model_type=args.model_type, 
        use_sentiment=args.sentiment,
        use_alternative_data=args.alternative_data,
        optimize_hyperparams=args.optimize
    )
    
    # Print summary
    logger.info("Training Summary:")
    
    for ticker, model_info in trained_models.items():
        performance = model_info.get('performance', {})
        logger.info(f"{ticker}:")
        logger.info(f"  Test Accuracy: {performance.get('test_accuracy', 0):.4f}")
        logger.info(f"  Number of Features: {len(model_info.get('features', []))}")

if __name__ == "__main__":
    main()