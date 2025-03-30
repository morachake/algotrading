import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from strategies.base_strategy import BaseStrategy

# Get logger
logger = logging.getLogger('strategies.ml')

class MLStrategy(BaseStrategy):
    """
    Machine learning based trading strategy
    """
    
    def __init__(self, name='ml'):
        """
        Initialize the ML strategy
        
        Parameters:
        name (str): Strategy name
        """
        super().__init__(name)
        self.model = None
        self.features = None
        self.prediction_horizon = 5  # Default to 5-day prediction
    
    def initialize(self, **kwargs):
        """
        Initialize the strategy with parameters
        
        Parameters:
        **kwargs: Strategy-specific parameters
            - model_type (str): Type of ML model to use
            - n_estimators (int): Number of estimators for RandomForest
            - max_depth (int): Max depth for RandomForest
            - prediction_horizon (int): Days to predict ahead
            - features (list): Features to use
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Get parameters
            model_type = kwargs.get('model_type', 'random_forest')
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', None)
            self.prediction_horizon = kwargs.get('prediction_horizon', 5)
            self.features = kwargs.get('features', None)
            
            # Initialize model
            if model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return False
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ML strategy: {e}")
            return False
    
    def train(self, data, **kwargs):
        """
        Train the ML model using historical data
        
        Parameters:
        data (DataFrame): Historical market data
        **kwargs: Training parameters
            - test_size (float): Size of test set
            - features (list): Features to use
            - target_type (str): Type of target ('binary', 'direction')
        
        Returns:
        dict: Training results
        """
        try:
            # Validate initialization
            if not self.is_initialized:
                if not self.initialize(**kwargs):
                    logger.error("Failed to initialize strategy")
                    return {}
            
            # Get parameters
            test_size = kwargs.get('test_size', 0.2)
            features = kwargs.get('features', self.features)
            target_type = kwargs.get('target_type', 'binary')
            
            # Prepare data
            processed_data = self.prepare_data(data)
            
            # Define features if not specified
            if features is None:
                features = [
                    'Returns', 'Price_to_SMA20', 'Price_to_SMA50', 
                    'Volatility', 'Return_Lag_1', 'Return_Lag_2',
                    'Return_Lag_3', 'Volume_Change', 'Volume_Ratio',
                    'RSI'
                ]
                
                # Add sentiment features if available
                sentiment_features = [
                    'sentiment_score', 'sentiment_score_3d', 'sentiment_score_7d',
                    'sentiment_positive', 'sentiment_negative'
                ]
                
                features.extend([f for f in sentiment_features if f in processed_data.columns])
            
            # Generate target based on future price movement
            if target_type == 'binary':
                # 1 if price goes up by threshold in next N days, 0 otherwise
                threshold = kwargs.get('threshold', 0.0)
                future_return = processed_data['Adj Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
                processed_data['Target'] = (future_return > threshold).astype(int)
            elif target_type == 'direction':
                # 1 if price goes up, 0 if price goes down
                future_price = processed_data['Adj Close'].shift(-self.prediction_horizon)
                processed_data['Target'] = (future_price > processed_data['Adj Close']).astype(int)
            else:
                logger.error(f"Unsupported target type: {target_type}")
                return {}
            
            # Remove NaN values
            processed_data = processed_data.dropna()
            
            # Split into features and target
            X = processed_data[features]
            y = processed_data['Target']
            
            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            train_preds = self.model.predict(X_train)
            test_preds = self.model.predict(X_test)
            
            # Evaluate performance
            train_accuracy = accuracy_score(y_train, train_preds)
            test_accuracy = accuracy_score(y_test, test_preds)
            
            # Store feature list
            self.features = features
            
            # Set trained flag
            self.is_trained = True
            
            # Log performance
            logger.info(f"Model trained - Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
            
            # Create detailed report
            test_report = classification_report(y_test, test_preds, output_dict=True)
            test_confusion = confusion_matrix(y_test, test_preds).tolist()
            
            # Store feature importance if available
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                for i, feature in enumerate(features):
                    feature_importance[feature] = float(self.model.feature_importances_[i])
            
            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'features': features,
                'feature_importance': feature_importance,
                'test_report': test_report,
                'test_confusion': test_confusion
            }
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return {}
    
    def generate_signals(self, data, **kwargs):
        """
        Generate trading signals using the trained model
        
        Parameters:
        data (DataFrame): Market data
        **kwargs: Signal generation parameters
            - threshold (float): Probability threshold for signals
        
        Returns:
        DataFrame: Trading signals
        """
        try:
            # Validate model
            if not self.is_trained or self.model is None:
                logger.error("Model not trained. Cannot generate signals.")
                return pd.DataFrame()
            
            # Get parameters
            threshold_buy = kwargs.get('threshold_buy', 0.7)
            threshold_sell = kwargs.get('threshold_sell', 0.3)
            
            # Prepare data
            processed_data = self.prepare_data(data)
            
            # Drop rows with NaN values
            processed_data = processed_data.dropna()
            
            # Create signals dataframe
            signals = pd.DataFrame(index=processed_data.index)
            signals['Adj Close'] = processed_data['Adj Close']
            
            # Extract features
            features = [f for f in self.features if f in processed_data.columns]
            
            if len(features) != len(self.features):
                logger.warning(f"Some features are missing: expected {len(self.features)}, got {len(features)}")
            
            X = processed_data[features]
            
            # Generate predictions
            signals['Probability'] = self.model.predict_proba(X)[:, 1]
            signals['Prediction'] = self.model.predict(X)
            
            # Generate trading signals based on probability
            signals['Signal'] = 0  # Default to hold
            signals.loc[signals['Probability'] >= threshold_buy, 'Signal'] = 1  # Buy signal
            signals.loc[signals['Probability'] <= threshold_sell, 'Signal'] = -1  # Sell signal
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return pd.DataFrame()