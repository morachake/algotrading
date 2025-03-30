import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import logging

# Set up logger
logger = logging.getLogger(__name__)

class MLTradingModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.features = None
        
    def prepare_features(self, df):
        """
        Create features for the model
        """
        # Make a copy to avoid changing original data
        data = df.copy()
        
        # Create features
        data['Returns'] = data['Adj Close'].pct_change()
        data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
        
        # Price distance from moving averages
        data['Price_to_SMA20'] = data['Adj Close'] / data['SMA_20'] - 1
        data['Price_to_SMA50'] = data['Adj Close'] / data['SMA_50'] - 1
        
        # Moving average crossover
        data['SMA_20_50_Crossover'] = (data['SMA_20'] > data['SMA_50']).astype(int)
        
        # Volatility (standard deviation of returns)
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Return lags
        for i in range(1, 6):
            data[f'Return_Lag_{i}'] = data['Returns'].shift(i)
            
        # Volume features
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Volume_SMA_5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_5']
        
        # Target: 1 if price goes up in next n days, 0 otherwise
        # Here using 5-day forward returns
        data['Target'] = (data['Adj Close'].shift(-5) > data['Adj Close']).astype(int)
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def train(self, data, features=None):
        """
        Train the model
        
        Parameters:
        data (DataFrame): Prepared data with features and target
        features (list): List of feature column names, if None uses default features
        
        Returns:
        dict: Training results
        """
        if features is None:
            features = [
                'Returns', 'Price_to_SMA20', 'Price_to_SMA50', 
                'Volatility', 'Return_Lag_1', 'Return_Lag_2',
                'Return_Lag_3', 'Volume_Change', 'Volume_Ratio'
            ]
        
        X = data[features]
        y = data['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_preds)
        test_accuracy = accuracy_score(y_test, test_preds)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, test_preds))
        
        # Feature importance
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(X.shape[1]), importance[indices], align='center')
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        self.features = features
        self.is_trained = True
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': dict(zip([features[i] for i in indices], importance[indices]))
        }
    
    def predict(self, data):
        """
        Generate predictions for new data
        
        Parameters:
        data (DataFrame): Prepared data with features
        
        Returns:
        DataFrame: Original data with predictions and signal columns
        """
        if not self.is_trained:
            raise Exception("Model has not been trained yet!")
            
        # Make a copy of the data
        result = data.copy()
        
        # Get model features
        X = result[self.features]
        
        # Generate predictions (probability of price going up)
        result['Prediction'] = self.model.predict(X)
        result['Probability'] = self.model.predict_proba(X)[:, 1]
        
        # Generate trading signals
        # 1 = Buy, 0 = Hold, -1 = Sell
        result['Signal'] = 0
        result.loc[result['Probability'] > 0.7, 'Signal'] = 1
        result.loc[result['Probability'] < 0.3, 'Signal'] = -1
        
        return result