import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import time
from datetime import datetime, timedelta
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Download NLTK data (uncomment and run once)
# nltk.download('vader_lexicon')
# nltk.download('punkt')

class FinancialSentimentAnalyzer:
    def __init__(self):
        try:
            self.sid = SentimentIntensityAnalyzer()
            
            # Add financial terms to the sentiment analyzer
            # These are domain-specific terms that VADER doesn't recognize correctly
            self.financial_dict = {
                "bullish": 3.0,
                "bearish": -3.0,
                "outperform": 2.0,
                "underperform": -2.0,
                "buy": 2.0,
                "sell": -2.0,
                "upgrade": 2.5,
                "downgrade": -2.5,
                "exceeded expectations": 3.0,
                "missed expectations": -3.0,
                "beat": 3.0,
                "missed": -3.0,
                "above estimates": 2.5,
                "below estimates": -2.5,
                "guidance raised": 3.0,
                "guidance lowered": -3.0,
                "positive outlook": 2.0,
                "negative outlook": -2.0,
                "strong growth": 3.0,
                "weak growth": -3.0,
                "dividend increase": 2.0,
                "dividend cut": -2.5,
                "layoffs": -2.0,
                "restructuring": -1.0,
                "expansion": 2.0,
                "acquisition": 1.0,
                "merger": 1.0,
                "partnership": 1.5,
                "lawsuit": -1.5,
                "investigation": -2.0,
                "fine": -1.5,
                "record high": 3.0,
                "record low": -3.0,
                "bankruptcy": -3.0,
                "default": -3.0
            }
            
            # Update the VADER lexicon
            self.sid.lexicon.update(self.financial_dict)
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            # Create a dummy analyzer if NLTK fails
            self.sid = None
    
    def clean_text(self, text):
        """Clean the text by removing unwanted characters"""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a piece of text
        
        Returns:
        dict: Sentiment scores using VADER and TextBlob
        """
        if self.sid is None:
            logger.warning("Sentiment analyzer not properly initialized. Using TextBlob only.")
            blob = TextBlob(self.clean_text(text))
            return {
                'text': text[:200] + "..." if len(text) > 200 else text,
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity,
                'combined_score': blob.sentiment.polarity,
                'sentiment': "positive" if blob.sentiment.polarity > 0.05 else "negative" if blob.sentiment.polarity < -0.05 else "neutral"
            }
        
        clean = self.clean_text(text)
        
        # VADER sentiment
        vader_scores = self.sid.polarity_scores(clean)
        
        # TextBlob sentiment
        blob = TextBlob(clean)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Combined sentiment score (weighted average)
        # Giving more weight to VADER as it's better for financial text
        combined_score = 0.7 * vader_scores['compound'] + 0.3 * textblob_polarity
        
        # Determine sentiment label
        if combined_score >= 0.05:
            sentiment = "positive"
        elif combined_score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            'text': text[:200] + "..." if len(text) > 200 else text,  # Preview
            'clean_text': clean[:200] + "..." if len(clean) > 200 else clean,  # Preview
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'vader_pos': vader_scores['pos'],
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'combined_score': combined_score,
            'sentiment': sentiment
        }
    
    def fetch_financial_news(self, ticker, num_articles=10):
        """
        Scrape financial news for a given ticker symbol
        NOTE: This is a basic implementation. In a real system, you would use APIs
        from news providers or specialized services.
        """
        print(f"Fetching news for {ticker}...")
        
        # Example: Fetch from Yahoo Finance
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract news article titles and links
            articles = []
            for item in soup.find_all('h3', limit=num_articles):
                title = item.text.strip()
                
                # Skip ads
                if "Ad" in title or len(title) < 10:
                    continue
                
                articles.append({
                    'title': title,
                    'date': datetime.now().strftime('%Y-%m-%d'),  # In real code, extract actual date
                    'ticker': ticker
                })
            
            print(f"Found {len(articles)} articles for {ticker}")
            return articles
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            # Return dummy data in case of failure
            return [
                {
                    'title': f"{ticker} stock continues to perform in market",
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'ticker': ticker
                },
                {
                    'title': f"Analysts remain optimistic about {ticker}'s future",
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'ticker': ticker
                }
            ]
    
    def analyze_news_sentiment(self, ticker, num_articles=10):
        """
        Fetch and analyze sentiment for recent news about a ticker
        
        Returns:
        DataFrame: News articles with sentiment analysis
        """
        articles = self.fetch_financial_news(ticker, num_articles)
        results = []
        
        for article in articles:
            # In a real system, you would fetch and analyze the full article text
            # Here we're just analyzing the title as a simple example
            sentiment_data = self.analyze_sentiment(article['title'])
            
            results.append({
                'date': article['date'],
                'ticker': article['ticker'],
                'title': article['title'],
                'combined_score': sentiment_data['combined_score'],
                'sentiment': sentiment_data['sentiment']
            })
        
        return pd.DataFrame(results)
    
    def generate_sentiment_features(self, df, ticker):
        """
        Generate sentiment features to incorporate into ML model
        
        Parameters:
        df (DataFrame): Price data DataFrame with dates as index
        ticker (str): Ticker symbol
        
        Returns:
        DataFrame: Original DataFrame with added sentiment features
        """
        # Get sentiment data
        try:
            sentiment_df = self.analyze_news_sentiment(ticker, num_articles=30)
            
            # Convert to daily aggregated sentiment
            if not sentiment_df.empty:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date).agg({
                    'combined_score': ['mean', 'min', 'max', 'count'],
                    'sentiment': lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else "neutral"  # Most common sentiment
                })
                
                # Flatten multi-index columns
                daily_sentiment.columns = ['sentiment_score_avg', 'sentiment_score_min', 
                                        'sentiment_score_max', 'article_count', 'dominant_sentiment']
                
                # Convert index to datetime
                daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
                
                # Merge with original dataframe
                result = df.copy()
                
                for date, row in daily_sentiment.iterrows():
                    if date in result.index:
                        result.loc[date, 'sentiment_score'] = row['sentiment_score_avg']
                        result.loc[date, 'sentiment_article_count'] = row['article_count']
                        
                        # Encode dominant sentiment
                        if row['dominant_sentiment'] == 'positive':
                            result.loc[date, 'sentiment_positive'] = 1
                            result.loc[date, 'sentiment_negative'] = 0
                        elif row['dominant_sentiment'] == 'negative':
                            result.loc[date, 'sentiment_positive'] = 0
                            result.loc[date, 'sentiment_negative'] = 1
                        else:
                            result.loc[date, 'sentiment_positive'] = 0
                            result.loc[date, 'sentiment_negative'] = 0
                
                # Fill missing values with 0 (days without news)
                result['sentiment_score'] = result['sentiment_score'].fillna(0)
                result['sentiment_article_count'] = result['sentiment_article_count'].fillna(0)
                result['sentiment_positive'] = result['sentiment_positive'].fillna(0)
                result['sentiment_negative'] = result['sentiment_negative'].fillna(0)
                
                # Create rolling sentiment features
                result['sentiment_score_3d'] = result['sentiment_score'].rolling(window=3).mean()
                result['sentiment_score_7d'] = result['sentiment_score'].rolling(window=7).mean()
                
                return result
            else:
                # If no sentiment data, return original dataframe
                logger.warning(f"No sentiment data found for {ticker}")
                return df
        except Exception as e:
            logger.error(f"Error generating sentiment features: {e}")
            return df  # Return original dataframe if there's an error