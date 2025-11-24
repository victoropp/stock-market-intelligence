import pandas as pd
import numpy as np

class FeatureEngineering:
    """Advanced feature engineering for ML models."""
    
    @staticmethod
    def add_lag_features(df, columns=['Close'], lags=[1, 2, 3, 5, 10]):
        """Add lag features for specified columns."""
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    @staticmethod
    def add_rolling_features(df, column='Close', windows=[5, 10, 20, 50]):
        """Add rolling statistics."""
        df = df.copy()
        
        for window in windows:
            df[f'{column}_Rolling_Mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_Rolling_Std_{window}'] = df[column].rolling(window=window).std()
            df[f'{column}_Rolling_Min_{window}'] = df[column].rolling(window=window).min()
            df[f'{column}_Rolling_Max_{window}'] = df[column].rolling(window=window).max()
        
        return df
    
    @staticmethod
    def add_price_changes(df, periods=[1, 5, 10, 20]):
        """Add price change percentages."""
        df = df.copy()
        
        for period in periods:
            df[f'Price_Change_{period}d'] = df['Close'].pct_change(periods=period)
        
        return df
    
    @staticmethod
    def detect_candlestick_patterns(df):
        """Detect basic candlestick patterns."""
        df = df.copy()
        
        # Doji: Open ≈ Close
        df['Doji'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1).astype(int)
        
        # Hammer: Long lower shadow, small body at top
        body = abs(df['Close'] - df['Open'])
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)
        
        # Bullish Engulfing
        prev_close = df['Close'].shift(1)
        prev_open = df['Open'].shift(1)
        df['Bullish_Engulfing'] = ((df['Close'] > df['Open']) & 
                                    (prev_close < prev_open) & 
                                    (df['Open'] < prev_close) & 
                                    (df['Close'] > prev_open)).astype(int)
        
        return df
    
    @staticmethod
    def create_ml_features(df):
        """Create comprehensive feature set for ML."""
        df = df.copy()
        
        # Lag features
        df = FeatureEngineering.add_lag_features(df, columns=['Close', 'Volume'], lags=[1, 2, 3, 5])
        
        # Rolling features
        df = FeatureEngineering.add_rolling_features(df, column='Close', windows=[5, 10, 20])
        
        # Price changes
        df = FeatureEngineering.add_price_changes(df, periods=[1, 5, 10])
        
        # Candlestick patterns
        df = FeatureEngineering.detect_candlestick_patterns(df)
        
        return df

if __name__ == "__main__":
    from data_loader import StockDataLoader
    
    loader = StockDataLoader()
    aapl = loader.load_stock('aapl')
    aapl = aapl.tail(100)
    
    # Add features
    aapl_features = FeatureEngineering.create_ml_features(aapl)
    print(f"Original columns: {len(aapl.columns)}")
    print(f"With features: {len(aapl_features.columns)}")
    print(f"\nNew features: {[c for c in aapl_features.columns if c not in aapl.columns]}")
