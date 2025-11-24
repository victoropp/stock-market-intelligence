import pandas as pd
import numpy as np
import ta

class TechnicalIndicators:
    """Calculate technical indicators for stock data."""
    
    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators to a dataframe."""
        df = df.copy()
        
        # Check if we have enough data
        if len(df) < 50:
            print(f"Warning: Only {len(df)} rows. Need at least 50 for indicators.")
            return df
        
        try:
            # Momentum Indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['Stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            
            # Trend Indicators
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
            df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_High'] = bollinger.bollinger_hband()
            df['BB_Low'] = bollinger.bollinger_lband()
            df['BB_Mid'] = bollinger.bollinger_mavg()
            
            # Volatility
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
            # Volume
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return df
    
    @staticmethod
    def add_price_features(df):
        """Add price-based features."""
        df = df.copy()
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price changes
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Range'] = df['High'] - df['Low']
        
        # Rolling statistics
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        df['Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()
        df['Rolling_Std_20'] = df['Close'].rolling(window=20).std()
        
        return df
    
    @staticmethod
    def generate_signals(df):
        """Generate basic trading signals."""
        df = df.copy()
        
        # Check if indicators exist
        required_indicators = ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50']
        missing = [ind for ind in required_indicators if ind not in df.columns]
        
        if missing:
            print(f"Warning: Missing indicators {missing}. Call add_all_indicators() first.")
            return df
        
        # RSI signals
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        
        # MACD crossover
        df['MACD_Bullish'] = ((df['MACD'] > df['MACD_Signal']) & 
                              (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
        df['MACD_Bearish'] = ((df['MACD'] < df['MACD_Signal']) & 
                              (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))).astype(int)
        
        # Moving Average crossover
        df['MA_Bullish'] = ((df['SMA_20'] > df['SMA_50']) & 
                            (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))).astype(int)
        df['MA_Bearish'] = ((df['SMA_20'] < df['SMA_50']) & 
                            (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))).astype(int)
        
        return df

if __name__ == "__main__":
    # Test with sample data
    from data_loader import StockDataLoader
    
    loader = StockDataLoader()
    aapl = loader.load_stock('aapl')
    
    # Add indicators
    aapl_with_indicators = TechnicalIndicators.add_all_indicators(aapl)
    aapl_with_indicators = TechnicalIndicators.add_price_features(aapl_with_indicators)
    aapl_with_indicators = TechnicalIndicators.generate_signals(aapl_with_indicators)
    
    print(aapl_with_indicators.tail())
    print(f"\nColumns: {aapl_with_indicators.columns.tolist()}")
