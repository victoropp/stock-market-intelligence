import pandas as pd
import os
import glob

class StockDataLoader:
    """Handles loading and preprocessing of stock market data."""
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Use absolute path to datasets
            self.data_dir = r"C:\Users\victo\Documents\Data_Science_Projects\Data Science Portfolio Projects\datasets\financial\stock_market\Stocks"
        else:
            self.data_dir = data_dir
    
    def load_stock(self, ticker):
        """Load a single stock's data."""
        filepath = os.path.join(self.data_dir, f"{ticker.lower()}.us.txt")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Stock file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df.set_index('Date')
        
        # Drop OpenInt column (usually all zeros)
        if 'OpenInt' in df.columns:
            df = df.drop('OpenInt', axis=1)
        
        return df
    
    def get_available_tickers(self, limit=None):
        """Get list of available ticker symbols."""
        files = glob.glob(os.path.join(self.data_dir, "*.us.txt"))
        tickers = [os.path.basename(f).replace('.us.txt', '').upper() for f in files]
        
        if limit:
            return sorted(tickers)[:limit]
        return sorted(tickers)
    
    def load_multiple_stocks(self, tickers, start_date=None, end_date=None):
        """Load multiple stocks and return a dictionary."""
        data = {}
        
        for ticker in tickers:
            try:
                df = self.load_stock(ticker)
                
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                data[ticker] = df
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
        
        return data

if __name__ == "__main__":
    # Test
    loader = StockDataLoader()
    print(f"Total stocks available: {len(loader.get_available_tickers())}")
    
    # Load AAPL
    aapl = loader.load_stock('aapl')
    print(f"\nAAPL data shape: {aapl.shape}")
    print(aapl.head())
