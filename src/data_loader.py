import pandas as pd
import os
import glob

class StockDataLoader:
    """Handles loading and preprocessing of stock market data.

    Supports two modes:
    1. Local mode: Load from CSV files (for development)
    2. Cloud mode: Fetch from yfinance (for Streamlit Cloud deployment)
    """

    def __init__(self, data_dir=None, use_yfinance=None):
        """Initialize the data loader.

        Args:
            data_dir: Path to local data directory (optional)
            use_yfinance: Force yfinance mode (True) or local mode (False).
                         If None, auto-detect based on environment.
        """
        self.use_yfinance = use_yfinance

        if data_dir is None:
            # Check if running on Streamlit Cloud or local
            if os.environ.get('STREAMLIT_RUNTIME_ENV') or not self._local_data_exists():
                self.use_yfinance = True
                self.data_dir = None
            else:
                self.use_yfinance = False
                # Try relative path first, then absolute
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                local_data_path = os.path.join(project_root, 'data', 'Stocks')

                if os.path.exists(local_data_path):
                    self.data_dir = local_data_path
                else:
                    # Fallback to absolute path for development
                    self.data_dir = r"C:\Users\victo\Documents\Data_Science_Projects\Data Science Portfolio Projects\datasets\financial\stock_market\Stocks"
        else:
            self.data_dir = data_dir
            self.use_yfinance = False

    def _local_data_exists(self):
        """Check if local data directory exists."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_data_path = os.path.join(project_root, 'data', 'Stocks')
        return os.path.exists(local_data_path)

    def load_stock(self, ticker, start_date='2010-01-01', end_date=None):
        """Load a single stock's data.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date for yfinance (default: 2010-01-01)
            end_date: End date for yfinance (default: today)

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        if self.use_yfinance:
            return self._load_from_yfinance(ticker, start_date, end_date)
        else:
            return self._load_from_file(ticker)

    def _load_from_yfinance(self, ticker, start_date, end_date):
        """Load stock data from Yahoo Finance."""
        try:
            import yfinance as yf

            # Use download() which is more reliable than Ticker.history()
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                raise ValueError(f"No data available for {ticker}")

            # Handle multi-level columns (yfinance 0.2.x returns MultiIndex columns)
            if isinstance(df.columns, pd.MultiIndex):
                # New yfinance format: ('Price', 'Ticker') - get first level (Price names)
                df.columns = df.columns.droplevel(1)

            # Ensure we have standard column names
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Check if columns exist (case-insensitive)
            col_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower()
                for expected in expected_cols:
                    if col_lower == expected.lower():
                        col_mapping[col] = expected
                        break

            df = df.rename(columns=col_mapping)

            # Keep only OHLCV columns that exist
            cols_to_keep = [c for c in expected_cols if c in df.columns]
            df = df[cols_to_keep]

            if len(cols_to_keep) == 0:
                raise ValueError(f"No valid OHLCV columns found for {ticker}")

            return df

        except ImportError:
            raise ImportError("yfinance is required for cloud mode. Install with: pip install yfinance")
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Error fetching {ticker} from Yahoo Finance: {e}")

    def _load_from_file(self, ticker):
        """Load stock data from local CSV file."""
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
        """Get list of available ticker symbols.

        For yfinance mode, returns a predefined list of popular stocks.
        For local mode, scans the data directory.
        """
        if self.use_yfinance:
            # Popular stocks for cloud mode
            popular = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                      'JPM', 'V', 'WMT', 'JNJ', 'PG', 'XOM', 'BAC', 'DIS']
            if limit:
                return popular[:limit]
            return popular

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
                if self.use_yfinance:
                    df = self.load_stock(ticker, start_date or '2010-01-01', end_date)
                else:
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
    print(f"Using yfinance: {loader.use_yfinance}")
    print(f"Available tickers: {loader.get_available_tickers(limit=10)}")

    # Load AAPL
    aapl = loader.load_stock('AAPL')
    print(f"\nAAPL data shape: {aapl.shape}")
    print(aapl.tail())
