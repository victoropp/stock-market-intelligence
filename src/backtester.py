import pandas as pd
import numpy as np

class Backtester:
    """Simple backtesting engine for trading strategies."""
    
    def __init__(self, initial_capital=10000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run_strategy(self, df, buy_signal_col, sell_signal_col):
        """
        Backtest a strategy based on buy/sell signals.
        
        Args:
            df: DataFrame with price data and signals
            buy_signal_col: Column name for buy signals (1 = buy)
            sell_signal_col: Column name for sell signals (1 = sell)
        
        Returns:
            Dictionary with performance metrics
        """
        df = df.copy()
        
        # Initialize tracking variables
        cash = self.initial_capital
        shares = 0
        portfolio_value = []
        trades = []
        
        for i, row in df.iterrows():
            # Buy signal
            if row[buy_signal_col] == 1 and shares == 0:
                shares_to_buy = cash // row['Close']
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['Close'] * (1 + self.commission)
                    if cost <= cash:
                        shares = shares_to_buy
                        cash -= cost
                        trades.append({'Date': i, 'Type': 'BUY', 'Price': row['Close'], 'Shares': shares})
            
            # Sell signal
            elif row[sell_signal_col] == 1 and shares > 0:
                proceeds = shares * row['Close'] * (1 - self.commission)
                cash += proceeds
                trades.append({'Date': i, 'Type': 'SELL', 'Price': row['Close'], 'Shares': shares})
                shares = 0
            
            # Calculate portfolio value
            current_value = cash + (shares * row['Close'])
            portfolio_value.append(current_value)
        
        # Final liquidation
        if shares > 0:
            final_price = df.iloc[-1]['Close']
            cash += shares * final_price * (1 - self.commission)
        
        # Calculate metrics
        df['Portfolio_Value'] = portfolio_value
        total_return = (cash - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe Ratio
        returns = df['Portfolio_Value'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = df['Portfolio_Value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        trade_returns = []
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                buy_price = trades[i]['Price']
                sell_price = trades[i + 1]['Price']
                trade_returns.append((sell_price - buy_price) / buy_price)
        
        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
        
        return {
            'Total Return (%)': total_return * 100,
            'Final Portfolio Value': cash,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Number of Trades': len(trades),
            'Win Rate (%)': win_rate * 100,
            'Trades': trades,
            'Portfolio History': df[['Close', 'Portfolio_Value']]
        }

if __name__ == "__main__":
    from data_loader import StockDataLoader
    from technical_indicators import TechnicalIndicators
    
    # Load and prepare data
    loader = StockDataLoader()
    aapl = loader.load_stock('aapl')
    aapl = aapl['2020-01-01':'2023-12-31']  # 4 years
    
    # Add indicators and signals
    aapl = TechnicalIndicators.add_all_indicators(aapl)
    aapl = TechnicalIndicators.add_price_features(aapl)
    aapl = TechnicalIndicators.generate_signals(aapl)
    
    # Run backtest
    backtester = Backtester(initial_capital=10000)
    results = backtester.run_strategy(aapl, 'MACD_Bullish', 'MACD_Bearish')
    
    print("Backtest Results:")
    for key, value in results.items():
        if key not in ['Trades', 'Portfolio History']:
            print(f"{key}: {value}")
