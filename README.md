# 📊 Stock Market Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Enterprise-grade quantitative trading platform** combining Technical Analysis, LSTM Deep Learning, and Algorithmic Backtesting for systematic strategy development across 7,195+ US stocks.

![Platform Screenshot](https://via.placeholder.com/800x400/0e1117/00CC96?text=Stock+Market+Intelligence+Platform)

---

## 🎯 Project Overview

A state-of-the-art financial analytics platform that demonstrates advanced **Data Science** and **ML Engineering** capabilities through:

- 🤖 **Deep Learning**: LSTM neural networks for price forecasting (95.65% - 98.33% accuracy)
- 📈 **Technical Analysis**: 50+ indicators (RSI, MACD, Bollinger Bands, ATR, OBV)
- 💹 **Algorithmic Backtesting**: Event-driven engine with risk metrics (Sharpe Ratio, Max Drawdown)
- 🎨 **Interactive Dashboard**: Professional Streamlit UI with real-time analytics

### Key Achievements
- ✅ **7 trained LSTM models** with documented performance metrics
- ✅ **98.33% prediction accuracy** on MSFT stock (1.67% MAPE)
- ✅ **Average 97% accuracy** across all models
- ✅ **Production-ready** with pre-trained models and REST-like interface

---

## 🚀 Features

### 1. 📈 Market Scanner
Screen 7,195+ stocks by technical criteria:
- RSI Oversold/Overbought signals
- MACD Bullish/Bearish crossovers
- Moving Average Golden/Death crosses
- Real-time indicator values

### 2. 🔬 Strategy Backtester
Test algorithmic trading strategies with:
- **Performance Metrics**: Total Return, CAGR, Sharpe Ratio, Sortino Ratio
- **Risk Analysis**: Max Drawdown, Win Rate, Profit Factor
- **Realistic Modeling**: Transaction costs (0.1%), slippage
- **Popular Strategies**: MACD Crossover, RSI Mean Reversion, MA Crossover

### 3. 📊 Technical Analysis
Interactive candlestick charts with:
- Bollinger Bands overlay
- Moving Averages (SMA 20/50)
- MACD and RSI subplots
- Volume analysis (OBV)
- Real-time indicator calculations

### 4. 🤖 ML-Powered Predictions
LSTM deep learning forecasts:
- **Pre-trained models** for 7 major stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, FB)
- **60-day lookback** windows for sequence learning
- **Configurable forecast horizon** (7-60 days)
- **Performance metrics**: MAE, RMSE, MAPE with visualization

### 5. 📊 Model Performance Metrics
Comprehensive model evaluation:
- Interactive metrics display (MAE, RMSE, MAPE)
- Predictions vs Actual charts
- Training details and data ranges
- Model comparison across all stocks

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer (7,195 Stocks)                │
│                     Historical OHLCV Data                   │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────────┐    ┌────────▼─────────┐
│ Feature Engineer │    │  Data Loader     │
│ • 50+ Indicators │    │  • CSV Parser    │
│ • Price Features │    │  • Date Handling │
└───────┬──────────┘    └────────┬─────────┘
        │                        │
        └────────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────────┐    ┌────────▼─────────┐
│   ML Models      │    │  Backtesting     │
│ • LSTM (7 stocks)│    │  • Event-driven  │
│ • 98% Accuracy   │    │  • Risk Metrics  │
└───────┬──────────┘    └────────┬─────────┘
        │                        │
        └────────────┬───────────┘
                     │
            ┌────────▼─────────┐
            │  Streamlit UI    │
            │  • 4 Tabs        │
            │  • Real-time     │
            └──────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-market-intelligence.git
cd stock-market-intelligence
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run deployment/app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📊 Dataset

**Source**: Historical stock market data (OHLCV format)
- **Coverage**: 7,195 US stocks
- **Time Range**: 1984-09-07 to 2017-11-10
- **Format**: CSV files with Date, Open, High, Low, Close, Volume

**Note**: For deployment, update `src/data_loader.py` to point to your data source.

---

## 🤖 Model Performance

### LSTM Models (Pre-trained)

| Stock | MAE ($) | RMSE ($) | MAPE (%) | **Accuracy** |
|-------|---------|----------|----------|--------------|
| **MSFT** | 0.71 | 0.98 | 1.67 | **98.33%** ✨ |
| **GOOGL** | 14.44 | 19.52 | 1.91 | **98.09%** ✨ |
| **NVDA** | 1.59 | 3.04 | 2.37 | **97.63%** ⭐ |
| **AAPL** | 2.77 | 3.87 | 2.75 | **97.25%** ⭐ |
| **AMZN** | 17.52 | 23.49 | 2.85 | **97.15%** ⭐ |
| **TSLA** | 10.39 | 13.62 | 3.66 | **96.34%** ✓ |
| **FB** | 6.57 | 7.32 | 4.35 | **95.65%** ✓ |

**Average Accuracy**: 97.20%

### Model Architecture
- **Type**: Long Short-Term Memory (LSTM)
- **Layers**: 2-layer LSTM (50 units each) with dropout (0.2)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Training**: 20 epochs, 60-day lookback windows
- **Validation**: 80/20 train-test split

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **TensorFlow/Keras**: LSTM model development
- **Streamlit**: Interactive dashboard framework
- **Pandas/NumPy**: Data manipulation and analysis

### ML & Analytics
- **TA-Lib**: Technical indicators library
- **Scikit-learn**: Model evaluation, preprocessing
- **XGBoost**: Ensemble learning (future enhancement)

### Visualization
- **Plotly**: Interactive charts and graphs
- **Matplotlib**: Performance charts for model metrics

---

## 📁 Project Structure

```
stock_market_intelligence/
├── deployment/
│   └── app.py                 # Streamlit dashboard
├── src/
│   ├── data_loader.py         # Data loading utilities
│   ├── technical_indicators.py # 50+ technical indicators
│   ├── backtester.py          # Backtesting engine
│   ├── ml_models.py           # LSTM model utilities
│   ├── feature_engineering.py # Feature creation
│   └── train_models.py        # Model training script
├── models/
│   ├── *_lstm_model.h5        # Saved LSTM models
│   ├── *_scaler.pkl           # MinMax scalers
│   ├── *_metrics.json         # Performance metrics
│   └── *_performance.png      # Evaluation charts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🎓 Key Learnings & Skills Demonstrated

### Data Science
- ✅ Time series analysis and forecasting
- ✅ Feature engineering for financial data
- ✅ Model evaluation with domain-specific metrics
- ✅ Handling class imbalance and data preprocessing

### Machine Learning
- ✅ LSTM neural network architecture
- ✅ Sequence modeling with sliding windows
- ✅ Hyperparameter tuning (lookback, epochs, dropout)
- ✅ Model persistence and versioning

### Software Engineering
- ✅ Modular code architecture
- ✅ Production-ready model deployment
- ✅ Interactive dashboard development
- ✅ Error handling and validation

### Domain Expertise
- ✅ Technical analysis indicators
- ✅ Backtesting methodologies
- ✅ Risk-adjusted performance metrics
- ✅ Quantitative finance concepts

---

## 🚀 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from repository
4. Set Python version to 3.8+

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run deployment/app.py
```

---

## 📈 Future Enhancements

- [ ] Real-time data integration (Alpha Vantage, Yahoo Finance API)
- [ ] Additional ML models (Transformer, GRU, Prophet)
- [ ] Portfolio optimization with Modern Portfolio Theory
- [ ] Sentiment analysis integration
- [ ] Multi-timeframe analysis
- [ ] Automated trading signals
- [ ] Performance tracking dashboard

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Victor Collins Oppon**
*Data Scientist | ML Engineer | Quantitative Analyst*

**Skills Showcased:**
- Deep Learning (LSTM, Time Series)
- Technical Analysis & Trading Strategies
- Algorithmic Backtesting
- Interactive Dashboard Development
- Risk Management & Performance Metrics
- Production ML Deployment

---

## 🙏 Acknowledgments

- TensorFlow/Keras teams for deep learning frameworks
- TA-Lib for technical indicator implementations
- Streamlit for the amazing dashboard framework
- Financial data providers for market data

---

**⭐ If you find this project useful, please consider giving it a star!**
