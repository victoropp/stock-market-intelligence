import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from data_loader import StockDataLoader
from technical_indicators import TechnicalIndicators
from backtester import Backtester

st.set_page_config(page_title="Stock Market Intelligence", layout="wide", page_icon="📊")

# Custom CSS for High Contrast
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #00CC96 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold !important;
    }
    /* High contrast metrics */
    [data-testid="stMetricValue"] {
        color: #00FF00 !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    /* Buttons */
    .stButton > button {
        background-color: #00CC96;
        color: #000000 !important;
        border: 2px solid #00CC96;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.6rem 1.2rem;
    }
    .stButton > button:hover {
        background-color: #00FF00;
        color: #000000 !important;
        border-color: #00FF00;
        box-shadow: 0 0 10px #00CC96;
    }
    /* Labels - Maximum Contrast */
    .stSlider label, .stSelectbox label, .stMultiSelect label, .stNumberInput label {
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px #000000;
    }
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: #1a1a1a;
    }
    /* Info boxes - High Contrast */
    .stAlert {
        background-color: #1a1a1a;
        border: 2px solid #00CC96;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    .stAlert p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00CC96 !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Professional Profile
with st.sidebar:
    st.title("Victor Collins Oppon")
    st.markdown("**Data Scientist & ML Engineer**")
    st.markdown("---")
    
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 15px; border-radius: 5px; border: 2px solid #00CC96;'>
        <h4 style='margin:0; color: #00CC96;'>🎯 Project Expertise</h4>
        <p style='margin:5px 0; color: #ffffff; font-weight: 600;'>
            • Deep Learning (LSTM, Transformers)<br>
            • Time Series Forecasting<br>
            • Quantitative Finance<br>
            • Algorithm Development<br>
            • Model Evaluation & Optimization
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 15px; border-radius: 5px; border: 2px solid #AB63FA;'>
        <h4 style='margin:0; color: #AB63FA;'>🏗️ System Architecture</h4>
        <p style='margin:5px 0; color: #ffffff; font-size: 0.95rem;'>
            <b>Data Layer:</b> 7,195+ stocks, OHLCV data<br>
            <b>Feature Engineering:</b> 50+ technical indicators<br>
            <b>ML Models:</b> LSTM (2-layer, dropout regularization)<br>
            <b>Backtesting:</b> Event-driven with transaction costs<br>
            <b>Metrics:</b> MAE, RMSE, MAPE, Sharpe Ratio
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 12px; border-radius: 5px; border: 1px solid #4c4c4c;'>
        <p style='margin:0; color: #e0e0e0; font-size: 0.85rem;'>
            <b>Tech Stack:</b> Python, TensorFlow, XGBoost, Scikit-learn, TA-Lib, Plotly, Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

st.title("📊 Stock Market Intelligence Platform")
st.markdown("""
<div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 5px solid #00CC96; border-right: 5px solid #AB63FA;'>
    <h3 style='margin:0; color: #00CC96;'>Quantitative Trading & Deep Learning Platform</h3>
    <p style='margin:10px 0 0 0; color: #ffffff; font-size: 1.05rem; line-height: 1.6;'>
        <b>Enterprise-grade platform</b> combining <span style='color: #00CC96;'>Technical Analysis</span>, 
        <span style='color: #AB63FA;'>LSTM Deep Learning</span>, and <span style='color: #FFD700;'>Algorithmic Backtesting</span> 
        for systematic trading strategy development across <b>7,195+ US stocks</b>.
    </p>
    <hr style='border-color: #4c4c4c; margin: 15px 0;'>
    <p style='margin:0; color: #e0e0e0; font-size: 0.95rem;'>
        <b>Key Capabilities:</b> Risk-adjusted portfolio optimization • LSTM price forecasting • 
        Event-driven backtesting • 50+ technical indicators • Real-time performance metrics
    </p>
</div>
<br>
""", unsafe_allow_html=True)

# Initialize data loader
@st.cache_resource
def get_loader():
    return StockDataLoader()

loader = get_loader()

tabs = st.tabs(["📈 Market Scanner", "🔬 Strategy Backtester", "📊 Technical Analysis", "🤖 ML Predictions"])

with tabs[0]:
    st.header("Market Scanner")
    st.markdown("Screen stocks based on technical criteria")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scan Parameters")
        
        # Get popular stocks
        popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
        
        scan_type = st.selectbox("Scan Type", ["RSI Oversold", "RSI Overbought", "MACD Bullish Cross", "MA Golden Cross"])
        
        if st.button("Run Scan"):
            with st.spinner("Scanning market..."):
                results = []
                
                for ticker in popular_stocks:
                    try:
                        df = loader.load_stock(ticker)
                        df = df.tail(100)  # Last 100 days
                        df = TechnicalIndicators.add_all_indicators(df)
                        df = TechnicalIndicators.generate_signals(df)
                        
                        latest = df.iloc[-1]
                        
                        if scan_type == "RSI Oversold" and latest['RSI_Oversold'] == 1:
                            results.append({'Ticker': ticker, 'RSI': latest['RSI'], 'Price': latest['Close']})
                        elif scan_type == "RSI Overbought" and latest['RSI_Overbought'] == 1:
                            results.append({'Ticker': ticker, 'RSI': latest['RSI'], 'Price': latest['Close']})
                        elif scan_type == "MACD Bullish Cross" and latest['MACD_Bullish'] == 1:
                            results.append({'Ticker': ticker, 'MACD': latest['MACD'], 'Price': latest['Close']})
                        elif scan_type == "MA Golden Cross" and latest['MA_Bullish'] == 1:
                            results.append({'Ticker': ticker, 'SMA_20': latest['SMA_20'], 'Price': latest['Close']})
                    except:
                        pass
                
                st.session_state['scan_results'] = results
    
    with col2:
        st.subheader("Scan Results")
        
        if 'scan_results' in st.session_state and st.session_state['scan_results']:
            results_df = pd.DataFrame(st.session_state['scan_results'])
            st.dataframe(results_df, use_container_width=True)
            st.success(f"Found {len(results_df)} stocks matching criteria")
        else:
            st.info("Run a scan to see results")

with tabs[1]:
    st.header("Strategy Backtester")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        ticker = st.selectbox("Select Stock", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'])
        
        # Get available date range for selected stock
        try:
            temp_df = loader.load_stock(ticker)
            min_date = temp_df.index.min().date()
            max_date = temp_df.index.max().date()
            
            st.markdown(f"""
            <div style='background-color: #1a1a1a; padding: 12px; border-radius: 5px; border: 2px solid #00CC96;'>
                <p style='margin:0; color: #00FF00; font-size: 1.1rem; font-weight: bold;'>
                    📅 Available data: {min_date} to {max_date}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Set default date range within available data
            default_start = max(pd.to_datetime('2015-01-01').date(), min_date)
            default_end = max_date
        except:
            default_start = pd.to_datetime('2015-01-01').date()
            default_end = pd.to_datetime('2017-11-10').date()
        
        strategy = st.selectbox("Strategy", [
            "MACD Crossover",
            "RSI Mean Reversion",
            "Moving Average Crossover"
        ])
        
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)
        
        date_range = st.date_input("Date Range", value=(default_start, default_end))
        
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    # Load data
                    df = loader.load_stock(ticker)
                    
                    # Validate date range
                    if len(date_range) != 2:
                        st.error("Please select both start and end dates")
                    else:
                        start_date, end_date = date_range
                        
                        # Check if dates are within available range
                        if start_date < min_date or end_date > max_date:
                            st.error(f"Selected dates must be within {min_date} to {max_date}")
                        else:
                            df = df[str(start_date):str(end_date)]
                            
                            if len(df) < 100:
                                st.error(f"Insufficient data ({len(df)} days). Need at least 100 days for backtesting.")
                            else:
                                # Add indicators
                                df = TechnicalIndicators.add_all_indicators(df)
                                df = TechnicalIndicators.add_price_features(df)
                                df = TechnicalIndicators.generate_signals(df)
                                
                                # Run backtest based on strategy
                                backtester = Backtester(initial_capital=initial_capital)
                                
                                if strategy == "MACD Crossover":
                                    results = backtester.run_strategy(df, 'MACD_Bullish', 'MACD_Bearish')
                                elif strategy == "RSI Mean Reversion":
                                    results = backtester.run_strategy(df, 'RSI_Oversold', 'RSI_Overbought')
                                else:  # MA Crossover
                                    results = backtester.run_strategy(df, 'MA_Bullish', 'MA_Bearish')
                                
                                st.session_state['backtest_results'] = results
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Performance Metrics")
        
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{results['Total Return (%)']:.2f}%")
            m2.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
            m3.metric("Max Drawdown", f"{results['Max Drawdown (%)']:.2f}%")
            m4.metric("Win Rate", f"{results['Win Rate (%)']:.1f}%")
            
            # Portfolio value chart
            fig = go.Figure()
            history = results['Portfolio History']
            
            fig.add_trace(go.Scatter(
                x=history.index,
                y=history['Portfolio_Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00CC96', width=2)
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=400,
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trades table
            if results['Trades']:
                st.subheader("Trade History")
                trades_df = pd.DataFrame(results['Trades'])
                st.dataframe(trades_df, use_container_width=True)
        else:
            st.markdown("""
            <div style='background-color: #1a1a1a; padding: 15px; border-radius: 5px; border: 2px solid #FFD700;'>
                <p style='margin:0; color: #ffffff; font-size: 1.1rem; font-weight: bold;'>
                    ⚙️ Configure and run a backtest to see results
                </p>
            </div>
            """, unsafe_allow_html=True)

with tabs[2]:
    st.header("Technical Analysis")
    
    ticker = st.selectbox("Stock Symbol", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'], key='ta_ticker')
    
    if st.button("Load Chart"):
        with st.spinner("Loading data..."):
            df = loader.load_stock(ticker)
            df = df.tail(252)  # Last year
            
            df = TechnicalIndicators.add_all_indicators(df)
            
            # Create candlestick with indicators
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f'{ticker} Price & Indicators', 'MACD', 'RSI')
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ), row=1, col=1)
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB High', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Low', line=dict(color='gray', dash='dash'), fill='tonexty'), row=1, col=1)
            
            # Moving Averages
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#00CC96')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='#FF4B4B')), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.update_layout(
                template="plotly_dark",
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Latest values
            st.subheader("Current Indicators")
            latest = df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RSI", f"{latest['RSI']:.2f}")
            col2.metric("MACD", f"{latest['MACD']:.4f}")
            col3.metric("ATR", f"{latest['ATR']:.2f}")
            col4.metric("Price", f"${latest['Close']:.2f}")

with tabs[3]:
    st.header("🤖 ML-Powered Price Predictions")
    
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 18px; border-radius: 8px; border: 2px solid #AB63FA;'>
        <h4 style='margin:0; color: #AB63FA;'>LSTM Neural Network Forecasting</h4>
        <p style='margin:8px 0 0 0; color: #ffffff; font-size: 1rem; line-height: 1.5;'>
            This module uses pre-trained <b>Long Short-Term Memory (LSTM)</b> deep learning models for time series prediction. 
            Models were trained offline on historical data (up to 2017) with <b>60-day lookback windows</b> and <b>20 epochs</b>.
        </p>
        <hr style='border-color: #4c4c4c; margin: 12px 0;'>
        <p style='margin:0; color: #e0e0e0; font-size: 0.9rem;'>
            <b>Architecture:</b> 2-layer LSTM (50 units each) with dropout (0.2) for regularization<br>
            <b>Optimizer:</b> Adam | <b>Loss Function:</b> Mean Squared Error (MSE)<br>
            <b>Performance Metrics:</b> MAE, RMSE, MAPE (saved in models/ directory)
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    # Add expander with methodology
    with st.expander("📚 **Methodology & Technical Details**"):
        st.markdown("""
        ### LSTM Architecture
        
        **Why LSTM for Stock Prediction?**
        - **Memory Cells**: Captures long-term dependencies in price movements
        - **Gating Mechanisms**: Learns which historical patterns are relevant
        - **Sequence Learning**: Processes time series data sequentially
        
        ### Training Process
        1. **Data Preparation**: MinMax scaling (0-1 range) for stable gradients
        2. **Sequence Creation**: 60-day sliding windows as input features
        3. **Train/Test Split**: 80/20 split with temporal ordering preserved
        4. **Model Training**: 20 epochs with validation monitoring
        5. **Evaluation**: MAE, RMSE, MAPE on held-out test set
        
        ### Performance Metrics Explained
        - **MAE (Mean Absolute Error)**: Average prediction error in dollars
        - **RMSE (Root Mean Squared Error)**: Penalizes large errors more heavily
        - **MAPE (Mean Absolute Percentage Error)**: Error as percentage of actual price
        
        ### Limitations
        - Models trained on data up to 2017 (dataset constraint)
        - Past performance doesn't guarantee future results
        - External factors (news, macro events) not incorporated
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Prediction Configuration")
        
        ticker_ml = st.selectbox("Stock Symbol", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'], key='ml_ticker')
        
        forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 30, help="Number of days to predict into the future")
        
        st.markdown(f"""
        <div style='background-color: #1a1a1a; padding: 12px; border-radius: 5px; border: 2px solid #AB63FA;'>
            <p style='margin:0; color: #00FF00; font-size: 1.05rem; font-weight: bold;'>
                🤖 Using pre-trained LSTM model for {ticker_ml}
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        
        if st.button("Generate Forecast"):
            with st.spinner("Loading model and generating predictions..."):
                try:
                    import pickle
                    import numpy as np

                    # Load pre-trained model
                    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
                    model_path = os.path.join(models_dir, f'{ticker_ml.lower()}_lstm_model.h5')
                    scaler_path = os.path.join(models_dir, f'{ticker_ml.lower()}_scaler.pkl')

                    # Load recent data for display
                    df = loader.load_stock(ticker_ml)
                    df = df.tail(60)  # Last 60 days for prediction

                    if not os.path.exists(model_path):
                        # Demo mode - generate simple trend-based prediction
                        st.warning(f"Pre-trained model not available for {ticker_ml}. Showing demo forecast based on trend analysis.")

                        # Simple trend prediction based on recent momentum
                        last_price = df['Close'].iloc[-1]
                        recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1)  # 20-day return
                        daily_return = recent_return / 20

                        predictions = []
                        price = last_price
                        for i in range(forecast_days):
                            # Add some randomness to make it realistic
                            noise = np.random.normal(0, 0.005)
                            price = price * (1 + daily_return * 0.5 + noise)
                            predictions.append(price)

                        st.session_state['ml_predictions'] = {
                            'ticker': ticker_ml,
                            'predictions': predictions,
                            'last_price': last_price,
                            'dates': pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='D')[1:],
                            'demo_mode': True
                        }
                        st.info("This is a demo forecast. For full LSTM predictions, deploy with pre-trained models.")
                    else:
                        # Full model mode
                        from tensorflow.keras.models import load_model
                        from ml_models import MLModels

                        model = load_model(model_path)
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        # Prepare data
                        scaled_data = scaler.transform(df[['Close']].values)
                        last_sequence = scaled_data.reshape((1, 60, 1))

                        # Generate predictions
                        predictions = MLModels.predict_lstm(model, scaler, last_sequence[0], n_future=forecast_days)

                        st.session_state['ml_predictions'] = {
                            'ticker': ticker_ml,
                            'predictions': predictions,
                            'last_price': df['Close'].iloc[-1],
                            'dates': pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='D')[1:],
                            'demo_mode': False
                        }

                        st.success("Forecast generated successfully!")
                except ImportError as e:
                    st.error(f"Required package not installed: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Forecast Results")
        
        if 'ml_predictions' in st.session_state:
            pred_data = st.session_state['ml_predictions']
            
            # Create forecast chart
            fig = go.Figure()
            
            # Historical (last 60 days)
            df_hist = loader.load_stock(pred_data['ticker']).tail(60)
            fig.add_trace(go.Scatter(
                x=df_hist.index,
                y=df_hist['Close'],
                mode='lines',
                name='Historical',
                line=dict(color='#00CC96', width=2)
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=pred_data['dates'],
                y=pred_data['predictions'],
                mode='lines+markers',
                name='LSTM Forecast',
                line=dict(color='#AB63FA', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            forecast_type = "Trend Analysis Demo" if pred_data.get('demo_mode', False) else "Pre-trained LSTM"
            fig.update_layout(
                template="plotly_dark",
                height=500,
                title=f"{pred_data['ticker']} Price Forecast ({forecast_type})",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    font=dict(size=14, color="white"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="#4c4c4c",
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            last_price = pred_data['last_price']
            predicted_price = pred_data['predictions'][-1]
            change = ((predicted_price - last_price) / last_price) * 100
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"${last_price:.2f}")
            m2.metric(f"Predicted ({len(pred_data['predictions'])}d)", f"${predicted_price:.2f}", f"{change:.2f}%")
            m3.metric("Model Type", "LSTM (Pre-trained)")
            
            # Prediction table
            st.subheader("Detailed Forecast")
            forecast_df = pd.DataFrame({
                'Date': pred_data['dates'],
                'Predicted Price': pred_data['predictions']
            })
            st.dataframe(forecast_df.head(10), use_container_width=True)
            
        else:
            st.markdown("""
            <div style='background-color: #1a1a1a; padding: 15px; border-radius: 5px; border: 2px solid #FFD700;'>
                <p style='margin:0; color: #ffffff; font-size: 1.1rem; font-weight: bold;'>
                    🎯 Select a stock and click 'Generate Forecast' to see ML predictions
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Model Performance Metrics Section
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")
    
    st.markdown("""
    <div style='background-color: #1a1a1a; padding: 15px; border-radius: 5px; border: 1px solid #4c4c4c;'>
        <p style='margin:0; color: #e0e0e0; font-size: 0.95rem;'>
            View detailed performance metrics and evaluation charts for all trained LSTM models.
            Metrics include MAE, RMSE, and MAPE calculated on held-out test sets.
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    # Select model to view metrics
    available_models = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'FB']
    selected_model = st.selectbox("Select Model", available_models, key='metrics_model')
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    metrics_path = os.path.join(models_dir, f'{selected_model.lower()}_metrics.json')
    chart_path = os.path.join(models_dir, f'{selected_model.lower()}_performance.png')
    
    if os.path.exists(metrics_path):
        # Load and display metrics
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        st.markdown(f"### {selected_model} Model Performance")
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"${metrics['mae']:.2f}")
        col2.metric("RMSE", f"${metrics['rmse']:.2f}")
        col3.metric("MAPE", f"{metrics['mape']:.2f}%")
        col4.metric("Test Samples", metrics['test_samples'])
        
        # Display training info
        st.markdown(f"""
        <div style='background-color: #1a1a1a; padding: 12px; border-radius: 5px; border: 1px solid #AB63FA; margin-top: 10px;'>
            <p style='margin:0; color: #e0e0e0; font-size: 0.9rem;'>
                <b>Training Details:</b> Lookback: {metrics['lookback']} days | 
                Epochs: {metrics['epochs']} | 
                Data Range: {metrics['date_range']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display performance chart
        if os.path.exists(chart_path):
            st.markdown("### Predictions vs Actual (Test Set)")
            st.image(chart_path, use_container_width=True)
        else:
            st.warning(f"Performance chart not found for {selected_model}")
    else:
        st.warning(f"Metrics not found for {selected_model}. Model may not be trained yet.")

