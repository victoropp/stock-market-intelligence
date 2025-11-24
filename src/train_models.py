"""
Train and save ML models for stock price prediction.
This script should be run offline to train models for multiple stocks.
"""

import os
import pickle
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from data_loader import StockDataLoader
from ml_models import MLModels
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Create models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_save_lstm_model(ticker, lookback=60, epochs=20):
    """Train LSTM model for a specific stock and save it with metrics."""
    print(f"\n{'='*50}")
    print(f"Training LSTM model for {ticker}")
    print(f"{'='*50}")
    
    try:
        # Load data
        loader = StockDataLoader()
        df = loader.load_stock(ticker)
        
        # Use all available data (dataset goes up to 2017)
        print(f"Data loaded: {len(df)} days (from {df.index[0].date()} to {df.index[-1].date()})")
        
        if len(df) < lookback + 100:
            print(f"Insufficient data for {ticker}. Need at least {lookback + 100} days, have {len(df)}. Skipping.")
            return False
        
        # Train model
        print("Training LSTM model...")
        model, scaler, test_data = MLModels.train_lstm(df, lookback=lookback, epochs=epochs)
        
        if model is None:
            print(f"Failed to train model for {ticker}")
            return False
        
        # Evaluate on test set
        X_test, y_test = test_data
        predictions_test = model.predict(X_test, verbose=0)
        
        # Inverse transform to get actual prices
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        predictions_actual = scaler.inverse_transform(predictions_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, predictions_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
        mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
        
        print(f"\n📊 Model Performance Metrics:")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # Save metrics to JSON
        metrics = {
            'ticker': ticker,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'test_samples': len(X_test),
            'lookback': lookback,
            'epochs': epochs,
            'date_range': f"{df.index[0].date()} to {df.index[-1].date()}"
        }
        
        metrics_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Metrics saved: {metrics_path}")
        
        # Create prediction chart
        plt.style.use('dark_background')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Predictions vs Actual
        ax1.plot(y_test_actual, label='Actual Price', color='#00CC96', linewidth=2)
        ax1.plot(predictions_actual, label='LSTM Predictions', color='#AB63FA', linewidth=2, linestyle='--')
        ax1.set_title(f'{ticker} - LSTM Model: Predictions vs Actual (Test Set)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Steps', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error
        error = y_test_actual.flatten() - predictions_actual.flatten()
        ax2.plot(error, color='#FF4B4B', linewidth=1.5)
        ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax2.set_title('Prediction Error', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Steps', fontsize=12)
        ax2.set_ylabel('Error ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f'MAE: ${mae:.2f}\nRMSE: ${rmse:.2f}\nMAPE: {mape:.2f}%'
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        chart_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_performance.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
        plt.close()
        
        print(f"✓ Chart saved: {chart_path}")
        
        # Save model and scaler
        model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_lstm_model.h5')
        scaler_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_scaler.pkl')
        
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
        
        # Test prediction
        last_sequence = X_test[-1]
        predictions = MLModels.predict_lstm(model, scaler, last_sequence, n_future=30)
        
        print(f"✓ Test prediction successful: 30-day forecast generated")
        print(f"  Current price: ${df['Close'].iloc[-1]:.2f}")
        print(f"  Predicted (30d): ${predictions[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error training {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Train models for popular stocks."""
    
    print("\n" + "="*70)
    print("STOCK MARKET INTELLIGENCE - MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Popular stocks to train models for
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    print(f"\nTraining LSTM models for {len(stocks)} stocks...")
    print(f"Stocks: {', '.join(stocks)}")
    
    results = {}
    
    for ticker in stocks:
        success = train_and_save_lstm_model(ticker, lookback=60, epochs=20)
        results[ticker] = success
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    successful = [t for t, s in results.items() if s]
    failed = [t for t, s in results.items() if not s]
    
    print(f"\n✓ Successfully trained: {len(successful)}/{len(stocks)}")
    if successful:
        print(f"  {', '.join(successful)}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        print(f"  {', '.join(failed)}")
    
    print(f"\nModels saved to: {MODELS_DIR}")
    print("\nYou can now run the Streamlit app to use these pre-trained models!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
