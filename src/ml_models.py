import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MLModels:
    """Machine learning models for stock prediction."""
    
    @staticmethod
    def prepare_lstm_data(df, lookback=60, forecast_horizon=1):
        """
        Prepare data for LSTM model.
        
        Args:
            df: DataFrame with 'Close' column
            lookback: Number of previous days to use
            forecast_horizon: Days ahead to predict
        
        Returns:
            X, y, scaler
        """
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']].values)
        
        X, y = [], []
        
        for i in range(lookback, len(scaled_data) - forecast_horizon + 1):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i + forecast_horizon - 1, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    @staticmethod
    def build_lstm_model(lookback=60):
        """Build LSTM model for price prediction."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        except ImportError:
            print("TensorFlow not installed. Install with: pip install tensorflow")
            return None
    
    @staticmethod
    def train_lstm(df, lookback=60, epochs=10, batch_size=32):
        """Train LSTM model on stock data."""
        X, y, scaler = MLModels.prepare_lstm_data(df, lookback=lookback)
        
        # Split train/test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build and train model
        model = MLModels.build_lstm_model(lookback=lookback)
        if model is None:
            return None, None, None
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  validation_data=(X_test, y_test), verbose=0)
        
        return model, scaler, (X_test, y_test)
    
    @staticmethod
    def predict_lstm(model, scaler, last_n_days, n_future=30):
        """Make future predictions with LSTM."""
        predictions = []
        current_batch = last_n_days.reshape((1, last_n_days.shape[0], 1))
        
        for i in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
    
    @staticmethod
    def train_random_forest_classifier(df, features, target='Signal'):
        """
        Train Random Forest for Buy/Sell/Hold classification.
        
        Args:
            df: DataFrame with features and target
            features: List of feature column names
            target: Target column name
        
        Returns:
            model, feature_importance
        """
        # Prepare data
        df = df.dropna()
        X = df[features]
        y = df[target]
        
        # Split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Accuracy
        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)
        
        print(f"Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}")
        
        return rf, importance
    
    @staticmethod
    def train_xgboost(df, features, target='Returns'):
        """Train XGBoost for regression."""
        df = df.dropna()
        X = df[features]
        y = df[target]
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return model, importance

if __name__ == "__main__":
    from data_loader import StockDataLoader
    
    loader = StockDataLoader()
    aapl = loader.load_stock('aapl')
    aapl = aapl['2020-01-01':'2023-12-31']
    
    print("Training LSTM model...")
    model, scaler, test_data = MLModels.train_lstm(aapl, lookback=60, epochs=5)
    
    if model:
        print("Model trained successfully!")
        
        # Make predictions
        last_60_days = test_data[0][-1]
        predictions = MLModels.predict_lstm(model, scaler, last_60_days, n_future=30)
        print(f"30-day forecast: {predictions[:5]}...")
