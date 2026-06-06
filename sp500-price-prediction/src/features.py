import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV dataframe."""
    
    df = df.copy()
    
    # Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Volatility
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Lag features
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_3'] = df['Close'].shift(3)
    df['Lag_7'] = df['Close'].shift(7)
    
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add target variables."""
    
    df = df.copy()
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    df['Target_Price'] = df['Close'].shift(-1)
    df['Target_Vol'] = df['Volatility_20'].shift(-1)
    df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
    
    return df