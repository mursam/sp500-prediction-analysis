import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_data(df, target_col, features, split):
    """Prepare train/test split with scaling."""
    
    df_copy = df.copy()
    
    X = df_copy[features]
    y = df_copy[target_col]
    
    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler