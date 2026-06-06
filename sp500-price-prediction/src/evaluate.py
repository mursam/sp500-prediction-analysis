import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, classification_report)


def regression_metrics(y_true, y_pred, label="Model"):
    """Print regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"── {label} ──────────────────────────────")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def classification_metrics(y_true, y_pred, label="Model"):
    """Print classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    
    print(f"── {label} ──────────────────────────────")
    print(f"Accuracy: {acc:.4f}")
    print()
    print(classification_report(y_true, y_pred, 
                                target_names=['Down', 'Up']))
    
    return {'accuracy': acc}