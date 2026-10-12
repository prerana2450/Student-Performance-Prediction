"""Model evaluation metrics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    """Return MAE, MSE, RMSE and R2 calculated on the held-out test set."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, y_pred)),
    }
