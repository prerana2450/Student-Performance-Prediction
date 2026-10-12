"""Random Forest Regressor: training and n_estimators experiments."""
from sklearn.ensemble import RandomForestRegressor

from evaluation import evaluate_model

N_ESTIMATORS = [50, 100, 150, 200]


def train_random_forest(X_train, y_train, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model


def tune_random_forest(X_train, y_train, X_test, y_test):
    """Try different n_estimators values. Returns a list of result rows and the models."""
    rows, models = [], {}
    for n in N_ESTIMATORS:
        model = train_random_forest(X_train, y_train, n_estimators=n)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"Setting": f"n_estimators={n}", **metrics})
        models[f"n_estimators={n}"] = model
    return rows, models
