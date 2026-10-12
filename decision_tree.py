"""Decision Tree Regressor: training and max_depth experiments."""
from sklearn.tree import DecisionTreeRegressor

from evaluation import evaluate_model

DEPTHS = [3, 5, 8, 10]


def train_decision_tree(X_train, y_train, max_depth=None):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def tune_decision_tree(X_train, y_train, X_test, y_test):
    """Try different max_depth values. Returns a list of result rows and the models."""
    rows, models = [], {}
    for depth in DEPTHS:
        model = train_decision_tree(X_train, y_train, max_depth=depth)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"Setting": f"max_depth={depth}", **metrics})
        models[f"max_depth={depth}"] = model
    return rows, models
