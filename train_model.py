"""Train, evaluate, compare and save the final model.

Run with:  python train_model.py
Creates:   model.pkl (final model) and model_results.json (metrics used by the app)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from decision_tree import train_decision_tree, tune_decision_tree
from evaluation import evaluate_model
from preprocess import FEATURES, TARGET, preprocess
from random_forest import train_random_forest, tune_random_forest

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
RESULTS_PATH = BASE_DIR / "model_results.json"


def main():
    pd.set_option("display.width", 120)

    print("=== 1. PREPROCESSING ===")
    X_train, X_test, y_train, y_test, report = preprocess(verbose=True)

    print("\n=== 2. TRAINING (same 80/20 split for all models) ===")
    lr = LinearRegression().fit(X_train, y_train)
    dt = train_decision_tree(X_train, y_train, max_depth=3)  # shallow tree (a fully grown tree overfits badly)
    rf = train_random_forest(X_train, y_train, 100)     # default: 100 trees

    base_models = {"Linear Regression": lr, "Decision Tree": dt, "Random Forest": rf}
    comparison = []
    for name, model in base_models.items():
        comparison.append({"Model": name, **evaluate_model(model, X_test, y_test)})
    comp_df = pd.DataFrame(comparison)
    print("\n=== 3. MODEL COMPARISON (test set) ===")
    print(comp_df.round(4).to_string(index=False))

    print("\n=== 4. HYPERPARAMETER EXPERIMENTS ===")
    dt_rows, dt_models = tune_decision_tree(X_train, y_train, X_test, y_test)
    rf_rows, rf_models = tune_random_forest(X_train, y_train, X_test, y_test)
    print("\nDecision Tree - max_depth")
    print(pd.DataFrame(dt_rows).round(4).to_string(index=False))
    print("\nRandom Forest - n_estimators")
    print(pd.DataFrame(rf_rows).round(4).to_string(index=False))

    # Candidates: the 3 base models + every tuned variant
    candidates = {name: (m, r) for (name, m), r in zip(base_models.items(), comparison)}
    for row in dt_rows:
        candidates[f"Decision Tree ({row['Setting']})"] = (dt_models[row["Setting"]], row)
    for row in rf_rows:
        candidates[f"Random Forest ({row['Setting']})"] = (rf_models[row["Setting"]], row)

    # Best = lowest MAE, ties broken by highest R2 (same test split for everyone)
    best_name = min(candidates, key=lambda n: (candidates[n][1]["MAE"], -candidates[n][1]["R2"]))
    best_model, best_metrics = candidates[best_name]
    best_metrics = {k: best_metrics[k] for k in ("MAE", "MSE", "RMSE", "R2")}

    print("\n=== 5. BEST MODEL SELECTION ===")
    print(f"Selected model: {best_name}")
    print({k: round(v, 4) for k, v in best_metrics.items()})

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH.name}")

    y_pred = best_model.predict(X_test)
    results = {
        "selected_model": best_name,
        "selected_metrics": best_metrics,
        "features": FEATURES,
        "target": TARGET,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "dataset_info": {k: v for k, v in report.items() if k != "dtypes"},
        "comparison": comparison,
        "decision_tree_tuning": dt_rows,
        "random_forest_tuning": rf_rows,
        "test_predictions": {
            "actual": [float(v) for v in y_test],
            "predicted": [float(v) for v in y_pred],
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics -> {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
