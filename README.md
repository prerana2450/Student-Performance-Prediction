# 🎓 Student Performance Prediction System

A Machine Learning web application (Streamlit + scikit-learn) that predicts a student's **expected Final Marks** from academic and learning-related parameters.

> **Note:** The dataset used in this project is a simulated dataset created for academic prototype and model demonstration purposes.

## Project Description
The user enters a student's academic details in a web form. A trained regression model (loaded from `model.pkl`) returns an **estimated** Final Marks value out of 100. The app also shows model information, a comparison of three regression models, optimization results, dataset information and evaluation charts. Predictions are estimates, not actual or guaranteed results.

## Problem Statement
Teachers and students often want an early idea of likely exam performance so weak areas can be addressed in time. This project builds a simple ML model that estimates final marks from attendance, previous performance, study habits and similar factors.

## Objectives
- Build a regression model that predicts Final Marks from 9 input features.
- Compare Linear Regression, Decision Tree and Random Forest using the same train/test split.
- Run basic hyperparameter experiments and select the best model by test-set MAE and R².
- Deploy a simple, working web application on the cloud.

## Features
- Input form for student details (Student Name is display-only, never passed to the model)
- Predicted Final Marks (clipped to 0–100) with a performance message (high / moderate / low)
- Model information (selected model, R², MAE on the test set)
- Model comparison table (MAE, MSE, RMSE, R²) with MAE and R² charts
- Hyperparameter optimization tables
- Actual-vs-Predicted chart using the held-out **test set** only
- Dataset information, project workflow and disclaimer

## Technology Stack
Python 3.11+, pandas, numpy, scikit-learn, matplotlib, joblib, Streamlit.

## Dataset Description
`dataset/student_data.csv` – 500 records, 12 columns, no missing values, no duplicate rows.

| Column | Role |
|---|---|
| Student_ID, Student_Name | Identification only – **not used as features** |
| Study_Hours, Attendance, Previous_Marks, Assignment_Score, Internal_Marks, Previous_Semester_GPA, Class_Participation, Learning_Hours, Backlogs | **Input features (9)** |
| Final_Marks | **Target** |

## Machine Learning Models
1. **Linear Regression** – fits a straight-line (weighted sum) relationship between features and marks.
2. **Decision Tree Regressor** – baseline uses `max_depth=3`.
3. **Random Forest Regressor** – baseline uses `n_estimators=100`.
All use `random_state=42`.

## Data Preprocessing (`preprocess.py`)
1. Load CSV with pandas; show shape, data types, missing values and duplicates.
2. Drop duplicate rows; fill missing numeric values with the median (none were needed for this dataset).
3. Select the 9 feature columns as `X` and `Final_Marks` as `y` (ID and name excluded).
4. `train_test_split(test_size=0.2, random_state=42)` → 400 training / 100 test records.

## Model Evaluation
Metrics (MAE, MSE, RMSE, R²) are **calculated on the 20% test set** by `train_model.py` – nothing is hard-coded.

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 3.27 | 18.43 | 4.29 | 0.66 |
| Decision Tree (max_depth=3) | 5.84 | 52.42 | 7.24 | 0.02 |
| Random Forest (100 trees) | 4.18 | 25.50 | 5.05 | 0.52 |

## Model Optimization
Decision Tree `max_depth` ∈ {3, 5, 8, 10} and Random Forest `n_estimators` ∈ {50, 100, 150, 200} were tested.

| Decision Tree | MAE | R² | | Random Forest | MAE | R² |
|---|---|---|---|---|---|---|
| max_depth=3 | 5.84 | 0.02 | | n_estimators=50 | 4.27 | 0.51 |
| max_depth=5 | 5.96 | -0.00 | | n_estimators=100 | 4.18 | 0.52 |
| max_depth=8 | 6.14 | -0.11 | | n_estimators=150 | 4.19 | 0.52 |
| max_depth=10 | 6.28 | -0.17 | | n_estimators=200 | 4.19 | 0.53 |

The final model is the candidate with the **lowest MAE** (ties broken by highest R²) on the same test split.

## Results
**Linear Regression** is selected (MAE 3.27, R² 0.66). It beats both tree-based models, which suggests the relationship between the features and Final Marks is mostly linear; the Decision Tree overfits easily on 400 training records.

## How to Run Locally
```bash
cd Student_Performance_Prediction
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py             # trains, compares, saves model.pkl + model_results.json
streamlit run app.py              # opens http://localhost:8501
```

## How to Deploy on Streamlit Community Cloud
1. Create a GitHub repository and push the whole project folder (including `model.pkl`, `model_results.json` and `dataset/student_data.csv`).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose your repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**. Streamlit installs `requirements.txt` and opens the app.

`requirements.txt` pins `scikit-learn==1.8.0` because `model.pkl` was saved with that version. If you retrain with a different version, update the pin to match.

## Project Structure
```
Student_Performance_Prediction/
├── dataset/student_data.csv
├── model.pkl                 # saved final model
├── model_results.json        # metrics saved by train_model.py (used by the app)
├── preprocess.py             # loading, cleaning, feature/target split, train-test split
├── evaluation.py             # MAE, MSE, RMSE, R²
├── decision_tree.py          # Decision Tree + max_depth experiments
├── random_forest.py          # Random Forest + n_estimators experiments
├── train_model.py            # trains all models, selects best, saves model.pkl
├── predict.py                # loads model.pkl and predicts for one student
├── app.py                    # Streamlit web app
├── requirements.txt
├── README.md
└── .gitignore
```

## Workflow
Student Data → Data Preprocessing → Feature Selection → Train-Test Split → ML Model Training → Model Evaluation → Best Model Selection → Student Input → Final Marks Prediction

## Limitations
- Simulated dataset; results may not reflect real students.
- Only 500 records and 9 features; many real factors (health, family, teaching quality) are missing.
- The best model is chosen using the same 20% test split used for reporting, so the reported scores can be slightly optimistic (no separate validation set / cross-validation).
- Predictions are estimates and should not be used for real academic decisions.

## Future Scope
- Train on real, anonymised institutional data.
- Add cross-validation and `GridSearchCV`.
- Try more models (Gradient Boosting, etc.) and feature scaling.
- Add batch prediction from a CSV upload and student progress tracking.

## Author
Name: ____________  |  Class: TE IT  |  College: ____________  |  Guide: ____________

---
## Academic Documentation Content

**1. Problem Definition** – Predict a student's expected Final Marks (a continuous value) from 9 academic and learning parameters; this is a supervised regression problem.

**2. Dataset Collection** – `student_data.csv` with 500 records and 12 columns (Student_ID, Student_Name, 9 features, Final_Marks). The dataset is a simulated dataset created for academic prototype and model demonstration purposes; it is not real student data.

**3. Data Preprocessing** – Loaded with pandas; checked shape, dtypes, missing values (0) and duplicates (0); features and target separated, ID/name excluded; 80/20 split with `random_state=42`.

**4. Model Development** – Linear Regression, Decision Tree Regressor and Random Forest Regressor trained on the same 400 training records.

**5. Performance Evaluation** – MAE, MSE, RMSE and R² computed on the 100 test records (see table above).

**6. Model Optimization** – Tested Decision Tree `max_depth` (3, 5, 8, 10) and Random Forest `n_estimators` (50–200). Tuning did not beat Linear Regression.

**7. Result Analysis and Documentation** – Linear Regression gave the lowest error (MAE 3.27, i.e. about ±3 marks on average) and highest R² (0.66, explaining about 66% of the variance). The Decision Tree performed worst; deeper trees overfit. The Random Forest was better than a single tree but behind Linear Regression.

**8. Conclusion** – A working ML web application was built that predicts expected Final Marks, compares three regression models and uses the best (Linear Regression) for prediction, deployed as a Streamlit app.

**9. Future Scope** – See above.

**10. Limitations** – See above.
