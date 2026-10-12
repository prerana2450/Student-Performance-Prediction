"""Data loading and preprocessing for the Student Performance Prediction System."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "student_data.csv"

# Identification / display columns - NEVER used as ML features
ID_COLUMNS = ["Student_ID", "Student_Name"]

# Input features used by the ML models (order matters, app uses the same order)
FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Semester_GPA",
    "Class_Participation",
    "Learning_Hours",
    "Backlogs",
]

# Target variable
TARGET = "Final_Marks"


def load_data(path=DATA_PATH):
    """Load the CSV dataset using pandas."""
    return pd.read_csv(path)


def dataset_report(df):
    """Summary of the raw dataset (shape, missing values, duplicates, dtypes)."""
    return {
        "records": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "features": len(FEATURES),
        "target": TARGET,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def clean_data(df):
    """Remove duplicates and handle missing values (median for numeric columns)."""
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])  # rows without a target are unusable
    for col in FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)


def split_data(df):
    """Separate X and y, then split 80% train / 20% test."""
    X = df[FEATURES]  # Student_ID and Student_Name are NOT included
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess(path=DATA_PATH, verbose=False):
    """Full preprocessing pipeline. Returns X_train, X_test, y_train, y_test, report."""
    df = load_data(path)
    report = dataset_report(df)
    if verbose:
        print("Dataset shape        :", df.shape)
        print("Missing values       :", report["missing_values"])
        print("Duplicate rows       :", report["duplicate_rows"])
        print("Data types           :")
        print(df.dtypes.to_string())
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    if verbose:
        print("Shape after cleaning :", df.shape)
        print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, report
