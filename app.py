
"""Student Performance Prediction System - Streamlit app.

Run with: streamlit run app.py
Loads the saved model and predicts marks for the selected student.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predict import load_model, predict_marks
from preprocess import FEATURES

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "model_results.json"
CSV_PATH = BASE_DIR / "student_data.csv"

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_results():
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def get_student_data():
    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "Student_ID",
        "Student_Name",
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

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns in student_data.csv: " + ", ".join(missing)
        )

    df["Student_ID"] = df["Student_ID"].astype(str).str.strip()
    df["Student_Name"] = df["Student_Name"].astype(str).str.strip()

    # Do not allow an ID to refer to multiple records.
    df = df.drop_duplicates(subset=["Student_ID"], keep="first")

    return df


# ---------------------------------------------------------------
# Load model and project results
# ---------------------------------------------------------------

st.title("🎓 Student Performance Prediction System")

st.write(
    "Select a student to load their academic details and predict "
    "their expected final marks using Machine Learning."
)

st.caption(
    "📝 This project uses a simulated dataset for academic prototype "
    "and model demonstration purposes."
)

try:
    model = get_model()
    results = get_results()
    student_df = get_student_data()

except FileNotFoundError as exc:
    st.error(
        f"Required file not found: {exc.filename or str(exc)}. "
        "Check that model.pkl, model_results.json and student_data.csv "
        "are uploaded to the repository."
    )
    st.stop()

except (ValueError, KeyError) as exc:
    st.error(f"Could not load project data: {exc}")
    st.stop()


# ---------------------------------------------------------------
# Student selection
# ---------------------------------------------------------------

st.header("👨‍🎓 Student Details")

st.write(f"**Total student records:** {len(student_df)}")

student_options = {
    f"{row['Student_ID']} - {row['Student_Name']}": row["Student_ID"]
    for _, row in student_df.iterrows()
}

option_labels = list(student_options.keys())

selected_student = st.selectbox(
    "Select Student (Student ID - Name)",
    options=["Select a student"] + option_labels,
    index=0,
)

if selected_student == "Select a student":
    st.info("Please select a student to load their academic details.")
    st.stop()

selected_id = student_options[selected_student]

matching_rows = student_df[student_df["Student_ID"] == selected_id]

if matching_rows.empty:
    st.error("Student record not found. Please select another student.")
    st.stop()

student = matching_rows.iloc[0]
name = str(student["Student_Name"])

st.success(f"Details loaded for {name} (ID: {selected_id})")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = float(student["Study_Hours"])
    st.number_input(
        "Study Hours per Day",
        value=study_hours,
        disabled=True,
    )

    attendance = float(student["Attendance"])
    st.number_input(
        "Attendance (%)",
        value=attendance,
        disabled=True,
    )

    previous_marks = float(student["Previous_Marks"])
    st.number_input(
        "Previous Marks (0-100)",
        value=previous_marks,
        disabled=True,
    )

with col2:
    assignment = float(student["Assignment_Score"])
    st.number_input(
        "Assignment Score (0-100)",
        value=assignment,
        disabled=True,
    )

    internal = float(student["Internal_Marks"])
    st.number_input(
        "Internal Marks (0-100)",
        value=internal,
        disabled=True,
    )

    gpa = float(student["Previous_Semester_GPA"])
    st.number_input(
        "Previous Semester GPA (0-10)",
        value=gpa,
        disabled=True,
    )

with col3:
    participation = float(student["Class_Participation"])
    st.number_input(
        "Class Participation (%)",
        value=participation,
        disabled=True,
    )

    learning_hours = float(student["Learning_Hours"])
    st.number_input(
        "Learning Hours per Day",
        value=learning_hours,
        disabled=True,
    )

    backlogs = int(student["Backlogs"])
    st.number_input(
        "Number of Backlogs",
        value=backlogs,
        disabled=True,
    )


# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------

st.header("🔮 Predict Final Marks")

if st.button("🔮 Predict Final Marks", type="primary"):
    inputs = {
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Marks": previous_marks,
        "Assignment_Score": assignment,
        "Internal_Marks": internal,
        "Previous_Semester_GPA": gpa,
        "Class_Participation": participation,
        "Learning_Hours": learning_hours,
        "Backlogs": backlogs,
    }

    try:
        marks = predict_marks(model, inputs)

        st.subheader("🎯 Prediction Result")
        st.write(f"**Student:** {name}")
        st.write(f"**Student ID:** {selected_id}")
        st.success(f"🎯 Predicted Final Marks: {marks:.2f} / 100")

        st.caption(
            "This is an estimated prediction from a Machine Learning model, "
            "not an actual or guaranteed examination result."
        )

        if marks >= 75:
            st.info("📚 The predicted performance is relatively high.")
        elif marks >= 50:
            st.info("📖 The predicted performance is moderate.")
        else:
            st.warning("⚠️ The predicted performance is relatively low.")

    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Prediction could not be completed: {exc}")


st.divider()


# ---------------------------------------------------------------
# Model information
# ---------------------------------------------------------------

st.header("🤖 Model Information")

m = results["selected_metrics"]

c1, c2, c3 = st.columns(3)
c1.metric("Selected Model", results["selected_model"])
c2.metric("R² Score", f"{m['R2']:.2f}")
c3.metric("Mean Absolute Error", f"{m['MAE']:.2f}")

st.caption(
    f"These metrics were calculated on the held-out test set "
    f"({results['test_size']} records, 20% of the data). "
    f"The model was trained on {results['train_size']} records (80%)."
)


# ---------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------

st.header("📊 Model Comparison")

comp = pd.DataFrame(results["comparison"]).rename(
    columns={"R2": "R² Score"}
)
st.dataframe(comp.round(4), hide_index=True)

ch1, ch2 = st.columns(2)

with ch1:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(comp["Model"], comp["MAE"])
    ax.set_title("MAE Comparison (lower is better)")
    ax.set_ylabel("MAE")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with ch2:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(comp["Model"], comp["R² Score"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("R² Score Comparison (higher is better)")
    ax.set_ylabel("R² Score")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.subheader("⚙️ Model Optimization Results")

o1, o2 = st.columns(2)

with o1:
    st.write("Decision Tree - `max_depth`")
    st.dataframe(
        pd.DataFrame(results["decision_tree_tuning"]).round(4),
        hide_index=True,
    )

with o2:
    st.write("Random Forest - `n_estimators`")
    st.dataframe(
        pd.DataFrame(results["random_forest_tuning"]).round(4),
        hide_index=True,
    )


# ---------------------------------------------------------------
# Evaluation chart
# ---------------------------------------------------------------

st.header("📈 Evaluation Chart")

actual = results["test_predictions"]["actual"]
pred = results["test_predictions"]["predicted"]

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(actual, pred, alpha=0.7)

lo, hi = min(actual + pred), max(actual + pred)
ax.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")

ax.set_xlabel("Actual Final Marks (test set)")
ax.set_ylabel("Predicted Final Marks")
ax.set_title(f"Actual vs Predicted - {results['selected_model']}")
ax.legend()
fig.tight_layout()

left, _ = st.columns([1, 1])
with left:
    st.pyplot(fig)

plt.close(fig)

st.caption(
    "This chart uses the held-out test data only."
)


# ---------------------------------------------------------------
# Dataset information
# ---------------------------------------------------------------

st.header("📁 Dataset Information")

d = results["dataset_info"]

d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Records", d["records"])
d2.metric("Features", d["features"])
d3.metric("Target", d["target"])
d4.metric("Missing Values", d["missing_values"])
d5.metric("Duplicate Rows", d["duplicate_rows"])

st.write("**Input features:** " + ", ".join(FEATURES))
st.caption(
    "Student_ID and Student_Name are identification fields and "
    "are not used as model features."
)


# ---------------------------------------------------------------
# Project workflow
# ---------------------------------------------------------------

st.header("🔄 Project Workflow")

st.code(
    "Student Data\n"
    "      ↓\n"
    "Data Preprocessing\n"
    "      ↓\n"
    "Feature Selection\n"
    "      ↓\n"
    "Train-Test Split\n"
    "      ↓\n"
    "ML Model Training\n"
    "      ↓\n"
    "Model Evaluation\n"
    "      ↓\n"
    "Best Model Selection\n"
    "      ↓\n"
    "Select Student by ID\n"
    "      ↓\n"
    "Load Academic Details\n"
    "      ↓\n"
    "Final Marks Prediction",
    language=None,
)


# ---------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------

st.divider()

st.caption(
    "ℹ️ Disclaimer: This system provides an estimated prediction "
    "based on a Machine Learning model trained on simulated academic "
    "data. The prediction is for educational and demonstration purposes "
    "only and should not be considered an actual or guaranteed result."
)

