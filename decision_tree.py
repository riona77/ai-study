"""
AI Study Lab - Decision Tree
Purpose: Help choose an appropriate study difficulty/learning level.
Dataset: decision/Learning_Pathway_Index.csv
"""

from pathlib import Path
import zipfile
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "decision"
ZIP_PATH = BASE_DIR / "decision.zip"
MODEL_PATH = BASE_DIR / "decision_tree_model.pkl"


def load_data():
    """Load Learning_Pathway_Index.csv from folder or ZIP."""
    csv_path = DATA_DIR / "Learning_Pathway_Index.csv"

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            with z.open("Learning_Pathway_Index.csv") as f:
                return pd.read_csv(f)

    raise FileNotFoundError(
        "Could not find decision/Learning_Pathway_Index.csv or decision.zip."
    )


def duration_to_minutes(value):
    """Convert values such as '20.0 minutes' to a numeric value."""
    text = str(value).lower().strip()
    if text in ("", "nan", "none"):
        return 0.0

    import re
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return 0.0

    number = float(match.group())

    if "hour" in text:
        number *= 60
    elif "day" in text:
        number *= 1440

    return number


from sklearn.feature_extraction.text import TfidfVectorizer

def clean_data(df):
    df = df.copy()

    required = [
        "Course_Level", "Type_Free_Paid", "Module",
        "Duration", "Difficulty_Level",
        "Keywords_Tags_Skills_Interests_Categories"
    ]
    df = df.dropna(subset=required)

    for col in required:
        df[col] = df[col].astype(str).str.strip()

    df["duration_minutes"] = df["Duration"].apply(duration_to_minutes)
    return df


def train_model():
    df = clean_data(load_data())

    feature_columns = [
        "Course_Level",
        "Type_Free_Paid",
        "Module",
        "duration_minutes",
        "Keywords_Tags_Skills_Interests_Categories"
    ]

    X = df[feature_columns]
    y = df["Difficulty_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             ["Course_Level", "Type_Free_Paid", "Module"]),
            ("num", "passthrough", ["duration_minutes"]),
            ("text", TfidfVectorizer(max_features=500, ngram_range=(1, 2)),
             "Keywords_Tags_Skills_Interests_Categories")
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                random_state=42
            ))
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    
    from sklearn.metrics import confusion_matrix
    classes = list(model.named_steps["classifier"].classes_)
    cm = confusion_matrix(y_test, predictions, labels=classes).tolist()

    print(f"Dataset shape: {df.shape}")
    print(f"Difficulty classes: {sorted(y.unique())}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

    import json
    metrics_path = BASE_DIR / "model_metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    metrics["decision_tree"] = {
        "accuracy": float(accuracy),
        "report": report,
        "classes": [str(c) for c in classes],
        "confusion_matrix": cm,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return model, accuracy


def predict_learning_level(
    course_level,
    type_free_paid,
    module,
    duration_minutes,
    keyword
):
    """Predict the difficulty of a resource for study-path planning."""
    if not MODEL_PATH.exists():
        train_model()

    model = joblib.load(MODEL_PATH)

    input_df = pd.DataFrame([{
        "Course_Level": course_level,
        "Type_Free_Paid": type_free_paid,
        "Module": module,
        "duration_minutes": float(duration_minutes),
        "Keywords_Tags_Skills_Interests_Categories": str(keyword).lower().strip()
    }])

    prediction = model.predict(input_df)[0]

    result = {"predicted_difficulty": prediction}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.named_steps["classifier"].classes_
        result["probabilities"] = {
            str(classes[i]): round(float(probabilities[i]), 4)
            for i in range(len(classes))
        }

    return result


if __name__ == "__main__":
    train_model()

