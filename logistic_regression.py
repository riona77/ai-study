"""
AI Study Lab - Logistic Regression
Purpose: Recommend a study course from a student's interests/preferences.
Dataset: logistic/recomdata.csv
"""

from pathlib import Path
import zipfile
import io
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "logistic"
ZIP_PATH = BASE_DIR / "logistic.zip"
MODEL_PATH = BASE_DIR / "logistic_regression_model.pkl"


def load_data():
    """Load recomdata.csv from the extracted folder or logistic.zip."""
    csv_path = DATA_DIR / "recomdata.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            with z.open("recomdata.csv") as f:
                return pd.read_csv(f)

    raise FileNotFoundError(
        "Could not find logistic/recomdata.csv or logistic.zip."
    )


def clean_data(df):
    df = df.copy()
    required = [
        "interests", "category", "preferred_difficulty",
        "past_experience", "preferred_language", "recommended_course"
    ]
    df = df.dropna(subset=required)

    for col in required:
        df[col] = df[col].astype(str).str.strip()

    return df


def train_model():
    df = clean_data(load_data())

    X = df[
        ["interests", "category", "preferred_difficulty",
         "past_experience", "preferred_language"]
    ]
    y = df["recommended_course"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    categorical_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             categorical_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=2000,
                C=2.0,
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
    print(f"Number of recommended courses: {y.nunique()}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

    # Save metrics
    metrics_path = BASE_DIR / "model_metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    metrics["logistic_regression"] = {
        "accuracy": float(accuracy),
        "report": report,
        "classes": [str(c) for c in classes],
        "confusion_matrix": cm,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return model, accuracy


def predict_course(
    interests,
    category,
    preferred_difficulty,
    past_experience,
    preferred_language
):
    """Return the recommended course for a student's study preferences."""
    if not MODEL_PATH.exists():
        train_model()

    model = joblib.load(MODEL_PATH)

    input_df = pd.DataFrame([{
        "interests": interests,
        "category": category,
        "preferred_difficulty": preferred_difficulty,
        "past_experience": past_experience,
        "preferred_language": preferred_language
    }])

    prediction = model.predict(input_df)[0]

    result = {"recommended_course": prediction}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.named_steps["classifier"].classes_
        top_indices = probabilities.argsort()[-5:][::-1]
        result["top_recommendations"] = [
            {
                "course": classes[i],
                "probability": round(float(probabilities[i]), 4)
            }
            for i in top_indices
        ]

    return result


if __name__ == "__main__":
    train_model()
