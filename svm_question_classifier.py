"""
AI Study Lab - SVM
Purpose: Classify a student's question by question type.
Dataset: question/train_5500.txt and question/test_data.txt

Labels in this dataset include:
ABBR, DESC, ENTY, HUM, LOC, NUM
"""

from pathlib import Path
import zipfile
import joblib
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "question"
ZIP_PATH = BASE_DIR / "question.zip"
MODEL_PATH = BASE_DIR / "svm_question_classifier.pkl"


def read_text_file(filename):
    folder_path = DATA_DIR / filename

    if folder_path.exists():
        return folder_path.read_text(encoding="utf-8", errors="replace")

    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            with z.open(filename) as f:
                return f.read().decode("utf-8", errors="replace")

    raise FileNotFoundError(
        f"Could not find {filename} in question/ or question.zip."
    )


def parse_question_file(text):
    """
    Each line has the form:
        LABEL:subcategory question text
    """
    questions = []
    labels = []

    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        label, question = line.split(":", 1)
        label = label.strip().upper()
        question = question.strip()

        # Remove the fine-grained subcategory, e.g. "dist", "city", "def".
        parts = question.split(maxsplit=1)
        if parts and re.fullmatch(r"[A-Za-z0-9_-]+", parts[0]):
            question = parts[1] if len(parts) > 1 else ""

        if question:
            labels.append(label)
            questions.append(question)

    return questions, labels


from sklearn.pipeline import FeatureUnion, Pipeline

def train_model():
    train_text = read_text_file("train_5500.txt")
    test_text = read_text_file("test_data.txt")

    X_train, y_train = parse_question_file(train_text)
    X_test, y_test = parse_question_file(test_text)

    word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, norm='l2')
    char_vec = TfidfVectorizer(ngram_range=(2, 6), analyzer='char', sublinear_tf=True, min_df=1, norm='l2')

    features = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec)
    ])

    model = Pipeline(
        steps=[
            ("features", features),
            ("classifier", LinearSVC(
                C=1.0,
                loss="squared_hinge",
                max_iter=2000,
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

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Question types: {sorted(set(y_train))}")
    print(f"Test accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

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

    metrics["svm"] = {
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


def predict_question_type(question):
    """Return the question type predicted by the SVM."""
    if not MODEL_PATH.exists():
        train_model()

    model = joblib.load(MODEL_PATH)

    question = str(question).strip()
    prediction = model.predict([question])[0]

    result = {"question_type": prediction}

    # LinearSVC has decision_function rather than predict_proba.
    if hasattr(model, "decision_function"):
        scores = model.decision_function([question])[0]
        classes = model.named_steps["classifier"].classes_

        if np.ndim(scores) == 0:
            scores = [float(scores)]

        order = np.argsort(scores)[::-1]
        result["top_types"] = [
            {
                "type": str(classes[i]),
                "score": round(float(scores[i]), 4)
            }
            for i in order[:3]
        ]

    return result


if __name__ == "__main__":
    train_model()
