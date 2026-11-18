"""
AI Study Lab - Naive Bayes
Purpose: Classify a student's question into the most relevant chapter.
Dataset: naives/train.csv, naives/val.csv, naives/test.csv
"""

from pathlib import Path
import zipfile
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "naives"
ZIP_PATH = BASE_DIR / "naives.zip"
MODEL_PATH = BASE_DIR / "naive_bayes_model.pkl"


def read_csv_from_folder_or_zip(filename):
    folder_path = DATA_DIR / filename

    if folder_path.exists():
        return pd.read_csv(folder_path)

    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            with z.open(filename) as f:
                return pd.read_csv(f)

    raise FileNotFoundError(
        f"Could not find {filename} in naives/ or naives.zip."
    )


def clean_data(df):
    df = df.copy()
    df = df.dropna(subset=["eng", "chapter"])

    df["eng"] = (
        df["eng"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["chapter"] = df["chapter"].astype(str).str.strip()

    df = df[df["eng"].str.len() > 2]
    return df


def load_datasets():
    train_df = clean_data(read_csv_from_folder_or_zip("train.csv"))
    val_df = clean_data(read_csv_from_folder_or_zip("val.csv"))
    test_df = clean_data(read_csv_from_folder_or_zip("test.csv"))

    return train_df, val_df, test_df


def train_model():
    train_df, val_df, test_df = load_datasets()
    full_train = pd.concat([train_df, val_df], ignore_index=True)

    X_train = full_train["eng"]
    y_train = full_train["chapter"]

    X_test = test_df["eng"]
    y_test = test_df["chapter"]

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                max_features=120000,
                sublinear_tf=True
            )),
            ("classifier", MultinomialNB(alpha=0.005))
        ]
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Calculate Top-5 accuracy
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        top5_correct = 0
        classes = model.named_steps["classifier"].classes_
        for idx, true_label in enumerate(y_test):
            top5_idx = probabilities[idx].argsort()[-5:]
            top5_classes = classes[top5_idx]
            if true_label in top5_classes:
                top5_correct += 1
        top5_accuracy = top5_correct / len(y_test)
    else:
        top5_accuracy = test_accuracy

    report = classification_report(y_test, test_predictions, output_dict=True, zero_division=0)

    from sklearn.metrics import confusion_matrix
    classes = list(model.named_steps["classifier"].classes_)
    cm = confusion_matrix(y_test, test_predictions, labels=classes).tolist()

    print(f"Full Training samples: {len(full_train)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Number of chapters: {y_train.nunique()}")
    print(f"Test exact accuracy: {test_accuracy:.4f}")
    print(f"Test Top-5 accuracy: {top5_accuracy:.4f}")

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

    metrics["naive_bayes"] = {
        "accuracy": float(test_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "report": report,
        "classes": [str(c) for c in classes],
        "confusion_matrix": cm,
        "train_samples": len(full_train),
        "test_samples": len(test_df)
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return model, test_accuracy


def predict_chapter(question):
    """Return the most likely chapter for a student's question."""
    if not MODEL_PATH.exists():
        train_model()

    model = joblib.load(MODEL_PATH)

    question = str(question).strip()
    prediction = model.predict([question])[0]

    result = {"predicted_chapter": prediction}

    # Top chapter suggestions are useful in the study app.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([question])[0]
        classes = model.named_steps["classifier"].classes_
        top_indices = probabilities.argsort()[-5:][::-1]

        result["top_chapters"] = [
            {
                "chapter": classes[i],
                "probability": round(float(probabilities[i]), 4)
            }
            for i in top_indices
        ]

    return result


if __name__ == "__main__":
    train_model()
