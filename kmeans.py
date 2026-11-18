"""
AI Study Lab - K-Means
Purpose: Group similar educational courses/resources into clusters.
Dataset: k means/edx_courses.json

The model uses course title, descriptions, subjects, skills and tags.
It is unsupervised: there is no target column.
"""

from pathlib import Path
import zipfile
import json
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "k means"
ZIP_PATH = BASE_DIR / "k means.zip"
MODEL_PATH = BASE_DIR / "kmeans_course_clusters.pkl"


def load_courses():
    """Load edX course data from folder or ZIP."""
    json_path = DATA_DIR / "edx_courses.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            with z.open("edx_courses.json") as f:
                return json.load(f)

    raise FileNotFoundError(
        "Could not find k means/edx_courses.json or k means.zip."
    )


def flatten(value):
    """Convert strings/lists/dicts to searchable text."""
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(flatten(item) for item in value)

    if isinstance(value, dict):
        return " ".join(flatten(v) for v in value.values())

    return str(value)


def clean_html(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataframe(records):
    rows = []

    for item in records:
        title = clean_html(flatten(item.get("title")))
        description = clean_html(
            flatten(item.get("primary_description"))
        )
        secondary = clean_html(
            flatten(item.get("secondary_description"))
        )
        subject = flatten(item.get("subject"))
        skills = flatten(item.get("skills"))
        tags = flatten(item.get("tags"))

        combined_text = " ".join([
            title,
            description,
            secondary,
            subject,
            skills,
            tags
        ]).strip()

        if combined_text:
            rows.append({
                "title": title if title else "Untitled course",
                "subject": subject,
                "level": flatten(item.get("level")),
                "url": flatten(item.get("marketing_url"))
                       or flatten(item.get("external_url")),
                "text": combined_text
            })

    return pd.DataFrame(rows)


def train_model(n_clusters=8):
    records = load_courses()
    df = prepare_dataframe(records)

    if len(df) < n_clusters:
        raise ValueError("Not enough courses for the requested number of clusters.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df["text"])

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=128,
        n_init=10
    )

    labels = model.fit_predict(X)
    df["cluster"] = labels

    silhouette = silhouette_score(X, labels)

    print(f"Courses used: {len(df)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette score: {silhouette:.4f}")

    cluster_counts = df["cluster"].value_counts().sort_index().to_dict()
    print("\nCourses per cluster:")
    print(cluster_counts)

    # Save the vectorizer, cluster model and course metadata together.
    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "courses": df[
            ["title", "subject", "level", "url", "text", "cluster"]
        ].copy()
    }

    joblib.dump(artifact, MODEL_PATH)
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

    metrics["kmeans"] = {
        "silhouette_score": float(silhouette),
        "n_clusters": int(n_clusters),
        "total_courses": len(df),
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()}
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return artifact, silhouette


def find_similar_resources(query, top_n=5):
    """
    Find courses in the same cluster as the query.
    This is the function the Streamlit app can call.
    """
    if not MODEL_PATH.exists():
        train_model()

    artifact = joblib.load(MODEL_PATH)
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    courses = artifact["courses"].copy()

    query_vector = vectorizer.transform([str(query)])
    cluster = int(model.predict(query_vector)[0])

    matching = courses[courses["cluster"] == cluster].copy()

    # Rank matching courses by TF-IDF cosine similarity.
    course_vectors = vectorizer.transform(
        matching["text"].fillna("")
    )

    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(query_vector, course_vectors)[0]
    matching["similarity"] = similarities

    matching = matching.sort_values(
        "similarity", ascending=False
    ).head(top_n)

    return {
        "cluster": cluster,
        "resources": matching[
            ["title", "subject", "level", "url", "similarity"]
        ].to_dict("records")
    }


if __name__ == "__main__":
    train_model()
