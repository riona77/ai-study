"""
AI Study Lab - Unified Pickle Deployment & Inference API

This module loads all serialized `.pkl` models and provides a standalone,
deployment-ready API for performing inference across all 5 machine learning models.
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent


class AIStudyLabPredictor:
    """
    Unified predictor that loads model pipelines from .pkl files.
    Does NOT require training datasets at inference time.
    """

    def __init__(self, model_dir=BASE_DIR):
        self.model_dir = Path(model_dir)

        # Load standard pickle artifacts
        self.logistic_model = self._load_pkl("logistic_regression_model.pkl")
        self.decision_model = self._load_pkl("decision_tree_model.pkl")
        self.naive_model = self._load_pkl("naive_bayes_model.pkl")
        self.svm_model = self._load_pkl("svm_question_classifier.pkl")
        self.kmeans_artifact = self._load_pkl("kmeans_course_clusters.pkl")

    def _load_pkl(self, filename):
        path = self.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Pickle model artifact missing: {path}. "
                f"Please run model training script first."
            )
        return joblib.load(path)

    # -------------------------------------------------------------
    # 1. Logistic Regression - Course Recommendation
    # -------------------------------------------------------------
    def predict_course(
        self,
        interests,
        category,
        preferred_difficulty,
        past_experience,
        preferred_language
    ):
        """Recommend course based on student preference features."""
        input_df = pd.DataFrame([{
            "interests": interests,
            "category": category,
            "preferred_difficulty": preferred_difficulty,
            "past_experience": past_experience,
            "preferred_language": preferred_language
        }])

        prediction = self.logistic_model.predict(input_df)[0]
        result = {"recommended_course": prediction}

        if hasattr(self.logistic_model, "predict_proba"):
            probs = self.logistic_model.predict_proba(input_df)[0]
            classes = self.logistic_model.named_steps["classifier"].classes_
            top_idx = probs.argsort()[-5:][::-1]
            result["top_recommendations"] = [
                {
                    "course": str(classes[i]),
                    "probability": round(float(probs[i]), 4)
                }
                for i in top_idx
            ]

        return result

    # -------------------------------------------------------------
    # 2. Decision Tree - Learning Path Difficulty Guide
    # -------------------------------------------------------------
    def predict_learning_path(
        self,
        course_level,
        type_free_paid,
        module,
        duration_minutes,
        keyword
    ):
        """Predict appropriate resource difficulty level for learning path."""
        input_df = pd.DataFrame([{
            "Course_Level": course_level,
            "Type_Free_Paid": type_free_paid,
            "Module": module,
            "duration_minutes": float(duration_minutes),
            "Keywords_Tags_Skills_Interests_Categories": str(keyword).lower().strip()
        }])

        prediction = self.decision_model.predict(input_df)[0]
        result = {"predicted_difficulty": prediction}

        if hasattr(self.decision_model, "predict_proba"):
            probs = self.decision_model.predict_proba(input_df)[0]
            classes = self.decision_model.named_steps["classifier"].classes_
            result["class_probabilities"] = {
                str(classes[i]): round(float(probs[i]), 4)
                for i in range(len(classes))
            }

        return result

    # -------------------------------------------------------------
    # 3. Naive Bayes - Question Topic Classifier
    # -------------------------------------------------------------
    def predict_question_chapter(self, question):
        """Classify student question text into textbook chapter/topic."""
        question_text = str(question).strip()
        prediction = self.naive_model.predict([question_text])[0]
        result = {"predicted_chapter": prediction}

        if hasattr(self.naive_model, "predict_proba"):
            probs = self.naive_model.predict_proba([question_text])[0]
            classes = self.naive_model.named_steps["classifier"].classes_
            top_idx = probs.argsort()[-5:][::-1]
            result["top_chapters"] = [
                {
                    "chapter": str(classes[i]),
                    "probability": round(float(probs[i]), 4)
                }
                for i in top_idx
            ]

        return result

    # -------------------------------------------------------------
    # 4. SVM - Question Type Detective
    # -------------------------------------------------------------
    def predict_question_type(self, question):
        """Detect question structural intent (LOC, DESC, HUM, NUM, ENTY, ABBR)."""
        question_text = str(question).strip()
        prediction = self.svm_model.predict([question_text])[0]
        result = {"question_type": prediction}

        if hasattr(self.svm_model, "decision_function"):
            scores = self.svm_model.decision_function([question_text])[0]
            classes = self.svm_model.named_steps["classifier"].classes_
            if np.ndim(scores) == 0:
                scores = [float(scores)]
            order = np.argsort(scores)[::-1]
            result["top_type_scores"] = [
                {
                    "type": str(classes[i]),
                    "score": round(float(scores[i]), 4)
                }
                for i in order[:3]
            ]

        return result

    # -------------------------------------------------------------
    # 5. K-Means - Course Cluster Explorer
    # -------------------------------------------------------------
    def predict_course_cluster(self, query, top_n=5):
        """Assign query to a K-Means cluster world and return top similar courses."""
        vectorizer = self.kmeans_artifact["vectorizer"]
        model = self.kmeans_artifact["model"]
        courses = self.kmeans_artifact["courses"].copy()

        query_vec = vectorizer.transform([str(query)])
        cluster_id = int(model.predict(query_vec)[0])

        matching = courses[courses["cluster"] == cluster_id].copy()
        course_vecs = vectorizer.transform(matching["text"].fillna(""))

        sims = cosine_similarity(query_vec, course_vecs)[0]
        matching["similarity"] = sims

        top_matches = matching.sort_values("similarity", ascending=False).head(top_n)

        return {
            "cluster_id": cluster_id,
            "matched_courses": top_matches[
                ["title", "subject", "level", "url", "similarity"]
            ].to_dict("records")
        }


# Quick test interface when executed directly
if __name__ == "__main__":
    print("[+] Initializing AI Study Lab Pickle Predictor...")
    predictor = AIStudyLabPredictor()

    print("\n--- 1. Logistic Regression ---")
    print(predictor.predict_course("Algebra", "Mathematics", "beginner", "none", "English"))

    print("\n--- 2. Decision Tree ---")
    print(predictor.predict_learning_path("Beginner", "Free", "Python Programming", 60, "python"))

    print("\n--- 3. Naive Bayes ---")
    print(predictor.predict_question_chapter("What is Newton's second law of motion?"))

    print("\n--- 4. SVM ---")
    print(predictor.predict_question_type("Where is Mount Everest located?"))

    print("\n--- 5. K-Means ---")
    print(predictor.predict_course_cluster("machine learning data science"))
