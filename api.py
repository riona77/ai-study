"""
AI Study Lab - FastAPI REST Server

Exposes RESTful API endpoints for the 5 trained ML models using FastAPI and Pydantic.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from predict import AIStudyLabPredictor

# Initialize FastAPI application
app = FastAPI(
    title="AI Study Lab ML REST API",
    description="FastAPI service for 5 machine learning models (Logistic Regression, Decision Tree, Naive Bayes, SVM, K-Means)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor instance
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        predictor = AIStudyLabPredictor()
        print("[+] FastAPI ML Predictor initialized successfully!")
    except Exception as e:
        print(f"[-] Error initializing predictor: {e}")


# Pydantic Schemas for Requests
class CourseRequest(BaseModel):
    interests: str = Field(..., example="Algebra")
    category: str = Field(..., example="Mathematics")
    preferred_difficulty: str = Field(..., example="beginner")
    past_experience: str = Field(..., example="none")
    preferred_language: str = Field(..., example="English")


class PathRequest(BaseModel):
    course_level: str = Field(..., example="Beginner")
    type_free_paid: str = Field(..., example="Free")
    module: str = Field(..., example="Python Programming")
    duration_minutes: float = Field(60.0, example=60.0)
    keyword: str = Field(..., example="python")


class QuestionRequest(BaseModel):
    question: str = Field(..., example="What is Newton's second law of motion?")


class ClusterRequest(BaseModel):
    query: str = Field(..., example="machine learning data science")
    top_n: Optional[int] = Field(5, example=5)


# API Routes
@app.get("/")
def read_root():
    return {
        "title": "AI Study Lab REST API",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": [
            "/api/predict/course",
            "/api/predict/path",
            "/api/predict/chapter",
            "/api/predict/question-type",
            "/api/predict/cluster",
            "/api/metrics"
        ]
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": predictor is not None}


@app.post("/api/predict/course")
def predict_course(req: CourseRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    try:
        return predictor.predict_course(
            req.interests,
            req.category,
            req.preferred_difficulty,
            req.past_experience,
            req.preferred_language
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict/path")
def predict_path(req: PathRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    try:
        return predictor.predict_learning_path(
            req.course_level,
            req.type_free_paid,
            req.module,
            req.duration_minutes,
            req.keyword
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict/chapter")
def predict_chapter(req: QuestionRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    try:
        return predictor.predict_question_chapter(req.question)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict/question-type")
def predict_question_type(req: QuestionRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    try:
        return predictor.predict_question_type(req.question)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict/cluster")
def predict_cluster(req: ClusterRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    try:
        return predictor.predict_course_cluster(req.query, top_n=req.top_n)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/metrics")
def get_metrics():
    metrics_path = Path(__file__).resolve().parent / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"error": "Metrics file not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)
