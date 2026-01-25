import re
import joblib
import mlflow
import numpy as np
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient


# -------------------- HARD-CODED CONFIG --------------------
MLFLOW_TRACKING_URI = "http://16.16.58.154:8000"
MODEL_NAME = "yt_chrome_plugin_model"
MAX_BATCH_SIZE = 1000


# -------------------- FASTAPI INIT --------------------
app = FastAPI(title="Sentiment Prediction API")


# -------------------- NLP GLOBAL OBJECTS --------------------
SENTIMENT_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'none',
    'but', 'however', 'although', 'though', 'yet', 'still',
    'very', 'extremely', 'absolutely', 'completely', 'totally',
    'quite', 'rather', 'somewhat', 'slightly', 'fairly',
    'good', 'bad', 'great', 'terrible', 'awesome', 'awful',
    'love', 'hate', 'like', 'dislike', 'amazing', 'horrible',
    'best', 'worst', 'better', 'worse', 'excellent', 'poor'
}

STOP_WORDS = set(stopwords.words("english")) - SENTIMENT_WORDS
LEMMATIZER = WordNetLemmatizer()


# -------------------- PREPROCESSING --------------------
def preprocess_comment(comment: str) -> str:
    if not isinstance(comment, str) or not comment.strip():
        return ""

    comment = re.sub(r'https?://\S+|www\.\S+', '', comment)
    comment = re.sub(r'&[a-z]+;', '', comment)
    comment = comment.lower()
    comment = re.sub(r'\n+', ' ', comment)
    comment = re.sub(r'\s+', ' ', comment)
    comment = re.sub(r'[^a-z0-9\s!?.,]', '', comment)

    words = comment.split()
    words = [
        w for w in words
        if w in SENTIMENT_WORDS or w not in STOP_WORDS
    ]

    words = [
        LEMMATIZER.lemmatize(w)
        for w in words
        if len(w) >= 2 or w in {"no", "ok"}
    ]

    return " ".join(words).strip()


# -------------------- MODEL + VECTORIZER LOADING --------------------
def load_model_and_vectorizer():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME)
    if not versions:
        raise RuntimeError("No model versions found in MLflow")

    run_id = versions[0].run_id

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

    vectorizer_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="tfidf_vectorizer.pkl"
    )
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


# FAIL FAST IF THIS BREAKS
model, vectorizer = load_model_and_vectorizer()


# -------------------- REQUEST / RESPONSE SCHEMAS --------------------
class PredictRequest(BaseModel):
    comments: List[str]


class PredictResponse(BaseModel):
    predictions: List[int]
    indices: List[int]


# -------------------- ENDPOINT --------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    comments = payload.comments

    if not comments:
        raise HTTPException(400, "No comments provided")

    if len(comments) > MAX_BATCH_SIZE:
        raise HTTPException(413, "Too many comments")

    processed = []
    valid_indices = []

    for i, comment in enumerate(comments):
        p = preprocess_comment(comment)
        if p:
            processed.append(p)
            valid_indices.append(i)

    if not processed:
        raise HTTPException(400, "No valid comments after preprocessing")

    try:
        features = vectorizer.transform(processed)
        preds = model.predict(features)

        if preds.ndim > 1:
            preds = preds.argmax(axis=1)

        return PredictResponse(
            predictions=preds.tolist(),
            indices=valid_indices
        )

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# -------------------- LOCAL RUN --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
