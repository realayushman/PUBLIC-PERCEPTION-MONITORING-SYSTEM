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
import pandas as pd

# -------------------- HARD-CODED CONFIG --------------------
MLFLOW_TRACKING_URI = "http://13.49.223.143:8000/"
MODEL_NAME = "yt_chrome_plugin_model"
MAX_BATCH_SIZE = 1000


# -------------------- FASTAPI INIT --------------------
app = FastAPI(title="Brand pulse")


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
    """MATCHES TRAINING PREPROCESSING EXACTLY"""
    if not isinstance(comment, str) or not comment.strip():
        return ""

    # Clean noise (URLs and HTML)
    comment = re.sub(r'https?://\S+|www\.\S+', '', comment)
    comment = re.sub(r'&[a-z]+;', '', comment)
    comment = comment.lower()
    # Keep the same punctuation you used in training!
    comment = re.sub(r'[^a-z0-9\s!?.,]', '', comment)

    words = comment.split()
    final_words = []
    for word in words:
        # Match your training variable names (STOP_WORDS vs stop_words)
        if word in SENTIMENT_WORDS or word not in STOP_WORDS:
            root_word = LEMMATIZER.lemmatize(word)
            # Match the 2-character rule from training
            if len(root_word) >= 2 or root_word in {'no', 'ok'}:
                final_words.append(root_word)

    return ' '.join(final_words).strip()


# -------------------- MODEL + VECTORIZER LOADING --------------------
def load_model_and_vectorizer():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Get the specific run_id of the LATEST version
    versions = client.get_latest_versions(MODEL_NAME, stages=["Staging", "None"])
    if not versions:
        raise RuntimeError("No model versions found in MLflow")
    
    # Grab the run_id from the actual registered model
    run_id = versions[0].run_id
    print(f"LOADING ASSETS FROM RUN: {run_id}")

    # Load both using the SAME RUN ID to ensure they match
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/lgbm_model")
    
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
import pandas as pd

import pandas as pd

@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    comments = payload.comments
    if not comments:
        raise HTTPException(400, "No comments provided")
    if len(comments) > MAX_BATCH_SIZE:
        raise HTTPException(413, "Too many comments")

    # Preprocess comments
    processed = []
    valid_indices = []
    for i, c in enumerate(payload.comments):
        p = preprocess_comment(c)
        if p:
            processed.append(p)
            valid_indices.append(i)

    try:
        features = vectorizer.transform(processed)
        
        # DEBUG: See what the model actually sees!
        # If this is empty, your vectorizer is the wrong version.
        print(f"WORDS RECOGNIZED: {vectorizer.inverse_transform(features)}")

        probs = model.predict_proba(features)
        pos_probs = probs[:, 1]
        
        final_preds = []
        for p in pos_probs:
            # Tuned thresholds for 95% model
            if p > 0.60:
                final_preds.append(1)
            elif p < 0.45:
                final_preds.append(0)
            else:
                final_preds.append(2)

        return PredictResponse(predictions=final_preds, indices=valid_indices)
    
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

# -------------------- LOCAL RUN --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
