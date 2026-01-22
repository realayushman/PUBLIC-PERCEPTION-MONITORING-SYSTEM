import re
import joblib
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

# -------------------- Preprocessing --------------------

STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
LEMMATIZER = WordNetLemmatizer()

def preprocess_comment(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s!?.,]", "", text)
    return " ".join(
        LEMMATIZER.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS
    )

# -------------------- Model Loading (ONCE) --------------------

def load_model_and_vectorizer():
    mlflow.set_tracking_uri("http://51.20.255.70:8000/")
    MlflowClient()  # sanity init

    model = mlflow.pyfunc.load_model("models:/yt_chrome_plugin_model/1")
    vectorizer = joblib.load("./tfidf_vectorizer.pkl")

    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()
FEATURES = vectorizer.get_feature_names_out()

# -------------------- Schema --------------------

class PredictRequest(BaseModel):
    comments: List[str]

# -------------------- Endpoint --------------------

@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        # ... (Preprocess steps) ...
        cleaned = [preprocess_comment(c) for c in payload.comments]

        # Transform to sparse
        X_sparse = vectorizer.transform(cleaned)
        
        # Convert to DataFrame using the exact feature names from the vectorizer
        X_df = pd.DataFrame(
            X_sparse.toarray(), 
            columns=vectorizer.get_feature_names_out()
        )
        
        # *** CRITICAL FIX: Ensure all column names are clean and match the schema exactly ***
        X_df.columns = X_df.columns.str.strip()

        # Predict
        preds = model.predict(X_df)

        return {"predictions": [int(p) for p in preds]}
    except Exception as e:
        # Log the exact error for debugging
        print(f"Prediction error: {e}")
        return {"error": str(e)}

