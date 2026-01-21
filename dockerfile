FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependency
RUN apt-get update && apt-get install -y libgomp1

# Copy app code
COPY fastapi/ /app/

# Copy vectorizer
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources
RUN python -m nltk.downloader stopwords wordnet

# Expose container port (matches FastAPI uvicorn)
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
