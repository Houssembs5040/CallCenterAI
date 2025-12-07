FROM python:3.11-slim

WORKDIR /app

# Copy the whole project (including .dvc files)
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC with local or cloud support
RUN pip install dvc[s3]

# Pull model from DVC storage
COPY models/tfidf/ ./models/tfidf/

EXPOSE 5000
CMD ["uvicorn", "app_tfidf:app", "--host", "0.0.0.0", "--port", "5000"]
