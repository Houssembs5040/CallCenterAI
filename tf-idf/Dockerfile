FROM python:3.11-slim

WORKDIR /app
COPY . /app
COPY model/tfidf ./models/tfidf/

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["uvicorn", "app_tfidf:app", "--host", "0.0.0.0", "--port", "5000"]
