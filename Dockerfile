FROM python:3.12-slim

WORKDIR /app

COPY main.py requirements.txt .env ./
COPY routes ./routes
COPY models ./models
COPY common ./common

RUN pip install -r requirements.txt

CMD ["fastapi", "run", "/app/main.py", "--port", "80"]