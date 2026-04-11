FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai openenv-core

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]