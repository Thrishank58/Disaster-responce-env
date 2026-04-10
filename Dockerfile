FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install pydantic openai

ENV PYTHONUNBUFFERED=1

CMD ["python", "inference.py"]
FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir openai fastapi uvicorn pydantic openenv-core

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
