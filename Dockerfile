FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install pydantic openai

ENV PYTHONUNBUFFERED=1

CMD ["python", "inference.py"]