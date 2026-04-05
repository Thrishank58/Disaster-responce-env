FROM python:3.10

WORKDIR /app

# 🔥 FORCE COPY EVERYTHING (THIS IS THE FIX)
COPY . .

# 🔥 DEBUG (will show folders in logs)
RUN ls -R

RUN pip install --no-cache-dir openai fastapi uvicorn pydantic

EXPOSE 7860

CMD ["python", "inference.py"]