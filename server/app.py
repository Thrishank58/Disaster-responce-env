from fastapi import FastAPI
from env import DisasterEnv
import tasks.easy as easy_task
import tasks.medium as medium_task
import tasks.hard as hard_task

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Disaster Response Env Running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/run/easy")
async def run_easy():
    env = DisasterEnv(easy_task)
    result = await env.reset()
    return {"observation": result["observation"].dict()}

@app.get("/run/medium")
async def run_medium():
    env = DisasterEnv(medium_task)
    result = await env.reset()
    return {"observation": result["observation"].dict()}

@app.get("/run/hard")
async def run_hard():
    env = DisasterEnv(hard_task)
    result = await env.reset()
    return {"observation": result["observation"].dict()}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()