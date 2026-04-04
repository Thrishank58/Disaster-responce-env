from fastapi import FastAPI
from env import DisasterEnv
import tasks.easy as task

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Disaster Response Env Running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/run")
async def run_env():
    env = DisasterEnv(task)
    result = await env.reset()
    return {"observation": result["observation"].dict()}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()