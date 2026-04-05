from fastapi import FastAPI
from env import DisasterEnv
from models import Action
import tasks.easy as easy_task
import tasks.medium as medium_task
import tasks.hard as hard_task

app = FastAPI(title="Disaster Response Env")

envs = {
    "easy":   DisasterEnv(easy_task),
    "medium": DisasterEnv(medium_task),
    "hard":   DisasterEnv(hard_task),
}

active_env = envs["easy"]

@app.get("/")
async def root():
    return {"message": "Disaster Response Env is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reset")
async def reset(task: str = "easy"):
    global active_env
    active_env = envs.get(task, envs["easy"])
    result = await active_env.reset()
    return {
        "observation": result["observation"].model_dump(),
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"],
    }

@app.post("/step")
async def step(action: Action):
    result = await active_env.step(action)
    return {
        "observation": result["observation"].model_dump(),
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"],
    }

@app.get("/state")
async def state():
    obs = await active_env.state()
    return {"observation": obs.model_dump()}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()