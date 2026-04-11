import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from env import DisasterEnv
from models import Action
import tasks.easy as easy_task
import tasks.medium as medium_task
import tasks.hard as hard_task

app = FastAPI(
    title="Disaster Response Coordinator Env",
    description="OpenEnv-compatible flood disaster response simulation. "
                "An AI agent allocates rescue teams, medical kits, food, helicopters, "
                "and flood barriers across zones under dynamic weather and access conditions.",
    version="2.0.0",
)

envs = {
    "easy":   DisasterEnv(easy_task),
    "medium": DisasterEnv(medium_task),
    "hard":   DisasterEnv(hard_task),
}

active_env = envs["easy"]


@app.get("/")
async def root():
    return {
        "message": "Disaster Response Coordinator Env is running",
        "tasks": list(envs.keys()),
        "version": "2.0.0",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(task: str = "easy"):
    global active_env
    if task not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Choose from: {list(envs.keys())}")
    active_env = envs[task]
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
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()