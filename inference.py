import asyncio
from openai import OpenAI
from env import DisasterEnv
from grader import grade
import tasks.easy as easy
import tasks.medium as medium
import tasks.hard as hard

async def run(task):
    env = DisasterEnv(task)
    result = await env.reset()
    print("Initial observation:", result["observation"])

    for _ in range(task.max_steps):
        from models import Action
        action = Action(
            allocate_rescue={"A": 2},
            send_food={"A": 10},
            send_medical={"A": 5}
        )
        result = await env.step(action)
        print(f"Step reward: {result['reward']}, Done: {result['done']}")
        if result["done"]:
            break

    final_state = result["observation"].model_dump()
    score = grade(final_state)
    print(f"Final score: {score}")

if __name__ == "__main__":
    asyncio.run(run(easy))