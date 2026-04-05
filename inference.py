import asyncio
import os
import json
from openai import OpenAI
from env import DisasterEnv
from tasks import easy, medium, hard
from grader import grade

# API config
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

MAX_STEPS = 15


def log_start(task_name):
    print(f"[START] {task_name}", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done}", flush=True)


def log_end(task_name, score):
    print(f"[END] {task_name} score={score:.2f}", flush=True)


def get_action_from_model(observation):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI disaster response coordinator."
                },
                {
                    "role": "user",
                    "content": f"State:\n{json.dumps(observation, indent=2)}\nReturn JSON action."
                }
            ],
            temperature=0.3,
        )

        text = response.choices[0].message.content.strip()
        action = json.loads(text)
        return action

    except Exception:
        # 🔥 Fallback logic (multi-zone aware)
        action = {
            "allocate_rescue": {},
            "send_food": {},
            "send_medical": {}
        }

        zones = observation.get("zones", [])

        for zone in zones:
            zid = zone["id"]
            injured = zone["injured"]
            flood = zone["flood_level"]
            access = zone["access"]

            priority = injured + (flood * 10)

            if priority > 150:
                rescue = 2
                food = 5
                medical = 4
            elif priority > 80:
                rescue = 1
                food = 4
                medical = 3
            else:
                rescue = 1
                food = 2
                medical = 1

            if access == "road_blocked":
                rescue = max(0, rescue - 1)

            action["allocate_rescue"][zid] = rescue
            action["send_food"][zid] = food
            action["send_medical"][zid] = medical

        return action


async def run_task(task_module, task_name):
    env = DisasterEnv(task_module)
    result = await env.reset()

    log_start(task_name)

    for step in range(1, MAX_STEPS + 1):
        obs = result["observation"].model_dump()

        action_dict = get_action_from_model(obs)

        action_obj = type("Action", (), action_dict)()

        result = await env.step(action_obj)

        reward = result["reward"]
        done = result["done"]

        log_step(step, action_dict, reward, done)

        if done:
            break

    final_state = (await env.state()).model_dump()
    score = grade(final_state)

    log_end(task_name, score)

    return score


async def main():
    tasks_list = [
        (easy, "easy"),
        (medium, "medium"),
        (hard, "hard")
    ]

    scores = []

    for task_module, name in tasks_list:
        score = await run_task(task_module, name)
        scores.append(score)

    avg_score = sum(scores) / len(scores)

    print(f"\n[FINAL] Average Score: {avg_score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())