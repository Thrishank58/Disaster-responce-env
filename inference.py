import asyncio
import os
import json
from typing import List, Optional

from openai import OpenAI

from env import DisasterEnv
from models import Action
from grader import grade
import tasks.easy as easy_task
import tasks.medium as medium_task
import tasks.hard as hard_task

# ── ENV VARS ────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "no-key")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "disaster-response-env"

# ── LOGGING ─────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── LLM CALL ────────────────────────────────────────────────────────────────
def get_action(client: OpenAI, observation: dict) -> Action:
    zones = observation.get("zones", [])
    resources = observation.get("resources", {})
    zone_ids = [z["id"] for z in zones]

    zone_summary = "\n".join(
        f"  Zone {z['id']}: population={z['population']}, injured={z['injured']}, "
        f"flood_level={z['flood_level']}, access={z['access']}"
        for z in zones
    )

    prompt = f"""You are an AI disaster response coordinator.

Current situation:
{zone_summary}

Available resources:
  rescue_teams={resources.get('rescue_teams', 0)}
  food_units={resources.get('food_units', 0)}
  medical_kits={resources.get('medical_kits', 0)}

Allocate resources across zones to minimize injuries and maximize survival.
Respond ONLY with a valid JSON object with these exact keys:
{{
  "allocate_rescue": {{"ZONE_ID": int, ...}},
  "send_food": {{"ZONE_ID": int, ...}},
  "send_medical": {{"ZONE_ID": int, ...}}
}}
Zone IDs are: {zone_ids}. No explanation, just JSON."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return Action(
            allocate_rescue=data.get("allocate_rescue", {}),
            send_food=data.get("send_food", {}),
            send_medical=data.get("send_medical", {}),
        )
    except Exception:
        return rule_based_action(observation)


def rule_based_action(observation: dict) -> Action:
    zones = observation.get("zones", [])
    resources = observation.get("resources", {})
    n = max(len(zones), 1)

    rescue  = resources.get("rescue_teams", 0)
    food    = resources.get("food_units", 0)
    medical = resources.get("medical_kits", 0)

    allocate_rescue, send_food, send_medical = {}, {}, {}
    for z in zones:
        zid = z["id"]
        allocate_rescue[zid] = max(1, rescue // n)
        send_food[zid]       = max(1, food // n)
        send_medical[zid]    = max(1, medical // n)

    return Action(
        allocate_rescue=allocate_rescue,
        send_food=send_food,
        send_medical=send_medical,
    )


# ── RUN ONE TASK ─────────────────────────────────────────────────────────────
async def run_task(client: OpenAI, task_module, task_name: str):
    env = DisasterEnv(task_module)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()

        for step in range(1, task_module.max_steps + 1):
            if result["done"]:
                break

            obs_dict = result["observation"].model_dump()
            action = get_action(client, obs_dict)
            action_str = f"rescue={action.allocate_rescue} food={action.send_food} medical={action.send_medical}"

            result = await env.step(action)

            reward = result["reward"]
            done   = result["done"]
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        final_state = result["observation"].model_dump()
        score = grade(final_state)
        score = max(0.0, min(1.0, score))
        success = score >= 0.5

    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── MAIN ─────────────────────────────────────────────────────────────────────
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_tasks = [
        (easy_task,   "easy"),
        (medium_task, "medium"),
        (hard_task,   "hard"),
    ]

    for task_module, task_name in all_tasks:
        await run_task(client, task_module, task_name)
        print("", flush=True)


if __name__ == "__main__":
    asyncio.run(main())