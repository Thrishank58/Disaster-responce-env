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


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "disaster-response-env"


# ── LOGGING ─────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── NORMALIZE OBS ───────────────────────────────────
def normalize_obs(raw):
    if isinstance(raw, dict):
        return raw.get("observation", raw)
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if hasattr(raw, "__dict__"):
        return raw.__dict__
    return {}


# ── SANITIZE ACTION ─────────────────────────────────
def sanitize(d):
    return {k: max(0, int(v)) for k, v in d.items() if isinstance(v, (int, float))}


# ── PRIORITY FUNCTION (SMART STRATEGY) ──────────────
def compute_priority(zone):
    pop = zone.get("population", 1)
    injured = zone.get("injured", 0)
    flood = zone.get("flood_level", 0)
    sheltered = zone.get("sheltered", 0)

    injury_ratio = injured / max(pop, 1)
    unsheltered = pop - sheltered

    return injury_ratio * 3 + flood * 0.8 + (unsheltered / pop) * 1.5


# ── LLM ACTION ──────────────────────────────────────
def get_action(client, observation):

    zones = observation.get("zones", [])
    zones = sorted(zones, key=compute_priority, reverse=True)

    resources = observation.get("resources", {})
    weather = observation.get("weather", "unknown")
    time_step = observation.get("time_step", 0)

    zone_summary = "\n".join(
        f"{z['id']}: pop={z['population']}, inj={z['injured']}, flood={z['flood_level']}, access={z['access']}"
        for z in zones
    )

    prompt = f"""
You are an expert disaster response AI maximizing survival score.

Rules:
- PRIORITIZE highest injury zones
- If flood >= 7 → deploy barriers
- If blocked → send helicopter FIRST
- Use medical kits for high injuries
- Avoid unnecessary food
- Evacuate high-risk zones

Step: {time_step}
Weather: {weather}

Zones:
{zone_summary}

Resources:
{resources}

Return ONLY JSON:
{{
  "allocate_rescue": {{}},
  "send_food": {{}},
  "send_medical": {{}},
  "deploy_helicopters": {{}},
  "deploy_barriers": {{}},
  "evacuate": {{}}
}}
"""

    for _ in range(2):
        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=10,
            )

            data = json.loads(res.choices[0].message.content)

            return Action(
                allocate_rescue=sanitize(data.get("allocate_rescue", {})),
                send_food=sanitize(data.get("send_food", {})),
                send_medical=sanitize(data.get("send_medical", {})),
                deploy_helicopters=sanitize(data.get("deploy_helicopters", {})),
                deploy_barriers=sanitize(data.get("deploy_barriers", {})),
                evacuate=sanitize(data.get("evacuate", {})),
            )

        except:
            continue

    return rule_based_action(observation)


# ── FALLBACK ────────────────────────────────────────
def rule_based_action(observation):
    zones = observation.get("zones", [])
    resources = observation.get("resources", {})

    action = Action()

    for z in zones:
        zid = z["id"]

        if resources.get("rescue_teams", 0) > 0:
            action.allocate_rescue[zid] = 1

        if z["flood_level"] >= 7:
            action.deploy_barriers[zid] = 1

        if z["access"] != "open":
            action.deploy_helicopters[zid] = 1

        action.evacuate[zid] = min(50, z["population"] - z["sheltered"])

    return action


# ── RUN TASK ────────────────────────────────────────
async def run_task(client, task_module, task_name):

    env = DisasterEnv(task_module)
    rewards = []
    steps = 0

    log_start(task_name, BENCHMARK, MODEL_NAME)

    try:
        result = await env.reset()

        for step in range(1, task_module.max_steps + 1):

            if result["done"]:
                break

            obs = normalize_obs(result["observation"])

            action = get_action(client, obs)

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))

            result = await env.step(action)

            reward = result["reward"]
            done = result["done"]
            error = result["info"].get("error")

            rewards.append(reward)
            steps = step

            log_step(step, action_str, reward, done, error)

            if done:
                break

        final_state = normalize_obs(result["observation"])
        score = grade(final_state)
        success = score >= 0.5

    except Exception as e:
        log_step(steps + 1, "error", 0.0, True, str(e))
        success = False

    finally:
        log_end(success, steps, rewards)

    return success


# ── MAIN ────────────────────────────────────────────
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks = [(easy_task, "easy"), (medium_task, "medium"), (hard_task, "hard")]

    for t, name in tasks:
        await run_task(client, t, name)
        print()


if __name__ == "__main__":
    asyncio.run(main())