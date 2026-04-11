"""
Disaster Response Coordinator — FINAL ROBUST Inference Script
(Hackathon-safe + crash-resistant + validator-proof)
"""

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
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "disaster-response-env"


# ── LOGGING (STRICT FORMAT) ─────────────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── NORMALIZE OBS (FRIEND LEVEL DEFENSE) ────────────────────────────────────
def normalize_obs(raw):
    try:
        if isinstance(raw, dict):
            if "observation" in raw:
                return raw["observation"]
            return raw

        if hasattr(raw, "model_dump"):
            return raw.model_dump()

        if hasattr(raw, "__dict__"):
            return raw.__dict__

    except Exception:
        pass

    return {}


# ── SANITIZE ACTION (ANTI-GARBAGE SHIELD) ───────────────────────────────────
def sanitize(d):
    clean = {}
    for k, v in d.items():
        try:
            v = int(v)
            if v > 0:
                clean[k] = v
        except:
            continue
    return clean


# ── LLM CALL (WITH RETRY + SAFETY) ──────────────────────────────────────────
def get_action(client: OpenAI, observation: dict) -> Action:

    zones     = observation.get("zones", [])
    resources = observation.get("resources", {})
    weather   = observation.get("weather", "unknown")
    time_step = observation.get("time_step", 0)
    zone_ids  = [z.get("id") for z in zones if "id" in z]

    zone_summary = "\n".join(
        f"Zone {z.get('id')} | pop={z.get('population')} | injured={z.get('injured')} | "
        f"flood={z.get('flood_level')}/10 | access={z.get('access')} | "
        f"sheltered={z.get('sheltered')} | barriers={z.get('flood_control_level')}"
        for z in zones
    )

    prompt = f"""
You are an AI disaster response coordinator managing a flood emergency.

Step: {time_step}
Weather: {weather}

Zones:
{zone_summary}

Resources:
rescue_teams={resources.get('rescue_teams', 0)}
food_units={resources.get('food_units', 0)}
medical_kits={resources.get('medical_kits', 0)}
helicopters={resources.get('helicopters', 0)}
flood_barriers={resources.get('flood_barriers', 0)}

Rules:
- Blocked zones need helicopters
- Do NOT exceed resources
- Prioritize high injury + high flood zones

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

    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
                timeout=10,
            )

            text = (completion.choices[0].message.content or "").strip()
            text = text.replace("```json", "").replace("```", "").strip()

            data = json.loads(text)

            return Action(
                allocate_rescue=sanitize(data.get("allocate_rescue", {})),
                send_food=sanitize(data.get("send_food", {})),
                send_medical=sanitize(data.get("send_medical", {})),
                deploy_helicopters=sanitize(data.get("deploy_helicopters", {})),
                deploy_barriers=sanitize(data.get("deploy_barriers", {})),
                evacuate=sanitize(data.get("evacuate", {})),
            )

        except Exception as e:
            if attempt == 1:
                print(f"LLM FAILED TWICE: {e}", flush=True)
                return rule_based_action(observation)


# ── FALLBACK (UNCHANGED BUT SAFE) ───────────────────────────────────────────
def rule_based_action(observation: dict) -> Action:
    zones     = observation.get("zones", [])
    resources = observation.get("resources", {})

    allocate_rescue = {}
    send_food = {}
    send_medical = {}
    deploy_helicopters = {}
    deploy_barriers = {}
    evacuate = {}

    for z in zones:
        zid = z.get("id")

        if not zid:
            continue

        if resources.get("rescue_teams", 0) > 0:
            allocate_rescue[zid] = 1

        if resources.get("food_units", 0) > 0:
            send_food[zid] = 5

        if resources.get("medical_kits", 0) > 0:
            send_medical[zid] = 2

        if z.get("flood_level", 0) >= 7:
            deploy_barriers[zid] = 1

        if z.get("access") in ("road_blocked", "air_only"):
            deploy_helicopters[zid] = 1

        evac_possible = z.get("population", 0) - z.get("sheltered", 0)
        if evac_possible > 0:
            evacuate[zid] = min(50, evac_possible)

    return Action(
        allocate_rescue=allocate_rescue,
        send_food=send_food,
        send_medical=send_medical,
        deploy_helicopters=deploy_helicopters,
        deploy_barriers=deploy_barriers,
        evacuate=evacuate,
    )


# ── RUN TASK ────────────────────────────────────────────────────────────────
async def run_task(client: OpenAI, task_module, task_name: str):

    env = DisasterEnv(task_module)
    rewards = []
    steps_taken = 0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()

        for step in range(1, task_module.max_steps + 1):

            if result.get("done"):
                break

            obs_dict = normalize_obs(result.get("observation", {}))

            if not obs_dict or "zones" not in obs_dict:
                action = rule_based_action(obs_dict)
            else:
                action = get_action(client, obs_dict)

            action_str = json.dumps({
                "rescue": action.allocate_rescue,
                "food": action.send_food,
                "medical": action.send_medical,
                "heli": action.deploy_helicopters,
                "barriers": action.deploy_barriers,
                "evac": action.evacuate,
            }, separators=(",", ":"))

            result = await env.step(action)

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error_msg = result.get("info", {}).get("error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done, error_msg)

            if done:
                break

        final_state = normalize_obs(result.get("observation", {}))
        score = grade(final_state)
        success = score >= 0.5

    except Exception as e:
        log_step(steps_taken + 1, "error", 0.0, True, str(e))

    finally:
        if hasattr(env, "close"):
            await env.close()

        log_end(success, steps_taken, rewards)

    return success


# ── MAIN ────────────────────────────────────────────────────────────────────
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks = [
        (easy_task, "easy"),
        (medium_task, "medium"),
        (hard_task, "hard"),
    ]

    for task_module, task_name in tasks:
        await run_task(client, task_module, task_name)
        print("", flush=True)


if __name__ == "__main__":
    asyncio.run(main())