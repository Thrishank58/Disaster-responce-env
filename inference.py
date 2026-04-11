"""
Disaster Response Coordinator — Baseline Inference Script

Environment variables:
  API_BASE_URL   : LLM API base URL       (default: https://api.openai.com/v1)
  MODEL_NAME     : model identifier       (default: gpt-4o-mini)
  HF_TOKEN       : Hugging Face API key   (REQUIRED — no default)
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

# ── ENV VARS ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "disaster-response-env"

# ── LOGGING — exact format required by hackathon checker ─────────────────────
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


# ── LLM CALL ─────────────────────────────────────────────────────────────────
def get_action(client: OpenAI, observation: dict) -> Action:
    zones     = observation.get("zones", [])
    resources = observation.get("resources", {})
    weather   = observation.get("weather", "unknown")
    time_step = observation.get("time_step", 0)
    zone_ids  = [z["id"] for z in zones]

    zone_summary = "\n".join(
        f"  Zone {z['id']}: pop={z['population']}, injured={z['injured']}, "
        f"flood={z['flood_level']}/10, access={z['access']}, "
        f"sheltered={z['sheltered']}, barriers_deployed={z['flood_control_level']}"
        for z in zones
    )

    prompt = f"""You are an AI disaster response coordinator managing a flood emergency.

=== CURRENT STATE (step {time_step}) ===
Weather: {weather}

Zones:
{zone_summary}

Available resources (totals across ALL zones must NOT exceed these):
  rescue_teams  : {resources.get('rescue_teams', 0)}
  food_units    : {resources.get('food_units', 0)}
  medical_kits  : {resources.get('medical_kits', 0)}
  helicopters   : {resources.get('helicopters', 0)}
  flood_barriers: {resources.get('flood_barriers', 0)}

=== KEY RULES ===
1. Zones with access=road_blocked or access=air_only CANNOT receive rescue/medical/food
   UNLESS you also send at least 1 helicopter to that zone via deploy_helicopters.
2. Total resource usage across all zones must not exceed available amounts.
3. deploy_barriers increases a zone's flood resistance (reduces future flood rise).
4. evacuate moves civilians to shelter, reducing future injury exposure.
5. Conserve some resources for future steps — episodes last {20} steps.

=== STRATEGY TIPS ===
- Prioritise zones with high injury counts AND high flood levels.
- Always send helicopters to blocked/air-only zones if you want to help them.
- Deploy barriers early to high-flood zones to prevent worsening.
- Spread medical kits across steps rather than using all at once.

Respond ONLY with a valid JSON object (no explanation, no markdown):
{{
  "allocate_rescue":    {{"ZONE_ID": int, ...}},
  "send_food":          {{"ZONE_ID": int, ...}},
  "send_medical":       {{"ZONE_ID": int, ...}},
  "deploy_helicopters": {{"ZONE_ID": int, ...}},
  "deploy_barriers":    {{"ZONE_ID": int, ...}},
  "evacuate":           {{"ZONE_ID": int, ...}}
}}
Zone IDs: {zone_ids}"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return Action(
            allocate_rescue=data.get("allocate_rescue", {}),
            send_food=data.get("send_food", {}),
            send_medical=data.get("send_medical", {}),
            deploy_helicopters=data.get("deploy_helicopters", {}),
            deploy_barriers=data.get("deploy_barriers", {}),
            evacuate=data.get("evacuate", {}),
        )
    except Exception as e:
        print(f"LLM ERROR: {e}", flush=True)
        return rule_based_action(observation)


# ── RULE-BASED FALLBACK ───────────────────────────────────────────────────────
def rule_based_action(observation: dict) -> Action:
    """
    Spreads resources conservatively across steps.
    Sends helicopters to blocked zones before rescue/medical.
    Deploys barriers to highest-flood zones.
    """
    zones     = observation.get("zones", [])
    resources = observation.get("resources", {})

    available_rescue  = resources.get("rescue_teams", 0)
    available_food    = resources.get("food_units", 0)
    available_medical = resources.get("medical_kits", 0)
    available_helis   = resources.get("helicopters", 0)
    available_barriers = resources.get("flood_barriers", 0)

    # Use at most half the available resources per step (save for later)
    budget_rescue  = max(1, available_rescue  // 2)
    budget_food    = max(1, available_food    // 2)
    budget_medical = max(1, available_medical // 2)
    budget_barriers = available_barriers // 2

    allocate_rescue    = {}
    send_food          = {}
    send_medical       = {}
    deploy_helicopters = {}
    deploy_barriers    = {}
    evacuate           = {}

    # Sort zones by injury severity descending for triage
    sorted_zones = sorted(zones, key=lambda z: z["injured"], reverse=True)
    n = max(len(sorted_zones), 1)

    helis_left    = available_helis
    barriers_left = budget_barriers
    rescue_left   = budget_rescue
    food_left     = budget_food
    medical_left  = budget_medical

    for zone in sorted_zones:
        zid = zone["id"]
        blocked  = zone["access"] in ("road_blocked", "air_only")

        # Send helicopter to blocked zone (1 per blocked zone while stock lasts)
        if blocked and helis_left > 0:
            deploy_helicopters[zid] = 1
            helis_left -= 1

        can_reach = (not blocked) or (zid in deploy_helicopters)

        # Allocate rescue proportionally
        share_rescue = max(1, rescue_left // n)
        if can_reach and zone["injured"] > 0:
            allocate_rescue[zid] = share_rescue
            rescue_left = max(0, rescue_left - share_rescue)

        # Food for all reachable zones
        share_food = max(1, food_left // n)
        if can_reach or not blocked:
            send_food[zid] = share_food
            food_left = max(0, food_left - share_food)

        # Medical kits
        share_medical = max(1, medical_left // n)
        if can_reach and zone["injured"] > 0:
            send_medical[zid] = share_medical
            medical_left = max(0, medical_left - share_medical)

        # Barrier to high-flood zones
        if barriers_left > 0 and zone["flood_level"] >= 7:
            deploy_barriers[zid] = 1
            barriers_left -= 1

        # Evacuate a small batch each step
        can_evacuate = zone["population"] - zone["sheltered"]
        if can_evacuate > 0:
            evacuate[zid] = min(50, can_evacuate)

    return Action(
        allocate_rescue=allocate_rescue,
        send_food=send_food,
        send_medical=send_medical,
        deploy_helicopters=deploy_helicopters,
        deploy_barriers=deploy_barriers,
        evacuate=evacuate,
    )


# ── RUN ONE TASK ──────────────────────────────────────────────────────────────
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
            action   = get_action(client, obs_dict)

            action_str = (
                f"rescue={action.allocate_rescue} food={action.send_food} "
                f"medical={action.send_medical} heli={action.deploy_helicopters} "
                f"barriers={action.deploy_barriers} evac={action.evacuate}"
            )

            result      = await env.step(action)
            reward      = result["reward"]
            done        = result["done"]
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        final_state = result["observation"].model_dump()
        score   = grade(final_state)
        score   = max(0.0, min(1.0, score))
        success = score >= 0.5

    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


# ── MAIN ──────────────────────────────────────────────────────────────────────
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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