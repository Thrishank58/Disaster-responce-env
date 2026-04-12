"""
Disaster Response Coordinator — Baseline Inference Script

STDOUT (OpenEnv): [START] … → [STEP] … per env.step() → [END] … after env.close().
[END] includes score=<episode score> and rewards=<comma-separated step rewards>, all 2 d.p.

Environment variables:
  API_BASE_URL          : LLM base URL (default: https://api.openai.com/v1)
  MODEL_NAME            : model id       (default: gpt-4.1-mini)
  HF_TOKEN              : HF API token   (required)
  INFERENCE_STEP_SLEEP  : seconds to sleep after each successful LLM call (default: 0).
                          Set e.g. 0.25 if you hit rate limits; keep 0 for eval time limits.
"""

import asyncio
import json
import os
import sys
import time
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
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

try:
    INFERENCE_STEP_SLEEP = float(os.getenv("INFERENCE_STEP_SLEEP") or "0")
except ValueError:
    INFERENCE_STEP_SLEEP = 0.0

BENCHMARK = "disaster-response-env"


# ── CLAMP HELPER ──────────────────────────────────────────────────────────────
def _clamp(v: float) -> float:
    """Strictly between 0 and 1 — never exactly 0.0 or 1.0 (platform task validation)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.01
    if x != x or x == float("inf") or x == float("-inf"):  # NaN / inf
        return 0.01
    return max(0.01, min(0.99, x))


# ── LOGGING — exact format required by hackathon checker ─────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={float(reward):.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """[END] must include episode score (task grader, in [0, 1] per OpenEnv spec)."""
    rewards_str = ",".join(f"{float(r):.2f}" for r in rewards)
    s = _clamp(score)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={s:.2f} rewards={rewards_str}",
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

    def _sanitize(d):
        return {k: int(v) for k, v in d.items() if v and int(v) > 0}

    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
                stream=False,
                timeout=10,
            )
            text = (completion.choices[0].message.content or "").strip()
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            action = Action(
                allocate_rescue=_sanitize(data.get("allocate_rescue", {})),
                send_food=_sanitize(data.get("send_food", {})),
                send_medical=_sanitize(data.get("send_medical", {})),
                deploy_helicopters=_sanitize(data.get("deploy_helicopters", {})),
                deploy_barriers=_sanitize(data.get("deploy_barriers", {})),
                evacuate=_sanitize(data.get("evacuate", {})),
            )
            if INFERENCE_STEP_SLEEP > 0:
                time.sleep(INFERENCE_STEP_SLEEP)
            return _fix_action(action, zones)
        except Exception as e:
            print(f"LLM ERROR (attempt {attempt+1}): {e}", file=sys.stderr, flush=True)
            if attempt == 1:
                return rule_based_action(observation)


# ── ACTION VALIDATOR ──────────────────────────────────────────────────────────
def _fix_action(action: Action, zones: list) -> Action:
    """
    Fixes common LLM mistakes:
    1. Removes rescue/medical/food allocations to blocked zones with no helicopter.
    2. Auto-assigns helicopters to blocked zones that need help but were forgotten.
    3. Redirects wasted resources to open zones.
    """
    zone_map = {z["id"]: z for z in zones}
    helis = dict(action.deploy_helicopters)

    for zid, z in zone_map.items():
        blocked = z["access"] in ("road_blocked", "air_only")
        has_allocation = (
            action.allocate_rescue.get(zid, 0) > 0
            or action.send_medical.get(zid, 0) > 0
        )
        if blocked and has_allocation and zid not in helis:
            helis[zid] = 1

    allocate_rescue = dict(action.allocate_rescue)
    send_medical    = dict(action.send_medical)
    send_food       = dict(action.send_food)
    open_zones      = [z["id"] for z in zones if z["access"] == "open"]

    for zid, z in zone_map.items():
        blocked = z["access"] in ("road_blocked", "air_only")
        if blocked and zid not in helis:
            wasted_rescue  = allocate_rescue.pop(zid, 0)
            wasted_medical = send_medical.pop(zid, 0)
            if open_zones and (wasted_rescue > 0 or wasted_medical > 0):
                best = open_zones[0]
                allocate_rescue[best] = allocate_rescue.get(best, 0) + wasted_rescue
                send_medical[best]    = send_medical.get(best, 0) + wasted_medical

    return Action(
        allocate_rescue=allocate_rescue,
        send_food=send_food,
        send_medical=send_medical,
        deploy_helicopters=helis,
        deploy_barriers=action.deploy_barriers,
        evacuate=action.evacuate,
    )


# ── RULE-BASED FALLBACK ───────────────────────────────────────────────────────
def rule_based_action(observation: dict) -> Action:
    """
    Triage-based resource allocation.
    Never over-allocates. Sends helicopters to blocked zones first.
    """
    zones     = observation.get("zones", [])
    resources = observation.get("resources", {})

    rescue   = resources.get("rescue_teams", 0)
    food     = resources.get("food_units", 0)
    medical  = resources.get("medical_kits", 0)
    helis    = resources.get("helicopters", 0)
    barriers = resources.get("flood_barriers", 0)

    allocate_rescue    = {}
    send_food          = {}
    send_medical       = {}
    deploy_helicopters = {}
    deploy_barriers    = {}
    evacuate           = {}

    sorted_zones = sorted(zones, key=lambda z: z["injured"], reverse=True)

    for zone in sorted_zones:
        zid = zone["id"]
        if zone["access"] in ("road_blocked", "air_only") and helis > 0:
            deploy_helicopters[zid] = 1
            helis -= 1

    reachable_zones = [
        z for z in sorted_zones
        if z["access"] == "open" or z["id"] in deploy_helicopters
    ]
    nr = max(len(reachable_zones), 1)

    for zone in reachable_zones:
        zid = zone["id"]

        if zone["injured"] > 0 and rescue > 0:
            share = min(max(1, rescue // nr), rescue)
            allocate_rescue[zid] = share
            rescue -= share

        if food > 0:
            share = min(max(1, food // nr), food)
            send_food[zid] = share
            food -= share

        if zone["injured"] > 0 and medical > 0:
            share = min(max(1, medical // nr), medical)
            send_medical[zid] = share
            medical -= share

    for zone in sorted_zones:
        zid = zone["id"]
        if barriers > 0 and zone["flood_level"] >= 7:
            deploy_barriers[zid] = 1
            barriers -= 1

    for zone in sorted_zones:
        zid = zone["id"]
        can_evac = zone["population"] - zone["sheltered"]
        if can_evac > 0:
            evacuate[zid] = min(60, can_evac)

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

            action_str = json.dumps({
                "rescue":   action.allocate_rescue,
                "food":     action.send_food,
                "medical":  action.send_medical,
                "heli":     action.deploy_helicopters,
                "barriers": action.deploy_barriers,
                "evac":     action.evacuate,
            }, separators=(",", ":"))

            result      = await env.step(action)
            reward      = result["reward"]
            done        = result["done"]
            error_msg   = result["info"].get("error", None)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        final_state = result["observation"].model_dump()
        score   = _clamp(grade(final_state))
        success = score >= 0.5

    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        if hasattr(env, "close"):
            await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return _clamp(score)


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


if __name__ == "__main__":
    asyncio.run(main())