---
title: Disaster Response Coordinator Env
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
tags:
  - openenv
---

# 🌊 Disaster Response Coordinator Agent

## Overview

A flood disaster is unfolding across multiple zones. You are the AI coordinator.
You have limited rescue teams, medical kits, food supplies, helicopters, and flood barriers.
Roads wash out. Floods intensify. Storms arrive without warning.

Your decisions directly determine how many people survive.

This is a real operational problem — disaster response coordinators face exactly
this resource-allocation-under-uncertainty challenge in real emergencies. The
environment models the core trade-offs: triage vs. fairness, flood prevention vs.
immediate rescue, and helicopter deployment for cut-off zones.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `zones[].id` | str | Zone identifier |
| `zones[].population` | int | Total residents |
| `zones[].injured` | int | Currently injured people |
| `zones[].flood_level` | int (0–10) | Flood severity; ≥8 causes rapid new casualties |
| `zones[].access` | str | `open` / `road_blocked` / `air_only` |
| `zones[].sheltered` | int | People moved to safe shelter |
| `zones[].flood_control_level` | int (0–5) | Deployed barrier strength |
| `resources.rescue_teams` | int | Available rescue teams |
| `resources.food_units` | int | Available food units |
| `resources.medical_kits` | int | Available medical kits |
| `resources.helicopters` | int | Helicopters (required for blocked zones) |
| `resources.flood_barriers` | int | Deployable flood barriers |
| `weather` | str | `clear` / `heavy_rain` / `storm` |
| `time_step` | int | Current step in episode |
| `total_rescued` | int | Cumulative rescues this episode |
| `total_casualties` | int | Cumulative fatalities this episode |

---

## Action Space

All fields are `Dict[zone_id, int]`. Total allocations across zones must not exceed available resources.

| Field | Effect |
|---|---|
| `allocate_rescue` | Each team rescues up to 10 injured (1.5× if helicopter co-deployed) |
| `send_food` | Provides survival support; bonus scales with injury ratio |
| `send_medical` | Each kit heals up to 3 injured |
| `deploy_helicopters` | **Required** to reach `road_blocked` or `air_only` zones; also boosts rescue efficiency |
| `deploy_barriers` | Increases `flood_control_level`, reducing future flood rise |
| `evacuate` | Moves civilians to shelter, reducing future injury exposure |

**Key constraint:** `road_blocked` and `air_only` zones receive no rescue, medical, or food unless helicopters are also sent there.

---

## Reward Function

Rewards are shaped across the full trajectory (not just end-of-episode):

- **+0.03 per person rescued** via rescue teams
- **+0.015 per person healed** via medical kits
- **+0.003–0.006 per food unit** (scaled by zone need)
- **+0.02 per flood barrier deployed**
- **+0.005 per person evacuated**
- **−0.05 to −0.25** per step for high flood levels (≥7 and ≥9)
- **−0.1 to −0.3** per step for high injury ratios (>15% and >30%)
- **−0.15** for over-allocating resources (exceeding available stock)
- **−0.2** for taking no action (idle penalty)

---

## Grader

Final episode score (0.0–1.0) combines:

| Component | Weight | Description |
|---|---|---|
| Survival rate | 35% | `1 - (total_injured / total_population)` |
| Casualty control | 20% | Penalises cumulative fatalities |
| Flood control | 20% | Average final flood level below critical |
| Equity | 15% | Worst-zone survival relative to average |
| Shelter rate | 10% | Proportion of population sheltered |

---

## Tasks

### 🟢 Easy
- **Zones:** 1 (open access)
- **Steps:** 10
- **Challenge:** Basic allocation, moderate flood
- **Baseline (rule-based):** ~0.50–0.60
- **Good agent target:** ~0.75–0.85

### 🟡 Medium
- **Zones:** 2 (one road-blocked)
- **Steps:** 15
- **Challenge:** Must use helicopters to reach blocked zone; heavier injuries
- **Baseline (rule-based):** ~0.35–0.45
- **Good agent target:** ~0.65–0.75

### 🔴 Hard
- **Zones:** 3 (one road-blocked, one air-only from start)
- **Steps:** 20
- **Challenge:** Active storm, only 2 helicopters for 2 blocked zones — agent must triage
- **Baseline (rule-based):** ~0.20–0.30
- **Good agent target:** ~0.55–0.65

---

## API

```
\POST /reset?task=easy|medium|hard   → initial observation
POST /step                          → next observation, reward, done, info
POST /grade                         → final episode score (0.01–0.99)
GET  /state                         → current observation
GET  /health                        → {"status": "ok"}
```

### Example: Reset and Step

```python
import requests, json

BASE = "http://localhost:7860"

obs = requests.post(f"{BASE}/reset?task=medium").json()

action = {
    "allocate_rescue":    {"A": 2, "B": 1},
    "send_food":          {"A": 20, "B": 15},
    "send_medical":       {"A": 10, "B": 8},
    "deploy_helicopters": {"B": 1},   # B is road_blocked — helicopter required!
    "deploy_barriers":    {"B": 1},
    "evacuate":           {"A": 50}
}

result = requests.post(f"{BASE}/step", json=action).json()
print(result["reward"], result["done"])
```

---

## Setup

### Docker

```bash
docker build -t disaster-response-env .
docker run -p 7860:7860 disaster-response-env
```

### Local (Python)

```bash
pip install fastapi uvicorn pydantic openai openenv-core
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=hf_...   # required — passed as OpenAI client api_key (see submission guidelines)

python inference.py
```

---

## Project Structure

```
.
├── Dockerfile
├── pyproject.toml
├── openenv.yaml
├── inference.py          # baseline inference script
├── env.py                # core DisasterEnv
├── models.py             # Observation, Action, Reward pydantic models
├── grader.py             # deterministic episode scorer
├── tasks/
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
└── server/
    └── app.py            # FastAPI server
```

---

## Author
Thrishank