---
title: Disaster Response Env
emoji: 🚑
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---
# 🌊 Disaster Response Coordinator Agent

## 🧠 Overview
This project simulates a **real-world disaster response system** where an AI agent must manage limited resources during a flood emergency.

The agent allocates:
- 🚑 Rescue teams  
- 🍞 Food supplies  
- 💊 Medical aid  

across multiple zones while conditions evolve dynamically (rising water levels, injuries, and access constraints).

---

## 🌍 Real-World Relevance
Disaster response is a **high-stakes decision-making problem** involving:
- Limited resources  
- Incomplete information  
- Rapidly changing conditions  

This environment models realistic trade-offs faced by:
- Government disaster agencies  
- NGOs  
- Emergency response teams  

---

## ⚙️ Environment Design

### 🧾 Observation Space
Each timestep provides:
- Zone-wise population  
- Number of injured people  
- Flood level severity  
- Accessibility (e.g., road blocked)  

---

### 🎯 Action Space
The agent decides:
- Allocation of rescue teams  
- Distribution of food supplies  
- Distribution of medical resources  

for each zone.

---

### 🔁 Environment API
The environment follows the **OpenEnv specification**:

- `reset()` → initializes environment  
- `step(action)` → advances simulation  
- `state()` → returns current state  

---

## 🧪 Tasks

### 🟢 Easy
- Single-zone environment  
- Basic allocation decisions  

### 🟡 Medium
- Multiple zones  
- Resource balancing required  

### 🔴 Hard
- Multiple zones with blocked access  
- Higher uncertainty and trade-offs  

---

## 📊 Reward Function

The reward function provides **continuous feedback**:

- ✅ Rewards reducing injuries  
- ✅ Rewards balanced resource allocation  
- ❌ Penalizes neglect and worsening conditions  

---

## 📏 Evaluation (Grader)

Score range: **0.0 – 1.0**

Based on:
- 🧍 Survival rate (minimizing injuries)  
- ⚖️ Fairness across zones  

The grader is:
- Deterministic  
- State-dependent  
- Non-trivial  

---

## 🤖 Baseline Agent

Includes a baseline agent that:
- Uses OpenAI API (if available)  
- Falls back to a rule-based strategy  
- Adapts resource allocation based on zone conditions  

---

## 🐳 Docker Support

Build and run:

```bash
docker build -t disaster-env .
docker run disaster-env
````

---

## 🚀 Running Locally

```bash
python inference.py
```

---

## 🌐 API Server

Start server:

```bash
python -m server.app
```

Open in browser:

```
http://localhost:7860
```

---

## 📦 Project Structure

```
disaster-env/
│
├── env.py
├── models.py
├── grader.py
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── uv.lock
│
├── tasks/
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
│
├── server/
│   └── app.py
```

---

## 🧠 Key Features

* Real-world disaster simulation
* Multi-step decision making
* Dynamic environment conditions
* Multi-task evaluation
* Deterministic grading system
* OpenEnv compliant

---

## 🏁 Baseline Performance

Typical scores:

* Easy: ~0.95
* Medium: ~0.95
* Hard: ~0.90–0.95

---

## 🔥 Why This Project Stands Out

* Models a **critical real-world problem**
* Includes **meaningful reward shaping**
* Evaluates agents under **uncertainty and constraints**
* Designed for **training and benchmarking AI agents**

---

## 👨‍💻 Author

**Thrishank**

---

```

