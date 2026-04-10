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
This project simulates a real-world disaster response system where an AI agent must manage limited resources during a flood emergency.

The agent allocates:
- 🚑 Rescue teams
- 🍞 Food supplies
- 💊 Medical aid

across multiple zones while conditions evolve dynamically (rising water levels, injuries, and access constraints).

## ⚙️ Environment Design

### Observation Space
- Zone-wise population
- Number of injured people
- Flood level severity
- Accessibility (e.g., road blocked)

### Action Space
- Allocation of rescue teams
- Distribution of food supplies
- Distribution of medical resources

## 🔁 Environment API
- `reset()` → initializes environment
- `step(action)` → advances simulation
- `state()` → returns current state

## 🧪 Tasks
- 🟢 Easy: Single zone, basic allocation
- 🟡 Medium: Multiple zones, resource balancing
- 🔴 Hard: Multiple zones with blocked access

## 📊 Baseline Performance
- Easy: ~0.95
- Medium: ~0.95
- Hard: ~0.90–0.95

## 👨‍💻 Author
Thrishank