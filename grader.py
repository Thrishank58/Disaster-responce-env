def grade(state):
    zones = state["zones"]

    total_population = sum(z["population"] for z in zones)
    total_injured = sum(z["injured"] for z in zones)

    # 🧠 Survival score (main objective)
    survival = 1 - (total_injured / total_population)

    # 🧠 Fairness score (no zone ignored)
    fairness = min(
        1 - (z["injured"] / z["population"])
        for z in zones
    )

    # ⚖️ Balanced scoring
    score = (survival * 0.6) + (fairness * 0.4)

    # 🔥 Realism penalty (prevents perfect scores too easily)
    if total_injured < 10:
        score -= 0.05

    # 🔒 Clamp score between 0 and 1
    score = max(0.0, min(1.0, score))

    return score