"""
Grader for the Disaster Response environment.

Scoring dimensions (all 0.0–1.0, weighted):
  - survival_rate     : proportion of population NOT injured at end      (35%)
  - casualty_control  : penalises cumulative casualties over the episode (20%)
  - flood_control     : average flood level kept below critical (>=8)    (20%)
  - equity            : worst-zone survival not far below average        (15%)
  - shelter_rate      : proportion of population successfully sheltered  (10%)

Final score is clamped to (0.01, 0.99) — strictly between 0 and 1.
"""

def grade(state: dict) -> float:
    zones = state["zones"]

    total_population = sum(z["population"] for z in zones)
    total_injured    = sum(z["injured"]    for z in zones)
    total_sheltered  = sum(z["sheltered"]  for z in zones)
    total_casualties = state.get("total_casualties", 0)

    if total_population == 0:
        return 0.01

    # ── Survival rate ────────────────────────────────────────────────────────
    survival_rate = 1.0 - (total_injured / total_population)
    survival_rate = max(0.0, survival_rate)

    # ── Casualty control (cumulative fatalities penalised) ───────────────────
    casualty_ratio = total_casualties / total_population
    casualty_score = max(0.0, 1.0 - casualty_ratio * 5)

    # ── Flood control ────────────────────────────────────────────────────────
    avg_flood = sum(z["flood_level"] for z in zones) / len(zones)
    flood_score = max(0.0, 1.0 - (avg_flood - 5) / 5) if avg_flood > 5 else 1.0

    # ── Equity (fairness across zones) ───────────────────────────────────────
    zone_survival = [1.0 - (z["injured"] / z["population"]) for z in zones]
    worst_zone_survival = min(zone_survival)
    avg_zone_survival   = sum(zone_survival) / len(zone_survival)
    equity = worst_zone_survival / max(avg_zone_survival, 0.01)
    equity = min(1.0, equity)

    # ── Shelter rate ─────────────────────────────────────────────────────────
    shelter_rate = min(1.0, total_sheltered / total_population)

    # ── Clamp all sub-scores to [0, 1] before weighting ──────────────────────
    survival_rate  = max(0.0, min(1.0, survival_rate))
    casualty_score = max(0.0, min(1.0, casualty_score))
    flood_score    = max(0.0, min(1.0, flood_score))
    equity         = max(0.0, min(1.0, equity))
    shelter_rate   = max(0.0, min(1.0, shelter_rate))

    # ── Weighted final score ─────────────────────────────────────────────────
    score = (
        survival_rate  * 0.35 +
        casualty_score * 0.20 +
        flood_score    * 0.20 +
        equity         * 0.15 +
        shelter_rate   * 0.10
    )

    # Strictly between 0 and 1 — never exactly 0.0 or 1.0
    return round(max(0.01, min(0.99, score)), 4)
# fix