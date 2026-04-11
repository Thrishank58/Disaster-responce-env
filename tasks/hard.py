"""
Hard Task: Three zones under active storm. Zone B road-blocked, Zone C air-only.
Only 2 helicopters for 2 inaccessible zones. Agent must triage ruthlessly.
Flood levels start at critical. Resources extremely scarce.
Rule-based baseline: ~0.20–0.30
Strong agent target:  ~0.50–0.65
"""

def initial_state():
    return {
        "zones": [
            {
                "id": "A",
                "population": 3000,
                "flood_level": 7,
                "injured": 300,
                "access": "open",
                "sheltered": 0,
                "flood_control_level": 0,
            },
            {
                "id": "B",
                "population": 2000,
                "flood_level": 8,
                "injured": 400,
                "access": "road_blocked",
                "sheltered": 0,
                "flood_control_level": 0,
            },
            {
                "id": "C",
                "population": 1200,
                "flood_level": 9,
                "injured": 350,
                "access": "air_only",
                "sheltered": 0,
                "flood_control_level": 0,
            },
        ],
        "resources": {
            "rescue_teams": 4,
            "food_units": 40,
            "medical_kits": 20,
            "helicopters": 2,
            "flood_barriers": 3,
        },
        "time_step": 0,
        "weather": "storm",
        "total_rescued": 0,
        "total_casualties": 0,
    }

max_steps = 20

resupply = {
    "rescue_teams": 1,
    "food_units": 3,
    "medical_kits": 2,
    "helicopters": 0,
    "flood_barriers": 0,
}

max_resources = {
    "rescue_teams": 6,
    "food_units": 60,
    "medical_kits": 30,
    "helicopters": 2,
    "flood_barriers": 4,
}