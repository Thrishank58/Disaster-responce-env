"""
Medium Task: Two zones — Zone B road-blocked from start, heavier injuries,
tighter resources. Agent must use helicopters strategically for zone B.
Rule-based baseline: ~0.35–0.45
Strong agent target:  ~0.60–0.75
"""

def initial_state():
    return {
        "zones": [
            {
                "id": "A",
                "population": 2000,
                "flood_level": 6,
                "injured": 200,
                "access": "open",
                "sheltered": 0,
                "flood_control_level": 0,
            },
            {
                "id": "B",
                "population": 1500,
                "flood_level": 8,
                "injured": 300,
                "access": "road_blocked",
                "sheltered": 0,
                "flood_control_level": 0,
            },
        ],
        "resources": {
            "rescue_teams": 4,
            "food_units": 50,
            "medical_kits": 25,
            "helicopters": 2,
            "flood_barriers": 3,
        },
        "time_step": 0,
        "weather": "heavy_rain",
        "total_rescued": 0,
        "total_casualties": 0,
    }

max_steps = 15

resupply = {
    "rescue_teams": 1,
    "food_units": 4,
    "medical_kits": 2,
    "helicopters": 0,
    "flood_barriers": 1,
}

max_resources = {
    "rescue_teams": 6,
    "food_units": 70,
    "medical_kits": 40,
    "helicopters": 2,
    "flood_barriers": 5,
}