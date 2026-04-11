"""
Easy Task: Single zone, open access, moderate flood, medium injuries.
Rule-based baseline: ~0.50–0.60
Strong agent target:  ~0.75–0.85
"""

def initial_state():
    return {
        "zones": [
            {
                "id": "A",
                "population": 1000,
                "flood_level": 6,
                "injured": 150,
                "access": "open",
                "sheltered": 0,
                "flood_control_level": 0,
            }
        ],
        "resources": {
            "rescue_teams": 3,
            "food_units": 40,
            "medical_kits": 20,
            "helicopters": 1,
            "flood_barriers": 3,
        },
        "time_step": 0,
        "weather": "heavy_rain",
        "total_rescued": 0,
        "total_casualties": 0,
    }

max_steps = 10

resupply = {
    "rescue_teams": 1,
    "food_units": 5,
    "medical_kits": 3,
    "helicopters": 0,
    "flood_barriers": 1,
}

max_resources = {
    "rescue_teams": 5,
    "food_units": 60,
    "medical_kits": 30,
    "helicopters": 1,
    "flood_barriers": 5,
}