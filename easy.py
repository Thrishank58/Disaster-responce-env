def initial_state():
    return {
        "zones": [
            {
                "id": "A",
                "population": 1000,
                "flood_level": 5,
                "injured": 50,
                "access": "open"
            }
        ],
        "resources": {
            "rescue_teams": 3,
            "food_units": 50,
            "medical_kits": 30
        },
        "time_step": 0
    }

max_steps = 10