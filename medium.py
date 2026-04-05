def initial_state():
    return {
        "zones": [
            {
                "id": "A",
                "population": 2000,
                "flood_level": 6,
                "injured": 80,
                "access": "open"
            },
            {
                "id": "B",
                "population": 1500,
                "flood_level": 7,
                "injured": 60,
                "access": "open"
            }
        ],
        "resources": {
            "rescue_teams": 4,
            "food_units": 70,
            "medical_kits": 40
        },
        "time_step": 0
    }

max_steps = 15