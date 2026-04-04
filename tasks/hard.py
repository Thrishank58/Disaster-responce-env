def initial_state():
    return {
        "zones": [
            {
                "id": "A",
                "population": 3000,
                "flood_level": 8,
                "injured": 150,
                "access": "open"
            },
            {
                "id": "B",
                "population": 2500,
                "flood_level": 9,
                "injured": 200,
                "access": "road_blocked"
            },
            {
                "id": "C",
                "population": 1800,
                "flood_level": 7,
                "injured": 120,
                "access": "open"
            }
        ],
        "resources": {
            "rescue_teams": 5,
            "food_units": 80,
            "medical_kits": 50
        },
        "time_step": 0
    }

max_steps = 20