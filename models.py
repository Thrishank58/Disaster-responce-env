from pydantic import BaseModel
from typing import Dict, List, Optional

class Zone(BaseModel):
    id: str
    population: int
    flood_level: int        # 0–10; >= 9 causes rapid casualties
    injured: int
    access: str             # "open" | "road_blocked" | "air_only"
    sheltered: int          # people moved to safe shelter
    flood_control_level: int  # sandbag/barrier investment 0–5

class Resources(BaseModel):
    rescue_teams: int
    food_units: int
    medical_kits: int
    helicopters: int        # required for air_only zones
    flood_barriers: int     # deployed to reduce flood_level rise

class Observation(BaseModel):
    zones: List[Zone]
    resources: Resources
    time_step: int
    weather: str            # "clear" | "storm" | "heavy_rain"
    total_rescued: int
    total_casualties: int

class Action(BaseModel):
    allocate_rescue: Dict[str, int] = {}      # zone_id -> rescue teams
    send_food: Dict[str, int] = {}             # zone_id -> food units
    send_medical: Dict[str, int] = {}          # zone_id -> medical kits
    deploy_helicopters: Dict[str, int] = {}    # zone_id -> helicopters (for blocked zones)
    deploy_barriers: Dict[str, int] = {}       # zone_id -> flood barriers
    evacuate: Dict[str, int] = {}              # zone_id -> num people to shelter

class Reward(BaseModel):
    value: float