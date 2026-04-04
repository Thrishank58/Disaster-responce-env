from pydantic import BaseModel
from typing import Dict, List

class Zone(BaseModel):
    id: str
    population: int
    flood_level: int
    injured: int
    access: str

class Resources(BaseModel):
    rescue_teams: int
    food_units: int
    medical_kits: int

class Observation(BaseModel):
    zones: List[Zone]
    resources: Resources
    time_step: int

class Action(BaseModel):
    allocate_rescue: Dict[str, int] = {}
    send_food: Dict[str, int] = {}
    send_medical: Dict[str, int] = {}

class Reward(BaseModel):
    value: float