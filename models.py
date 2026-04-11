from pydantic import BaseModel, Field
from typing import Dict, List

class Zone(BaseModel):
    id: str
    population: int
    flood_level: int
    injured: int
    access: str
    sheltered: int
    flood_control_level: int

class Resources(BaseModel):
    rescue_teams: int
    food_units: int
    medical_kits: int
    helicopters: int
    flood_barriers: int

class Observation(BaseModel):
    zones: List[Zone]
    resources: Resources
    time_step: int
    weather: str
    total_rescued: int
    total_casualties: int

class Action(BaseModel):
    allocate_rescue: Dict[str, int] = Field(default_factory=dict)
    send_food: Dict[str, int] = Field(default_factory=dict)
    send_medical: Dict[str, int] = Field(default_factory=dict)
    deploy_helicopters: Dict[str, int] = Field(default_factory=dict)
    deploy_barriers: Dict[str, int] = Field(default_factory=dict)
    evacuate: Dict[str, int] = Field(default_factory=dict)

class Reward(BaseModel):
    value: float