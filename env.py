import random
from models import Observation, Action, Reward

class DisasterEnv:
    def __init__(self, task):
        self.task = task
        self.state_data = None
        self.done = False

    async def reset(self):
        random.seed(42)
        self.state_data = self.task.initial_state()
        self.done = False

        return {
            "observation": Observation(**self.state_data),
            "reward": 0.0,
            "done": False,
            "info": {}
        }

    async def step(self, action: Action):
        if self.done:
            return {
                "observation": Observation(**self.state_data),
                "reward": 0.0,
                "done": True,
                "info": {}
            }

        reward = 0.0

        for zone in self.state_data["zones"]:
            zid = zone["id"]

            if zid in action.allocate_rescue:
                teams = action.allocate_rescue[zid]
                rescued = min(teams * 10, zone["injured"])
                zone["injured"] -= rescued
                reward += rescued * 0.02

            if zid in action.send_food:
                reward += action.send_food[zid] * 0.005

            if zid in action.send_medical:
                healed = min(action.send_medical[zid] * 2, zone["injured"])
                zone["injured"] -= healed
                reward += healed * 0.01

        # dynamic events
        for zone in self.state_data["zones"]:
            if random.random() < 0.3:
                zone["flood_level"] += 1
            if random.random() < 0.2:
                zone["access"] = "road_blocked"

        # penalties
        for zone in self.state_data["zones"]:
            if zone["flood_level"] > 8:
                reward -= 0.2
            if zone["injured"] > 100:
                reward -= 0.3

        self.state_data["time_step"] += 1

        if self.state_data["time_step"] >= self.task.max_steps:
            self.done = True

        return {
            "observation": Observation(**self.state_data),
            "reward": reward,
            "done": self.done,
            "info": {}
        }

    async def state(self):
        return Observation(**self.state_data)