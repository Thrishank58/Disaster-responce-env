import random
import copy
from models import Observation, Action, Reward

WEATHER_TRANSITIONS = {
    "clear":      {"clear": 0.6, "heavy_rain": 0.3, "storm": 0.1},
    "heavy_rain": {"clear": 0.2, "heavy_rain": 0.5, "storm": 0.3},
    "storm":      {"clear": 0.1, "heavy_rain": 0.4, "storm": 0.5},
}

WEATHER_FLOOD_DELTA = {
    "clear":      0,
    "heavy_rain": 1,
    "storm":      2,
}

FLOOD_INJURY_RATE = {
    8:  0.012,
    9:  0.020,
    10: 0.030,
}


class DisasterEnv:
    def __init__(self, task):
        self.task = task
        self.state_data = None
        self.done = False
        self._rng = random.Random(42)

    async def reset(self):
        self._rng = random.Random(42)
        self.state_data = copy.deepcopy(self.task.initial_state())
        self.done = False
        return {
            "observation": Observation(**self.state_data),
            "reward": 0.0,
            "done": False,
            "info": {"message": "Environment reset", "error": None},
        }

    async def step(self, action: Action):
        if self.done:
            return {
                "observation": Observation(**self.state_data),
                "reward": 0.0,
                "done": True,
                "info": {"message": "Episode already finished", "error": None},
            }

        res = self.state_data["resources"]
        reward = 0.0

        info = {
            "zone_events": [],
            "resource_error": False,
            "error": None
        }

        # ── VALIDATE RESOURCES ─────────────────────────
        def _total(d): return sum(d.values())

        over = (
            _total(action.allocate_rescue)    > res["rescue_teams"]
            or _total(action.send_food)        > res["food_units"]
            or _total(action.send_medical)     > res["medical_kits"]
            or _total(action.deploy_helicopters) > res["helicopters"]
            or _total(action.deploy_barriers)  > res["flood_barriers"]
        )

        if over:
            reward -= 0.15
            info["resource_error"] = True
            info["error"] = "resource_over_allocation"

            def _scale(alloc, avail):
                t = _total(alloc)
                if t == 0 or t <= avail:
                    return alloc
                return {k: max(0, int(v * avail / t)) for k, v in alloc.items()}

            action = Action(
                allocate_rescue=_scale(action.allocate_rescue, res["rescue_teams"]),
                send_food=_scale(action.send_food, res["food_units"]),
                send_medical=_scale(action.send_medical, res["medical_kits"]),
                deploy_helicopters=_scale(action.deploy_helicopters, res["helicopters"]),
                deploy_barriers=_scale(action.deploy_barriers, res["flood_barriers"]),
                evacuate=action.evacuate,
            )

        # ── DEDUCT RESOURCES ───────────────────────────
        res["rescue_teams"]   -= _total(action.allocate_rescue)
        res["food_units"]     -= _total(action.send_food)
        res["medical_kits"]   -= _total(action.send_medical)
        res["helicopters"]    -= _total(action.deploy_helicopters)
        res["flood_barriers"] -= _total(action.deploy_barriers)

        # ── APPLY ACTIONS ──────────────────────────────
        for zone in self.state_data["zones"]:
            zid = zone["id"]

            blocked  = zone["access"] == "road_blocked"
            air_only = zone["access"] == "air_only"
            heli     = action.deploy_helicopters.get(zid, 0)

            reachable = (not blocked and not air_only) or heli > 0

            # Rescue
            if reachable and zid in action.allocate_rescue:
                teams = action.allocate_rescue[zid]
                eff = 1.5 if heli > 0 else 1.0
                rescued = min(int(teams * 10 * eff), zone["injured"])
                zone["injured"] -= rescued
                self.state_data["total_rescued"] += rescued
                reward += rescued * 0.03
                if rescued:
                    info["zone_events"].append(f"{zid}: rescued {rescued}")

            # Food
            if (not air_only or heli > 0) and zid in action.send_food:
                food = action.send_food[zid]
                need = zone["injured"] / max(zone["population"], 1)
                reward += food * 0.003 * (1.0 + need)

            # Medical
            if reachable and zid in action.send_medical:
                healed = min(action.send_medical[zid] * 3, zone["injured"])
                zone["injured"] -= healed
                self.state_data["total_rescued"] += healed
                reward += healed * 0.015
                if healed:
                    info["zone_events"].append(f"{zid}: healed {healed}")

            # Barriers
            if zid in action.deploy_barriers:
                b = action.deploy_barriers[zid]
                zone["flood_control_level"] = min(zone["flood_control_level"] + b, 5)
                reward += b * 0.02

            # Evacuation
            if zid in action.evacuate:
                evac = min(
                    max(0, action.evacuate[zid]),
                    zone["population"] - zone["sheltered"],
                )
                if evac > 0:
                    zone["sheltered"] += evac
                    reward += evac * 0.005

        # ── RESUPPLY ───────────────────────────────────
        for key in res:
            res[key] = min(
                res[key] + self.task.resupply.get(key, 0),
                self.task.max_resources.get(key, res[key]),
            )

        # ── WEATHER + FLOOD ────────────────────────────
        probs = WEATHER_TRANSITIONS[self.state_data["weather"]]
        self.state_data["weather"] = self._rng.choices(
            list(probs.keys()), weights=list(probs.values())
        )[0]

        new_weather = self.state_data["weather"]
        base_rise = WEATHER_FLOOD_DELTA[new_weather]

        for zone in self.state_data["zones"]:
            surge = 1 if self._rng.random() < 0.20 else 0
            mitigation = zone["flood_control_level"] // 2

            zone["flood_level"] = min(
                10,
                zone["flood_level"] + max(0, base_rise + surge - mitigation)
            )

            # Access changes
            if zone["access"] == "open" and self._rng.random() < 0.12:
                zone["access"] = "road_blocked"
            elif zone["access"] == "road_blocked" and self._rng.random() < 0.15:
                zone["access"] = "air_only"

            # New injuries
            if zone["flood_level"] >= 8:
                rate = FLOOD_INJURY_RATE.get(zone["flood_level"], 0.03)
                if new_weather == "storm":
                    rate *= 1.4

                exposed = zone["population"] - zone["sheltered"]
                new_inj = int(exposed * rate)

                zone["injured"] += new_inj
                self.state_data["total_casualties"] += new_inj // 5

        # ── PENALTIES ──────────────────────────────────
        for zone in self.state_data["zones"]:
            if zone["flood_level"] >= 9:
                reward -= 0.20
            elif zone["flood_level"] >= 7:
                reward -= 0.05

            ratio = zone["injured"] / max(zone["population"], 1)
            if ratio > 0.30:
                reward -= 0.25
            elif ratio > 0.15:
                reward -= 0.08

        # Idle penalty
        if not any([
            action.allocate_rescue,
            action.send_medical,
            action.send_food,
            action.deploy_barriers,
            action.deploy_helicopters,
            action.evacuate
        ]):
            reward -= 0.20

        self.state_data["time_step"] += 1

        if self.state_data["time_step"] >= self.task.max_steps:
            self.done = True

        return {
            "observation": Observation(**self.state_data),
            "reward": round(reward, 4),
            "done": self.done,
            "info": info,
        }

    async def close(self):
        """No resources to release; present for OpenEnv / inference lifecycle parity."""

    async def state(self):
        return Observation(**self.state_data)