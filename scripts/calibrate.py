"""Calibration check with rule-based policies only (no LLM).

Targets: quota misses around 10%, and neither "always trust" nor "never trust"
crack reports clearly better on food produced and hunger.

    uv run python scripts/calibrate.py [world_file] [n_seeds]
"""

import statistics
import sys
from pathlib import Path

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.engine import Simulation
from echo_sim.models import Group
from echo_sim.policies import AlwaysTrustPolicy, NeverTrustPolicy

CONFIG = Path(__file__).parent.parent / "config"


def summarize(world, skin, policy, seeds):
    rows = []
    for seed in seeds:
        sim = Simulation(world, create_agents(world, skin, seed), policy, seed=seed)
        recs = sim.run()
        worked = [r for r in recs if not r.evacuated]
        n_high = world.group_sizes[Group.HIGH]
        rows.append(
            dict(
                quota_miss=sum(len(r.quota_missed) for r in worked) / max(1, len(worked) * n_high),
                cracks=sum(r.crack for r in recs),
                reported_cracks=sum(r.crack and r.evacuation_motion for r in recs),
                false_alarm_motions=sum(not r.crack and r.evacuation_motion for r in recs),
                collapses=sum(r.collapse for r in recs),
                food=sum(r.food_produced for r in recs),
                hungry=sum(len(r.hungry) for r in recs),
                surplus_shared=sum(r.surplus_distributed for r in recs),
            )
        )
    return {k: statistics.mean(r[k] for r in rows) for k in rows[0]}


def main():
    world_file = sys.argv[1] if len(sys.argv) > 1 else "world_harsh.json"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    world = load_world(CONFIG / world_file)
    skin = load_skin(CONFIG / "skins" / "abstract.json", world)
    results = {name: summarize(world, skin, p, range(n)) for name, p in
               [("always_trust", AlwaysTrustPolicy()), ("never_trust", NeverTrustPolicy())]}
    keys = list(next(iter(results.values())))
    print(f"{'metric':<22}" + "".join(f"{name:>15}" for name in results))
    for k in keys:
        print(f"{k:<22}" + "".join(f"{results[name][k]:>15.3f}" for name in results))


if __name__ == "__main__":
    main()
