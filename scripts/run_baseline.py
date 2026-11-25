"""Run the rule-based (non-LLM) baseline and store each run under runs/.

    uv run python scripts/run_baseline.py [world_file] [skin_file] [n_seeds]
"""

import sys
from pathlib import Path

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.engine import Simulation
from echo_sim.policies import RuleBasedPolicy
from echo_sim.storage import write_run

ROOT = Path(__file__).parent.parent


def main():
    world_file = sys.argv[1] if len(sys.argv) > 1 else "world_harsh.json"
    skin_file = sys.argv[2] if len(sys.argv) > 2 else "echo.json"
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    world = load_world(ROOT / "config" / world_file)
    skin = load_skin(ROOT / "config" / "skins" / skin_file, world)

    for seed in range(n_seeds):
        sim = Simulation(world, create_agents(world, skin, seed), RuleBasedPolicy(), seed=seed)
        sim.run()
        run_id = f"{Path(world_file).stem}-{skin.id}-A-rule-s{seed:03d}"
        path = write_run(
            sim,
            ROOT / "runs",
            run_id,
            meta={"world_file": world_file, "skin": skin.id, "condition": "A", "policy": "rule", "model": None},
        )
        print(path)


if __name__ == "__main__":
    main()
