import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.engine import Simulation
from echo_sim.policies import RuleBasedPolicy
from echo_sim.storage import write_run

CONFIG = Path(__file__).parent.parent / "config"


def test_write_run_roundtrip(tmp_path):
    world = load_world(CONFIG / "world_harsh.json").model_copy(update={"cycles": 10})
    skin = load_skin(CONFIG / "skins" / "echo.json", world)
    sim = Simulation(world, create_agents(world, skin, 0), RuleBasedPolicy(), seed=0)
    sim.run()

    path = write_run(sim, tmp_path, "test-run", meta={"skin": skin.id})

    meta = json.loads((path / "meta.json").read_text())
    assert meta["seed"] == 0 and meta["skin"] == "echo" and len(meta["final_laws"]) == 5
    assert pq.read_table(path / "events.parquet").num_rows == 10
    assert pq.read_table(path / "cycles.parquet").num_rows == 10
    assert pq.read_table(path / "agent_cycles.parquet").num_rows == 10 * len(sim.agents)
    assert pq.read_table(path / "decisions.parquet").num_rows == len(sim.decisions)
    values = pq.read_table(path / "values.parquet")
    assert values.num_rows == len(sim.agents)  # rule-based policy never revises
    assert "legitimacy.birth" in values.column_names

    with pytest.raises(FileExistsError):
        write_run(sim, tmp_path, "test-run", meta={})


def test_runs_share_schema(tmp_path):
    world = load_world(CONFIG / "world_harsh.json").model_copy(update={"cycles": 5})
    skin = load_skin(CONFIG / "skins" / "echo.json", world)
    files = []
    for seed in range(3):
        sim = Simulation(world, create_agents(world, skin, seed), RuleBasedPolicy(), seed=seed)
        sim.run()
        files.append(write_run(sim, tmp_path, f"r{seed}", meta={}) / "cycles.parquet")
    assert len({pq.read_schema(f) for f in files}) == 1
    assert pq.ParquetDataset(files).read().num_rows == 15


def test_checkpoint_overwrites_only_when_asked(tmp_path):
    world = load_world(CONFIG / "world_harsh.json").model_copy(update={"cycles": 3})
    skin = load_skin(CONFIG / "skins" / "echo.json", world)
    sim = Simulation(world, create_agents(world, skin, 0), RuleBasedPolicy(), seed=0)
    for n in range(1, 4):
        sim.step()
        path = write_run(sim, tmp_path, "r.partial", meta={}, overwrite=True)
        assert pq.read_table(path / "cycles.parquet").num_rows == n
