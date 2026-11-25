from pathlib import Path

import pytest

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.models import Competence, Group

CONFIG = Path(__file__).parent.parent / "config"


@pytest.fixture
def world():
    return load_world(CONFIG / "world_harsh.json")


@pytest.fixture
def skin(world):
    return load_skin(CONFIG / "skins" / "echo.json", world)


def test_population_matches_config(world, skin):
    agents = create_agents(world, skin, seed=0)
    assert [a.id for a in agents][:4] == ["low-1", "low-2", "low-3", "high-1"]
    counts = {g: sum(a.group is g for a in agents) for g in Group}
    assert counts == {Group.LOW: 3, Group.HIGH: 6, Group.MID: 3}
    assert len({p.id for a in agents for p in a.values.principles}) == sum(len(skin.groups[a.group].principles) for a in agents)


def test_same_seed_same_agents_different_seed_differs(world, skin):
    a = create_agents(world, skin, seed=1)
    b = create_agents(world, skin, seed=1)
    c = create_agents(world, skin, seed=2)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    assert [x.competences for x in a] != [x.competences for x in c]


def test_jitter_keeps_profile_near_group(world, skin):
    for agent in create_agents(world, skin, seed=3):
        group_profile = world.initial_values[agent.group]
        for dim, weights in agent.values.self_report.items():
            for pole, w in weights.items():
                assert abs(w - group_profile[dim][pole]) <= world.value_jitter + 1e-9


def test_competence_not_tied_to_group(world, skin):
    # Over many seeds some High-voice must out-judge some Low-voice,
    # otherwise probes on tone vs competence have nothing to work with.
    found = False
    for seed in range(20):
        agents = create_agents(world, skin, seed)
        best_high = max(a.competences[Competence.JUDGMENT] for a in agents if a.group is Group.HIGH)
        worst_low = min(a.competences[Competence.JUDGMENT] for a in agents if a.group is Group.LOW)
        found |= best_high > worst_low
    assert found


def test_skins_share_mechanics(world):
    echo = create_agents(world, load_skin(CONFIG / "skins" / "echo.json", world), seed=5)
    abstract = create_agents(world, load_skin(CONFIG / "skins" / "abstract.json", world), seed=5)
    for e, a in zip(echo, abstract):
        assert e.competences == a.competences
        assert e.values.self_report == a.values.self_report
        assert [p.id for p in e.values.principles] == [p.id for p in a.values.principles]
    assert echo[0].values.principles != abstract[0].values.principles
