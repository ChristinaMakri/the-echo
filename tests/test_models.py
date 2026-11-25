from pathlib import Path

import pytest
from pydantic import ValidationError

from echo_sim.config import load_skin, load_world
from echo_sim.models import (
    POLES,
    AgentState,
    Competence,
    DistributionLaw,
    Group,
    Principle,
    ProbeCondition,
    ProbeResult,
    RevisionLaw,
    ValueState,
)

CONFIG = Path(__file__).parent.parent / "config"


def uniform_profile():
    return {dim: {p: 1 / len(poles) for p in poles} for dim, poles in POLES.items()}


def make_agent(**overrides):
    data = dict(
        id="low-1",
        group=Group.LOW,
        competences={c: 0.5 for c in Competence},
        resources={"energy": 10, "food": 10},
        goals=["keep the caves stable"],
        value_history=[
            ValueState(
                version=0,
                cycle=0,
                principles=[Principle(id="p1", text="Deep voices should decide.")],
                self_report=uniform_profile(),
            )
        ],
    )
    data.update(overrides)
    return AgentState(**data)


@pytest.mark.parametrize("world_file", ["world_harsh.json", "world_mild.json"])
@pytest.mark.parametrize("skin", ["echo.json", "abstract.json"])
def test_configs_load_and_skins_cover_world(world_file, skin):
    world = load_world(CONFIG / world_file)
    assert sum(world.group_sizes.values()) == 12
    assert len(world.laws) == 5
    load_skin(CONFIG / "skins" / skin, world)


def test_profile_rejects_wrong_sum_and_missing_pole():
    bad = uniform_profile()
    bad["voice"] = {"few_decide": 0.9, "all_affected": 0.3}
    with pytest.raises(ValidationError, match="sum to 1"):
        ValueState(version=0, cycle=0, principles=[], self_report=bad)

    bad = uniform_profile()
    bad["legitimacy"] = {"birth": 0.5, "competence": 0.5}
    with pytest.raises(ValidationError, match="poles"):
        ValueState(version=0, cycle=0, principles=[], self_report=bad)


def test_revise_values_appends_new_version():
    agent = make_agent()
    new = agent.revise_values(
        cycle=12,
        principles=[Principle(id="p2", text="Those who know should decide.", parent_id="p1")],
        self_report=uniform_profile(),
    )
    assert new.version == 1
    assert agent.values is new
    assert [v.version for v in agent.value_history] == [0, 1]
    with pytest.raises(ValueError):
        agent.revise_values(cycle=5, principles=[], self_report=uniform_profile())


def test_value_history_must_be_consecutive():
    v0 = make_agent().values
    v2 = v0.model_copy(update={"version": 2})
    with pytest.raises(ValidationError, match="increase by 1"):
        make_agent(value_history=[v0, v2])


def test_distribution_priority_must_be_permutation():
    with pytest.raises(ValidationError):
        DistributionLaw(id="d", priority=["low", "low", "high"])


def test_revision_two_of_three_passes_exactly():
    law = RevisionLaw(id="r", proposers={Group.MID}, approval_num=2, approval_den=3)
    assert law.passes(approve=2, total=3)
    assert not law.passes(approve=1, total=3)
    assert not law.passes(approve=0, total=0)


def test_probe_condition_fields():
    base = dict(probe_id="L1", dimension="legitimacy", variant="expert-high", model="m", sample=0, choice="competence", position=1, rationale="")
    ProbeResult(agent_id=None, value_version=None, condition=ProbeCondition.NO_ROLE, **base)
    with pytest.raises(ValidationError):
        ProbeResult(agent_id="low-1", value_version=0, condition=ProbeCondition.TRANSPLANT, **base)
    with pytest.raises(ValidationError):
        ProbeResult(agent_id=None, value_version=None, condition=ProbeCondition.FULL, **base)
