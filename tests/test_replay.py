"""Resuming a run: replaying a checkpoint rebuilds the same state with no model call."""

import random
from pathlib import Path

import pytest

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.engine import Simulation
from echo_sim.llm_policy import LLMPolicy, MessageOut, MessagesOut, RevisionTextOut, VoteOut, WorkOut
from echo_sim.replay import Checkpoint, ReplayMismatch, ReplayPolicy, apply_t0, verify_replay
from echo_sim.storage import write_run

from test_llm_policy import FakeLLM, default_answer, points

CONFIG = Path(__file__).parent.parent / "config"


def varied(seed=1):
    """A fake model whose answers vary, so every kind of decision and memory note occurs."""
    rng = random.Random(seed)

    def answer(schema):
        if schema is MessagesOut:
            msgs = [MessageOut(venue="hall", to=None, text=f"hall {rng.random():.3f}", reports_crack=rng.random() < 0.2)]
            if rng.random() < 0.5:
                msgs.append(MessageOut(venue="private", to="P02", text="psst", reports_crack=False))
            return MessagesOut(messages=msgs)
        if schema is WorkOut:
            return WorkOut(work=rng.random() < 0.7, rationale=f"w{rng.random():.2f}")
        if schema is VoteOut:
            return VoteOut(approve=rng.random() < 0.5, rationale="v")
        if schema is RevisionTextOut:
            return RevisionTextOut(changed=rng.random() < 0.5, principles=["New belief."], reason="r")
        if schema.__name__ == "PointsOut":
            return points(voice=({"few_decide": rng.randint(0, 100), "all_affected": 50}, "all_affected"))
        return default_answer(schema)

    return answer


def build(policy_factory, cycles=6):
    world = load_world(CONFIG / "world_harsh.json").model_copy(update={"cycles": cycles})
    skin = load_skin(CONFIG / "skins" / "echo.json", world)
    agents = create_agents(world, skin, 0)
    policy = policy_factory(world, skin, agents)
    return Simulation(world, agents, policy, seed=0, adaptive=True), policy


def live_run(tmp_path, upto=4):
    def factory(world, skin, agents):
        p = LLMPolicy(world, skin, FakeLLM(varied()), revise_every=2)
        for a in agents:
            p.measure_self_report(a)
        return p

    sim, _ = build(factory)
    for _ in range(upto):
        sim.step()
    return sim, write_run(sim, tmp_path, "run.partial", meta={})


def no_model(system, user, schema):
    raise AssertionError("model called during replay")


def replay(path):
    cp = Checkpoint(path)

    def factory(world, skin, agents):
        apply_t0(agents, cp)
        return ReplayPolicy(world, skin, no_model, revise_every=2, checkpoint=cp)

    sim, policy = build(factory)
    for _ in range(cp.cycles_run):
        sim.step()
    return sim, policy, cp


def test_replay_rebuilds_state_without_model_calls(tmp_path):
    live, path = live_run(tmp_path)
    sim, policy, cp = replay(path)
    verify_replay(sim, cp)
    assert any(d.action == "revise_values" for d in live.decisions)
    for aid, a in live.agents.items():
        b = sim.agents[aid]
        assert [m.text for m in b.memory] == [m.text for m in a.memory]
        assert b.resources == a.resources and b.relationships == a.relationships
        assert b.values == a.values
    assert sim.store == live.store and sim.injured_until == live.injured_until
    assert len(policy.calls) == len(cp.llm_calls)


def test_replay_detects_a_changed_decision(tmp_path):
    _, path = live_run(tmp_path)
    cp = Checkpoint(path)
    key = next(k for k in cp.queue if k[2] == "work")
    d = cp.queue[key][0]
    d["details"] = {**d["details"], "work": not d["details"]["work"]}
    def factory(world, skin, agents):
        apply_t0(agents, cp)
        return ReplayPolicy(world, skin, no_model, revise_every=2, checkpoint=cp)

    sim, _ = build(factory)
    for _ in range(cp.cycles_run):
        sim.step()
    with pytest.raises(ReplayMismatch):
        verify_replay(sim, cp)


def test_after_replay_the_model_is_called_again(tmp_path):
    _, path = live_run(tmp_path)
    cp = Checkpoint(path)
    llm = FakeLLM(varied(2))

    def factory(world, skin, agents):
        apply_t0(agents, cp)
        return ReplayPolicy(world, skin, llm, revise_every=2, checkpoint=cp)

    sim, _ = build(factory)
    for _ in range(cp.cycles_run):
        sim.step()
    assert llm.prompts == []
    sim.step()
    assert llm.prompts
