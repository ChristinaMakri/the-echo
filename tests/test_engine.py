from pathlib import Path

import pytest

from echo_sim.agents import create_agents
from echo_sim.config import load_skin, load_world
from echo_sim.engine import Simulation
from echo_sim.events import generate_events
from echo_sim.models import (
    POLES,
    Competence,
    Event,
    EventKind,
    Group,
    Message,
    Proposal,
    Resource,
    ValueRevision,
    Venue,
    Vote,
)
from echo_sim.policies import RuleBasedPolicy

CONFIG = Path(__file__).parent.parent / "config"


@pytest.fixture
def world():
    return load_world(CONFIG / "world_harsh.json")


@pytest.fixture
def agents(world):
    return create_agents(world, load_skin(CONFIG / "skins" / "echo.json", world), seed=0)


def sim_with_event(world, agents, event, policy=None, **kw):
    sim = Simulation(world, agents, policy or RuleBasedPolicy(), seed=0, **kw)
    sim.events = {1: event}
    return sim


def test_event_schedule_depends_only_on_seed(world):
    a, b = generate_events(world, 7), generate_events(world, 7)
    assert a == b and len(a) == world.cycles
    assert a != generate_events(world, 8)
    for e in a:
        if e.kind is EventKind.DISPUTE:
            assert len(set(e.involved)) == 3 and e.involved[2].startswith("mid-")


def test_full_run_is_deterministic(world, agents):
    runs = []
    for _ in range(2):
        sim = Simulation(world, agents, RuleBasedPolicy(), seed=3)
        runs.append((sim.run(), sim.decisions))
    assert len(runs[0][0]) == world.cycles
    assert runs[0] == runs[1]
    assert runs[0][1], "decisions must be logged"


def test_engine_does_not_mutate_input_agents(world, agents):
    before = [a.model_dump() for a in agents]
    Simulation(world, agents, RuleBasedPolicy(), seed=0).run()
    assert [a.model_dump() for a in agents] == before


def test_crack_warning_leads_to_evacuation(world, agents):
    for a in agents:
        if a.group is Group.HIGH:
            a.competences[Competence.CRACK_LISTENING] = 1.0
    world = world.model_copy(update={"economy": world.economy.model_copy(update={"detection_rate": 1.0})})
    crack = Event(id="c", kind=EventKind.CRACK, cycle=1, involved=["high-1"], severity=0.5)
    rec = sim_with_event(world, agents, crack).step()
    assert rec.detectors and rec.evacuation_motion and rec.evacuated
    assert not rec.collapse
    assert all(v == 0 for v in rec.mined.values())
    assert rec.quota_missed == []  # waived when evacuated


def test_undetected_crack_collapses(world, agents):
    for a in agents:
        a.competences[Competence.CRACK_LISTENING] = 0.0
    crack = Event(id="c", kind=EventKind.CRACK, cycle=1, involved=["high-1"], severity=0.5)
    sim = sim_with_event(world, agents, crack)
    rec = sim.step()
    assert rec.detectors == [] and not rec.evacuation_motion and rec.collapse
    assert sim.injured_until["high-1"] > 1


class LoudPolicy(RuleBasedPolicy):
    def messages(self, agent, obs):
        return [Message(sender=agent.id, venue=Venue.HALL, text="hello")]


def test_silence_violations_recorded(world, agents):
    quiet = Event(id="q", kind=EventKind.NEW_VEIN, cycle=1, severity=0.2)
    rec = sim_with_event(world, agents, quiet, LoudPolicy()).step()
    assert sorted(rec.silence_violations) == sorted(a.id for a in agents if a.group is Group.HIGH)


class ReformPolicy(RuleBasedPolicy):
    def __init__(self, yes_votes):
        self.yes_votes = yes_votes

    def propose(self, agent, obs):
        if agent.id != "mid-1":
            return None
        law = next(l for l in obs.laws if l.kind == "silence")
        return Proposal(law_id=law.id, new_law=law.model_copy(update={"silenced": set()}))

    def vote_proposal(self, agent, obs, proposal):
        return Vote(approve=agent.id in self.yes_votes)


@pytest.mark.parametrize("yes, passes", [({"low-1", "low-2"}, True), ({"low-1"}, False)])
def test_law_revision_needs_two_thirds(world, agents, yes, passes):
    ev = Event(id="v", kind=EventKind.NEW_VEIN, cycle=1, severity=0.2)
    sim = sim_with_event(world, agents, ev, ReformPolicy(yes))
    rec = sim.step()
    assert rec.proposals == ["silence"]
    assert (rec.law_changes == ["amend:silence"]) is passes
    assert (sim.laws["silence"].silenced == set()) is passes


def test_distribution_priority_under_scarcity(world, agents):
    blight = Event(id="b", kind=EventKind.BLIGHT, cycle=1, severity=0.9)
    sim = sim_with_event(world, agents, blight)
    sim.store = {Resource.ENERGY: 0.0, Resource.FOOD: 4.0}
    for a in sim.group(Group.HIGH):  # nothing mined, so only the store feeds people
        a.competences[Competence.MINING] = 0.0
    sim.world = world.model_copy(update={"economy": world.economy.model_copy(update={"base_yield": 1e-9})})
    rec = sim.step()
    assert all(rec.rations[a.id] == 1.0 for a in sim.group(Group.LOW))
    assert all(rec.rations[a.id] < 1.0 for a in sim.group(Group.HIGH))


class RevisingPolicy(RuleBasedPolicy):
    def revise(self, agent, obs):
        profile = {d: {p: 1 / len(ps) for p in ps} for d, ps in POLES.items()}
        return ValueRevision(principles=[], self_report=profile)


@pytest.mark.parametrize("adaptive", [False, True])
def test_value_revision_only_when_adaptive(world, agents, adaptive):
    ev = Event(id="v", kind=EventKind.NEW_VEIN, cycle=1, severity=0.2)
    sim = sim_with_event(world, agents, ev, RevisingPolicy(), adaptive=adaptive)
    rec = sim.step()
    expected = 1 if adaptive else 0
    assert all(a.values.version == expected for a in sim.agents.values())
    assert len(rec.value_revisions) == (len(agents) if adaptive else 0)


def test_false_alarm_evacuates_without_crack_and_is_hidden(world, agents):
    for a in agents:
        a.competences[Competence.CRACK_LISTENING] = 0.0  # always imagines cracks
    world = world.model_copy(update={"economy": world.economy.model_copy(update={"false_alarm_rate": 1.0})})
    quiet = Event(id="q", kind=EventKind.NEW_VEIN, cycle=1, severity=0.2)
    sim = sim_with_event(world, agents, quiet)
    rec = sim.step()
    assert not rec.crack and rec.false_alarms and rec.evacuated and not rec.collapse
    seen = sim._observe("low-1", rec, last=rec).last_cycle
    assert seen.false_alarms == [] and seen.detectors == [] and seen.evacuated


class SharePolicy(RuleBasedPolicy):
    def __init__(self, share):
        self.share = share

    def vote_surplus(self, agent, obs, surplus):
        return Vote(approve=self.share)


@pytest.mark.parametrize("share", [True, False])
def test_surplus_vote(world, agents, share):
    quiet = Event(id="q", kind=EventKind.NEW_VEIN, cycle=1, severity=0.2)
    sim = sim_with_event(world, agents, quiet, SharePolicy(share))
    sim.store = {Resource.ENERGY: 0.0, Resource.FOOD: 24.0}
    rec = sim.step()
    assert rec.surplus > world.economy.surplus_vote_min
    assert rec.surplus_distributed is share
    spoil = 1 - world.economy.spoilage[Resource.FOOD]
    expected_store = 0.0 if share else rec.surplus * spoil
    assert sim.store[Resource.FOOD] == pytest.approx(expected_store)


class CrackClaimPolicy(RuleBasedPolicy):
    """Everyone claims a crack in a private message to the first Council member."""

    def messages(self, agent, obs):
        to = next(c for c in obs.council if c != agent.id)
        return [Message(sender=agent.id, venue=Venue.PRIVATE, to=to, text="Crack!", reports_crack=True)]


def test_only_those_who_heard_can_raise_a_crack_alarm(world, agents):
    for a in agents:
        a.competences[Competence.CRACK_LISTENING] = 0.0
    quiet = Event(id="q", kind=EventKind.NEW_VEIN, cycle=1, severity=0.2)
    sim = sim_with_event(world, agents, quiet, CrackClaimPolicy())
    sim.world = world.model_copy(update={"economy": world.economy.model_copy(update={"false_alarm_rate": 0.0})})  # nobody hears
    rec = sim.step()
    assert not rec.evacuation_motion
    assert sorted(rec.unfounded_crack_reports) == sorted(a.id for a in agents)
    assert sim._observe("low-1", rec, last=rec).last_cycle.unfounded_crack_reports == []
    logged = [d for d in sim.decisions if d.action == "message"]
    assert all(d.details["claimed_crack"] and not d.details["reports_crack"] for d in logged)


def test_false_alarm_hearer_can_still_raise_alarm(world, agents):
    for a in agents:
        a.competences[Competence.CRACK_LISTENING] = 0.0
    world = world.model_copy(update={"economy": world.economy.model_copy(update={"false_alarm_rate": 1.0})})
    quiet = Event(id="q", kind=EventKind.NEW_VEIN, cycle=1, severity=0.2)
    rec = sim_with_event(world, agents, quiet, CrackClaimPolicy()).step()
    assert rec.evacuation_motion
    assert not any(a.startswith("high") for a in rec.unfounded_crack_reports)  # High-voices heard (falsely)
